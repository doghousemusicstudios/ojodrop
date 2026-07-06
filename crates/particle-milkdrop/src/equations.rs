// EEL2 per-frame equation evaluator for MilkDrop presets.
//
// Targets Butterchurn's ns-eel2 semantics (milkdrop-eel-parser grammar +
// presetBase.js runtime). All values are JS doubles → we use f64.
//
// Supports the subset actually used by presets:
//   - Variable assignment:   var = expr
//   - Compound assignment:   var += -= *= /= %= expr
//   - Buffer access/assign:  megabuf(i), gmegabuf(i), megabuf(i) = v, ...
//   - Arithmetic:            + - * / % ^ (^ = power, not xor)
//   - Comparison:            < > <= >= == !=  (return 0.0 or 1.0, EPSILON for ==/!=)
//   - Logical:               && ||  !
//   - Control flow:          if(c,t,f), exec2/exec3, loop(n,..), while(..)
//   - Functions:             above, below, equal, bnot, band, bor, abs, sin, cos,
//                            tan, asin, acos, atan, atan2, sqrt, sqr, pow, log,
//                            log10, exp, min, max, floor, ceil, int, sign, clamp,
//                            lerp, rand, randint, bitor, bitand, sigmoid,
//                            megabuf, gmegabuf
//   - Comments:              // to end of line

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

pub type Env = HashMap<String, f64>;

/// EEL value comparison epsilon (ns-eel2 uses 1e-5 for ==/!=/if/bnot etc.).
const EPS: f64 = 1e-5;
/// Maximum addressable megabuf/gmegabuf index (Butterchurn pre-fills 1<<20).
const MEGABUF_MAX: i64 = 1_048_576;
/// Per-loop cap plus per-program cumulative budget for loop/while. Butterchurn's
/// 1<<20 guard is too large for a render-thread evaluator; normal MilkDrop loops
/// are tiny, and hostile/buggy presets should yield quickly.
const LOOP_CAP: u64 = 16_384;
const LOOP_ITERATION_BUDGET: u64 = 16_384;
const EVAL_DEPTH_CAP: u32 = 256;
const PARSE_DEPTH_CAP: u32 = 256;

/// Sparse backing store for megabuf / gmegabuf (avoids 8 MB dense allocs).
#[derive(Default)]
pub struct MegaBuf {
    map: HashMap<i64, f64>,
}

impl MegaBuf {
    fn read(&self, idx: f64) -> f64 {
        let i = idx.floor() as i64;
        if i < 0 || i >= MEGABUF_MAX {
            return 0.0;
        }
        self.map.get(&i).copied().unwrap_or(0.0)
    }
    fn write(&mut self, idx: f64, v: f64) -> f64 {
        let i = idx.floor() as i64;
        if i >= 0 && i < MEGABUF_MAX {
            self.map.insert(i, v);
        }
        v
    }
}

/// Runtime state threaded through an EelProgram run:
///   - `megabuf` is PER-POOL (each per-frame / shape / wave context has its own).
///   - `gmegabuf` is SHARED across the whole preset (clone the Rc into each pool).
pub struct EelState {
    pub megabuf: MegaBuf,
    pub gmegabuf: Rc<RefCell<MegaBuf>>,
}

impl EelState {
    /// New per-pool state with a private gmegabuf (use [`with_gmegabuf`] to share).
    pub fn new() -> Self {
        Self {
            megabuf: MegaBuf::default(),
            gmegabuf: Rc::new(RefCell::new(MegaBuf::default())),
        }
    }
    /// New per-pool state sharing the given preset-wide gmegabuf.
    pub fn with_gmegabuf(gmegabuf: Rc<RefCell<MegaBuf>>) -> Self {
        Self {
            megabuf: MegaBuf::default(),
            gmegabuf,
        }
    }
}

impl Default for EelState {
    fn default() -> Self {
        Self::new()
    }
}

// ── AST ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Expr {
    Num(f64),
    Var(String),
    Assign(String, Box<Expr>),
    /// megabuf/gmegabuf write: (is_global, index, value). Returns value.
    BufAssign(bool, Box<Expr>, Box<Expr>),
    BinOp(BinKind, Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Not(Box<Expr>),
    Call(String, Vec<Expr>),
}

#[derive(Debug, Clone, Copy)]
enum BinKind {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Lt,
    Gt,
    Le,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}

impl BinKind {
    fn from_assign_op(op: &str) -> Option<BinKind> {
        match op {
            "+=" => Some(BinKind::Add),
            "-=" => Some(BinKind::Sub),
            "*=" => Some(BinKind::Mul),
            "/=" => Some(BinKind::Div),
            "%=" => Some(BinKind::Mod),
            _ => None,
        }
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

pub struct EelProgram {
    stmts: Vec<Expr>,
}

impl EelProgram {
    pub fn parse(src: &str) -> Self {
        let stmts = Parser::new(src).parse_program();
        Self { stmts }
    }

    /// Run with a throwaway per-call megabuf/gmegabuf (back-compat path: presets
    /// that don't use buffers behave identically; buffer state is not persisted).
    /// Used by unit tests; the renderer always uses [`run_with`].
    #[allow(dead_code)]
    pub fn run(&self, env: &mut Env) {
        let mut state = EelState::new();
        self.run_with(env, &mut state);
    }

    /// Run with explicit, caller-owned buffer state (megabuf persists across
    /// frames for a pool; gmegabuf is shared across pools).
    pub fn run_with(&self, env: &mut Env, state: &mut EelState) {
        let mut budget = EvalBudget::default();
        for s in &self.stmts {
            eval(s, env, state, &mut budget);
        }
    }
}

// ── Evaluator ────────────────────────────────────────────────────────────────

struct EvalBudget {
    depth: u32,
    remaining_loop_iters: u64,
}

impl Default for EvalBudget {
    fn default() -> Self {
        Self {
            depth: 0,
            remaining_loop_iters: LOOP_ITERATION_BUDGET,
        }
    }
}

impl EvalBudget {
    fn enter(&mut self) -> bool {
        if self.depth >= EVAL_DEPTH_CAP {
            return false;
        }
        self.depth += 1;
        true
    }

    fn exit(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    fn spend_loop_iter(&mut self) -> bool {
        if self.remaining_loop_iters == 0 {
            return false;
        }
        self.remaining_loop_iters -= 1;
        true
    }
}

fn eval(e: &Expr, env: &mut Env, st: &mut EelState, budget: &mut EvalBudget) -> f64 {
    if !budget.enter() {
        return 0.0;
    }
    let value = match e {
        Expr::Num(v) => *v,
        Expr::Var(n) => *env.get(n.as_str()).unwrap_or(&0.0),
        Expr::Assign(n, rhs) => {
            let v = eval(rhs, env, st, budget);
            env.insert(n.clone(), v);
            v
        }
        Expr::BufAssign(is_global, idx, val) => {
            let i = eval(idx, env, st, budget);
            let v = eval(val, env, st, budget);
            if *is_global {
                st.gmegabuf.borrow_mut().write(i, v)
            } else {
                st.megabuf.write(i, v)
            }
        }
        Expr::Neg(e) => -eval(e, env, st, budget),
        Expr::Not(e) => {
            if eval(e, env, st, budget).abs() > EPS {
                0.0
            } else {
                1.0
            }
        }
        Expr::BinOp(op, l, r) => {
            let lv = eval(l, env, st, budget);
            let rv = eval(r, env, st, budget);
            match op {
                BinKind::Add => lv + rv,
                BinKind::Sub => lv - rv,
                BinKind::Mul => lv * rv,
                // ns-eel2 div(): y==0 ? 0 : x/y
                BinKind::Div => {
                    if rv == 0.0 {
                        0.0
                    } else {
                        lv / rv
                    }
                }
                // ns-eel2 mod(): INTEGER mod — y==0 ? 0 : floor(x) % floor(y)
                BinKind::Mod => {
                    let d = rv.floor();
                    // checked_rem avoids the `i64::MIN % -1` overflow panic (debug/test builds);
                    // 0 is the correct result for any `x % -1`.
                    if d == 0.0 {
                        0.0
                    } else {
                        (lv.floor() as i64).checked_rem(d as i64).unwrap_or(0) as f64
                    }
                }
                BinKind::Pow => lv.powf(rv),
                BinKind::Lt => (lv < rv) as i32 as f64,
                BinKind::Gt => (lv > rv) as i32 as f64,
                BinKind::Le => (lv <= rv) as i32 as f64,
                BinKind::Ge => (lv >= rv) as i32 as f64,
                // ns-eel2 ==/!= use EPSILON tolerance
                BinKind::Eq => ((lv - rv).abs() < EPS) as i32 as f64,
                BinKind::Ne => ((lv - rv).abs() >= EPS) as i32 as f64,
                BinKind::And => ((lv != 0.0) && (rv != 0.0)) as i32 as f64,
                BinKind::Or => ((lv != 0.0) || (rv != 0.0)) as i32 as f64,
            }
        }
        Expr::Call(name, args) => eval_call_node(name, args, env, st, budget),
    };
    budget.exit();
    value
}

/// Handles control-flow functions that need LAZY / ordered evaluation, then
/// falls back to eager math functions in `eval_call`.
fn eval_call_node(
    name: &str,
    args: &[Expr],
    env: &mut Env,
    st: &mut EelState,
    budget: &mut EvalBudget,
) -> f64 {
    match name {
        // if(c, t, f): only the taken branch runs (side-effect safety).
        "if" | "If" | "IF" => {
            let c = eval(args.first().unwrap_or(&Expr::Num(0.0)), env, st, budget);
            if c.abs() > EPS {
                eval(args.get(1).unwrap_or(&Expr::Num(0.0)), env, st, budget)
            } else {
                eval(args.get(2).unwrap_or(&Expr::Num(0.0)), env, st, budget)
            }
        }
        // exec2(a, b) → eval in order, return last. exec3(a, b, c) likewise.
        "exec2" | "exec3" => {
            let mut last = 0.0;
            for a in args {
                last = eval(a, env, st, budget);
            }
            last
        }
        // loop(n, body...) → for(i=0; i<n; i++) body. n evaluated ONCE.
        // Multi-statement bodies parse as args[1..].
        "loop" => {
            let n = eval(args.first().unwrap_or(&Expr::Num(0.0)), env, st, budget);
            let mut last = 0.0;
            let mut i = 0u64;
            // Butterchurn compares i<n with float n (loop(3.9,..) → 4 iters).
            while (i as f64) < n && i < LOOP_CAP && budget.spend_loop_iter() {
                for a in &args[1..] {
                    last = eval(a, env, st, budget);
                }
                i += 1;
            }
            last
        }
        // while(body...) → run body; repeat while |last|>1e-5, hard cap.
        "while" => {
            let mut last = 0.0;
            let mut c = 0u64;
            loop {
                if !budget.spend_loop_iter() {
                    break;
                }
                for a in args {
                    last = eval(a, env, st, budget);
                }
                c += 1;
                if last.abs() <= EPS || c >= LOOP_CAP {
                    break;
                }
            }
            last
        }
        // megabuf/gmegabuf READ (write form is handled via BufAssign at parse).
        "megabuf" => {
            let i = eval(args.first().unwrap_or(&Expr::Num(0.0)), env, st, budget);
            st.megabuf.read(i)
        }
        "gmegabuf" => {
            let i = eval(args.first().unwrap_or(&Expr::Num(0.0)), env, st, budget);
            st.gmegabuf.borrow().read(i)
        }
        // Everything else: eager args.
        _ => {
            let a: Vec<f64> = args.iter().map(|e| eval(e, env, st, budget)).collect();
            eval_call(name, &a)
        }
    }
}

fn eval_call(name: &str, a: &[f64]) -> f64 {
    let get = |i: usize| a.get(i).copied().unwrap_or(0.0);
    match name {
        "above" => (get(0) > get(1)) as i32 as f64,
        "below" => (get(0) < get(1)) as i32 as f64,
        "equal" => ((get(0) - get(1)).abs() < EPS) as i32 as f64,
        // ns-eel2 div(x,y): y==0 ? 0 : x/y (matches the `/` BinKind::Div semantics).
        // Butterchurn's JS transpiler emits `div(a,b)` for `a/b` (EEL division).
        "div" => {
            let y = get(1);
            if y == 0.0 {
                0.0
            } else {
                get(0) / y
            }
        }
        // ns-eel2 mod(x,y): INTEGER mod — y==0 ? 0 : floor(x) % floor(y).
        // Mirrors the `%` BinKind::Mod logic exactly.
        "mod" => {
            let d = get(1).floor();
            // checked_rem avoids the `i64::MIN % -1` overflow panic; 0 is correct for `x % -1`.
            if d == 0.0 {
                0.0
            } else {
                (get(0).floor() as i64).checked_rem(d as i64).unwrap_or(0) as f64
            }
        }
        // ns-eel2 boolean ops (bnot was missing → ORB's edge-detect collapsed q3,
        // killing the warp-feedback tunnel accumulation).
        "bnot" => (get(0).abs() < EPS) as i32 as f64,
        "band" => ((get(0) != 0.0) && (get(1) != 0.0)) as i32 as f64,
        "bor" => ((get(0) != 0.0) || (get(1) != 0.0)) as i32 as f64,
        "abs" => get(0).abs(),
        "sin" => get(0).sin(),
        "cos" => get(0).cos(),
        "tan" => get(0).tan(),
        "asin" => get(0).asin(),
        "acos" => get(0).acos(),
        "atan" => get(0).atan(),
        "atan2" => get(0).atan2(get(1)),
        // ns-eel2 sqrt(): sqrt(abs(x))
        "sqrt" => get(0).abs().sqrt(),
        "invsqrt" => {
            let s = get(0).sqrt();
            if s == 0.0 {
                0.0
            } else {
                1.0 / s
            }
        }
        "sqr" => get(0) * get(0),
        "pow" => {
            let z = get(0).powf(get(1));
            if z.is_finite() {
                z
            } else {
                0.0
            }
        }
        "exp" => get(0).exp(),
        "log" => get(0).ln(),
        "log10" => get(0).log10(),
        "min" => get(0).min(get(1)),
        "max" => get(0).max(get(1)),
        "floor" => get(0).floor(),
        "ceil" => get(0).ceil(),
        "int" => get(0).trunc(),
        // ns-eel2 sign(): x>0?1 : x<0?-1 : 0  (signum() returns ±1 at 0 — differs)
        "sign" => {
            let x = get(0);
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        "clamp" => get(0).clamp(get(1), get(2)),
        "lerp" => get(0) + (get(1) - get(0)) * get(2),
        // ns-eel2 sigmoid(x,y): t=1+exp(-x*y); |t|>1e-5 ? 1/t : 0
        "sigmoid" => {
            let t = 1.0 + (-get(0) * get(1)).exp();
            if t.abs() > EPS {
                1.0 / t
            } else {
                0.0
            }
        }
        "bitor" => ((get(0).floor() as i64) | (get(1).floor() as i64)) as f64,
        "bitand" => ((get(0).floor() as i64) & (get(1).floor() as i64)) as f64,
        "rand" => rand_eel(get(0)),
        "randint" => rand_eel(get(0)).floor(),
        _ => 0.0,
    }
}

/// ns-eel2 rand(x): xf=floor(x); xf<1 ? random() : random()*xf  (in [0, xf)).
fn rand_eel(x: f64) -> f64 {
    static SEED: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(0x123456789abcdef0);
    let old = SEED.load(std::sync::atomic::Ordering::Relaxed);
    let new = old
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    SEED.store(new, std::sync::atomic::Ordering::Relaxed);
    // Uniform [0,1) from the high bits.
    let u = (new >> 11) as f64 / (1u64 << 53) as f64;
    let xf = x.floor();
    if xf < 1.0 {
        u
    } else {
        u * xf
    }
}

// ── Tokenizer ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Op(String), // multi-char operators stored as string
    LParen,
    RParen,
    Comma,
    Semi,
    Eof,
}

struct Lexer {
    chars: Vec<char>,
    pos: usize,
}

impl Lexer {
    fn new(src: &str) -> Self {
        Self {
            chars: src.chars().collect(),
            pos: 0,
        }
    }

    fn peek(&self) -> char {
        self.chars.get(self.pos).copied().unwrap_or('\0')
    }
    fn next(&mut self) -> char {
        let c = self.peek();
        self.pos += 1;
        c
    }
    fn eat_if(&mut self, c: char) -> bool {
        if self.peek() == c {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            while self.peek().is_ascii_whitespace() {
                self.next();
            }
            if self.peek() == '/' && self.chars.get(self.pos + 1) == Some(&'/') {
                while self.peek() != '\n' && self.peek() != '\0' {
                    self.next();
                }
            } else {
                break;
            }
        }
    }

    fn next_tok(&mut self) -> Tok {
        self.skip_whitespace_and_comments();
        let c = self.peek();
        if c == '\0' {
            return Tok::Eof;
        }

        // Number
        if c.is_ascii_digit()
            || (c == '.'
                && self
                    .chars
                    .get(self.pos + 1)
                    .map(|x| x.is_ascii_digit())
                    .unwrap_or(false))
        {
            let start = self.pos;
            while self.peek().is_ascii_digit() || self.peek() == '.' {
                self.next();
            }
            if self.peek() == 'e' || self.peek() == 'E' {
                self.next();
                if self.peek() == '+' || self.peek() == '-' {
                    self.next();
                }
                while self.peek().is_ascii_digit() {
                    self.next();
                }
            }
            let s: String = self.chars[start..self.pos].iter().collect();
            return Tok::Num(s.parse().unwrap_or(0.0));
        }

        // Identifier
        if c.is_ascii_alphabetic() || c == '_' {
            let start = self.pos;
            while self.peek().is_ascii_alphanumeric() || self.peek() == '_' {
                self.next();
            }
            let s: String = self.chars[start..self.pos].iter().collect();
            return Tok::Ident(s);
        }

        self.next(); // consume c
        match c {
            '(' => Tok::LParen,
            ')' => Tok::RParen,
            ',' => Tok::Comma,
            ';' => Tok::Semi,
            // Compound-assignment operators desugar in the parser.
            '+' => {
                if self.eat_if('=') {
                    Tok::Op("+=".into())
                } else {
                    Tok::Op("+".into())
                }
            }
            '-' => {
                if self.eat_if('=') {
                    Tok::Op("-=".into())
                } else {
                    Tok::Op("-".into())
                }
            }
            '*' => {
                if self.eat_if('=') {
                    Tok::Op("*=".into())
                } else {
                    Tok::Op("*".into())
                }
            }
            '/' => {
                if self.eat_if('=') {
                    Tok::Op("/=".into())
                } else {
                    Tok::Op("/".into())
                }
            }
            '%' => {
                if self.eat_if('=') {
                    Tok::Op("%=".into())
                } else {
                    Tok::Op("%".into())
                }
            }
            '^' => Tok::Op("^".into()),
            '!' => {
                if self.eat_if('=') {
                    Tok::Op("!=".into())
                } else {
                    Tok::Op("!".into())
                }
            }
            '<' => {
                if self.eat_if('=') {
                    Tok::Op("<=".into())
                } else {
                    Tok::Op("<".into())
                }
            }
            '>' => {
                if self.eat_if('=') {
                    Tok::Op(">=".into())
                } else {
                    Tok::Op(">".into())
                }
            }
            '=' => {
                if self.eat_if('=') {
                    Tok::Op("==".into())
                } else {
                    Tok::Op("=".into())
                }
            }
            '&' => {
                if self.eat_if('&') {
                    Tok::Op("&&".into())
                } else {
                    Tok::Op("&".into())
                }
            }
            '|' => {
                if self.eat_if('|') {
                    Tok::Op("||".into())
                } else {
                    Tok::Op("|".into())
                }
            }
            // Ternary punctuation (handled by parse_ternary).
            '?' => Tok::Op("?".into()),
            ':' => Tok::Op(":".into()),
            // Ignore unknown chars silently
            _ => self.next_tok(),
        }
    }
}

// ── Parser ───────────────────────────────────────────────────────────────────

struct Parser {
    tokens: Vec<Tok>,
    pos: usize,
    parse_depth: u32,
}

fn is_assign_op(s: &str) -> bool {
    matches!(s, "=" | "+=" | "-=" | "*=" | "/=" | "%=")
}

impl Parser {
    fn new(src: &str) -> Self {
        let mut lex = Lexer::new(src);
        let mut tokens = Vec::new();
        loop {
            let t = lex.next_tok();
            let done = t == Tok::Eof;
            tokens.push(t);
            if done {
                break;
            }
        }
        Self {
            tokens,
            pos: 0,
            parse_depth: 0,
        }
    }

    fn peek(&self) -> &Tok {
        self.tokens.get(self.pos).unwrap_or(&Tok::Eof)
    }
    fn peek2(&self) -> &Tok {
        self.tokens.get(self.pos + 1).unwrap_or(&Tok::Eof)
    }

    fn eat(&mut self) -> Tok {
        let t = self.tokens.get(self.pos).cloned().unwrap_or(Tok::Eof);
        self.pos += 1;
        t
    }

    fn eat_semi(&mut self) {
        while matches!(self.peek(), Tok::Semi) {
            self.eat();
        }
    }

    fn enter_parse(&mut self) -> bool {
        if self.parse_depth >= PARSE_DEPTH_CAP {
            return false;
        }
        self.parse_depth += 1;
        true
    }

    fn exit_parse(&mut self) {
        self.parse_depth = self.parse_depth.saturating_sub(1);
    }

    fn skip_depth_limited_expr(&mut self) {
        let mut depth = 0i32;
        while !matches!(self.peek(), Tok::Eof) {
            match self.peek() {
                Tok::Comma | Tok::Semi if depth == 0 => break,
                Tok::RParen if depth == 0 => break,
                Tok::LParen => {
                    depth += 1;
                    self.eat();
                }
                Tok::RParen => {
                    depth -= 1;
                    self.eat();
                }
                _ => {
                    self.eat();
                }
            }
        }
    }

    fn parse_program(&mut self) -> Vec<Expr> {
        let mut stmts = Vec::new();
        while !matches!(self.peek(), Tok::Eof) {
            self.eat_semi();
            if matches!(self.peek(), Tok::Eof) {
                break;
            }
            let e = self.parse_expr();
            stmts.push(e);
            self.eat_semi();
        }
        stmts
    }

    // Highest-level: assignment (right-assoc). Handles `=` and compound `+= …`.
    fn parse_expr(&mut self) -> Expr {
        if !self.enter_parse() {
            self.skip_depth_limited_expr();
            return Expr::Num(0.0);
        }
        let expr = self.parse_expr_inner();
        self.exit_parse();
        expr
    }

    fn parse_expr_inner(&mut self) -> Expr {
        // Look-ahead: `IDENT <assign-op> ...`  (plain var assignment).
        if let Tok::Ident(name) = self.peek().clone() {
            if let Tok::Op(op) = self.peek2().clone() {
                if is_assign_op(&op) {
                    let name = name.clone();
                    self.eat(); // ident
                    self.eat(); // assign-op
                    let rhs = self.parse_expr(); // right-associative
                    return match BinKind::from_assign_op(&op) {
                        // x += rhs  →  x = x <op> rhs
                        Some(k) => Expr::Assign(
                            name.clone(),
                            Box::new(Expr::BinOp(k, Box::new(Expr::Var(name)), Box::new(rhs))),
                        ),
                        None => Expr::Assign(name, Box::new(rhs)), // plain '='
                    };
                }
            }
        }

        // Look-ahead: buffer assignment `megabuf(idx) <assign-op> rhs`.
        if let Tok::Ident(name) = self.peek().clone() {
            if (name == "megabuf" || name == "gmegabuf") && matches!(self.peek2(), Tok::LParen) {
                if let Some(close) = self.matching_paren(self.pos + 1) {
                    if let Tok::Op(op) = self.tokens.get(close + 1).cloned().unwrap_or(Tok::Eof) {
                        if is_assign_op(&op) {
                            let is_global = name == "gmegabuf";
                            self.eat(); // ident
                            self.eat(); // '('
                            let idx = self.parse_expr();
                            if matches!(self.peek(), Tok::RParen) {
                                self.eat();
                            }
                            self.eat(); // assign-op
                            let rhs = self.parse_expr();
                            return match BinKind::from_assign_op(&op) {
                                Some(k) => {
                                    // buf(i) op= rhs → buf(i) = buf(i) op rhs
                                    let read = Expr::Call(name.clone(), vec![idx.clone()]);
                                    Expr::BufAssign(
                                        is_global,
                                        Box::new(idx),
                                        Box::new(Expr::BinOp(k, Box::new(read), Box::new(rhs))),
                                    )
                                }
                                None => Expr::BufAssign(is_global, Box::new(idx), Box::new(rhs)),
                            };
                        }
                    }
                }
            }
        }

        self.parse_ternary()
    }

    /// C-style ternary `cond ? a : b`, looser than `||` (the lowest binary level).
    /// Right-associative so `a ? b : c ? d : e` parses as `a ? b : (c ? d : e)`.
    /// Desugars to the existing lazy `if(cond, a, b)` so ONLY the taken branch
    /// evaluates (same side-effect safety as `if`). The JS-transpiled Butterchurn
    /// equations use ternaries heavily (e.g. `.00001<abs(above(d,r))?0:sin(...)`).
    fn parse_ternary(&mut self) -> Expr {
        let cond = self.parse_or();
        if matches!(self.peek(), Tok::Op(s) if s == "?") {
            self.eat(); // '?'
                        // Branches can themselves contain assignments/ternaries.
            let then_branch = self.parse_expr();
            if matches!(self.peek(), Tok::Op(s) if s == ":") {
                self.eat(); // ':'
            }
            let else_branch = self.parse_expr();
            Expr::Call("if".to_string(), vec![cond, then_branch, else_branch])
        } else {
            cond
        }
    }

    /// Given index of an LParen, return index of the matching RParen.
    fn matching_paren(&self, lparen: usize) -> Option<usize> {
        let mut depth = 0i32;
        let mut i = lparen;
        while i < self.tokens.len() {
            match self.tokens[i] {
                Tok::LParen => depth += 1,
                Tok::RParen => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                Tok::Eof => return None,
                _ => {}
            }
            i += 1;
        }
        None
    }

    fn parse_or(&mut self) -> Expr {
        let mut lhs = self.parse_and();
        while matches!(self.peek(), Tok::Op(s) if s == "||") {
            self.eat();
            let rhs = self.parse_and();
            lhs = Expr::BinOp(BinKind::Or, Box::new(lhs), Box::new(rhs));
        }
        lhs
    }

    fn parse_and(&mut self) -> Expr {
        let mut lhs = self.parse_cmp();
        while matches!(self.peek(), Tok::Op(s) if s == "&&") {
            self.eat();
            let rhs = self.parse_cmp();
            lhs = Expr::BinOp(BinKind::And, Box::new(lhs), Box::new(rhs));
        }
        lhs
    }

    fn parse_cmp(&mut self) -> Expr {
        let mut lhs = self.parse_add();
        loop {
            let op = match self.peek() {
                Tok::Op(s) if s == "<" => BinKind::Lt,
                Tok::Op(s) if s == ">" => BinKind::Gt,
                Tok::Op(s) if s == "<=" => BinKind::Le,
                Tok::Op(s) if s == ">=" => BinKind::Ge,
                Tok::Op(s) if s == "==" => BinKind::Eq,
                Tok::Op(s) if s == "!=" => BinKind::Ne,
                _ => break,
            };
            self.eat();
            let rhs = self.parse_add();
            lhs = Expr::BinOp(op, Box::new(lhs), Box::new(rhs));
        }
        lhs
    }

    fn parse_add(&mut self) -> Expr {
        let mut lhs = self.parse_mul();
        loop {
            let op = match self.peek() {
                Tok::Op(s) if s == "+" => BinKind::Add,
                Tok::Op(s) if s == "-" => BinKind::Sub,
                _ => break,
            };
            self.eat();
            let rhs = self.parse_mul();
            lhs = Expr::BinOp(op, Box::new(lhs), Box::new(rhs));
        }
        lhs
    }

    fn parse_mul(&mut self) -> Expr {
        let mut lhs = self.parse_unary();
        loop {
            let op = match self.peek() {
                Tok::Op(s) if s == "*" => BinKind::Mul,
                Tok::Op(s) if s == "/" => BinKind::Div,
                Tok::Op(s) if s == "%" => BinKind::Mod,
                _ => break,
            };
            self.eat();
            let rhs = self.parse_unary();
            lhs = Expr::BinOp(op, Box::new(lhs), Box::new(rhs));
        }
        lhs
    }

    fn parse_unary(&mut self) -> Expr {
        match self.peek() {
            Tok::Op(s) if s == "-" => {
                self.eat();
                Expr::Neg(Box::new(self.parse_unary()))
            }
            Tok::Op(s) if s == "!" => {
                self.eat();
                Expr::Not(Box::new(self.parse_unary()))
            }
            _ => self.parse_pow(),
        }
    }

    fn parse_pow(&mut self) -> Expr {
        let base = self.parse_atom();
        if matches!(self.peek(), Tok::Op(s) if s == "^") {
            self.eat();
            let exp = self.parse_unary(); // right-associative
            Expr::BinOp(BinKind::Pow, Box::new(base), Box::new(exp))
        } else {
            base
        }
    }

    fn parse_atom(&mut self) -> Expr {
        match self.peek().clone() {
            Tok::Num(v) => {
                self.eat();
                Expr::Num(v)
            }
            Tok::LParen => {
                self.eat();
                let e = self.parse_expr();
                if matches!(self.peek(), Tok::RParen) {
                    self.eat();
                }
                e
            }
            Tok::Ident(name) => {
                self.eat();
                if matches!(self.peek(), Tok::LParen) {
                    self.eat(); // consume '('
                    let mut args = Vec::new();
                    while !matches!(self.peek(), Tok::RParen | Tok::Eof) {
                        args.push(self.parse_expr());
                        if matches!(self.peek(), Tok::Comma) {
                            self.eat();
                        }
                    }
                    if matches!(self.peek(), Tok::RParen) {
                        self.eat();
                    }
                    Expr::Call(name, args)
                } else {
                    Expr::Var(name)
                }
            }
            _ => {
                self.eat(); // skip unexpected token
                Expr::Num(0.0)
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn run(src: &str) -> Env {
        let prog = EelProgram::parse(src);
        let mut env = Env::new();
        prog.run(&mut env);
        env
    }

    fn run_st(src: &str, st: &mut EelState) -> Env {
        let prog = EelProgram::parse(src);
        let mut env = Env::new();
        prog.run_with(&mut env, st);
        env
    }

    #[test]
    fn basic_assign() {
        let env = run("x = 3.0 + 4.0;");
        assert_eq!(env["x"], 7.0);
    }

    #[test]
    fn chained() {
        let env = run("x = 2; y = x * 3;");
        assert_eq!(env["y"], 6.0);
    }

    #[test]
    fn power() {
        let env = run("x = 2^10;");
        assert_eq!(env["x"], 1024.0);
    }

    #[test]
    fn if_fn() {
        let e1 = run("x = if(1, 42, 0);");
        let e2 = run("x = if(0, 42, 99);");
        assert_eq!(e1["x"], 42.0);
        assert_eq!(e2["x"], 99.0);
    }

    #[test]
    fn above_below() {
        let env = run("a = above(5, 3); b = below(5, 3);");
        assert_eq!(env["a"], 1.0);
        assert_eq!(env["b"], 0.0);
    }

    #[test]
    fn comments() {
        let env = run("x = 1; // this is a comment\ny = 2;");
        assert_eq!(env["x"], 1.0);
        assert_eq!(env["y"], 2.0);
    }

    #[test]
    fn neg_unary() {
        let env = run("x = -3; y = -x;");
        assert_eq!(env["x"], -3.0);
        assert_eq!(env["y"], 3.0);
    }

    #[test]
    fn spring_step() {
        // Simulate one step of jelly spring
        let src = "
            spring = 18; resist = 5; dt = 0.0003; grav = 1;
            x1 = 0.5; x2 = 0.4; x3 = 0.6; x4 = 0.5;
            y1 = 0.5; y2 = 0.4; y3 = 0.6; y4 = 0.5;
            vx2 = 0; vy2 = 0;
            vx2 = vx2*(1-resist*dt) + dt*((x1+x3-2*x2)*spring);
            vy2 = vy2*(1-resist*dt) + dt*((y1+y3-2*y2)*spring-grav);
            x2 = x2 + vx2;
            y2 = y2 + vy2;
        ";
        let env = run(src);
        assert!(env["vx2"] > 0.0, "vx2={}", env["vx2"]);
        assert!(env["vy2"].is_finite());
    }

    // ── New-feature tests ────────────────────────────────────────────────────

    #[test]
    fn compound_assign() {
        let env = run("x = 5; x += 3;");
        assert_eq!(env["x"], 8.0);
        let env = run("x = 5; x -= 2;");
        assert_eq!(env["x"], 3.0);
        let env = run("x = 5; x *= 4;");
        assert_eq!(env["x"], 20.0);
        let env = run("x = 20; x /= 4;");
        assert_eq!(env["x"], 5.0);
        let env = run("x = 17; x %= 5;");
        assert_eq!(env["x"], 2.0);
    }

    #[test]
    fn compound_assign_accumulate() {
        // The 82%-of-corpus pattern: an accumulator updated each statement.
        let env = run("t = 0; t += 1; t += 1; t += 0.5;");
        assert_eq!(env["t"], 2.5);
    }

    #[test]
    fn compound_assign_does_not_drop() {
        // Prior bug: `x += 1` was parsed as `x` then `+= 1` dropped → x stayed 0.
        let env = run("x = 10; x += 1;");
        assert_eq!(env["x"], 11.0, "compound-assign was dropped");
    }

    #[test]
    fn if_laziness_skips_untaken_branch() {
        // Only the taken branch should execute its side effects.
        let env = run(
            "taken = 0; skipped = 0; r = if(1, exec2(taken = 1, 100), exec2(skipped = 1, 200));",
        );
        assert_eq!(env["r"], 100.0);
        assert_eq!(env["taken"], 1.0);
        assert_eq!(env["skipped"], 0.0, "untaken branch ran its side effect");

        let env = run(
            "taken = 0; skipped = 0; r = if(0, exec2(taken = 1, 100), exec2(skipped = 1, 200));",
        );
        assert_eq!(env["r"], 200.0);
        assert_eq!(env["taken"], 0.0, "untaken branch ran its side effect");
        assert_eq!(env["skipped"], 1.0);
    }

    #[test]
    fn megabuf_round_trip() {
        let mut st = EelState::new();
        let env = run_st("megabuf(5) = 42; x = megabuf(5);", &mut st);
        assert_eq!(env["x"], 42.0);
        // Out-of-range reads → 0.
        let env = run_st("x = megabuf(-1); y = megabuf(2000000);", &mut st);
        assert_eq!(env["x"], 0.0);
        assert_eq!(env["y"], 0.0);
    }

    #[test]
    fn megabuf_persists_across_runs() {
        let mut st = EelState::new();
        run_st("megabuf(3) = 7;", &mut st);
        let env = run_st("x = megabuf(3);", &mut st);
        assert_eq!(env["x"], 7.0);
    }

    #[test]
    fn megabuf_compound_assign() {
        let mut st = EelState::new();
        let env = run_st("megabuf(2) = 10; megabuf(2) += 5; x = megabuf(2);", &mut st);
        assert_eq!(env["x"], 15.0);
    }

    #[test]
    fn gmegabuf_shared() {
        let g = Rc::new(RefCell::new(MegaBuf::default()));
        let mut st1 = EelState::with_gmegabuf(g.clone());
        let mut st2 = EelState::with_gmegabuf(g.clone());
        run_st("gmegabuf(1) = 99;", &mut st1);
        let env = run_st("x = gmegabuf(1);", &mut st2);
        assert_eq!(env["x"], 99.0, "gmegabuf not shared across pools");
    }

    #[test]
    fn while_loop_counts() {
        // Decrement i until 0; n counts iterations. while repeats while |last|>1e-5.
        let env = run("i = 5; n = 0; while(exec2(n = n + 1, i = i - 1));");
        assert_eq!(env["i"], 0.0);
        assert_eq!(env["n"], 5.0);
    }

    #[test]
    fn loop_runs_n_times() {
        let env = run("n = 0; loop(4, n = n + 1);");
        assert_eq!(env["n"], 4.0);
        // loop(3.9,..) runs 4 times (i<n with float n).
        let env = run("n = 0; loop(3.9, n = n + 1);");
        assert_eq!(env["n"], 4.0);
    }

    #[test]
    fn loop_multi_statement_body() {
        let env = run("a = 0; b = 0; loop(3, a = a + 1, b = b + 2);");
        assert_eq!(env["a"], 3.0);
        assert_eq!(env["b"], 6.0);
    }

    #[test]
    fn loops_are_capped_by_cumulative_render_budget() {
        let env = run("n = 0; loop(1048576, n = n + 1);");
        assert_eq!(env["n"], LOOP_ITERATION_BUDGET as f64);

        let env = run("n = 0; loop(1048576, loop(1048576, n = n + 1));");
        assert_eq!(
            env["n"],
            (LOOP_ITERATION_BUDGET - 1) as f64,
            "outer loop consumes one iteration budget entry before inner body runs"
        );
    }

    #[test]
    fn deeply_nested_parse_and_eval_are_bounded() {
        let mut src = String::from("x = ");
        for _ in 0..(PARSE_DEPTH_CAP + 64) {
            src.push_str("if(1,");
        }
        src.push('7');
        for _ in 0..(PARSE_DEPTH_CAP + 64) {
            src.push_str(",0)");
        }
        src.push(';');
        let env = run(&src);
        assert!(env.get("x").copied().unwrap_or(0.0).is_finite());
    }

    #[test]
    fn exec2_exec3_return_last() {
        let env = run("x = exec2(10, 20);");
        assert_eq!(env["x"], 20.0);
        let env = run("x = exec3(1, 2, 3);");
        assert_eq!(env["x"], 3.0);
    }

    #[test]
    fn sigmoid_fn() {
        // sigmoid(0,y) = 1/(1+e^0) = 0.5
        let env = run("x = sigmoid(0, 1);");
        assert!((env["x"] - 0.5).abs() < 1e-9, "sigmoid(0,1)={}", env["x"]);
        // Large positive x*y → near 1.
        let env = run("x = sigmoid(10, 1);");
        assert!(env["x"] > 0.999);
    }

    #[test]
    fn integer_mod() {
        // ns-eel2 mod is integer: floor(x) % floor(y).
        let env = run("x = 7.8 % 3.2;");
        assert_eq!(env["x"], 1.0); // 7 % 3 = 1
    }

    #[test]
    fn sqrt_of_negative() {
        // ns-eel2 sqrt(abs(x)).
        let env = run("x = sqrt(-9);");
        assert_eq!(env["x"], 3.0);
    }

    #[test]
    fn bitops() {
        let env = run("a = bitor(5, 2); b = bitand(6, 3);");
        assert_eq!(env["a"], 7.0); // 101 | 010 = 111
        assert_eq!(env["b"], 2.0); // 110 & 011 = 010
    }

    // ── Ternary + div/mod (Butterchurn JS-transpiled equation support) ──────────

    #[test]
    fn ternary_basic() {
        let env = run("x = 1 ? 42 : 99; y = 0 ? 42 : 99;");
        assert_eq!(env["x"], 42.0);
        assert_eq!(env["y"], 99.0);
    }

    #[test]
    fn ternary_with_condition_expr() {
        // The exact shape from sherwin_maxawow's pixel_eqs.
        let env = run("d = 5; r = 3; x1 = .00001 < abs(above(d, r)) ? 0 : 7;");
        // above(5,3)=1, abs(1)=1, .00001<1 → true → 0.
        assert_eq!(env["x1"], 0.0);
        let env = run("d = 2; r = 3; x1 = .00001 < abs(above(d, r)) ? 0 : 7;");
        // above(2,3)=0, abs(0)=0, .00001<0 → false → 7.
        assert_eq!(env["x1"], 7.0);
    }

    #[test]
    fn ternary_nested_right_assoc() {
        // a ? b : c ? d : e  parses as  a ? b : (c ? d : e)
        let env = run("x = 0 ? 1 : 1 ? 2 : 3;");
        assert_eq!(env["x"], 2.0);
        let env = run("x = 0 ? 1 : 0 ? 2 : 3;");
        assert_eq!(env["x"], 3.0);
        let env = run("x = 1 ? 5 : 0 ? 2 : 3;");
        assert_eq!(env["x"], 5.0);
    }

    #[test]
    fn ternary_is_lazy() {
        // Only the taken branch should run its side effects.
        let env =
            run("taken = 0; skipped = 0; r = 1 ? exec2(taken = 1, 100) : exec2(skipped = 1, 200);");
        assert_eq!(env["r"], 100.0);
        assert_eq!(env["taken"], 1.0);
        assert_eq!(
            env["skipped"], 0.0,
            "untaken ternary branch ran its side effect"
        );
    }

    #[test]
    fn ternary_looser_than_comparison() {
        // `1 < 2 ? 10 : 20` must parse as `(1 < 2) ? 10 : 20`, not `1 < (2 ? 10 : 20)`.
        let env = run("x = 1 < 2 ? 10 : 20;");
        assert_eq!(env["x"], 10.0);
    }

    #[test]
    fn div_fn() {
        let env = run("x = div(10, 4);");
        assert_eq!(env["x"], 2.5);
        // div by zero → 0 (ns-eel2 / BinKind::Div semantics).
        let env = run("x = div(10, 0);");
        assert_eq!(env["x"], 0.0);
    }

    #[test]
    fn mod_fn() {
        // mod(x,y) is integer mod: floor(x) % floor(y).
        let env = run("x = mod(7.8, 3.2);");
        assert_eq!(env["x"], 1.0); // 7 % 3 = 1
                                   // mod by zero → 0.
        let env = run("x = mod(7, 0);");
        assert_eq!(env["x"], 0.0);
    }
}
