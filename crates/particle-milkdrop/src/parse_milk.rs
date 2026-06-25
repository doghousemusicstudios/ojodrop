// Parses raw MilkDrop .milk preset files.
//
// The warp and comp shader sections are stored as numbered lines:
//   warp_1=`shader_body
//   warp_2=`{
//   warp_3=`   float3 dx = ...
//   warp_N=`}
//
// Each line has the prefix `warp_N=`` (with literal backtick) and contributes
// one line of shader code. We reassemble them in order and strip the outer
// `shader_body { }` wrapper to get just the inner body code.

use std::collections::BTreeMap;

pub struct MilkShaders {
    pub warp:      Option<String>,
    pub comp:      Option<String>,
    /// When true, `warp`/`comp` hold already-GLSL shader BODIES (Butterchurn
    /// converted-JSON presets), so the renderer compiles them via the GLSL-body
    /// path (glsl_milk_*_to_naga) instead of the HLSL→GLSL path. .milk presets
    /// leave this false (HLSL bodies). Additive — never set for the .milk path.
    pub shaders_glsl: bool,
    pub per_frame: Option<String>,
    /// Per-frame INIT EEL program (per_frame_init_N lines) — run once at load.
    pub per_frame_init: Option<String>,
    /// Per-vertex warp EEL program (per_pixel_N lines).
    pub per_pixel: Option<String>,
    /// 0.0–1.0; default 0.98
    pub decay:     f32,
    /// brightness exponent; default 1.5
    pub gamma_adj: f32,

    // ── Video echo (in-comp-shader feedback look) ─────────────────────────────
    pub echo_zoom:  f32, // fVideoEchoZoom,        default 1.0
    pub echo_alpha: f32, // fVideoEchoAlpha,       default 0.0
    pub echo_orient: f32, // nVideoEchoOrientation, default 0.0

    // ── Comp post-FX flags ────────────────────────────────────────────────────
    pub brighten: bool, // bBrighten
    pub darken:   bool, // bDarken
    pub solarize: bool, // bSolarize
    pub invert:   bool, // bInvert

    // ── Per-frame warp base scalars (overridable by the per-frame EEL program) ──
    pub warpscale:     f32, // fWarpScale,     default 1.0
    pub warpanimspeed: f32, // fWarpAnimSpeed, default 1.0
    pub zoom:    f32, // default 1.0
    pub zoomexp: f32, // default 1.0
    pub rot:     f32, // default 0.0
    pub warp_amount: f32, // `warp` scalar, default 1.0 (renamed to avoid colliding with the warp shader field)
    pub cx:      f32, // default 0.5
    pub cy:      f32, // default 0.5
    pub dx:      f32, // default 0.0
    pub dy:      f32, // default 0.0
    pub sx:      f32, // default 1.0
    pub sy:      f32, // default 1.0
    pub wrap:    bool, // bTexWrap, default true

    // ── Built-in waveform scalars/bools ──────────────────────────────────────
    pub wave_mode:        f32, // nWaveMode
    pub wave_x:           f32, // wave_x
    pub wave_y:           f32, // wave_y
    pub wave_r:           f32, // wave_r
    pub wave_g:           f32, // wave_g
    pub wave_b:           f32, // wave_b
    pub wave_a:           f32, // fWaveAlpha (base alpha)
    pub wave_mystery:     f32, // fWaveParam
    pub wave_scale:       f32, // fWaveScale
    pub wave_smoothing:   f32, // fWaveSmoothing
    pub wave_dots:        bool, // bWaveDots
    pub wave_thick:       bool, // bWaveThick
    pub additive_wave:    bool, // bAdditiveWaves
    pub wave_brighten:    bool, // bMaximizeWaveColor
    pub modwavealphabyvolume: bool, // bModWaveAlphaByVolume
    pub modwavealphastart: f32, // fModWaveAlphaStart
    pub modwavealphaend:   f32, // fModWaveAlphaEnd

    // ── Motion vectors (butterchurn runtime defaults: ON, mv_l 0.9, mv_a 1) ───
    pub mv_on: bool, // bMotionVectorsOn (DEFAULT true — render default frame)
    pub mv_x:  f32,  // nMotionVectorsX, default 12
    pub mv_y:  f32,  // nMotionVectorsY, default 9
    pub mv_dx: f32,  // mv_dx, default 0
    pub mv_dy: f32,  // mv_dy, default 0
    pub mv_l:  f32,  // mv_l,  default 0.9
    pub mv_r:  f32,  // mv_r,  default 1
    pub mv_g:  f32,  // mv_g,  default 1
    pub mv_b:  f32,  // mv_b,  default 1
    pub mv_a:  f32,  // mv_a,  default 1

    // ── Borders (outer/inner colored frame) ──────────────────────────────────
    pub ob_size: f32, pub ob_r: f32, pub ob_g: f32, pub ob_b: f32, pub ob_a: f32,
    pub ib_size: f32, pub ib_r: f32, pub ib_g: f32, pub ib_b: f32, pub ib_a: f32,

    // ── Darken center ─────────────────────────────────────────────────────────
    pub darken_center: bool, // bDarkenCenter

    // ── Blur min/max (per-level range remap; butterchurn b1n/b1x …) ───────────
    pub b1n: f32, pub b1x: f32, // blur1 min/max (default 0 / 1)
    pub b2n: f32, pub b2x: f32, // blur2 min/max
    pub b3n: f32, pub b3x: f32, // blur3 min/max

    // ── Custom shapes (up to 4) ──────────────────────────────────────────────
    pub shapes: Vec<ShapeCode>,
    // ── Custom waveforms (up to 4) ───────────────────────────────────────────
    pub waves:  Vec<CustomWaveDef>,
}

// ── Custom shape definitions ─────────────────────────────────────────────────

#[derive(Clone)]
pub struct ShapeBaseVals {
    pub enabled:      i32,
    pub sides:        f32,
    pub additive:     i32,
    pub thick_outline: i32,
    pub textured:     i32,
    pub num_inst:     i32,
    pub x:            f32,
    pub y:            f32,
    pub rad:          f32,
    pub ang:          f32,
    pub tex_ang:      f32,
    pub tex_zoom:     f32,
    pub r:  f32, pub g:  f32, pub b:  f32, pub a:  f32,
    pub r2: f32, pub g2: f32, pub b2: f32, pub a2: f32,
    pub border_r: f32, pub border_g: f32, pub border_b: f32, pub border_a: f32,
}

impl Default for ShapeBaseVals {
    fn default() -> Self {
        // Mirrors butterchurn shapeBaseValsDefaults (butterchurn.js ~12062).
        ShapeBaseVals {
            enabled: 0,
            sides: 4.0,
            additive: 0,
            thick_outline: 0,
            textured: 0,
            num_inst: 1,
            x: 0.5,
            y: 0.5,
            rad: 0.1,
            ang: 0.0,
            tex_ang: 0.0,
            tex_zoom: 1.0,
            r: 1.0, g: 0.0, b: 0.0, a: 1.0,
            r2: 0.0, g2: 1.0, b2: 0.0, a2: 0.0,
            border_r: 1.0, border_g: 1.0, border_b: 1.0, border_a: 0.1,
        }
    }
}

pub struct ShapeCode {
    pub base:      ShapeBaseVals,
    pub per_frame: Option<String>,
    /// shape_N_per_frame_init lines — run once into the shape env at load.
    pub per_frame_init: Option<String>,
}

// ── Custom waveform definitions ──────────────────────────────────────────────

pub struct CustomWaveDef {
    pub index:     u32,
    pub enabled:   bool,
    pub samples:   u32,
    pub sep:       i32,
    pub spectrum:  bool,
    pub use_dots:  bool,
    pub draw_thick: bool,
    pub additive:  bool,
    pub scaling:   f32,
    pub smoothing: f32,
    pub r: f32, pub g: f32, pub b: f32, pub a: f32,
    pub per_frame: Option<String>,
    /// wave_N_per_frame_init_M lines — run once into the wave env at load.
    pub per_frame_init: Option<String>,
    pub per_point: Option<String>,
}

impl Default for CustomWaveDef {
    fn default() -> Self {
        CustomWaveDef {
            index: 0,
            enabled: false,
            samples: 512,
            sep: 0,
            spectrum: false,
            use_dots: false,
            draw_thick: false,
            additive: false,
            scaling: 1.0,
            smoothing: 0.5,
            r: 1.0, g: 1.0, b: 1.0, a: 1.0,
            per_frame: None,
            per_frame_init: None,
            per_point: None,
        }
    }
}

pub fn parse(content: &str) -> MilkShaders {
    MilkShaders {
        warp:      extract_section(content, "warp_"),
        comp:      extract_section(content, "comp_"),
        shaders_glsl: false, // .milk path = HLSL bodies (unchanged behavior).
        per_frame: extract_per_frame(content),
        per_frame_init: extract_per_frame_init(content),
        per_pixel: extract_per_pixel(content),
        decay:     parse_float(content, "fDecay",    0.98),
        gamma_adj: parse_float(content, "fGammaAdj", 1.5),

        echo_zoom:   parse_float(content, "fVideoEchoZoom",        1.0),
        echo_alpha:  parse_float(content, "fVideoEchoAlpha",       0.0),
        echo_orient: parse_float(content, "nVideoEchoOrientation", 0.0),

        brighten: parse_bool(content, "bBrighten", false),
        darken:   parse_bool(content, "bDarken",   false),
        solarize: parse_bool(content, "bSolarize", false),
        invert:   parse_bool(content, "bInvert",   false),

        warpscale:     parse_float(content, "fWarpScale",     1.0),
        warpanimspeed: parse_float(content, "fWarpAnimSpeed", 1.0),
        zoom:        parse_float(content, "zoom",    1.0),
        zoomexp:     parse_float(content, "zoomexp", 1.0),
        rot:         parse_float(content, "rot",     0.0),
        warp_amount: parse_float(content, "warp",    1.0),
        cx:          parse_float(content, "cx",      0.5),
        cy:          parse_float(content, "cy",      0.5),
        dx:          parse_float(content, "dx",      0.0),
        dy:          parse_float(content, "dy",      0.0),
        sx:          parse_float(content, "sx",      1.0),
        sy:          parse_float(content, "sy",      1.0),
        wrap:        parse_bool (content, "bTexWrap", true),

        wave_mode:      parse_float(content, "nWaveMode",      0.0),
        wave_x:         parse_float(content, "wave_x",         0.5),
        wave_y:         parse_float(content, "wave_y",         0.5),
        wave_r:         parse_float(content, "wave_r",         1.0),
        wave_g:         parse_float(content, "wave_g",         1.0),
        wave_b:         parse_float(content, "wave_b",         1.0),
        wave_a:         parse_float(content, "fWaveAlpha",     1.0),
        wave_mystery:   parse_float(content, "fWaveParam",     0.0),
        wave_scale:     parse_float(content, "fWaveScale",     1.0),
        wave_smoothing: parse_float(content, "fWaveSmoothing", 0.75),
        wave_dots:      parse_bool (content, "bWaveDots",      false),
        wave_thick:     parse_bool (content, "bWaveThick",     false),
        additive_wave:  parse_bool (content, "bAdditiveWaves", false),
        wave_brighten:  parse_bool (content, "bMaximizeWaveColor", false),
        modwavealphabyvolume: parse_bool(content, "bModWaveAlphaByVolume", false),
        modwavealphastart: parse_float(content, "fModWaveAlphaStart", 0.75),
        modwavealphaend:   parse_float(content, "fModWaveAlphaEnd",   0.95),

        // Motion vectors. butterchurn RENDER default frame (visualizer.js): ON,
        // mv_x 12, mv_y 9, mv_l 0.9, mv_a 1 (NOT the blankPreset parse-fallbacks).
        mv_on: parse_bool (content, "bMotionVectorsOn", true),
        mv_x:  parse_float(content, "nMotionVectorsX", 12.0),
        mv_y:  parse_float(content, "nMotionVectorsY", 9.0),
        mv_dx: parse_float(content, "mv_dx", 0.0),
        mv_dy: parse_float(content, "mv_dy", 0.0),
        mv_l:  parse_float(content, "mv_l",  0.9),
        mv_r:  parse_float(content, "mv_r",  1.0),
        mv_g:  parse_float(content, "mv_g",  1.0),
        mv_b:  parse_float(content, "mv_b",  1.0),
        mv_a:  parse_float(content, "mv_a",  1.0),

        // Borders (visualizer.js defaults: ob_size/ib_size 0.01, ob 0, ib 0.25, a 0).
        ob_size: parse_float(content, "ob_size", 0.01),
        ob_r:    parse_float(content, "ob_r", 0.0),
        ob_g:    parse_float(content, "ob_g", 0.0),
        ob_b:    parse_float(content, "ob_b", 0.0),
        ob_a:    parse_float(content, "ob_a", 0.0),
        ib_size: parse_float(content, "ib_size", 0.01),
        ib_r:    parse_float(content, "ib_r", 0.25),
        ib_g:    parse_float(content, "ib_g", 0.25),
        ib_b:    parse_float(content, "ib_b", 0.25),
        ib_a:    parse_float(content, "ib_a", 0.0),

        // Darken center (default off).
        darken_center: parse_bool(content, "bDarkenCenter", false),

        // Blur min/max (butterchurn visualizer.js defaults: min 0, max 1 = identity).
        b1n: parse_float(content, "b1n", 0.0),
        b1x: parse_float(content, "b1x", 1.0),
        b2n: parse_float(content, "b2n", 0.0),
        b2x: parse_float(content, "b2x", 1.0),
        b3n: parse_float(content, "b3n", 0.0),
        b3x: parse_float(content, "b3x", 1.0),

        shapes: parse_shapes(content),
        waves:  parse_waves(content),
    }
}

fn parse_bool(content: &str, key: &str, default: bool) -> bool {
    // Bool keys are stored as 0/1 floats in .milk.
    let v = parse_float(content, key, if default { 1.0 } else { 0.0 });
    v != 0.0
}

fn parse_float(content: &str, key: &str, default: f32) -> f32 {
    for line in content.lines() {
        // Case-insensitive key match: fDecay= or fdecay= etc.
        let lc = line.to_ascii_lowercase();
        let key_lc = key.to_ascii_lowercase();
        if let Some(rest) = lc.strip_prefix(&key_lc) {
            if let Some(val) = rest.strip_prefix('=') {
                if let Ok(v) = val.trim().parse::<f32>() {
                    return v;
                }
            }
        }
    }
    default
}

fn extract_per_frame(content: &str) -> Option<String> {
    let prefix = "per_frame_";
    let mut lines: BTreeMap<u32, String> = BTreeMap::new();
    for line in content.lines() {
        if !line.starts_with(prefix) { continue; }
        // Skip per_frame_init_N lines (they share the per_frame_ prefix) — they
        // are collected separately by extract_per_frame_init. Without this guard
        // the `n.parse()` below failed on "init_1" and silently aborted the whole
        // per-frame collection.
        let rest = &line[prefix.len()..];
        if rest.starts_with("init_") { continue; }
        let eq = match rest.find('=') { Some(e) => e, None => continue };
        let n: u32 = match rest[..eq].parse() { Ok(n) => n, Err(_) => continue };
        lines.insert(n, rest[eq + 1..].to_string());
    }
    if lines.is_empty() { return None; }
    Some(lines.values().cloned().collect::<Vec<_>>().join("\n"))
}

/// Per-frame INIT equations: `per_frame_init_N=<code>`. Run ONCE at preset load
/// (before frame 0) so per-frame equations see initialized vars (q1-q32, regs…).
fn extract_per_frame_init(content: &str) -> Option<String> {
    collect_numbered_eel(content, "per_frame_init_")
}

/// Per-vertex warp equations: `per_pixel_N=<code>`. Mirror of extract_per_frame.
fn extract_per_pixel(content: &str) -> Option<String> {
    let prefix = "per_pixel_";
    let mut lines: BTreeMap<u32, String> = BTreeMap::new();
    for line in content.lines() {
        if !line.starts_with(prefix) { continue; }
        let rest = &line[prefix.len()..];
        // Skip a single off-spec line instead of `?`-aborting the whole collection
        // (which would silently discard every already-parsed line in this section).
        let Some(eq) = rest.find('=') else { continue };
        let Ok(n) = rest[..eq].parse::<u32>() else { continue };
        lines.insert(n, rest[eq + 1..].to_string());
    }
    if lines.is_empty() { return None; }
    Some(lines.values().cloned().collect::<Vec<_>>().join("\n"))
}

/// Collect plain-EEL numbered lines of the form `<prefix><N>=<code>` (N ascending),
/// joined by '\n'. Used for shape_N_per_frame, wave_N_per_frame, wave_N_per_point.
/// Returns None if no matching line was present.
fn collect_numbered_eel(content: &str, prefix: &str) -> Option<String> {
    let mut lines: BTreeMap<u32, String> = BTreeMap::new();
    for line in content.lines() {
        if !line.starts_with(prefix) { continue; }
        let rest = &line[prefix.len()..];
        // rest is like "3=code" (the trailing index then '=')
        let eq = match rest.find('=') { Some(e) => e, None => continue };
        let n: u32 = match rest[..eq].parse() { Ok(n) => n, Err(_) => continue };
        lines.insert(n, rest[eq + 1..].to_string());
    }
    if lines.is_empty() { return None; }
    Some(lines.values().cloned().collect::<Vec<_>>().join("\n"))
}

/// Collect `<prefix><N>=code` numbered lines but SKIP any whose remainder begins
/// with `_init` (so `shape_0_per_frame_init_*` lines aren't swept into the regular
/// `shape_0_per_frame*` collection). Mirrors collect_numbered_eel otherwise.
fn collect_per_frame_no_init(content: &str, prefix: &str) -> Option<String> {
    let mut lines: BTreeMap<u32, String> = BTreeMap::new();
    for line in content.lines() {
        if !line.starts_with(prefix) { continue; }
        let rest = &line[prefix.len()..];
        if rest.starts_with("_init") { continue; }
        let eq = match rest.find('=') { Some(e) => e, None => continue };
        let n: u32 = match rest[..eq].parse() { Ok(n) => n, Err(_) => continue };
        lines.insert(n, rest[eq + 1..].to_string());
    }
    if lines.is_empty() { return None; }
    Some(lines.values().cloned().collect::<Vec<_>>().join("\n"))
}

/// Read a per-shape / per-wave scalar key like `shapecode_0_rad`. Case-insensitive
/// on the field name. Returns None if absent.
fn parse_key_opt(content: &str, key: &str) -> Option<f32> {
    let key_lc = key.to_ascii_lowercase();
    for line in content.lines() {
        let lc = line.to_ascii_lowercase();
        if let Some(rest) = lc.strip_prefix(&key_lc) {
            if let Some(val) = rest.strip_prefix('=') {
                if let Ok(v) = val.trim().parse::<f32>() {
                    return Some(v);
                }
            }
        }
    }
    None
}

fn parse_shapes(content: &str) -> Vec<ShapeCode> {
    let mut out = Vec::new();
    for i in 0..4u32 {
        let pfx = format!("shapecode_{i}_");
        // Was any shapecode_i_* key present at all?
        let lc_pfx = pfx.to_ascii_lowercase();
        let present = content.lines().any(|l| l.to_ascii_lowercase().starts_with(&lc_pfx));
        if !present { continue; }

        let mut base = ShapeBaseVals::default();
        let g = |field: &str| parse_key_opt(content, &format!("{pfx}{field}"));
        if let Some(v) = g("enabled")      { base.enabled = v as i32; }
        if let Some(v) = g("sides")        { base.sides = v; }
        if let Some(v) = g("additive")     { base.additive = v as i32; }
        if let Some(v) = g("thickOutline") { base.thick_outline = v as i32; }
        if let Some(v) = g("textured")     { base.textured = v as i32; }
        if let Some(v) = g("num_inst")     { base.num_inst = v as i32; }
        if let Some(v) = g("x")            { base.x = v; }
        if let Some(v) = g("y")            { base.y = v; }
        if let Some(v) = g("rad")          { base.rad = v; }
        if let Some(v) = g("ang")          { base.ang = v; }
        if let Some(v) = g("tex_ang")      { base.tex_ang = v; }
        if let Some(v) = g("tex_zoom")     { base.tex_zoom = v; }
        if let Some(v) = g("r")            { base.r = v; }
        if let Some(v) = g("g")            { base.g = v; }
        if let Some(v) = g("b")            { base.b = v; }
        if let Some(v) = g("a")            { base.a = v; }
        if let Some(v) = g("r2")           { base.r2 = v; }
        if let Some(v) = g("g2")           { base.g2 = v; }
        if let Some(v) = g("b2")           { base.b2 = v; }
        if let Some(v) = g("a2")           { base.a2 = v; }
        if let Some(v) = g("border_r")     { base.border_r = v; }
        if let Some(v) = g("border_g")     { base.border_g = v; }
        if let Some(v) = g("border_b")     { base.border_b = v; }
        if let Some(v) = g("border_a")     { base.border_a = v; }

        // Init lines use `shape_N_per_frame_init_M=` (trailing underscore); the
        // regular per-frame lines use `shape_N_per_frameM=` (no separator).
        let per_frame_init = collect_numbered_eel(content, &format!("shape_{i}_per_frame_init_"));
        let per_frame = collect_per_frame_no_init(content, &format!("shape_{i}_per_frame"));
        out.push(ShapeCode { base, per_frame, per_frame_init });
    }
    out
}

fn parse_waves(content: &str) -> Vec<CustomWaveDef> {
    let mut out = Vec::new();
    for n in 0..4u32 {
        let pfx = format!("wavecode_{n}_");
        let lc_pfx = pfx.to_ascii_lowercase();
        let present = content.lines().any(|l| l.to_ascii_lowercase().starts_with(&lc_pfx));
        if !present { continue; }

        let mut w = CustomWaveDef { index: n, ..Default::default() };
        let g = |field: &str| parse_key_opt(content, &format!("{pfx}{field}"));
        if let Some(v) = g("enabled")    { w.enabled = v != 0.0; }
        if let Some(v) = g("samples")    { w.samples = v.max(0.0) as u32; }
        if let Some(v) = g("sep")        { w.sep = v as i32; }
        if let Some(v) = g("bSpectrum")  { w.spectrum = v != 0.0; }
        if let Some(v) = g("bUseDots")   { w.use_dots = v != 0.0; }
        if let Some(v) = g("bDrawThick") { w.draw_thick = v != 0.0; }
        if let Some(v) = g("bAdditive")  { w.additive = v != 0.0; }
        if let Some(v) = g("scaling")    { w.scaling = v; }
        if let Some(v) = g("smoothing")  { w.smoothing = v; }
        if let Some(v) = g("r")          { w.r = v; }
        if let Some(v) = g("g")          { w.g = v; }
        if let Some(v) = g("b")          { w.b = v; }
        if let Some(v) = g("a")          { w.a = v; }

        w.per_frame_init = collect_numbered_eel(content, &format!("wave_{n}_per_frame_init_"));
        w.per_frame = collect_per_frame_no_init(content, &format!("wave_{n}_per_frame"));
        w.per_point = collect_numbered_eel(content, &format!("wave_{n}_per_point"));
        out.push(w);
    }
    out
}

fn extract_section(content: &str, prefix: &str) -> Option<String> {
    // Collect lines: `warp_N=`` or `comp_N=``
    let mut lines: BTreeMap<u32, String> = BTreeMap::new();

    for line in content.lines() {
        // Line format: `warp_42=`actual code here`  (backtick immediately after `=`)
        if !line.starts_with(prefix) { continue; }
        let rest = &line[prefix.len()..];
        // rest is like "42=`code"
        // Skip a single off-spec line instead of `?`-aborting the whole collection
        // (which would silently discard every already-parsed warp_/comp_ line).
        let Some(eq) = rest.find('=') else { continue };
        let Ok(n) = rest[..eq].parse::<u32>() else { continue };
        let after_eq = &rest[eq + 1..];
        // Strip the leading backtick
        let code = after_eq.strip_prefix('`').unwrap_or(after_eq);
        lines.insert(n, code.to_string());
    }

    if lines.is_empty() { return None; }

    // Reassemble
    let raw: String = lines.values().cloned().collect::<Vec<_>>().join("\n");

    // Strip the `shader_body` header and outer `{ }` braces.
    // The first line is typically "shader_body", second is "{", last is "}".
    let trimmed = raw.trim();
    let inner = trimmed
        .strip_prefix("shader_body")
        .unwrap_or(trimmed)
        .trim();
    let inner = inner.strip_prefix('{').unwrap_or(inner);
    let inner = inner.strip_suffix('}').unwrap_or(inner);

    Some(inner.trim().to_string())
}
