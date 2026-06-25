// Safe Rust wrapper around the milk_converter_shim C-ABI.
//
// Primary entry points:
//   convert_milk_shader(hlsl_body) -> Result<String, String>
//     Converts a raw MilkDrop HLSL body (already stripped of "shader_body{}"
//     wrapper) to GLSL ES 3.00, runs glsl-optimizer, then extracts the inner
//     body so the result can be fed directly into glsl_milk_body_to_naga.
//
//   convert_milk_shader_raw(hlsl_body, optimize) -> Result<String, String>
//     Returns the full GLSL ES 3.00 program string without post-processing.

/// `true` when this build linked the native converter (hlsl2glslfork +
/// glsl-optimizer). When `false`, the conversion entry points return `Err` and
/// the host should fall back to the JSON-only path. Lets callers surface a
/// clear "converter unavailable in this build" message instead of guessing.
pub const NATIVE_CONVERTER_AVAILABLE: bool = cfg!(milk_converter_native);

#[cfg(milk_converter_native)]
use std::ffi::{CStr, CString};
#[cfg(milk_converter_native)]
use std::os::raw::{c_char, c_int};

#[cfg(milk_converter_native)]
extern "C" {
    fn milk_convert_shader(hlsl: *const c_char, optimize: c_int, out_glsl: *mut *mut c_char) -> c_int;
    fn milk_convert_shader_ex(file_globals: *const c_char, hlsl: *const c_char, optimize: c_int, out_glsl: *mut *mut c_char) -> c_int;
    fn milk_convert_free(p: *mut c_char);
}

/// Convert a raw MilkDrop HLSL shader body to GLSL ES 3.00, then post-process
/// it into the inner body format expected by `glsl_milk_body_to_naga`.
///
/// `hlsl_body` must not contain the `shader_body { }` wrapper — pass the
/// stripped inner code.  Returns the GLSL body (ready for `glsl_milk_body_to_naga`)
/// or an error string.
pub fn convert_milk_shader(hlsl_body: &str) -> Result<String, String> {
    let glsl_es300 = convert_milk_shader_raw(hlsl_body, true)?;
    Ok(process_native_glsl(&glsl_es300))
}

/// Convert to GLSL ES 3.00 without post-processing.  `optimize` controls
/// whether glsl-optimizer is run on the hlsl2glsl output.
pub fn convert_milk_shader_raw(hlsl_body: &str, optimize: bool) -> Result<String, String> {
    call_c_convert(None, hlsl_body, optimize)
}

/// Like [`convert_milk_shader`] but accepts the pre-`shader_body` file-scope
/// globals (variable declarations, `static` initialisers, etc.) separately.
/// They are inserted between the MilkDrop HLSL prefix and the `shader_body()`
/// wrapper at file scope — matching MilkDrop's original HLSL layout and
/// preventing redeclaration errors when the inner body reuses the same names.
pub fn convert_milk_shader_ex(file_globals: &str, body_inner: &str, optimize: bool) -> Result<String, String> {
    let glsl_es300 = call_c_convert(Some(file_globals), body_inner, optimize)?;
    Ok(process_native_glsl(&glsl_es300))
}

/// Stub used when the crate was built without the native converter (sources
/// absent at build time). Keeps the crate linkable with no C++ dependency; the
/// host falls back to JSON-only ingestion.
#[cfg(not(milk_converter_native))]
fn call_c_convert(_file_globals: Option<&str>, _hlsl_body: &str, _optimize: bool) -> Result<String, String> {
    Err("native .milk converter not built into this binary (hlsl2glslfork + \
         glsl-optimizer sources were absent at build time). Load a pre-converted \
         .json preset, or rebuild with the converter sources present."
        .to_string())
}

#[cfg(milk_converter_native)]
fn call_c_convert(file_globals: Option<&str>, hlsl_body: &str, optimize: bool) -> Result<String, String> {
    // QA: dump the exact HLSL handed to the native converter (MILK_DUMP_NATIVE_IN=1).
    if std::env::var("MILK_DUMP_NATIVE_IN").is_ok() {
        eprintln!(
            "==== NATIVE file_globals ====\n{}\n==== NATIVE body ====\n{}\n==== END NATIVE IN ====",
            file_globals.unwrap_or("(none)"),
            hlsl_body
        );
    }
    let c_hlsl = CString::new(hlsl_body)
        .map_err(|e| format!("CString::new failed: {e}"))?;
    let mut out_ptr: *mut c_char = std::ptr::null_mut();

    let rc = match file_globals {
        None => unsafe {
            milk_convert_shader(c_hlsl.as_ptr(), optimize as c_int, &mut out_ptr)
        },
        Some(globals) => {
            let c_globals = CString::new(globals)
                .map_err(|e| format!("CString::new (globals) failed: {e}"))?;
            unsafe {
                milk_convert_shader_ex(c_globals.as_ptr(), c_hlsl.as_ptr(), optimize as c_int, &mut out_ptr)
            }
        }
    };

    if out_ptr.is_null() {
        return Err(format!("milk_convert_shader returned null output (rc={rc})"));
    }

    // Copy the C string out (as checked UTF-8 where it matters) BEFORE freeing it.
    let decoded = unsafe {
        let cstr = CStr::from_ptr(out_ptr);
        let d = match cstr.to_str() {
            Ok(s) => Ok(s.to_owned()),
            // Non-UTF-8: keep a lossy copy only for diagnostics.
            Err(_) => Err(cstr.to_string_lossy().into_owned()),
        };
        milk_convert_free(out_ptr);
        d
    };

    match (rc, decoded) {
        // Success path: require valid UTF-8 so a non-UTF-8 result is REJECTED rather
        // than silently corrupted (U+FFFD substitution) and fed as mangled GLSL to naga.
        (0, Ok(s)) => Ok(s),
        (0, Err(lossy)) => Err(format!("converter returned non-UTF-8 GLSL output: {lossy}")),
        // Error path: the string is just a diagnostic; lossy is acceptable.
        (_, Ok(s)) | (_, Err(s)) => Err(s),
    }
}

/// Replace the final `_glesFragData[0] = expr;\n}` with `ret = expr.xyz;\n`.
/// Matches the JS regex `/_glesFragData\[0\] = (.+);\n\}/` in processOptimizedShader.
fn transform_frag_output(s: &str) -> String {
    const NEEDLE: &str = "_glesFragData[0] = ";
    if let Some(pos) = s.rfind(NEEDLE) {
        let rest = &s[pos + NEEDLE.len()..];
        if let Some(semi) = rest.find(';') {
            let expr = rest[..semi].trim();
            let before = &s[..pos];
            let after = rest[semi + 1..].trim_start_matches('\n');
            // after should start with '}' (end of void main)
            return format!("{before}ret = {expr}.xyz;\n{after}");
        }
    }
    s.to_string()
}

/// Rust equivalent of milkdrop-preset-utils `processOptimizedShader`.
///
/// Takes the full GLSL ES 3.00 program output from the native converter and
/// returns just the inner body (local declarations + statements) with:
///   - precision qualifiers stripped (`lowp`, `highp`, `mediump`)
///   - `xlv_TEXCOORD0` renamed to `uv`
///   - final `_glesFragData[0] = expr;` replaced with `ret = expr.xyz;`
///
/// The returned string has no `shader_body { }` wrapper — strip_shader_body_wrapper
/// in glsl_milk_body_to_naga is a no-op on it.
pub fn process_native_glsl(glsl: &str) -> String {
    // Locate `void main` — everything before it is declarations we discard.
    let main_start = glsl.find("void main").unwrap_or(glsl.len());

    // glsl-optimizer hoists *mutable copies* of read-only globals (the `qN` macros,
    // `rad`, `uv_orig`, …) to FILE SCOPE as `xlat_mutableX` declarations, then assigns
    // them inside main().  We discard everything before `void main`, which drops those
    // declarations → the in-body assignments/reads become naga `UnknownVariable`.
    // Re-inject any `xlat_mutable*` global var declaration at the top of the returned
    // body so they resolve as locals (main assigns before first read).
    let mut hoisted_mut = String::new();
    for line in glsl[..main_start].lines() {
        let t = line.trim();
        let t = t
            .strip_prefix("lowp ")
            .or_else(|| t.strip_prefix("mediump "))
            .or_else(|| t.strip_prefix("highp "))
            .unwrap_or(t);
        // A global variable declaration: ends in `;`, mentions an xlat_mutable name,
        // is not a uniform/in/out/layout, and is not a one-line function (no `{`).
        if t.contains("xlat_mutable")
            && t.ends_with(';')
            && !t.contains('{')
            && !t.starts_with("uniform")
            && !t.starts_with("in ")
            && !t.starts_with("out ")
            && !t.starts_with("layout")
        {
            hoisted_mut.push_str("    ");
            hoisted_mut.push_str(t);
            hoisted_mut.push('\n');
        }
    }

    // Extract the body between the first `{` and the matching final `}`.
    let after_main = &glsl[main_start..];
    let brace_open = after_main.find('{').unwrap_or(after_main.len());
    let body_with_braces = &after_main[brace_open..];

    // Strip outer braces (first `{` … last `}`), ignoring trailing whitespace.
    let trimmed = body_with_braces.trim_end();
    let inner: &str = if trimmed.starts_with('{') && trimmed.ends_with('}') {
        &trimmed[1..trimmed.len() - 1]
    } else {
        body_with_braces
    };

    // Apply whole-string transformations first.
    // Convert the final frag output assignment before the closing brace:
    // `_glesFragData[0] = expr;\n}` → `ret = expr.xyz;\n`
    // This mirrors the JS regex: /_glesFragData\[0\] = (.+);\n\}/
    let inner_str = inner.to_string();
    let inner_str = transform_frag_output(&inner_str);

    // Apply line-by-line transformations.
    let mut out = String::with_capacity(inner_str.len());
    for line in inner_str.lines() {
        let mut l = line.to_string();

        // Strip precision qualifiers.
        l = l.replace("lowp ", "").replace("highp ", "").replace("mediump ", "");

        // Rename TEXCOORD0 varying to `uv`.
        l = l.replace("xlv_TEXCOORD0", "uv");

        // NOTE: do NOT rename `xlat_mutabletexsize` back to `texsize` — glsl-optimizer
        // creates it as a WRITABLE local copy (hoisted_mut above re-declares it), and a
        // shader that writes it (`xlat_mutabletexsize.x = …`) would then store into the
        // read-only `texsize` uniform → naga "pointer … not a valid store destination".
        // Keeping the local name makes reads and writes consistent.


        out.push_str(&l);
        out.push('\n');
    }

    let out = expand_swizzle_stores(&out);
    if hoisted_mut.is_empty() {
        out
    } else {
        format!("{hoisted_mut}{out}")
    }
}

/// Expand a multi-component swizzle store (`v.xyz = expr;`) into per-component stores
/// via a temporary. glsl-optimizer builds vec temporaries this way (`vec4 t; t.w = 1.0;
/// t.xyz = rgb;`), but naga's IR rejects a multi-component swizzle as a store
/// destination ("pointer doesn't relate to a valid destination for a store").
fn expand_swizzle_stores(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut counter = 0usize;
    for line in src.lines() {
        match try_expand_swizzle_store(line, &mut counter) {
            Some(expanded) => out.push_str(&expanded),
            None => out.push_str(line),
        }
        out.push('\n');
    }
    out
}

fn try_expand_swizzle_store(line: &str, counter: &mut usize) -> Option<String> {
    let trimmed = line.trim_start();
    let indent = &line[..line.len() - trimmed.len()];
    let stmt = trimmed.strip_suffix(';')?;
    let eq = find_simple_assign(stmt)?;
    let lhs = stmt[..eq].trim();
    let rhs = stmt[eq + 1..].trim();
    let dot = lhs.rfind('.')?;
    let swiz = &lhs[dot + 1..];
    if swiz.len() < 2
        || swiz.len() > 4
        || !swiz.bytes().all(|c| matches!(c, b'x' | b'y' | b'z' | b'w'))
    {
        return None;
    }
    let place = lhs[..dot].trim();
    // Only a simple lvalue (identifier, optionally `[index]`/`.member`). Skip anything
    // with arithmetic or a call so we never mis-handle a non-store expression.
    if place.is_empty() || place.contains(['+', '-', '*', '/', '?', '(', ' ']) {
        return None;
    }
    let vtype = match swiz.len() {
        2 => "vec2",
        3 => "vec3",
        4 => "vec4",
        _ => return None,
    };
    *counter += 1;
    let tmp = format!("_sws{counter}");
    let comps = [b'x', b'y', b'z', b'w'];
    let mut s = format!("{indent}{vtype} {tmp} = {rhs};");
    for (k, sc) in swiz.bytes().enumerate() {
        s.push_str(&format!(" {place}.{} = {tmp}.{};", sc as char, comps[k] as char));
    }
    Some(s)
}

/// First top-level `=` that is a plain assignment (not `==`/`<=`/`>=`/`!=`).
fn find_simple_assign(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    let mut depth = 0i32;
    for i in 0..b.len() {
        match b[i] {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b'=' if depth == 0 => {
                let prev = if i > 0 { b[i - 1] } else { b' ' };
                let next = if i + 1 < b.len() { b[i + 1] } else { b' ' };
                if next != b'=' && !matches!(prev, b'=' | b'<' | b'>' | b'!') {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}
