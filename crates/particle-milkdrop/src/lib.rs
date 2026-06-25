//! # particle-milkdrop
//!
//! A native Rust + wgpu MilkDrop / Butterchurn preset player. This crate is the
//! **single source of truth** for the render engine; the `particle-milkdrop`
//! binary and the OjoDrop app built from it consume it as a library. Keep this
//! surface reusable — do not fork the engine.
//!
//! ## Public API surface (the seam other crates build on)
//!
//! - [`MilkdropRenderer`] — the wgpu render path (feedback ring → warp → comp).
//!   Construct with [`MilkdropRenderer::new`], drive per-frame with
//!   `set_audio`/`set_waveform`/`set_freq_spectrum` then `render`.
//! - [`MilkShaders`] — the parsed preset (per-frame/per-vertex EEL + warp/comp
//!   shader bodies + base values) fed to the renderer.
//! - [`load_preset_path`] / [`load_preset_str`] — high-level ingest that routes
//!   `.milk` (raw HLSL bodies, run through the native converter when the
//!   `milk-native-converter` feature is on) and `.json` (Butterchurn-converted
//!   GLSL bodies) to the same `MilkShaders`.
//! - [`fallback_preset`] — a known-good passthrough preset, used as a graceful
//!   fallback when an untrusted file fails to load or compile.
//! - [`native_converter_available`] — whether this build can ingest raw `.milk`.
//!
//! Lower-level modules ([`parse_milk`], [`load_json`], [`preprocess`],
//! [`equations`], [`renderer`]) are public for advanced consumers but most callers
//! only need the items re-exported at the crate root.

pub mod equations;
pub mod load_json;
pub mod parse_milk;
pub mod preprocess;
pub mod renderer;

use std::path::Path;

pub use parse_milk::{parse, CustomWaveDef, MilkShaders, ShapeBaseVals, ShapeCode};
pub use renderer::{compile_glsl, MilkdropRenderer};

/// Whether this build can ingest raw `.milk` presets — i.e. the native
/// HLSL→GLSL converter (hlsl2glslfork + glsl-optimizer) was compiled and linked.
///
/// When `false`, only pre-converted `.json` presets load; a dropped `.milk`
/// should report that the converter is unavailable rather than rendering a blank.
/// This is `true` only when the `milk-native-converter` feature is enabled **and**
/// the C++ sources were present at build time (the sys-crate degrades to a stub
/// otherwise — see `particle-milkdrop-converter-sys`).
pub fn native_converter_available() -> bool {
    #[cfg(feature = "milk-native-converter")]
    {
        particle_milkdrop_converter_sys::NATIVE_CONVERTER_AVAILABLE
    }
    #[cfg(not(feature = "milk-native-converter"))]
    {
        false
    }
}

/// Load a preset from a string, dispatching on `is_json`:
/// `true` → the Butterchurn converted-JSON loader (GLSL shader bodies);
/// `false` → the raw `.milk` parser (HLSL shader bodies, run through the native
/// converter when `milk-native-converter` is enabled). Both yield a
/// [`MilkShaders`] for the same renderer.
///
/// Returns `Err` only for malformed JSON. The `.milk` parser is infallible at the
/// parse stage — an unsupported shader is caught later by the renderer, which
/// falls back rather than erroring here.
pub fn load_preset_str(content: &str, is_json: bool) -> Result<MilkShaders, String> {
    if is_json {
        load_json::load(content).map_err(|e| format!("cannot load JSON preset: {e}"))
    } else {
        Ok(parse_milk::parse(content))
    }
}

/// Load a preset from disk, dispatching on file extension (`.json` → Butterchurn
/// loader, anything else → raw `.milk` parser). See [`load_preset_str`].
pub fn load_preset_path(path: &Path) -> Result<MilkShaders, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    // Match the `.json` predicate used at the call sites (and the foundation
    // `load_preset`): a full, case-insensitive ".json" suffix, so dispatch and the
    // UI/CLI gates never disagree on a pathological name.
    let is_json = path.to_string_lossy().to_ascii_lowercase().ends_with(".json");
    load_preset_str(&content, is_json).map_err(|e| format!("{} ({})", e, path.display()))
}

/// A known-good, always-compilable passthrough preset used as a graceful fallback
/// when an untrusted file fails to load or its shader fails to compile. `parse("")`
/// yields all-default scalars with `warp`/`comp` = `None`, which the renderer
/// renders as a plain feedback passthrough rather than crashing.
pub fn fallback_preset() -> MilkShaders {
    parse_milk::parse("")
}
