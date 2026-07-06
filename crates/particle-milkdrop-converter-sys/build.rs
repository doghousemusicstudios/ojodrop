use std::path::{Path, PathBuf};

// The native `.milk` HLSL→GLSL converter links two upstream C++ libraries
// (hlsl2glslfork + glsl-optimizer/Mesa). This script resolves their sources in
// priority order and never `panic!`s when they are absent — a clean clone must
// still build.
//
//   1. Prebuilt static libs under `references/…` (an optional local fast path,
//      used when present unless MILK_CONVERTER_FROM_SOURCE forces the source build).
//   2. Build-from-source from the vendored git submodule under `vendor/…` via
//      CMake (the clean-clone / public-repo path). Requires `cmake` + a C/C++
//      toolchain; the submodule must be initialised
//      (`git submodule update --init --recursive`).
//   3. Neither present → graceful stub: emit a `cargo:warning`, build nothing,
//      leave the `milk_converter_native` cfg UNSET. `src/lib.rs` then compiles a
//      stub that returns an error at runtime, so the crate (and the app) still
//      build and run; dropping a `.milk` simply reports the converter is
//      unavailable while `.json` presets keep working.

const LIB_HLSL: &str = "libhlsl2glsl.a";
const LIB_GLSLOPT: &str = "libglsl_optimizer.a";
const LIB_GLCPP: &str = "libglcpp-library.a";
const LIB_MESA: &str = "libmesa.a";

fn main() {
    println!("cargo:rustc-check-cfg=cfg(milk_converter_native)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/shim/milk_converter_shim.cpp");
    println!("cargo:rerun-if-changed=cmake/CMakeLists.txt");
    println!("cargo:rerun-if-env-changed=MILK_CONVERTER_FORCE_STUB");
    println!("cargo:rerun-if-env-changed=MILK_CONVERTER_FROM_SOURCE");

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    // Escape hatch: force the JSON-only stub even when sources are present.
    if std::env::var("MILK_CONVERTER_FORCE_STUB").is_ok() {
        warn("MILK_CONVERTER_FORCE_STUB set; building the no-op stub (JSON-only).");
        return;
    }
    let force_from_source = std::env::var("MILK_CONVERTER_FROM_SOURCE").is_ok();

    let Some(resolved) = resolve(&manifest_dir, force_from_source) else {
        warn(
            "native converter sources not found; building a no-op stub. The app runs and loads \
             JSON presets, but dropping a raw .milk reports that the converter is unavailable. To \
             enable it, initialise the vendored submodules: \
             `git submodule update --init --recursive`.",
        );
        return;
    };

    // Compile the C++ shim that wraps hlsl2glslfork + glsl-optimizer.
    let shim = manifest_dir.join("src/shim/milk_converter_shim.cpp");
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file(&shim)
        .flag_if_supported("-std=c++14")
        .flag_if_supported("-Wno-deprecated-declarations")
        .flag_if_supported("-Wno-unused-parameter");
    for inc in &resolved.include_dirs {
        build.include(inc);
    }
    build.compile("milk_converter_shim");

    // Link the four static libs.
    for dir in &resolved.lib_search_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    println!("cargo:rustc-link-lib=static=hlsl2glsl");
    println!("cargo:rustc-link-lib=static=glsl_optimizer");
    println!("cargo:rustc-link-lib=static=glcpp-library");
    println!("cargo:rustc-link-lib=static=mesa");

    // C++ stdlib (macOS: libc++; Linux: libstdc++).
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=c++");
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=stdc++");

    println!("cargo:rustc-cfg=milk_converter_native");
}

/// Resolved include + link locations for the converter's C++ deps.
struct Resolved {
    include_dirs: Vec<PathBuf>,
    lib_search_dirs: Vec<PathBuf>,
}

fn resolve(manifest_dir: &Path, force_from_source: bool) -> Option<Resolved> {
    let submodule = manifest_dir.join("vendor/milkdrop-shader-converter");
    let references = manifest_dir.join(
        "../../references/milkdrop-preset-converter-node/node_modules/milkdrop-shader-converter",
    );

    // Prefer the fast prebuilt path unless explicitly forced to build from source.
    if !force_from_source {
        if let Some(r) = prebuilt(&references) {
            return Some(r);
        }
    }
    if let Some(r) = from_source(&submodule) {
        return Some(r);
    }
    // Fall back to prebuilt even if from_source was preferred but unavailable.
    prebuilt(&references)
}

/// A converter root with the four prebuilt static libs already present.
fn prebuilt(root: &Path) -> Option<Resolved> {
    let root = root.canonicalize().ok()?;
    let hlsl_dir = root.join("build/hlsl2glslfork");
    let glslopt_dir = root.join("build/glsl-optimizer");
    let libs = [
        hlsl_dir.join(LIB_HLSL),
        glslopt_dir.join(LIB_GLSLOPT),
        glslopt_dir.join(LIB_GLCPP),
        glslopt_dir.join(LIB_MESA),
    ];
    if !libs.iter().all(|p| p.exists()) {
        return None;
    }
    Some(Resolved {
        include_dirs: header_dirs(&root),
        lib_search_dirs: vec![hlsl_dir, glslopt_dir],
    })
}

/// Build the static libs from the vendored submodule sources via CMake.
fn from_source(converter_dir: &Path) -> Option<Resolved> {
    // Submodule initialised? (a sentinel source file the build needs)
    if !converter_dir.join("glsl-optimizer/CMakeLists.txt").exists()
        || !converter_dir.join("hlsl2glslfork/CMakeLists.txt").exists()
    {
        return None;
    }
    let converter_dir = converter_dir.canonicalize().ok()?;
    let cmake_proj = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("cmake");

    // The vendored glsl-optimizer / hlsl2glslfork ship flex/bison-GENERATED lexer
    // and parser sources whose inter-file dependencies are under-declared in their
    // CMakeLists. A high-parallelism build (`--parallel <ncpu>`) can therefore RACE
    // — compiling a unit before its generated header exists — and fail on the FIRST
    // cold build, only to succeed on a retry. That flaky first-clone experience is
    // unacceptable for the public repo, so pin the CMake build to a single job (the
    // `cmake` crate reads NUM_JOBS to choose `--parallel`). One-time cost per clone;
    // the prebuilt `references/` path is unaffected.
    std::env::set_var("NUM_JOBS", "1");
    println!(
        "cargo:warning=particle-milkdrop-converter-sys: compiling the native .milk \
         converter from source (one-time; serialized to avoid an upstream flex/bison \
         parallel-build race)…"
    );

    // `cmake` crate builds into OUT_DIR/build and skips install via build_target.
    let dst = cmake::Config::new(&cmake_proj)
        .define("CONVERTER_DIR", &converter_dir)
        // The upstream sub-projects declare `cmake_minimum_required` < 3.5, which
        // CMake ≥ 4 refuses outright. This compatibility shim lets them configure.
        .define("CMAKE_POLICY_VERSION_MINIMUM", "3.5")
        .build_target("milk_converter_libs")
        .build();
    let build_root = dst.join("build");

    // Collect the directory of each produced static lib (locations can vary by
    // generator, so search rather than hardcode).
    let mut dirs = Vec::new();
    for name in [LIB_HLSL, LIB_GLSLOPT, LIB_GLCPP, LIB_MESA] {
        let found = find_file(&build_root, name)?;
        let dir = found.parent()?.to_path_buf();
        if !dirs.contains(&dir) {
            dirs.push(dir);
        }
    }

    Some(Resolved {
        include_dirs: header_dirs(&converter_dir),
        lib_search_dirs: dirs,
    })
}

/// The three header search dirs the shim + libs need, relative to a converter
/// root (same layout for the prebuilt `references/` tree and the submodule).
fn header_dirs(root: &Path) -> Vec<PathBuf> {
    vec![
        root.join("hlsl2glslfork/include"),
        root.join("glsl-optimizer/src/glsl"),
        root.join("glsl-optimizer/src"),
    ]
}

/// Depth-first search for a file by exact name under `root`. Returns the first
/// match (the four lib names are unique across the build tree).
fn find_file(root: &Path, name: &str) -> Option<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.file_name().map(|n| n == name).unwrap_or(false) {
                return Some(path);
            }
        }
    }
    None
}

fn warn(msg: &str) {
    println!("cargo:warning=particle-milkdrop-converter-sys: {msg}");
}
