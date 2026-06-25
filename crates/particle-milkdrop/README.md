# OjoDrop

**Drop a MilkDrop preset, watch it play.** OjoDrop is a standalone, open-source
(MIT) macOS app: a window where you drag a `.milk` MilkDrop preset (or a
pre-converted `.json`) and it renders, live, reacting to your microphone. It is a
native Rust + [wgpu](https://wgpu.rs) reimplementation of the MilkDrop /
[Butterchurn](https://github.com/jberg/butterchurn) engine — no browser, no
Node, no Winamp. The `.milk` → GLSL converter is **baked into the binary**.

> OjoDrop is the app layer over the `particle-milkdrop` engine crate, which is the
> single source of truth for the renderer. OjoDrop adds drag-and-drop, in-process
> `.milk` ingestion, crash-safety against arbitrary files, packaging, and the
> open-source scaffolding.

## Use

- **Launch it** → an empty window: *"drag a .milk or .json preset here."*
- **Drag a `.milk`** onto the window → it's converted in-process and rendered.
- **Drag a `.json`** (Butterchurn-exported preset) → loaded directly.
- A malformed or hostile file shows an error in the title bar and the app keeps
  running — it never crashes on bad input.
- `Esc` quits. `--about` prints credits and license info.

Audio comes from your default input device (the room mic); if none is available
the engine falls back to synthetic reactivity so presets still move.

## Build & run from source

OjoDrop links two upstream C++ shader libraries
([hlsl2glslfork](https://github.com/aras-p/hlsl2glslfork) +
[glsl-optimizer](https://github.com/aras-p/glsl-optimizer)) to convert `.milk`
HLSL into GLSL. They are vendored as **git submodules** and built from source.

```sh
# 1. Clone with submodules (or `git submodule update --init --recursive` after).
git clone --recurse-submodules <repo-url>
cd ojodrop

# 2. Build & run. The native converter is ON by default.
cargo run --release                 # opens the empty-state window
cargo run --release -- preset.milk  # boots straight into a preset
```

**Build dependencies:** a C/C++ toolchain and `cmake` (for the converter
sources). On macOS: `xcode-select --install` and `brew install cmake`.

### JSON-only build (no C++ toolchain)

If you don't want the converter, build without it — `.json` presets still load,
and dropping a `.milk` reports the converter is unavailable rather than failing:

```sh
cargo run --release --no-default-features
```

The converter sys-crate also **degrades gracefully**: if the submodules aren't
initialized, it compiles a no-op stub instead of failing the build, so a clean
clone always builds even before you fetch the converter sources.

## macOS .app bundle

```sh
cargo install cargo-bundle      # once
cargo bundle --release          # produces target/release/bundle/osx/OjoDrop.app
```

The bundle metadata and icon live under [`bundle/`](./bundle/). App signing and
notarization for distribution are out of scope here (note for later).

## Credits

OjoDrop exists because of **Ryan Geiss** (MilkDrop), **Jordan "jberg" Berg**
(Butterchurn + the shader converter), **Nullsoft / Winamp**, and the
hlsl2glslfork / glsl-optimizer / Mesa / MojoShader authors. See
[`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md) for full attribution and the
licensing verdict. Thank you, all of you.

## License

MIT — see [`LICENSE`](./LICENSE). Bundled third-party components retain their own
permissive licenses (BSD-3 / zlib / MIT), reproduced in `THIRD_PARTY_NOTICES.md`.
