#![allow(dead_code)]
use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::equations::{EelProgram, EelState, Env, MegaBuf};
use std::cell::RefCell;
use std::rc::Rc;
use crate::preprocess::{
    hlsl_milk_body_to_naga, hlsl_milk_warp_body_to_naga,
    glsl_milk_body_to_naga, glsl_milk_warp_body_to_naga, fix_glsl_vector_types, MILKDROP_SAMPLERS,
};
use crate::parse_milk::{MilkShaders, ShapeBaseVals, CustomWaveDef};

// ── Warp mesh constants ──────────────────────────────────────────────────────

const GRID_W: u32 = 48;
const GRID_H: u32 = 32;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct WarpVert {
    pos:   [f32; 2], // NDC screen position
    uv:    [f32; 2], // warped UV (sample coord) into the previous frame [0,1], DirectX-UV
    decay: [f32; 4], // per-vertex decay rgb (a unused = 1.0)
}
const _: () = assert!(std::mem::size_of::<WarpVert>() == 32);

// Per-frame warp base values (from MilkShaders), overridable by the per-frame EEL.
#[derive(Copy, Clone)]
struct WarpBase {
    zoom: f32, zoomexp: f32, rot: f32, warp: f32,
    cx: f32, cy: f32, dx: f32, dy: f32, sx: f32, sy: f32,
    warpscale: f32, warpanimspeed: f32, decay: f32, wrap: bool,
}

// ── Custom-shape vertex (interleaved pos/color/uv) ───────────────────────────
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ShapeVert {
    pos:   [f32; 2],
    color: [f32; 4],
    uv:    [f32; 2],
}
const _: () = assert!(std::mem::size_of::<ShapeVert>() == 32);

// ── Border vertex (pos only; color via uniform) ──────────────────────────────
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BorderVert { pos: [f32; 2] }

// ── ShapeU uniform (textured flag) ───────────────────────────────────────────
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ShapeU { textured: f32, _pad: [f32; 3] }

// ── BorderU uniform (color + thick offset) ───────────────────────────────────
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BorderU { color: [f32; 4], offset: [f32; 4] }

// ── Waveform vertex (pos + color) ────────────────────────────────────────────
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct WaveVert { pos: [f32; 2], color: [f32; 4] }
const _: () = assert!(std::mem::size_of::<WaveVert>() == 24);

// ── Motion-vector vertex (pos only; color via uniform) ───────────────────────
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MVVert { pos: [f32; 2] }
// maxX*maxY*2 verts (butterchurn caps the grid at 64x48, 2 verts per arrow).
const MV_VERT_CAP: usize = 64 * 48 * 2;

// ── MV color uniform (vec4) ──────────────────────────────────────────────────
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MVColor { color: [f32; 4] }

// ── Darken-center vertex (pos + color) ───────────────────────────────────────
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DarkenVert { pos: [f32; 2], color: [f32; 4] }
const _: () = assert!(std::mem::size_of::<DarkenVert>() == 24);

const SIDES_MAX: usize = 100;
// Each shape instance contributes (sides+2) fill verts.
const SHAPE_FILL_VERTS_MAX: usize = SIDES_MAX + 2;
// Static fan index count = sides*3 for sides<=100 → 300.
const SHAPE_FAN_IDX_MAX: usize = SIDES_MAX * 3;
// Custom-shape fill geometry capacity (verts for ALL shapes×instances of a frame).
const SHAPE_VERT_CAP: usize = 8192;
// Waveform vertex capacity (built-in + custom). 4 waves × ~1023 verts (512 samples,
// line-strip) ≈ 4092 — right at the old 4096 cap, so the 4th wave of multi-wave
// presets overflowed the upload and drew from a stale buffer tail.
const WAVE_VERT_CAP: usize = 16384;
// Border vertex capacity (per-frame across all shapes).
const BORDER_VERT_CAP: usize = 4096;

// Runtime state for one custom shape (base vals + per-frame program + var pool).
struct ShapeRT {
    base: ShapeBaseVals,
    prog: Option<EelProgram>,
    env:  Env,
    /// Per-pool megabuf (private) sharing the preset-wide gmegabuf.
    state: EelState,
}

// Runtime state for one custom waveform.
struct WaveRT {
    def:           CustomWaveDef,
    per_frame_prog: Option<EelProgram>,
    per_point_prog: Option<EelProgram>,
    env:           Env,
    /// Per-pool megabuf (private) sharing the preset-wide gmegabuf.
    state: EelState,
}

// One fill draw (a shape instance). base_vertex = vertex offset into shape_vert_buf.
struct ShapeFillDraw {
    base_vertex: i32,
    sides:       u32,   // index count = sides*3
    additive:    bool,
}
// One border source (rim verts already appended to border_vert_buf).
struct BorderDraw {
    start_vert: u32,
    count:      u32,    // = sides+1
    color:      [f32; 4],
    thick:      bool,
}
// One waveform draw record.
struct WaveDraw {
    start_vert: u32,
    count:      u32,
    points:     bool,   // PointList vs LineStrip
    additive:   bool,
    thick:      bool,   // 4-pass thick offset expansion
}

fn build_warp_indices() -> Vec<u32> {
    let mut idx = Vec::with_capacity((GRID_W * GRID_H * 6) as usize);
    for j in 0..GRID_H {
        for i in 0..GRID_W {
            let a = j * (GRID_W + 1) + i;
            let b = a + 1;
            let c = a + (GRID_W + 1);
            let d = c + 1;
            idx.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    idx
}

// PerFrame uniform buffer — layout must exactly match the WGSL PerFrame struct
// emitted by naga (16 vec4s + 29 f32s, padded to 384 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PerFrame {
    texsize:          [f32; 4],  //   0 — (w, h, 1/w, 1/h)
    aspect:           [f32; 4],  //  16 — (asp, 1/asp, 1, 1)
    slow_roam_cos:    [f32; 4],  //  32
    roam_cos:         [f32; 4],  //  48
    slow_roam_sin:    [f32; 4],  //  64
    roam_sin:         [f32; 4],  //  80
    rand_frame:       [f32; 4],  //  96
    rand_preset:      [f32; 4],  // 112
    _qa:              [f32; 4],  // 128 — q1..q4
    _qb:              [f32; 4],  // 144 — q5..q8
    _qc:              [f32; 4],  // 160
    _qd:              [f32; 4],  // 176
    _qe:              [f32; 4],  // 192
    _qf:              [f32; 4],  // 208
    _qg:              [f32; 4],  // 224
    _qh:              [f32; 4],  // 240
    time:             f32,       // 256
    fps:              f32,       // 260
    frame:            f32,       // 264
    progress:         f32,       // 268
    bass:             f32,       // 272
    mid:              f32,       // 276
    treb:             f32,       // 280
    vol:              f32,       // 284
    bass_att:         f32,       // 288
    mid_att:          f32,       // 292
    treb_att:         f32,       // 296
    vol_att:          f32,       // 300
    f_shader:         f32,       // 304
    gamma_adj:        f32,       // 308
    echo_zoom:        f32,       // 312
    echo_alpha:       f32,       // 316
    echo_orientation: f32,       // 320
    blur1_min:        f32,       // 324
    blur1_max:        f32,       // 328
    blur2_min:        f32,       // 332
    blur2_max:        f32,       // 336
    blur3_min:        f32,       // 340
    blur3_max:        f32,       // 344
    scale1:           f32,       // 348
    scale2:           f32,       // 352
    scale3:           f32,       // 356
    bias1:            f32,       // 360
    bias2:            f32,       // 364
    bias3:            f32,       // 368
    brighten:         f32,       // 372 — comp post-FX flags (bBrighten/bDarken/bSolarize/bInvert)
    darken:           f32,       // 376
    solarize:         f32,       // 380
    invert:           f32,       // 384
    _pad:             [f32; 3],  // 388 → pad to 400
}
const _: () = assert!(std::mem::size_of::<PerFrame>() == 400);

// ----- texture helpers -------------------------------------------------------

fn make_tex2d(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    w: u32, h: u32,
    usage: wgpu::TextureUsages,
    data: Option<&[u8]>,
) -> wgpu::Texture {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage,
        view_formats: &[],
    });
    if let Some(pixels) = data {
        queue.write_texture(
            tex.as_image_copy(),
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
    }
    tex
}

fn make_tex3d(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    s: u32,
    data: &[u8],
) -> wgpu::Texture {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: s, height: s, depth_or_array_layers: s },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        tex.as_image_copy(),
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(s * 4),
            rows_per_image: Some(s),
        },
        wgpu::Extent3d { width: s, height: s, depth_or_array_layers: s },
    );
    tex
}

/// Derive a per-preset hue seed (Butterchurn's `rand_start`, normally 4× Math.random()
/// chosen at load). We hash the preset's shader/equation text so each preset gets a
/// distinct but reproducible hue (vs the old fixed 0.5 that biased everything green).
fn preset_hue_seed(s: &str) -> [f32; 4] {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    let mut out = [0.0f32; 4];
    for slot in out.iter_mut() {
        h ^= h << 13; h ^= h >> 7; h ^= h << 17; // xorshift64
        *slot = ((h >> 40) as f32) / ((1u64 << 24) as f32); // → [0,1)
    }
    out
}

fn noise_bytes(n: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * 4);
    let mut x: u32 = 0xdeadbeef;
    for _ in 0..n {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        let r = ((x & 0xff) as u8, ((x >> 8) & 0xff) as u8, ((x >> 16) & 0xff) as u8, 255u8);
        v.extend_from_slice(&[r.0, r.1, r.2, r.3]);
    }
    v
}

fn noise_bytes_scaled(n: usize, max_val: u8) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * 4);
    let mut x: u32 = 0xcafebabe;
    for _ in 0..n {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        let scale = |b: u8| ((b as u32 * max_val as u32) / 255) as u8;
        v.push(scale((x & 0xff) as u8));
        v.push(scale(((x >> 8) & 0xff) as u8));
        v.push(scale(((x >> 16) & 0xff) as u8));
        v.push(255u8);
    }
    v
}

// ----- Butterchurn-faithful value/lattice noise (noise.js) -------------------
//
// Reproduces the createNoiseTex / createNoiseVolTex algorithm:
//   * random lattice fill (texRange 256 for zoom==1, 216 for zoom>1) with the JS
//     Uint8Array `& 0xFF` wrap emulated exactly (NOT clamping),
//   * separable per-axis Catmull-Rom cubic smoothing between lattice anchors
//     spaced `zoom` texels apart, wrapping (tiling) via modulo.
// RNG is a fixed-seed xorshift32 — Butterchurn uses non-deterministic Math.random
// but no preset depends on exact noise values (only on having structured value
// noise), so a deterministic seed is correct and reproducible for testing.

/// xorshift32 PRNG returning values in [0, 1).
fn bc_rng() -> impl FnMut() -> f32 {
    let mut x: u32 = 0x1234_5678;
    move || {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        // 24 high bits → [0,1) (matches Math.random precision adequately)
        ((x >> 8) as f32) / ((1u32 << 24) as f32)
    }
}

/// fCubicInterpolate (noise.js 158-170): Catmull-Rom-like cubic on scalar values.
fn cubic_interp(y0: f32, y1: f32, y2: f32, y3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t * t2;
    let a0 = y3 - y2 - y0 + y1;
    let a1 = y0 - y1 - a0;
    let a2 = y2 - y0;
    let a3 = y1;
    a0 * t3 + a1 * t2 + a2 * t + a3
}

/// dwCubicInterpolate (noise.js 172-184): per-channel cubic on 4 RGBA bytes.
/// Stores `f * 255` (0..255 after clamp) — JS truncates to Uint8Array, `as u8` matches.
fn dw_cubic(y0: &[u8; 4], y1: &[u8; 4], y2: &[u8; 4], y3: &[u8; 4], t: f32) -> [u8; 4] {
    let mut o = [0u8; 4];
    for c in 0..4 {
        let f = cubic_interp(
            y0[c] as f32 / 255.0,
            y1[c] as f32 / 255.0,
            y2[c] as f32 / 255.0,
            y3[c] as f32 / 255.0,
            t,
        )
        .clamp(0.0, 1.0);
        o[c] = (f * 255.0) as u8;
    }
    o
}

/// Read an RGBA texel from a flat byte buffer at texel index `i`.
fn rd4(buf: &[u8], i: usize) -> [u8; 4] {
    [buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2], buf[i * 4 + 3]]
}

/// createNoiseTex (noise.js 318-399): size×size RGBA8 tiling value noise.
fn create_noise_tex(size: usize, zoom: usize, rng: &mut impl FnMut() -> f32) -> Vec<u8> {
    let n = size; // noiseSize
    let mut buf = vec![0u8; n * n * 4];

    // Random lattice fill.
    let range: f32 = if zoom > 1 { 216.0 } else { 256.0 };
    let half = range * 0.5;
    for px in 0..(n * n) {
        for c in 0..4 {
            let v = (rng() * range + half).floor() as i64;
            // JS Uint8Array wrap (& 0xFF), NOT clamp.
            buf[px * 4 + c] = (v as u32 & 0xFF) as u8;
        }
    }

    if zoom > 1 {
        // Pass 1 — interpolate along X (rows that are multiples of zoom).
        let mut y = 0usize;
        while y < n {
            for x in 0..n {
                if x % zoom != 0 {
                    let base_x = (x / zoom) * zoom + n; // +n keeps (base-zoom) non-negative
                    let base_y = y * n;
                    let y0 = rd4(&buf, base_y + ((base_x - zoom) % n));
                    let y1 = rd4(&buf, base_y + (base_x % n));
                    let y2 = rd4(&buf, base_y + ((base_x + zoom) % n));
                    let y3 = rd4(&buf, base_y + ((base_x + zoom * 2) % n));
                    let t = (x % zoom) as f32 / zoom as f32;
                    let r = dw_cubic(&y0, &y1, &y2, &y3, t);
                    let dst = (y * n + x) * 4;
                    buf[dst..dst + 4].copy_from_slice(&r);
                }
            }
            y += zoom;
        }
        // Pass 2 — interpolate along Y (all columns, all rows).
        for x in 0..n {
            for y in 0..n {
                if y % zoom != 0 {
                    let base_y = (y / zoom) * zoom + n;
                    let y0 = rd4(&buf, ((base_y - zoom) % n) * n + x);
                    let y1 = rd4(&buf, (base_y % n) * n + x);
                    let y2 = rd4(&buf, ((base_y + zoom) % n) * n + x);
                    let y3 = rd4(&buf, ((base_y + zoom * 2) % n) * n + x);
                    let t = (y % zoom) as f32 / zoom as f32;
                    let r = dw_cubic(&y0, &y1, &y2, &y3, t);
                    let dst = (y * n + x) * 4;
                    buf[dst..dst + 4].copy_from_slice(&r);
                }
            }
        }
    }

    buf
}

/// createNoiseVolTex (noise.js 183-318): size³ RGBA8 tiling value noise.
fn create_noise_vol_tex(size: usize, zoom: usize, rng: &mut impl FnMut() -> f32) -> Vec<u8> {
    let n = size;
    let words_per_slice = n * n;
    let words_per_line = n;
    let mut buf = vec![0u8; n * n * n * 4];

    // Random lattice fill.
    let range: f32 = if zoom > 1 { 216.0 } else { 256.0 };
    let half = range * 0.5;
    for px in 0..(n * n * n) {
        for c in 0..4 {
            let v = (rng() * range + half).floor() as i64;
            buf[px * 4 + c] = (v as u32 & 0xFF) as u8;
        }
    }

    if zoom > 1 {
        // Pass X (z,y step by zoom; x over all).
        let mut z = 0usize;
        while z < n {
            let mut y = 0usize;
            while y < n {
                for x in 0..n {
                    if x % zoom != 0 {
                        let base_x = (x / zoom) * zoom + n;
                        let base = z * words_per_slice + y * words_per_line;
                        let y0 = rd4(&buf, base + ((base_x - zoom) % n));
                        let y1 = rd4(&buf, base + (base_x % n));
                        let y2 = rd4(&buf, base + ((base_x + zoom) % n));
                        let y3 = rd4(&buf, base + ((base_x + zoom * 2) % n));
                        let t = (x % zoom) as f32 / zoom as f32;
                        let r = dw_cubic(&y0, &y1, &y2, &y3, t);
                        let dst = (z * words_per_slice + y * words_per_line + x) * 4;
                        buf[dst..dst + 4].copy_from_slice(&r);
                    }
                }
                y += zoom;
            }
            z += zoom;
        }
        // Pass Y (z steps by zoom; x,y over all).
        let mut z = 0usize;
        while z < n {
            for x in 0..n {
                for y in 0..n {
                    if y % zoom != 0 {
                        let base_y = (y / zoom) * zoom + n;
                        let base_z = z * words_per_slice;
                        // sample index = ((base_y±k)%n)*words_per_line + base_z + x
                        let y0 = rd4(&buf, ((base_y - zoom) % n) * words_per_line + base_z + x);
                        let y1 = rd4(&buf, (base_y % n) * words_per_line + base_z + x);
                        let y2 = rd4(&buf, ((base_y + zoom) % n) * words_per_line + base_z + x);
                        let y3 = rd4(&buf, ((base_y + zoom * 2) % n) * words_per_line + base_z + x);
                        let t = (y % zoom) as f32 / zoom as f32;
                        let r = dw_cubic(&y0, &y1, &y2, &y3, t);
                        let dst = (z * words_per_slice + y * words_per_line + x) * 4;
                        buf[dst..dst + 4].copy_from_slice(&r);
                    }
                }
            }
            z += zoom;
        }
        // Pass Z (x,y over all; z over all). FAITHFUL QUIRK: t uses (y%zoom), not z
        // (noise.js line 305) — replicate exactly to match Butterchurn.
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    if z % zoom != 0 {
                        let base_z = (z / zoom) * zoom + n;
                        let base_y = y * words_per_line;
                        let y0 = rd4(&buf, ((base_z - zoom) % n) * words_per_slice + base_y + x);
                        let y1 = rd4(&buf, (base_z % n) * words_per_slice + base_y + x);
                        let y2 = rd4(&buf, ((base_z + zoom) % n) * words_per_slice + base_y + x);
                        let y3 = rd4(&buf, ((base_z + zoom * 2) % n) * words_per_slice + base_y + x);
                        let t = (y % zoom) as f32 / zoom as f32; // QUIRK: y, not z
                        let r = dw_cubic(&y0, &y1, &y2, &y3, t);
                        let dst = (z * words_per_slice + y * words_per_line + x) * 4;
                        buf[dst..dst + 4].copy_from_slice(&r);
                    }
                }
            }
        }
    }

    buf
}

// ----- bind group layout helpers --------------------------------------------

fn sampler_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let mut entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::with_capacity(28);
    for (i, name) in MILKDROP_SAMPLERS.iter().enumerate() {
        let tex_bind  = (i * 2) as u32;
        let samp_bind = tex_bind + 1;
        let dim = if name.contains("vol") {
            wgpu::TextureViewDimension::D3
        } else {
            wgpu::TextureViewDimension::D2
        };
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: tex_bind,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: dim,
                multisampled: false,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: samp_bind,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("milk-samplers-bgl"),
        entries: &entries,
    })
}

fn perframe_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let ubo_binding = (MILKDROP_SAMPLERS.len() * 2) as u32; // = 28
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("perframe-bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: ubo_binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

fn blur_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("blur-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

#[allow(clippy::too_many_arguments)]
fn build_sampler_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    main_view: &wgpu::TextureView,
    blur1_view: &wgpu::TextureView,
    blur2_view: &wgpu::TextureView,
    blur3_view: &wgpu::TextureView,
    noise2d_view: &wgpu::TextureView, // placeholder for fw/pw/pc slots (2/6/8)
    noise_lq_view: &wgpu::TextureView,
    noise_mq_view: &wgpu::TextureView,
    noise_hq_view: &wgpu::TextureView,
    noise_lite_view: &wgpu::TextureView,
    noisevol_lq_view: &wgpu::TextureView,
    noisevol_hq_view: &wgpu::TextureView,
    samp: &wgpu::Sampler,
    samp_clamp: &wgpu::Sampler,
) -> wgpu::BindGroup {
    use wgpu::{BindGroupEntry, BindingResource};
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: bgl,
        entries: &[
            // 0,1  sampler_main
            BindGroupEntry { binding:  0, resource: BindingResource::TextureView(main_view) },
            BindGroupEntry { binding:  1, resource: BindingResource::Sampler(samp) },
            // 2,3  sampler_fw_main (placeholder)
            BindGroupEntry { binding:  2, resource: BindingResource::TextureView(noise2d_view) },
            BindGroupEntry { binding:  3, resource: BindingResource::Sampler(samp) },
            // 4,5  sampler_fc_main → "force clamp": ALWAYS ClampToEdge (MilkDrop/Butterchurn).
            // The parade warp samples this without frac(), so Repeat here wraps opposite-edge
            // content into the border every frame → the vertical seam streaks. Clamp fixes it.
            BindGroupEntry { binding:  4, resource: BindingResource::TextureView(main_view) },
            BindGroupEntry { binding:  5, resource: BindingResource::Sampler(samp_clamp) },
            // 6,7  sampler_pw_main (placeholder)
            BindGroupEntry { binding:  6, resource: BindingResource::TextureView(noise2d_view) },
            BindGroupEntry { binding:  7, resource: BindingResource::Sampler(samp) },
            // 8,9  sampler_pc_main (placeholder) — "force clamp" → ClampToEdge, like fc.
            BindGroupEntry { binding:  8, resource: BindingResource::TextureView(noise2d_view) },
            BindGroupEntry { binding:  9, resource: BindingResource::Sampler(samp_clamp) },
            // 10,11 sampler_blur1
            BindGroupEntry { binding: 10, resource: BindingResource::TextureView(blur1_view) },
            BindGroupEntry { binding: 11, resource: BindingResource::Sampler(samp) },
            // 12,13 sampler_blur2
            BindGroupEntry { binding: 12, resource: BindingResource::TextureView(blur2_view) },
            BindGroupEntry { binding: 13, resource: BindingResource::Sampler(samp) },
            // 14,15 sampler_blur3
            BindGroupEntry { binding: 14, resource: BindingResource::TextureView(blur3_view) },
            BindGroupEntry { binding: 15, resource: BindingResource::Sampler(samp) },
            // 16,17 noise_lq (256² zoom1)
            BindGroupEntry { binding: 16, resource: BindingResource::TextureView(noise_lq_view) },
            BindGroupEntry { binding: 17, resource: BindingResource::Sampler(samp) },
            // 18,19 noise_mq (256² zoom4)
            BindGroupEntry { binding: 18, resource: BindingResource::TextureView(noise_mq_view) },
            BindGroupEntry { binding: 19, resource: BindingResource::Sampler(samp) },
            // 20,21 noise_hq (256² zoom8)
            BindGroupEntry { binding: 20, resource: BindingResource::TextureView(noise_hq_view) },
            BindGroupEntry { binding: 21, resource: BindingResource::Sampler(samp) },
            // 22,23 noise_hq_lite / noise_lq_lite (32² zoom1)
            BindGroupEntry { binding: 22, resource: BindingResource::TextureView(noise_lite_view) },
            BindGroupEntry { binding: 23, resource: BindingResource::Sampler(samp) },
            // 24,25 noisevol_lq (32³ zoom1, 3D)
            BindGroupEntry { binding: 24, resource: BindingResource::TextureView(noisevol_lq_view) },
            BindGroupEntry { binding: 25, resource: BindingResource::Sampler(samp) },
            // 26,27 noisevol_hq (32³ zoom4, 3D)
            BindGroupEntry { binding: 26, resource: BindingResource::TextureView(noisevol_hq_view) },
            BindGroupEntry { binding: 27, resource: BindingResource::Sampler(samp) },
        ],
    })
}

// ----- naga compilation ------------------------------------------------------

pub fn compile_glsl(glsl: &str) -> Result<String, String> {
    use naga::{
        back::wgsl as wgsl_out,
        front::glsl as glsl_in,
        valid::{Capabilities, ValidationFlags, Validator},
    };
    // Repair HLSL-permissive type mismatches (vec<scalar comparisons, …) that naga
    // rejects. Conservative: only confidently-typed constructs are rewritten.
    let glsl_fixed = fix_glsl_vector_types(glsl);
    if std::env::var("MILKDROP_DUMP_FIXED").is_ok() {
        eprintln!("==== type-fixed GLSL ====\n{glsl_fixed}\n==== end fixed ====");
    }
    let glsl = glsl_fixed.as_str();
    let mut parser = glsl_in::Frontend::default();
    let opts = glsl_in::Options {
        stage: naga::ShaderStage::Fragment,
        defines: Default::default(),
    };
    let module = parser
        .parse(&opts, glsl)
        .map_err(|e| format!("{e:?}"))?;
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .map_err(|e| {
            // naga's Display for a validation error stops at "Function 'main' is
            // invalid" — the actual cause (bad expression/type) is in the error
            // source chain. Append it so triage can see WHY validation failed.
            use std::error::Error;
            let mut msg = format!("{e}");
            let mut src = e.source();
            while let Some(s) = src {
                msg.push_str(&format!("  ->  {s}"));
                src = s.source();
            }
            msg
        })?;
    let wgsl = wgsl_out::write_string(&module, &info, wgsl_out::WriterFlags::empty())
        .map_err(|e| format!("{e}"))?;
    Ok(wgsl)
}

// compute_warp_verts is now a method on MilkdropRenderer (see impl block) — it
// runs the per_pixel EEL program per vertex and composes the butterchurn warped UV.

// ----- main renderer struct --------------------------------------------------

pub struct MilkdropRenderer {
    device: Arc<wgpu::Device>,
    queue:  Arc<wgpu::Queue>,

    // which warp/comp path to use
    has_custom_warp: bool,
    has_custom_comp: bool,
    /// preset's decay value (used in warp mesh pass)
    preset_decay: f32,
    /// per-preset hue seed (Butterchurn rand_start) feeding the comp hue_shader
    rand_preset: [f32; 4],

    // ping-pong feedback textures (both RGBA8, same size as render)
    tex_a:      wgpu::Texture,
    tex_b:      wgpu::Texture,
    view_a:     wgpu::TextureView,
    view_b:     wgpu::TextureView,
    write_to_a: bool, // true → write_to_a, read from b

    // blur textures (half-res, quarter-res, eighth-res)
    blur1: wgpu::Texture,
    blur2: wgpu::Texture,
    blur3: wgpu::Texture,
    view_blur1: wgpu::TextureView,
    view_blur2: wgpu::TextureView,
    view_blur3: wgpu::TextureView,
    // separable-blur horizontal-pass intermediates (same res as blur1/2/3)
    btemp1: wgpu::Texture,
    btemp2: wgpu::Texture,
    btemp3: wgpu::Texture,
    view_btemp1: wgpu::TextureView,
    view_btemp2: wgpu::TextureView,
    view_btemp3: wgpu::TextureView,

    // noise textures (Butterchurn-faithful; kept alive — views borrowed by bind groups)
    #[allow(dead_code)] noise2d:      wgpu::Texture, // placeholder for fw/pw/pc slots
    #[allow(dead_code)] noise_lq:     wgpu::Texture,
    #[allow(dead_code)] noise_mq:     wgpu::Texture,
    #[allow(dead_code)] noise_hq:     wgpu::Texture,
    #[allow(dead_code)] noise_lite:   wgpu::Texture,
    #[allow(dead_code)] noisevol_lq:  wgpu::Texture,
    #[allow(dead_code)] noisevol_hq:  wgpu::Texture,
    #[allow(dead_code)] view_noise2d:      wgpu::TextureView,
    #[allow(dead_code)] view_noise_lq:     wgpu::TextureView,
    #[allow(dead_code)] view_noise_mq:     wgpu::TextureView,
    #[allow(dead_code)] view_noise_hq:     wgpu::TextureView,
    #[allow(dead_code)] view_noise_lite:   wgpu::TextureView,
    #[allow(dead_code)] view_noisevol_lq:  wgpu::TextureView,
    #[allow(dead_code)] view_noisevol_hq:  wgpu::TextureView,

    // samplers
    linear_samp: wgpu::Sampler,
    clamp_samp: wgpu::Sampler,

    // UBO
    perframe_buf: wgpu::Buffer,

    // blur pass uniform buffers (one per pass, holds texel size of source)
    blur1_ubo: wgpu::Buffer,
    blur2_ubo: wgpu::Buffer,
    blur3_ubo: wgpu::Buffer,

    // pipelines
    warp_pipeline: wgpu::RenderPipeline,
    /// Custom-warp FS driven by the warped MESH VS (per-pixel warp + decay path).
    warp_custom_pipeline: wgpu::RenderPipeline,
    comp_pipeline: wgpu::RenderPipeline,
    blur_h_pipeline: wgpu::RenderPipeline,
    blur_v_pipeline: wgpu::RenderPipeline,
    // FXAA output pass: COMP → comp_view (offscreen Rgba8Unorm) → FXAA → swapchain.
    #[allow(dead_code)] comp_tex: wgpu::Texture, // kept alive; comp_view borrows it
    comp_view: wgpu::TextureView,
    output_pipeline: wgpu::RenderPipeline,
    #[allow(dead_code)] fxaa_bgl: wgpu::BindGroupLayout,
    #[allow(dead_code)] fxaa_ubo: wgpu::Buffer,
    fxaa_bg: wgpu::BindGroup,
    // standard warp mesh (used when no custom warp shader)
    warp_mesh_pipeline: wgpu::RenderPipeline,
    warp_mesh_bg_a:     wgpu::BindGroup, // reads from tex_a
    warp_mesh_bg_b:     wgpu::BindGroup, // reads from tex_b
    warp_mesh_bgl:      wgpu::BindGroupLayout,
    warp_vert_buf:      wgpu::Buffer,    // updated per frame
    warp_idx_buf:       wgpu::Buffer,    // static
    warp_idx_count:     u32,

    // bind group layouts
    sampler_bgl: wgpu::BindGroupLayout,
    perframe_bgl: wgpu::BindGroupLayout,
    blur_bgl: wgpu::BindGroupLayout,

    // sampler bind groups — one per ping-pong side, for WARP reading the OTHER side
    // bg_read_a: sampler_main = view_a  (use when comp reads curr=a, or warp reads prev=a)
    // bg_read_b: sampler_main = view_b
    bg_read_a: wgpu::BindGroup,
    bg_read_b: wgpu::BindGroup,

    // perframe bind group
    perframe_bg: wgpu::BindGroup,

    // blur bind groups for the separable H/V chain. blur1's H pass reads the warp
    // output (write_view) so it is rebuilt each frame in render(); the rest are static.
    blur1_v_bg: wgpu::BindGroup,
    blur2_h_bg: wgpu::BindGroup,
    blur2_v_bg: wgpu::BindGroup,
    blur3_h_bg: wgpu::BindGroup,
    blur3_v_bg: wgpu::BindGroup,

    // EEL2 per-frame equations
    eel_program: Option<EelProgram>,
    eel_env:     Env,
    /// Per-frame megabuf pool (private) + shared preset-wide gmegabuf handle.
    eel_state:   EelState,
    /// Preset-wide gmegabuf shared by all pools (per-frame/per-pixel/shape/wave).
    #[allow(dead_code)]
    gmegabuf:    Rc<RefCell<MegaBuf>>,
    /// q1..q32 post-init snapshot — re-applied at the top of every frame so
    /// accumulator-q presets don't drift (Butterchurn's per-frame q reset).
    q_init:      Vec<(String, f64)>,

    // Per-vertex warp (per_pixel) program + per-frame warp base values.
    per_pixel_prog: Option<EelProgram>,
    base_warp:      WarpBase,
    /// Scratch EEL env reused across warp vertices (avoids per-vertex alloc).
    warp_env:       Env,
    /// Per-pixel megabuf pool (private) sharing the preset-wide gmegabuf.
    warp_state:     EelState,

    // frame state
    frame_idx: u32,
    start: std::time::Instant,
    /// When Some(dt), time advances by `dt` seconds per rendered frame instead
    /// of using the wall clock. Used for deterministic offscreen animation export.
    time_per_frame: Option<f32>,
    /// Live audio reactivity. When Some([bass, mid, treb, vol]), these drive the
    /// per-frame audio uniforms instead of the synthetic sine-wave fallback.
    audio: Option<[f32; 4]>,
    /// Live attenuated (smoothed) reactivity [bass_att, mid_att, treb_att, vol_att].
    /// When None (headless/synthetic), `*_att` falls back to the non-att values so
    /// deterministic renders stay bit-identical to before this wiring existed.
    audio_att: Option<[f32; 4]>,
    /// Butterchurn-shaped 512-bin FFT magnitude array for `bSpectrum` custom
    /// waveforms. Empty when no live audio (built-in/synthetic path uses time data).
    freq_spectrum: Vec<f32>,
    pub width:  u32,
    pub height: u32,

    pub surface_format: wgpu::TextureFormat,

    // ── Custom shapes ────────────────────────────────────────────────────────
    shapes: Vec<ShapeRT>,
    shapes_fill_pipeline_alpha:    wgpu::RenderPipeline,
    shapes_fill_pipeline_additive: wgpu::RenderPipeline,
    shapes_border_pipeline:        wgpu::RenderPipeline,
    shape_bgl:    wgpu::BindGroupLayout,
    border_bgl:   wgpu::BindGroupLayout,
    shape_vert_buf: wgpu::Buffer,
    shape_idx_buf:  wgpu::Buffer,  // static fan triangulation, 300 u32
    shape_uniform_buf: wgpu::Buffer, // ShapeU (textured)
    border_vert_buf: wgpu::Buffer,
    // border uniforms: dyn-offset buffer (4 slots of 256B = up to 4 thick passes)
    border_uniform_buf: wgpu::Buffer,
    border_bg: wgpu::BindGroup,
    // shape fill bind groups, one per ping-pong read side (prev-frame texture)
    shape_bg_read_a: wgpu::BindGroup,
    shape_bg_read_b: wgpu::BindGroup,

    // ── Waveforms (built-in + custom) ────────────────────────────────────────
    waves: Vec<WaveRT>,
    wave_pipeline_lines_alpha:    wgpu::RenderPipeline,
    wave_pipeline_lines_additive: wgpu::RenderPipeline,
    wave_pipeline_points_alpha:   wgpu::RenderPipeline,
    wave_pipeline_points_additive: wgpu::RenderPipeline,
    wave_bgl:      wgpu::BindGroupLayout,
    wave_vert_buf: wgpu::Buffer,
    wave_off_buf:  wgpu::Buffer,   // dyn-offset, 4 slots of 256B (thick offsets)
    wave_bg:       wgpu::BindGroup,

    // built-in waveform scalar/bool state (parsed)
    bw_mode: f32,
    bw_x: f32, bw_y: f32,
    bw_r: f32, bw_g: f32, bw_b: f32, bw_a: f32,
    bw_mystery: f32, bw_scale: f32, bw_smoothing: f32,
    bw_dots: bool, bw_thick: bool, bw_additive: bool, bw_brighten: bool,
    bw_modalphavol: bool, bw_modalphastart: f32, bw_modalphaend: f32,

    // comp post-FX flags (bBrighten/bDarken/bSolarize/bInvert) for the built-in comp body
    comp_brighten: bool, comp_darken: bool, comp_solarize: bool, comp_invert: bool,

    // ── Motion vectors ───────────────────────────────────────────────────────
    mv_pipeline:  wgpu::RenderPipeline,   // LineList, alpha blend, Rgba8Unorm
    mv_bgl:       wgpu::BindGroupLayout,
    mv_vert_buf:  wgpu::Buffer,
    mv_color_buf: wgpu::Buffer,           // 16-byte uniform (vec4 color)
    mv_bg:        wgpu::BindGroup,
    mv_on: bool, mv_x: f32, mv_y: f32, mv_dx: f32, mv_dy: f32, mv_l: f32,
    mv_r: f32, mv_g: f32, mv_b: f32, mv_a: f32,

    // ── Frame borders (outer/inner) ──────────────────────────────────────────
    // Reuses border_bgl (BorderU) + a triangle-list pipeline. 24 verts/border.
    frame_border_pipeline: wgpu::RenderPipeline,
    frame_border_vert_buf: wgpu::Buffer,   // up to 2 borders * 24 verts
    frame_border_uniform_buf: wgpu::Buffer, // dyn-offset, 2 slots of 256B
    frame_border_bg: wgpu::BindGroup,
    ob_size: f32, ob_r: f32, ob_g: f32, ob_b: f32, ob_a: f32,
    ib_size: f32, ib_r: f32, ib_g: f32, ib_b: f32, ib_a: f32,

    // ── Darken center ────────────────────────────────────────────────────────
    darken_pipeline: wgpu::RenderPipeline, // TriangleList, alpha blend
    darken_vert_buf: wgpu::Buffer,         // 12 verts (4 fan tris)
    darken_center: bool,

    // previous-frame volume, used to derive the MilkDrop-style `diff` pseudo-var
    // (frame-to-frame volume delta) so presets like orb_waaa can gate mv_a on it.
    vol_prev: f64,

    // ── Blur min/max (per-level range remap base; overridable per-frame via EEL) ─
    b1n: f32, b1x: f32, b2n: f32, b2x: f32, b3n: f32, b3x: f32,

    // per-sample audio waveform (range ~[-1,1]); filled by set_waveform or synthesized.
    wave_l: Vec<f32>,
    wave_r: Vec<f32>,
}

impl MilkdropRenderer {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
        shaders: &MilkShaders,
    ) -> Result<Self, String> {
        let (w, h) = (width.max(1), height.max(1));

        let has_custom_warp = shaders.warp.is_some();
        let has_custom_comp = shaders.comp.is_some();

        // Compile warp/comp shaders. Fallback body passes through sampler_main.
        // The legacy fullscreen warp FS (quad VS) is retained only as a dead
        // fallback; the live custom-warp path uses warp_custom_wgsl (mesh VS).
        //
        // shaders_glsl path (Butterchurn converted-JSON): the custom warp/comp
        // bodies are ALREADY GLSL — compile them via glsl_milk_*_to_naga (no HLSL
        // conversion). When no custom shader is present we fall back to the HLSL
        // path's default bodies (which emit valid GLSL either way). The .milk path
        // keeps shaders_glsl=false → identical to before (byte-for-byte).
        let warp_default = "ret = GetMain(uv);";
        let comp_default =
            "float _eh = mod(echo_orientation, 2.0); \
             float _ex = (_eh != 0.0) ? -1.0 : 1.0; \
             float _ey = (echo_orientation >= 2.0) ? -1.0 : 1.0; \
             vec2 uv_echo = ((uv - 0.5) * (1.0 / echo_zoom) * vec2(_ex, _ey)) + 0.5; \
             ret = mix(GetMain(uv), GetMain(uv_echo), echo_alpha); \
             ret = ret * gammaAdj; \
             ret = ret * hue_shader; \
             if (brighten != 0.0) ret = sqrt(ret); \
             if (darken   != 0.0) ret = ret * ret; \
             if (solarize != 0.0) ret = ret * (1.0 - ret) * 4.0; \
             if (invert   != 0.0) ret = 1.0 - ret;";

        // Legacy fullscreen warp FS (quad VS) — dead fallback, but still compiled.
        // Must use the GLSL-body comp template for JSON presets (the body is GLSL,
        // not HLSL) or naga chokes on the un-stripped `shader_body` wrapper.
        let warp_legacy_glsl = match (shaders.shaders_glsl, shaders.warp.as_deref()) {
            (true, Some(body)) => glsl_milk_body_to_naga(body),
            _ => hlsl_milk_body_to_naga(shaders.warp.as_deref().unwrap_or(warp_default)),
        };
        // Custom-warp FS driven by the warped mesh VS (vUv@0, vWarpUv@1, vDecay@2).
        let warp_custom_glsl = match (shaders.shaders_glsl, shaders.warp.as_deref()) {
            (true, Some(body)) => glsl_milk_warp_body_to_naga(body),
            _ => hlsl_milk_warp_body_to_naga(shaders.warp.as_deref().unwrap_or(warp_default)),
        };
        // Built-in (fallback) comp body: video echo applied BEFORE gamma, matching
        // Butterchurn's compShader (echo mix -> gamma -> post-FX flags). The GLSL
        // template (preprocess.rs) provides the y-flipped `uv`, GetMain(), and the
        // echo_*/gammaAdj/brighten/darken/solarize/invert uniforms in the PerFrame UBO.
        let comp_glsl = match (shaders.shaders_glsl, shaders.comp.as_deref()) {
            (true, Some(body)) => glsl_milk_body_to_naga(body),
            _ => hlsl_milk_body_to_naga(shaders.comp.as_deref().unwrap_or(comp_default)),
        };
        // Dump ALL generated GLSL before any compile, so a naga failure in one
        // shader still leaves the others visible for triage (MILKDROP_DUMP_GLSL).
        if std::env::var("MILKDROP_DUMP_GLSL").is_ok() {
            eprintln!("==== legacy warp GLSL ====\n{warp_legacy_glsl}\n==== end legacy warp ====");
            eprintln!("==== custom warp GLSL ====\n{warp_custom_glsl}\n==== end custom warp ====");
            eprintln!("==== comp GLSL ====\n{comp_glsl}\n==== end comp ====");
        }
        let warp_wgsl = compile_glsl(&warp_legacy_glsl)?;
        let warp_custom_wgsl = compile_glsl(&warp_custom_glsl)?;
        let comp_wgsl = compile_glsl(&comp_glsl)?;
        if std::env::var("MILKDROP_DUMP_WARP_WGSL").is_ok() {
            eprintln!("==== custom warp WGSL (naga) ====\n{warp_custom_wgsl}\n==== end warp WGSL ====");
        }

        // Create feedback textures — seed with low-level noise so the warp shader
        // has non-uniform gradients to amplify from the very first frame.
        let fb_usage = wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST;

        let seed_a = noise_bytes_scaled((w * h) as usize, 16); // 0–15 range (subtle)
        let seed_b = noise_bytes_scaled((w * h) as usize, 16);
        let tex_a = make_tex2d(&device, &queue, w, h, fb_usage, Some(&seed_a));
        let tex_b = make_tex2d(&device, &queue, w, h, fb_usage, Some(&seed_b));
        let view_a = tex_a.create_view(&Default::default());
        let view_b = tex_b.create_view(&Default::default());

        // Blur textures: 1/2, 1/4, 1/8 of main size
        let blur_usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let (bw1, bh1) = ((w / 2).max(1), (h / 2).max(1));
        let (bw2, bh2) = ((w / 4).max(1), (h / 4).max(1));
        let (bw3, bh3) = ((w / 8).max(1), (h / 8).max(1));

        let blur1 = make_tex2d(&device, &queue, bw1, bh1, blur_usage, None);
        let blur2 = make_tex2d(&device, &queue, bw2, bh2, blur_usage, None);
        let blur3 = make_tex2d(&device, &queue, bw3, bh3, blur_usage, None);
        let view_blur1 = blur1.create_view(&Default::default());
        let view_blur2 = blur2.create_view(&Default::default());
        let view_blur3 = blur3.create_view(&Default::default());

        // Separable blur needs a horizontal-pass intermediate per level (same res).
        let btemp1 = make_tex2d(&device, &queue, bw1, bh1, blur_usage, None);
        let btemp2 = make_tex2d(&device, &queue, bw2, bh2, blur_usage, None);
        let btemp3 = make_tex2d(&device, &queue, bw3, bh3, blur_usage, None);
        let view_btemp1 = btemp1.create_view(&Default::default());
        let view_btemp2 = btemp2.create_view(&Default::default());
        let view_btemp3 = btemp3.create_view(&Default::default());

        // Offscreen full-res comp target (Rgba8Unorm). COMP now writes here; the FXAA
        // OUTPUT pass reads it and resolves into the swapchain.
        let comp_tex = make_tex2d(&device, &queue, w, h, blur_usage, None);
        let comp_view = comp_tex.create_view(&Default::default());

        // Noise textures — Butterchurn-faithful value/lattice noise (noise.js).
        // LQ 256² zoom1 (random), MQ 256² zoom4 (smoothed), HQ 256² zoom8 (smoothed),
        // LQ-lite 32² zoom1, noisevol_lq 32³ zoom1, noisevol_hq 32³ zoom4 (smoothed).
        let tex_binding = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let mut rng = bc_rng();
        let n_lq   = create_noise_tex(256, 1, &mut rng);
        let n_mq   = create_noise_tex(256, 4, &mut rng);
        let n_hq   = create_noise_tex(256, 8, &mut rng);
        let n_lite = create_noise_tex(32, 1, &mut rng);
        let nv_lq  = create_noise_vol_tex(32, 1, &mut rng);
        let nv_hq  = create_noise_vol_tex(32, 4, &mut rng);

        let noise_lq    = make_tex2d(&device, &queue, 256, 256, tex_binding, Some(&n_lq));
        let noise_mq    = make_tex2d(&device, &queue, 256, 256, tex_binding, Some(&n_mq));
        let noise_hq    = make_tex2d(&device, &queue, 256, 256, tex_binding, Some(&n_hq));
        let noise_lite  = make_tex2d(&device, &queue, 32, 32, tex_binding, Some(&n_lite));
        let noisevol_lq = make_tex3d(&device, &queue, 32, &nv_lq);
        let noisevol_hq = make_tex3d(&device, &queue, 32, &nv_hq);

        let view_noise_lq   = noise_lq.create_view(&Default::default());
        let view_noise_mq   = noise_mq.create_view(&Default::default());
        let view_noise_hq   = noise_hq.create_view(&Default::default());
        let view_noise_lite = noise_lite.create_view(&Default::default());
        let view_noisevol_lq = noisevol_lq.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });
        let view_noisevol_hq = noisevol_hq.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });
        // Placeholder view for the unrelated fw/pw/pc sampler slots (2/6/8) — keep
        // a small 2D random texture for those, matching the old behaviour.
        let n_placeholder = noise_bytes(64 * 64);
        let noise2d = make_tex2d(&device, &queue, 64, 64, tex_binding, Some(&n_placeholder));
        let view_noise2d = noise2d.create_view(&Default::default());

        // Sampler
        let linear_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // Clamp sampler — MilkDrop "force clamp" (sampler_fc_main/pc) + the blur passes,
        // which must not wrap opposite-edge content into the borders.
        let clamp_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // UBO
        let perframe_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("perframe-ubo"),
            size: std::mem::size_of::<PerFrame>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Blur uniform buffers: BlurU { texel: vec4 (1/levelW, 1/levelH, 0, 0), edge: vec4 }.
        // Offsets are in the LEVEL's own texels (blur1 = half-res, etc.). Edge decay
        // (ed1=1-b1ed, ed2=b1ed, ed3=5) fades the blur toward the borders, per Butterchurn.
        let b1ed = 0.25f32; // both jelly presets set b1ed=0.25 (default until parsed)
        let edge = [1.0f32 - b1ed, b1ed, 5.0f32, 0.0f32];
        // BlurU = { texel:vec4, edge:vec4, sb:vec4 } (12 floats / 48 bytes). sb (scale,
        // bias) is rewritten per-frame (offset 32B) from the blur min/max range remap.
        let blur_ubo_contents = |bw: u32, bh: u32| -> [f32; 12] {
            [1.0 / bw as f32, 1.0 / bh as f32, 0.0, 0.0,
             edge[0], edge[1], edge[2], edge[3],
             1.0, 0.0, 0.0, 0.0] // sb = identity (scale 1, bias 0) until updated
        };
        let blur1_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blur1-ubo"),
            contents: bytemuck::cast_slice(&blur_ubo_contents(bw1, bh1)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let blur2_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blur2-ubo"),
            contents: bytemuck::cast_slice(&blur_ubo_contents(bw2, bh2)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let blur3_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("blur3-ubo"),
            contents: bytemuck::cast_slice(&blur_ubo_contents(bw3, bh3)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Bind group layouts
        let sampler_bgl  = sampler_bgl(&device);
        let perframe_bgl = perframe_bgl(&device);
        let blur_bgl     = blur_bgl(&device);

        // Pipeline layout for warp/comp
        let milk_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("milk-pl"),
            bind_group_layouts: &[Some(&sampler_bgl), Some(&perframe_bgl)],
            immediate_size: 0,
        });
        // Pipeline layout for blur
        let blur_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blur-pl"),
            bind_group_layouts: &[Some(&blur_bgl)],
            immediate_size: 0,
        });

        // Vertex shader (shared by all passes)
        let quad_src = include_str!("shaders/quad.wgsl");
        let quad_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quad-vs"),
            source: wgpu::ShaderSource::Wgsl(quad_src.into()),
        });

        // Warp pipeline
        let warp_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("warp-fs"),
            source: wgpu::ShaderSource::Wgsl(warp_wgsl.into()),
        });
        let warp_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("warp-pipeline"),
            layout: Some(&milk_pl),
            vertex: wgpu::VertexState {
                module: &quad_mod,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &warp_mod,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Comp pipeline
        let comp_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("comp-fs"),
            source: wgpu::ShaderSource::Wgsl(comp_wgsl.into()),
        });
        let comp_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("comp-pipeline"),
            layout: Some(&milk_pl),
            vertex: wgpu::VertexState {
                module: &quad_mod,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &comp_mod,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    // COMP now writes the offscreen Rgba8Unorm target (FXAA reads it).
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Blur pipeline
        let blur_src = include_str!("shaders/blur.wgsl");
        let blur_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blur-fs"),
            source: wgpu::ShaderSource::Wgsl(blur_src.into()),
        });
        let make_blur_pipeline = |label: &str, entry: &'static str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&blur_pl),
                vertex: wgpu::VertexState {
                    module: &quad_mod,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &blur_mod,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let blur_h_pipeline = make_blur_pipeline("blur-h-pipeline", "fs_blur_h");
        let blur_v_pipeline = make_blur_pipeline("blur-v-pipeline", "fs_blur_v");

        // FXAA OUTPUT pass: reads the offscreen comp result, resolves edges → swapchain.
        // Same BGL pattern as blur (texture/sampler/UBO); own self-contained VS+FS.
        // UBO = texsize vec4 (W, H, 1/W, 1/H). Static at construction size (internal
        // targets aren't recreated on window resize — consistent with existing tex_a/b).
        let fxaa_ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fxaa-ubo"),
            contents: bytemuck::cast_slice(&[w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let fxaa_bgl = crate::renderer::blur_bgl(&device); // identical layout: {0:tex D2}{1:sampler}{2:uniform}
        let fxaa_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fxaa-pl"),
            bind_group_layouts: &[Some(&fxaa_bgl)],
            immediate_size: 0,
        });
        let fxaa_src = include_str!("shaders/fxaa.wgsl");
        let fxaa_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fxaa"),
            source: wgpu::ShaderSource::Wgsl(fxaa_src.into()),
        });
        let output_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fxaa-output-pipeline"),
            layout: Some(&fxaa_pl),
            vertex: wgpu::VertexState {
                module: &fxaa_mod,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fxaa_mod,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format, // ← the swapchain format
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let fxaa_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fxaa-bg"),
            layout: &fxaa_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&comp_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&linear_samp) },
                wgpu::BindGroupEntry { binding: 2, resource: fxaa_ubo.as_entire_binding() },
            ],
        });

        // Perframe bind group
        let ubo_binding = (MILKDROP_SAMPLERS.len() * 2) as u32;
        let perframe_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("perframe-bg"),
            layout: &perframe_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: ubo_binding,
                resource: perframe_buf.as_entire_binding(),
            }],
        });

        // Sampler bind groups (two, one per ping-pong side)
        let bg_read_a = build_sampler_bg(
            &device, &sampler_bgl, &view_a,
            &view_blur1, &view_blur2, &view_blur3,
            &view_noise2d,
            &view_noise_lq, &view_noise_mq, &view_noise_hq, &view_noise_lite,
            &view_noisevol_lq, &view_noisevol_hq,
            &linear_samp, &clamp_samp,
        );
        let bg_read_b = build_sampler_bg(
            &device, &sampler_bgl, &view_b,
            &view_blur1, &view_blur2, &view_blur3,
            &view_noise2d,
            &view_noise_lq, &view_noise_mq, &view_noise_hq, &view_noise_lite,
            &view_noisevol_lq, &view_noisevol_hq,
            &linear_samp, &clamp_samp,
        );

        // Blur bind groups — separable: each level does H (src→temp) then V (temp→level).
        // Clamp sampler avoids wrapping opposite-edge content into the blur near borders.
        let make_blur_bg = |src_view: &wgpu::TextureView, ubo: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &blur_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&clamp_samp) },
                    wgpu::BindGroupEntry { binding: 2, resource: ubo.as_entire_binding() },
                ],
            })
        };
        // blur1 H reads the warp output (write_view) → rebuilt each frame in render().
        let blur1_v_bg = make_blur_bg(&view_btemp1, &blur1_ubo);
        let blur2_h_bg = make_blur_bg(&view_blur1,  &blur2_ubo);
        let blur2_v_bg = make_blur_bg(&view_btemp2, &blur2_ubo);
        let blur3_h_bg = make_blur_bg(&view_blur2,  &blur3_ubo);
        let blur3_v_bg = make_blur_bg(&view_btemp3, &blur3_ubo);

        // ── Warp mesh pipeline (used when no custom warp shader) ─────────────
        // Decay is now per-vertex (vertex buffer attribute 2), so the old decay
        // UBO is gone; the mesh bind group is just {texture, sampler}.
        let warp_mesh_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("warp-mesh-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // bTexWrap: wrap=1 → repeat (linear_samp); wrap=0 → clamp (clamp_samp).
        let mesh_samp: &wgpu::Sampler = if shaders.wrap { &linear_samp } else { &clamp_samp };
        let make_mesh_bg = |tv: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &warp_mesh_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(tv) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(mesh_samp) },
                ],
            })
        };
        let warp_mesh_bg_a = make_mesh_bg(&view_a);
        let warp_mesh_bg_b = make_mesh_bg(&view_b);

        let num_verts = (GRID_W + 1) * (GRID_H + 1);
        let warp_vert_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("warp-verts"),
            size: (num_verts as usize * std::mem::size_of::<WarpVert>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let warp_indices = build_warp_indices();
        let warp_idx_count = warp_indices.len() as u32;
        let warp_idx_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("warp-indices"),
            contents: bytemuck::cast_slice(&warp_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let warp_mesh_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("warp-mesh-pl"),
            bind_group_layouts: &[Some(&warp_mesh_bgl)],
            immediate_size: 0,
        });
        let warp_mesh_src = include_str!("shaders/warp_mesh.wgsl");
        let warp_mesh_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("warp-mesh"),
            source: wgpu::ShaderSource::Wgsl(warp_mesh_src.into()),
        });
        // WarpVert attributes shared by both warp pipelines: pos@0, uv@1, decay@2.
        let warp_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<WarpVert>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { offset: 0,  shader_location: 0, format: wgpu::VertexFormat::Float32x2 },
                wgpu::VertexAttribute { offset: 8,  shader_location: 1, format: wgpu::VertexFormat::Float32x2 },
                wgpu::VertexAttribute { offset: 16, shader_location: 2, format: wgpu::VertexFormat::Float32x4 },
            ],
        };
        let warp_mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("warp-mesh-pipeline"),
            layout: Some(&warp_mesh_pl),
            vertex: wgpu::VertexState {
                module: &warp_mesh_mod,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[warp_vbl.clone()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &warp_mesh_mod,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // ── Custom-warp pipeline: warped mesh VS + the per-preset custom warp FS.
        // Uses the milk_pl layout (sampler_bgl + perframe_bgl) so it can sample
        // the full MilkDrop sampler set (sampler_main/fc_main/blur*) like comp.
        let warp_mesh_vs_src = include_str!("shaders/warp_mesh_vs.wgsl");
        let warp_mesh_vs_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("warp-mesh-vs"),
            source: wgpu::ShaderSource::Wgsl(warp_mesh_vs_src.into()),
        });
        let warp_custom_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("warp-custom-fs"),
            source: wgpu::ShaderSource::Wgsl(warp_custom_wgsl.into()),
        });
        let warp_custom_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("warp-custom-pipeline"),
            layout: Some(&milk_pl),
            vertex: wgpu::VertexState {
                module: &warp_mesh_vs_mod,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[warp_vbl.clone()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &warp_custom_mod,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // ── Custom-shape pipelines/buffers ───────────────────────────────────
        let shape_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shape-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let border_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("border-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<BorderU>() as u64),
                },
                count: None,
            }],
        });

        let shapes_src = include_str!("shaders/shapes.wgsl");
        let shapes_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shapes"),
            source: wgpu::ShaderSource::Wgsl(shapes_src.into()),
        });

        let shape_fill_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shape-fill-pl"),
            bind_group_layouts: &[Some(&shape_bgl)],
            immediate_size: 0,
        });
        let shape_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ShapeVert>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { offset: 0,  shader_location: 0, format: wgpu::VertexFormat::Float32x2 },
                wgpu::VertexAttribute { offset: 8,  shader_location: 1, format: wgpu::VertexFormat::Float32x4 },
                wgpu::VertexAttribute { offset: 24, shader_location: 2, format: wgpu::VertexFormat::Float32x2 },
            ],
        };
        let blend_alpha = wgpu::BlendState {
            color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::SrcAlpha, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
            alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, operation: wgpu::BlendOperation::Add },
        };
        let blend_additive = wgpu::BlendState {
            color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::SrcAlpha, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
            alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
        };
        let make_shape_fill = |label: &str, blend: wgpu::BlendState| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&shape_fill_pl),
                vertex: wgpu::VertexState {
                    module: &shapes_mod,
                    entry_point: Some("vs_shape"),
                    compilation_options: Default::default(),
                    buffers: &[shape_vbl.clone()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shapes_mod,
                    entry_point: Some("fs_shape"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let shapes_fill_pipeline_alpha    = make_shape_fill("shape-fill-alpha", blend_alpha);
        let shapes_fill_pipeline_additive = make_shape_fill("shape-fill-additive", blend_additive);

        let border_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shape-border-pl"),
            bind_group_layouts: &[Some(&border_bgl)],
            immediate_size: 0,
        });
        let border_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<BorderVert>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x2 }],
        };
        let shapes_border_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shape-border"),
            layout: Some(&border_pl),
            vertex: wgpu::VertexState {
                module: &shapes_mod,
                entry_point: Some("vs_border"),
                compilation_options: Default::default(),
                buffers: &[border_vbl],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shapes_mod,
                entry_point: Some("fs_border"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(blend_alpha),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineStrip, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let shape_vert_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shape-verts"),
            size: (SHAPE_VERT_CAP * std::mem::size_of::<ShapeVert>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let border_vert_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("border-verts"),
            size: (BORDER_VERT_CAP * std::mem::size_of::<BorderVert>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Static fan triangulation: [0, k, k+1] for k in 1..=SIDES_MAX.
        let mut fan_idx: Vec<u32> = Vec::with_capacity(SHAPE_FAN_IDX_MAX);
        for k in 1..=(SIDES_MAX as u32) { fan_idx.extend_from_slice(&[0, k, k + 1]); }
        let shape_idx_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shape-fan-idx"),
            contents: bytemuck::cast_slice(&fan_idx),
            usage: wgpu::BufferUsages::INDEX,
        });
        let shape_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shape-u"),
            size: std::mem::size_of::<ShapeU>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // border dyn-offset uniform: 4 slots of 256 bytes
        let border_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("border-u"),
            size: 4 * 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let border_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("border-bg"),
            layout: &border_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &border_uniform_buf, offset: 0,
                    size: std::num::NonZeroU64::new(std::mem::size_of::<BorderU>() as u64),
                }),
            }],
        });
        let make_shape_bg = |tv: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shape-bg"),
                layout: &shape_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(tv) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&linear_samp) },
                    wgpu::BindGroupEntry { binding: 2, resource: shape_uniform_buf.as_entire_binding() },
                ],
            })
        };
        let shape_bg_read_a = make_shape_bg(&view_a);
        let shape_bg_read_b = make_shape_bg(&view_b);

        // Preset-wide gmegabuf, shared by every EEL pool (per-frame, per-pixel,
        // each shape, each wave). megabuf is per-pool (private to each EelState).
        let gmegabuf: Rc<RefCell<MegaBuf>> = Rc::new(RefCell::new(MegaBuf::default()));

        // build ShapeRT list from parsed shapes
        let shapes: Vec<ShapeRT> = shaders.shapes.iter().map(|sc| {
            let mut env = Env::new();
            let mut state = EelState::with_gmegabuf(gmegabuf.clone());
            // Run shape per-frame-init ONCE into the shape env/megabuf at load.
            if let Some(init) = sc.per_frame_init.as_deref() {
                EelProgram::parse(init).run_with(&mut env, &mut state);
            }
            ShapeRT {
                base: sc.base.clone(),
                prog: sc.per_frame.as_deref().map(EelProgram::parse),
                env,
                state,
            }
        }).collect();

        // ── Waveform pipelines/buffers ───────────────────────────────────────
        let wave_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("wave-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(16),
                },
                count: None,
            }],
        });
        let wave_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("wave-pl"),
            bind_group_layouts: &[Some(&wave_bgl)],
            immediate_size: 0,
        });
        let wave_src = include_str!("shaders/wave.wgsl");
        let wave_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wave"),
            source: wgpu::ShaderSource::Wgsl(wave_src.into()),
        });
        let wave_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<WaveVert>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x2 },
                wgpu::VertexAttribute { offset: 8, shader_location: 1, format: wgpu::VertexFormat::Float32x4 },
            ],
        };
        let make_wave_pipeline = |label: &str, topo: wgpu::PrimitiveTopology, blend: wgpu::BlendState| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&wave_pl),
                vertex: wgpu::VertexState {
                    module: &wave_mod,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[wave_vbl.clone()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &wave_mod,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState { topology: topo, ..Default::default() },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let wave_pipeline_lines_alpha     = make_wave_pipeline("wave-lines-alpha",  wgpu::PrimitiveTopology::LineStrip, blend_alpha);
        let wave_pipeline_lines_additive  = make_wave_pipeline("wave-lines-add",    wgpu::PrimitiveTopology::LineStrip, blend_additive);
        let wave_pipeline_points_alpha    = make_wave_pipeline("wave-points-alpha", wgpu::PrimitiveTopology::PointList, blend_alpha);
        let wave_pipeline_points_additive = make_wave_pipeline("wave-points-add",   wgpu::PrimitiveTopology::PointList, blend_additive);

        let wave_vert_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wave-verts"),
            size: (WAVE_VERT_CAP * std::mem::size_of::<WaveVert>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let wave_off_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wave-off"),
            size: 4 * 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let wave_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wave-bg"),
            layout: &wave_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &wave_off_buf, offset: 0,
                    size: std::num::NonZeroU64::new(16),
                }),
            }],
        });

        // ── Motion-vectors pipeline (LineList, single flat color uniform) ─────
        let mv_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mv-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(16),
                },
                count: None,
            }],
        });
        let mv_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mv-pl"),
            bind_group_layouts: &[Some(&mv_bgl)],
            immediate_size: 0,
        });
        let mv_src = include_str!("shaders/motion_vectors.wgsl");
        let mv_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("motion-vectors"),
            source: wgpu::ShaderSource::Wgsl(mv_src.into()),
        });
        let mv_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MVVert>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x2 }],
        };
        let mv_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("motion-vectors"),
            layout: Some(&mv_pl),
            vertex: wgpu::VertexState {
                module: &mv_mod, entry_point: Some("vs_main"),
                compilation_options: Default::default(), buffers: &[mv_vbl],
            },
            fragment: Some(wgpu::FragmentState {
                module: &mv_mod, entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(blend_alpha),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let mv_vert_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mv-verts"),
            size: (MV_VERT_CAP * std::mem::size_of::<MVVert>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mv_color_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mv-color"),
            size: std::mem::size_of::<MVColor>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mv_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mv-bg"),
            layout: &mv_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: mv_color_buf.as_entire_binding() }],
        });

        // ── Frame-border pipeline (TriangleList; reuses border_bgl / BorderU) ─
        let frame_border_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("frame-border-pl"),
            bind_group_layouts: &[Some(&border_bgl)],
            immediate_size: 0,
        });
        let frame_border_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<BorderVert>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x2 }],
        };
        let frame_border_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("frame-border"),
            layout: Some(&frame_border_pl),
            vertex: wgpu::VertexState {
                module: &shapes_mod, entry_point: Some("vs_border"),
                compilation_options: Default::default(), buffers: &[frame_border_vbl],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shapes_mod, entry_point: Some("fs_border"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(blend_alpha),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        // up to 2 borders (outer + inner), 24 verts each.
        let frame_border_vert_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame-border-verts"),
            size: (2 * 24 * std::mem::size_of::<BorderVert>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // dyn-offset uniform: 2 slots of 256B (outer color in slot 0, inner in slot 1)
        let frame_border_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame-border-u"),
            size: 2 * 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_border_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame-border-bg"),
            layout: &border_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &frame_border_uniform_buf, offset: 0,
                    size: std::num::NonZeroU64::new(std::mem::size_of::<BorderU>() as u64),
                }),
            }],
        });

        // ── Darken-center pipeline (TriangleList, per-vertex color, alpha blend) ─
        let darken_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("darken-pl"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let darken_src = include_str!("shaders/darken_center.wgsl");
        let darken_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("darken-center"),
            source: wgpu::ShaderSource::Wgsl(darken_src.into()),
        });
        let darken_vbl = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<DarkenVert>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x2 },
                wgpu::VertexAttribute { offset: 8, shader_location: 1, format: wgpu::VertexFormat::Float32x4 },
            ],
        };
        let darken_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("darken-center"),
            layout: Some(&darken_pl),
            vertex: wgpu::VertexState {
                module: &darken_mod, entry_point: Some("vs_main"),
                compilation_options: Default::default(), buffers: &[darken_vbl],
            },
            fragment: Some(wgpu::FragmentState {
                module: &darken_mod, entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(blend_alpha),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        // 4 fan triangles expanded to a triangle list = 12 verts.
        let darken_vert_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("darken-verts"),
            size: (12 * std::mem::size_of::<DarkenVert>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let waves: Vec<WaveRT> = shaders.waves.iter().map(|wd| {
            let mut env = Env::new();
            let mut state = EelState::with_gmegabuf(gmegabuf.clone());
            // Run wave per-frame-init ONCE into the wave env/megabuf at load.
            if let Some(init) = wd.per_frame_init.as_deref() {
                EelProgram::parse(init).run_with(&mut env, &mut state);
            }
            WaveRT {
                def: CustomWaveDef {
                    index: wd.index, enabled: wd.enabled, samples: wd.samples, sep: wd.sep,
                    spectrum: wd.spectrum, use_dots: wd.use_dots, draw_thick: wd.draw_thick,
                    additive: wd.additive, scaling: wd.scaling, smoothing: wd.smoothing,
                    r: wd.r, g: wd.g, b: wd.b, a: wd.a,
                    per_frame: wd.per_frame.clone(),
                    per_frame_init: wd.per_frame_init.clone(),
                    per_point: wd.per_point.clone(),
                },
                per_frame_prog: wd.per_frame.as_deref().map(EelProgram::parse),
                per_point_prog: wd.per_point.as_deref().map(EelProgram::parse),
                env,
                state,
            }
        }).collect();

        // EEL2 per-frame equations
        let eel_program = shaders.per_frame.as_deref().map(EelProgram::parse);
        let mut eel_env = Env::new();
        let mut eel_state = EelState::with_gmegabuf(gmegabuf.clone());
        // Seed header echo/gamma defaults so header-only echo presets work and the
        // values persist across frames if the per-frame program never assigns them.
        // EEL/Butterchurn use the var name "echo_orient" (NOT "echo_orientation").
        eel_env.insert("echo_zoom".into(),   shaders.echo_zoom   as f64);
        eel_env.insert("echo_alpha".into(),  shaders.echo_alpha  as f64);
        eel_env.insert("echo_orient".into(), shaders.echo_orient as f64);
        eel_env.insert("gamma".into(),       shaders.gamma_adj   as f64);

        // Run per-frame INIT equations ONCE before frame 0, into the persistent
        // per-frame env/megabuf. per_frame then sees the initialized vars. We then
        // snapshot q1..q32 so we can RESET them to their post-init values at the
        // top of every frame (Butterchurn's mdVS = {...mdVS, ...mdVSQInit}).
        if let Some(init) = shaders.per_frame_init.as_deref() {
            EelProgram::parse(init).run_with(&mut eel_env, &mut eel_state);
        }
        let q_init: Vec<(String, f64)> = (1..=32)
            .filter_map(|i| {
                let k = format!("q{i}");
                eel_env.get(&k).copied().map(|v| (k, v))
            })
            .collect();

        // Per-vertex warp (per_pixel) program + per-frame warp base values.
        let per_pixel_prog = shaders.per_pixel.as_deref().map(EelProgram::parse);
        let warp_state = EelState::with_gmegabuf(gmegabuf.clone());
        let base_warp = WarpBase {
            zoom: shaders.zoom, zoomexp: shaders.zoomexp, rot: shaders.rot,
            warp: shaders.warp_amount,
            cx: shaders.cx, cy: shaders.cy, dx: shaders.dx, dy: shaders.dy,
            sx: shaders.sx, sy: shaders.sy,
            warpscale: shaders.warpscale, warpanimspeed: shaders.warpanimspeed,
            decay: shaders.decay, wrap: shaders.wrap,
        };
        let warp_env = Env::new();

        // Per-preset hue seed (Butterchurn rand_start). Hash the shader/equation text so
        // each preset gets a distinct, reproducible hue instead of the green-biased 0.5.
        let mut seed_src = String::new();
        seed_src.push_str(shaders.warp.as_deref().unwrap_or(""));
        seed_src.push_str(shaders.comp.as_deref().unwrap_or(""));
        seed_src.push_str(shaders.per_frame.as_deref().unwrap_or(""));
        let rand_preset = preset_hue_seed(&seed_src);

        Ok(Self {
            device, queue,
            has_custom_warp,
            has_custom_comp,
            preset_decay: shaders.decay,
            rand_preset,
            tex_a, tex_b, view_a, view_b,
            write_to_a: true,
            blur1, blur2, blur3,
            view_blur1, view_blur2, view_blur3,
            btemp1, btemp2, btemp3,
            view_btemp1, view_btemp2, view_btemp3,
            noise2d, noise_lq, noise_mq, noise_hq, noise_lite, noisevol_lq, noisevol_hq,
            view_noise2d, view_noise_lq, view_noise_mq, view_noise_hq, view_noise_lite,
            view_noisevol_lq, view_noisevol_hq,
            linear_samp,
            clamp_samp,
            perframe_buf,
            blur1_ubo, blur2_ubo, blur3_ubo,
            warp_pipeline, warp_custom_pipeline, comp_pipeline, blur_h_pipeline, blur_v_pipeline,
            comp_tex, comp_view, output_pipeline, fxaa_bgl, fxaa_ubo, fxaa_bg,
            warp_mesh_pipeline,
            warp_mesh_bg_a,
            warp_mesh_bg_b,
            warp_mesh_bgl,
            warp_vert_buf,
            warp_idx_buf,
            warp_idx_count,
            sampler_bgl, perframe_bgl, blur_bgl,
            bg_read_a, bg_read_b,
            perframe_bg,
            blur1_v_bg, blur2_h_bg, blur2_v_bg, blur3_h_bg, blur3_v_bg,
            eel_program,
            eel_env,
            eel_state,
            gmegabuf,
            q_init,
            per_pixel_prog,
            base_warp,
            warp_env,
            warp_state,
            frame_idx: 0,
            start: std::time::Instant::now(),
            time_per_frame: None,
            audio: None,
            audio_att: None,
            freq_spectrum: Vec::new(),
            width: w, height: h,
            surface_format,

            shapes,
            shapes_fill_pipeline_alpha,
            shapes_fill_pipeline_additive,
            shapes_border_pipeline,
            shape_bgl,
            border_bgl,
            shape_vert_buf,
            shape_idx_buf,
            shape_uniform_buf,
            border_vert_buf,
            border_uniform_buf,
            border_bg,
            shape_bg_read_a,
            shape_bg_read_b,

            waves,
            wave_pipeline_lines_alpha,
            wave_pipeline_lines_additive,
            wave_pipeline_points_alpha,
            wave_pipeline_points_additive,
            wave_bgl,
            wave_vert_buf,
            wave_off_buf,
            wave_bg,

            bw_mode: shaders.wave_mode,
            bw_x: shaders.wave_x, bw_y: shaders.wave_y,
            bw_r: shaders.wave_r, bw_g: shaders.wave_g, bw_b: shaders.wave_b, bw_a: shaders.wave_a,
            bw_mystery: shaders.wave_mystery, bw_scale: shaders.wave_scale, bw_smoothing: shaders.wave_smoothing,
            bw_dots: shaders.wave_dots, bw_thick: shaders.wave_thick,
            bw_additive: shaders.additive_wave, bw_brighten: shaders.wave_brighten,
            bw_modalphavol: shaders.modwavealphabyvolume,
            bw_modalphastart: shaders.modwavealphastart,
            bw_modalphaend: shaders.modwavealphaend,

            comp_brighten: shaders.brighten,
            comp_darken:   shaders.darken,
            comp_solarize: shaders.solarize,
            comp_invert:   shaders.invert,

            mv_pipeline, mv_bgl, mv_vert_buf, mv_color_buf, mv_bg,
            mv_on: shaders.mv_on, mv_x: shaders.mv_x, mv_y: shaders.mv_y,
            mv_dx: shaders.mv_dx, mv_dy: shaders.mv_dy, mv_l: shaders.mv_l,
            mv_r: shaders.mv_r, mv_g: shaders.mv_g, mv_b: shaders.mv_b, mv_a: shaders.mv_a,

            frame_border_pipeline, frame_border_vert_buf,
            frame_border_uniform_buf, frame_border_bg,
            ob_size: shaders.ob_size, ob_r: shaders.ob_r, ob_g: shaders.ob_g,
            ob_b: shaders.ob_b, ob_a: shaders.ob_a,
            ib_size: shaders.ib_size, ib_r: shaders.ib_r, ib_g: shaders.ib_g,
            ib_b: shaders.ib_b, ib_a: shaders.ib_a,

            darken_pipeline, darken_vert_buf,
            darken_center: shaders.darken_center,
            vol_prev: 0.0,

            b1n: shaders.b1n, b1x: shaders.b1x, b2n: shaders.b2n,
            b2x: shaders.b2x, b3n: shaders.b3n, b3x: shaders.b3x,

            wave_l: Vec::new(),
            wave_r: Vec::new(),
        })
    }

    /// Switch to deterministic fixed-timestep timing (for offscreen animation
    /// export). Each rendered frame advances `time` by `1/fps` seconds.
    pub fn set_fixed_fps(&mut self, fps: f32) {
        self.time_per_frame = Some(1.0 / fps.max(1.0));
    }

    /// Feed live audio reactivity for the next frame. Values are MilkDrop-style
    /// band levels (~1.0 = average energy, 0 = silent, >1 = loud). Once set, the
    /// synthetic sine-wave fallback is disabled.
    pub fn set_audio(&mut self, bass: f32, mid: f32, treb: f32, vol: f32) {
        self.audio = Some([bass, mid, treb, vol]);
    }

    /// Feed the attenuated (smoothed) reactivity envelopes for the next frame —
    /// the MilkDrop `bass_att`/`mid_att`/`treb_att`/`vol_att` inputs. These lag
    /// peaks relative to [`set_audio`]. If never called, `*_att` mirrors the
    /// non-att values (preserving the prior headless behavior bit-for-bit).
    pub fn set_audio_att(&mut self, bass_att: f32, mid_att: f32, treb_att: f32, vol_att: f32) {
        self.audio_att = Some([bass_att, mid_att, treb_att, vol_att]);
    }

    /// Feed the Butterchurn-shaped 512-bin FFT magnitude array (`freqArray`) for
    /// the next frame. Used by `bSpectrum` custom waveforms. Pass an empty slice
    /// (or never call) to keep the time-domain fallback.
    pub fn set_freq_spectrum(&mut self, spectrum: &[f32]) {
        self.freq_spectrum.clear();
        self.freq_spectrum.extend_from_slice(spectrum);
    }

    /// Feed per-sample PCM waveform for the next frame (range ~[-1,1]). Used by
    /// the built-in and custom waveforms. Length equals the audio buffer length.
    pub fn set_waveform(&mut self, left: &[f32], right: &[f32]) {
        self.wave_l.clear();
        self.wave_l.extend_from_slice(left);
        self.wave_r.clear();
        self.wave_r.extend_from_slice(right);
    }

    /// Get the per-sample waveform for this frame, synthesizing a deterministic
    /// animated waveform when no real audio is available (headless/anim).
    fn frame_waveform(&self, t: f32) -> (Vec<f32>, Vec<f32>) {
        if !self.wave_l.is_empty() {
            return (self.wave_l.clone(), self.wave_r.clone());
        }
        // Synthesize 512 samples in [-1,1] that animate with time so the wave moves.
        // 512 (matching real butterchurn-parity feeds) is required so built-in modes
        // 1/2/3/5 (which index wave[i+32]) and 4/6/7 (capped at ~width/3) have enough
        // samples and don't degenerate or index out of bounds.
        let n = 512usize;
        let mut l = Vec::with_capacity(n);
        let mut r = Vec::with_capacity(n);
        for i in 0..n {
            let fi = i as f32;
            let a = 0.5 * (t * 6.0 + fi * 0.49).sin()
                  + 0.3 * (t * 2.1 + fi * 0.21).sin()
                  + 0.18 * (t * 11.3 + fi * 0.83).sin();
            let b = 0.5 * (t * 5.3 + fi * 0.55 + 1.7).sin()
                  + 0.3 * (t * 2.7 + fi * 0.19 + 0.4).sin()
                  + 0.18 * (t * 9.1 + fi * 0.77 + 2.1).sin();
            l.push(a.clamp(-1.0, 1.0));
            r.push(b.clamp(-1.0, 1.0));
        }
        (l, r)
    }

    /// Build CPU-side fill + border geometry for all enabled shapes this frame.
    /// Returns (fill_verts, fill_draws, border_verts, border_draws).
    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    fn build_shape_geometry(
        &mut self,
        t: f32,
        bass: f64, mid: f64, treb: f64, vol: f64,
        bass_att: f64, mid_att: f64, treb_att: f64,
        aspectx: f32, aspecty: f32,
        q: &[f64; 32],
    ) -> (Vec<ShapeVert>, Vec<ShapeFillDraw>, Vec<BorderVert>, Vec<BorderDraw>) {
        use std::f32::consts::PI;
        let mut fill_verts: Vec<ShapeVert> = Vec::new();
        let mut fill_draws: Vec<ShapeFillDraw> = Vec::new();
        let mut border_verts: Vec<BorderVert> = Vec::new();
        let mut border_draws: Vec<BorderDraw> = Vec::new();

        for s in self.shapes.iter_mut() {
            if s.base.enabled == 0 { continue; }
            let num_inst = (s.base.num_inst.max(1)).min(1024);

            for j in 0..num_inst {
                // Resolve per-instance vals: run per-frame eqs if present, else base.
                let (sides_f, rad, ang, x, y,
                     r, g, b, a, r2, g2, b2, a2,
                     border_r, border_g, border_b, border_a,
                     thick, textured, tex_ang, tex_zoom, additive);

                if let Some(prog) = &s.prog {
                    // Reset shape vars from base each instance (butterchurn semantics).
                    let env = &mut s.env;
                    env.insert("time".into(), t as f64);
                    env.insert("frame".into(), self.frame_idx as f64);
                    env.insert("fps".into(), 60.0);
                    env.insert("bass".into(), bass);  env.insert("bass_att".into(), bass_att);
                    env.insert("mid".into(), mid);    env.insert("mid_att".into(), mid_att);
                    env.insert("treb".into(), treb);  env.insert("treb_att".into(), treb_att);
                    env.insert("vol".into(), vol);
                    env.insert("aspectx".into(), aspectx as f64);
                    env.insert("aspecty".into(), aspecty as f64);
                    for (i, qv) in q.iter().enumerate() { env.insert(format!("q{}", i + 1), *qv); }
                    env.insert("instance".into(), j as f64);
                    env.insert("num_inst".into(), num_inst as f64);
                    let bv = &s.base;
                    env.insert("sides".into(), bv.sides as f64);
                    env.insert("rad".into(), bv.rad as f64);
                    env.insert("ang".into(), bv.ang as f64);
                    env.insert("x".into(), bv.x as f64);
                    env.insert("y".into(), bv.y as f64);
                    env.insert("r".into(), bv.r as f64);   env.insert("g".into(), bv.g as f64);
                    env.insert("b".into(), bv.b as f64);   env.insert("a".into(), bv.a as f64);
                    env.insert("r2".into(), bv.r2 as f64); env.insert("g2".into(), bv.g2 as f64);
                    env.insert("b2".into(), bv.b2 as f64); env.insert("a2".into(), bv.a2 as f64);
                    env.insert("border_r".into(), bv.border_r as f64);
                    env.insert("border_g".into(), bv.border_g as f64);
                    env.insert("border_b".into(), bv.border_b as f64);
                    env.insert("border_a".into(), bv.border_a as f64);
                    env.insert("thickoutline".into(), bv.thick_outline as f64);
                    env.insert("textured".into(), bv.textured as f64);
                    env.insert("tex_ang".into(), bv.tex_ang as f64);
                    env.insert("tex_zoom".into(), bv.tex_zoom as f64);
                    env.insert("additive".into(), bv.additive as f64);
                    prog.run_with(env, &mut s.state);
                    let rd = |k: &str, d: f64| env.get(k).copied().unwrap_or(d) as f32;
                    sides_f = rd("sides", bv.sides as f64);
                    rad = rd("rad", bv.rad as f64);
                    ang = rd("ang", bv.ang as f64);
                    x = rd("x", bv.x as f64);
                    y = rd("y", bv.y as f64);
                    r = rd("r", bv.r as f64); g = rd("g", bv.g as f64);
                    b = rd("b", bv.b as f64); a = rd("a", bv.a as f64);
                    r2 = rd("r2", bv.r2 as f64); g2 = rd("g2", bv.g2 as f64);
                    b2 = rd("b2", bv.b2 as f64); a2 = rd("a2", bv.a2 as f64);
                    border_r = rd("border_r", bv.border_r as f64);
                    border_g = rd("border_g", bv.border_g as f64);
                    border_b = rd("border_b", bv.border_b as f64);
                    border_a = rd("border_a", bv.border_a as f64);
                    thick = rd("thickoutline", bv.thick_outline as f64);
                    textured = rd("textured", bv.textured as f64);
                    tex_ang = rd("tex_ang", bv.tex_ang as f64);
                    tex_zoom = rd("tex_zoom", bv.tex_zoom as f64);
                    additive = rd("additive", bv.additive as f64);
                } else {
                    let bv = &s.base;
                    sides_f = bv.sides; rad = bv.rad; ang = bv.ang; x = bv.x; y = bv.y;
                    r = bv.r; g = bv.g; b = bv.b; a = bv.a;
                    r2 = bv.r2; g2 = bv.g2; b2 = bv.b2; a2 = bv.a2;
                    border_r = bv.border_r; border_g = bv.border_g; border_b = bv.border_b; border_a = bv.border_a;
                    thick = bv.thick_outline as f32; textured = bv.textured as f32;
                    tex_ang = bv.tex_ang; tex_zoom = bv.tex_zoom; additive = bv.additive as f32;
                }

                let blend_progress = 1.0f32;
                let sides = (sides_f.clamp(3.0, 100.0)).floor() as u32;
                let x_ndc = x * 2.0 - 1.0;
                let y_ndc = y * (-2.0) + 1.0;
                let is_additive = additive.abs() >= 1.0;
                let is_textured = textured.abs() >= 1.0;
                let is_thick = thick.abs() >= 1.0;
                let border_alpha = border_a * blend_progress;
                let has_border = border_alpha > 0.0;
                let quarter_pi = PI * 0.25;

                let base_vertex = fill_verts.len() as i32;

                // center vertex (uv sentinel (-1,-1) when untextured → solid color in FS)
                fill_verts.push(ShapeVert {
                    pos: [x_ndc, y_ndc],
                    color: [r, g, b, a * blend_progress],
                    uv: if is_textured { [0.5, 0.5] } else { [-1.0, -1.0] },
                });

                let border_start = border_verts.len() as u32;
                // rim vertices k = 1..=sides+1 (last duplicates first to close)
                for k in 1..=(sides + 1) {
                    let p = (k - 1) as f32 / sides as f32;
                    let p_two_pi = p * 2.0 * PI;
                    let ang_sum = p_two_pi + ang + quarter_pi;
                    let px = x_ndc + rad * ang_sum.cos() * aspecty;
                    let py = y_ndc + rad * ang_sum.sin();
                    let (uu, vv) = if is_textured {
                        let tex_ang_sum = p_two_pi + tex_ang + quarter_pi;
                        let z = if tex_zoom.abs() < 1e-6 { 1.0 } else { tex_zoom };
                        (0.5 + (0.5 * tex_ang_sum.cos() / z) * aspecty,
                         0.5 + (0.5 * tex_ang_sum.sin() / z))
                    } else { (-1.0, -1.0) };
                    fill_verts.push(ShapeVert {
                        pos: [px, py],
                        color: [r2, g2, b2, a2 * blend_progress],
                        uv: [uu, vv],
                    });
                    if has_border { border_verts.push(BorderVert { pos: [px, py] }); }
                }

                fill_draws.push(ShapeFillDraw { base_vertex, sides, additive: is_additive });

                if has_border {
                    border_draws.push(BorderDraw {
                        start_vert: border_start,
                        count: sides + 1,
                        color: [border_r, border_g, border_b, border_alpha],
                        thick: is_thick,
                    });
                }
            }
            // textured flag is per-shape (we honor the first instance's via uniform).
            // jelly_space is untextured, so the textured path is wired but uses uv=0.5.
        }

        (fill_verts, fill_draws, border_verts, border_draws)
    }

    /// Build CPU-side waveform geometry (built-in + custom). Returns the packed
    /// vertex list and the draw records.
    #[allow(clippy::too_many_arguments)]
    fn build_wave_geometry(
        &mut self,
        t: f32,
        bass: f64, mid: f64, treb: f64, vol: f64,
        bass_att: f64, mid_att: f64, treb_att: f64,
        inv_aspectx: f32, inv_aspecty: f32,
        wave_l: &[f32], wave_r: &[f32], freq: &[f32],
    ) -> (Vec<WaveVert>, Vec<WaveDraw>) {
        let mut verts: Vec<WaveVert> = Vec::new();
        let mut draws: Vec<WaveDraw> = Vec::new();
        let audio_len = wave_l.len();

        // ── Custom waveforms first (index order), then built-in last ─────────
        self.build_custom_waves(t, bass, mid, treb, vol, bass_att, mid_att, treb_att,
            inv_aspectx, inv_aspecty, wave_l, wave_r, freq, &mut verts, &mut draws);

        if audio_len > 0 {
            // Built-in waveform alpha is the post-per-frame `wave_a` (butterchurn reads
            // mdVSFrame.wave_a, the value AFTER frame_eqs). Both jelly_space and parade
            // set wave_a=0 in per-frame, which correctly gates the wave off — matching
            // butterchurn. Fall back to the parsed base fWaveAlpha when per-frame never
            // touched wave_a.
            let live_wave_a = self.eel_env.get("wave_a").copied()
                .map(|v| v as f32).unwrap_or(self.bw_a);
            self.build_basic_waveform(t, bass, mid, treb, inv_aspectx, inv_aspecty,
                live_wave_a, wave_l, wave_r, &mut verts, &mut draws);
        }

        (verts, draws)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_basic_waveform(
        &self,
        t: f32,
        bass: f64, mid: f64, treb: f64,
        aspectx: f32, aspecty: f32,
        live_wave_a: f32,
        time_l: &[f32], time_r: &[f32],
        verts: &mut Vec<WaveVert>, draws: &mut Vec<WaveDraw>,
    ) {
        use std::f32::consts::PI;
        // alpha gate: built-in reads the post-per-frame wave_a (butterchurn behavior).
        let base_alpha = live_wave_a;
        let vol = ((bass + mid + treb) / 3.0) as f32;
        if !(vol > -0.01 && base_alpha > 0.001 && !time_l.is_empty()) { return; }

        // processWaveform (butterchurn 4520-4533): scale = wave_scale/128 on Int8.
        // Our samples are f32 in [-1,1] (== Int8/128), so the effective scale on the
        // f32 data is simply wave_scale.
        let process = |src: &[f32]| -> Vec<f32> {
            let scale = self.bw_scale;
            let smooth = self.bw_smoothing;
            let smooth2 = scale * (1.0 - smooth);
            let n = src.len();
            let mut out = vec![0.0f32; n];
            if n == 0 { return out; }
            out[0] = src[0] * scale;
            for i in 1..n { out[i] = src[i] * smooth2 + out[i - 1] * smooth; }
            out
        };
        let wave_l = process(time_l);
        let wave_r = process(time_r);

        let new_wave_mode = (self.bw_mode.floor() as i32).rem_euclid(8);
        let wave_pos_x = self.bw_x * 2.0 - 1.0;
        let wave_pos_y = self.bw_y * 2.0 - 1.0;

        let mut param2 = self.bw_mystery;
        if (new_wave_mode == 0 || new_wave_mode == 1 || new_wave_mode == 4) && param2.abs() > 1.0 {
            param2 = param2 * 0.5 + 0.5;
            param2 -= param2.floor();
            param2 = param2.abs();
            param2 = param2 * 2.0 - 1.0;
        }

        let nlen = wave_l.len();
        let mut positions: Vec<[f32; 2]> = Vec::new();
        // Mode 7 emits a SECOND polyline (R-channel line). Populated only by mode 7.
        let mut positions2: Option<Vec<[f32; 2]>> = None;
        let mut alpha = base_alpha;

        // mod-wave-alpha-by-volume (every mode applies this). Guarded divide like mode 0.
        let mod_alpha = |alpha: &mut f32| {
            if self.bw_modalphavol {
                let diff = self.bw_modalphaend - self.bw_modalphastart;
                if diff.abs() > 1e-9 {
                    *alpha *= (vol - self.bw_modalphastart) / diff;
                }
            }
        };

        // texsizeX / texsizeY (butterchurn) == internal render size, as f32.
        let texsize_x = self.width as f32;

        match new_wave_mode {
            0 => {
                // circle
                if self.bw_modalphavol {
                    let diff = self.bw_modalphaend - self.bw_modalphastart;
                    if diff.abs() > 1e-9 {
                        alpha *= (vol - self.bw_modalphastart) / diff;
                    }
                }
                alpha = alpha.clamp(0.0, 1.0);
                let num_vert = (nlen / 2) + 1;
                if num_vert < 2 { return; }
                let num_vert_inv = 1.0 / (num_vert - 1) as f32;
                let sample_offset = (nlen.saturating_sub(num_vert)) / 2;
                positions.resize(num_vert, [0.0, 0.0]);
                for i in 0..num_vert - 1 {
                    let mut rad = 0.5 + 0.4 * wave_r[(i + sample_offset).min(nlen - 1)] + param2;
                    let ang = i as f32 * num_vert_inv * 2.0 * PI + t * 0.2;
                    if (i as f32) < num_vert as f32 / 10.0 {
                        let mut mix = i as f32 / (num_vert as f32 * 0.1);
                        mix = 0.5 - 0.5 * (mix * PI).cos();
                        let idx2 = (i + num_vert + sample_offset).min(nlen - 1);
                        let rad2 = 0.5 + 0.4 * wave_r[idx2] + param2;
                        rad = (1.0 - mix) * rad2 + rad * mix;
                    }
                    positions[i] = [rad * ang.cos() * aspecty + wave_pos_x,
                                    rad * ang.sin() * aspectx + wave_pos_y];
                }
                positions[num_vert - 1] = positions[0];
            }
            1 => {
                // rotating circle, ang driven by L
                alpha *= 1.25;
                mod_alpha(&mut alpha);
                alpha = alpha.clamp(0.0, 1.0);
                let num_vert = nlen / 2;
                if num_vert < 1 { return; }
                positions.resize(num_vert, [0.0, 0.0]);
                for i in 0..num_vert {
                    let rad = 0.53 + 0.43 * wave_r[i] + param2;
                    let ang = wave_l[(i + 32).min(nlen - 1)] * 0.5 * PI + t * 2.3;
                    positions[i] = [rad * ang.cos() * aspecty + wave_pos_x,
                                    rad * ang.sin() * aspectx + wave_pos_y];
                }
            }
            2 => {
                // X/Y scatter, faint
                alpha *= if texsize_x < 1024.0 { 0.09 }
                         else if texsize_x < 2048.0 { 0.11 } else { 0.13 };
                mod_alpha(&mut alpha);
                alpha = alpha.clamp(0.0, 1.0);
                let num_vert = nlen;
                positions.resize(num_vert, [0.0, 0.0]);
                for i in 0..num_vert {
                    positions[i] = [wave_r[i] * aspecty + wave_pos_x,
                                    wave_l[(i + 32) % nlen] * aspectx + wave_pos_y];
                }
            }
            3 => {
                // X/Y scatter, treble-gated (same geometry as mode 2)
                alpha *= if texsize_x < 1024.0 { 0.15 }
                         else if texsize_x < 2048.0 { 0.22 } else { 0.33 };
                alpha *= 1.3;
                alpha *= (treb * treb) as f32;
                mod_alpha(&mut alpha);
                alpha = alpha.clamp(0.0, 1.0);
                let num_vert = nlen;
                positions.resize(num_vert, [0.0, 0.0]);
                for i in 0..num_vert {
                    positions[i] = [wave_r[i] * aspecty + wave_pos_x,
                                    wave_l[(i + 32) % nlen] * aspectx + wave_pos_y];
                }
            }
            4 => {
                // horizontal scope with momentum
                mod_alpha(&mut alpha);
                alpha = alpha.clamp(0.0, 1.0);
                let mut num_vert = nlen;
                if num_vert > (texsize_x / 3.0) as usize {
                    num_vert = (texsize_x / 3.0).floor() as usize;
                }
                if num_vert < 2 { return; }
                let num_vert_inv = 1.0 / num_vert as f32; // NOT num_vert-1
                let sample_offset = nlen.saturating_sub(num_vert) / 2;
                let w1 = 0.45 + 0.5 * (param2 * 0.5 + 0.5);
                let w2 = 1.0 - w1;
                positions.resize(num_vert, [0.0, 0.0]);
                for i in 0..num_vert {
                    let mut x = 2.0 * (i as f32) * num_vert_inv + (wave_pos_x - 1.0)
                        + wave_r[(i + 25 + sample_offset) % nlen] * 0.44;
                    let mut y = wave_l[i + sample_offset] * 0.47 + wave_pos_y;
                    if i > 1 {
                        x = x * w2 + w1 * (positions[i - 1][0] * 2.0 - positions[i - 2][0]);
                        y = y * w2 + w1 * (positions[i - 1][1] * 2.0 - positions[i - 2][1]);
                    }
                    positions[i] = [x, y];
                }
            }
            5 => {
                // Lissajous-ish rotating
                alpha *= if texsize_x < 1024.0 { 0.09 }
                         else if texsize_x < 2048.0 { 0.11 } else { 0.13 };
                mod_alpha(&mut alpha);
                alpha = alpha.clamp(0.0, 1.0);
                let cos_rot = (t * 0.3).cos();
                let sin_rot = (t * 0.3).sin();
                let num_vert = nlen;
                positions.resize(num_vert, [0.0, 0.0]);
                for i in 0..num_vert {
                    let ioff = (i + 32) % nlen;
                    let x0 = wave_r[i] * wave_l[ioff] + wave_l[i] * wave_r[ioff];
                    let y0 = wave_r[i] * wave_r[i] - wave_l[ioff] * wave_l[ioff];
                    positions[i] = [(x0 * cos_rot - y0 * sin_rot) * (aspecty + wave_pos_x),
                                    (x0 * sin_rot + y0 * cos_rot) * (aspectx + wave_pos_y)];
                }
            }
            6 | 7 => {
                // angled line through screen, clipped to [-1.1, 1.1] box.
                mod_alpha(&mut alpha);
                alpha = alpha.clamp(0.0, 1.0);
                let mut num_vert = nlen / 2;
                if num_vert > (texsize_x / 3.0) as usize {
                    num_vert = (texsize_x / 3.0).floor() as usize;
                }
                if num_vert < 1 { return; }
                let sample_offset = nlen.saturating_sub(num_vert) / 2;
                let ang = PI * 0.5 * param2;
                let mut dx = ang.cos();
                let mut dy = ang.sin();
                // Both edgex AND edgey seed from wave_pos_x (butterchurn quirk — literal).
                let mut edgex = [wave_pos_x * (ang + PI * 0.5).cos() - dx * 3.0,
                                 wave_pos_x * (ang + PI * 0.5).cos() + dx * 3.0];
                let mut edgey = [wave_pos_x * (ang + PI * 0.5).sin() - dy * 3.0,
                                 wave_pos_x * (ang + PI * 0.5).sin() + dy * 3.0];
                for i in 0..2 {
                    for j in 0..4 {
                        let mut tt = 0.0f32;
                        let mut clip = false;
                        match j {
                            0 => if edgex[i] >  1.1 { tt = ( 1.1 - edgex[1 - i]) / (edgex[i] - edgex[1 - i]); clip = true; },
                            1 => if edgex[i] < -1.1 { tt = (-1.1 - edgex[1 - i]) / (edgex[i] - edgex[1 - i]); clip = true; },
                            2 => if edgey[i] >  1.1 { tt = ( 1.1 - edgey[1 - i]) / (edgey[i] - edgey[1 - i]); clip = true; },
                            3 => if edgey[i] < -1.1 { tt = (-1.1 - edgey[1 - i]) / (edgey[i] - edgey[1 - i]); clip = true; },
                            _ => {}
                        }
                        if clip {
                            let dxi = edgex[i] - edgex[1 - i];
                            let dyi = edgey[i] - edgey[1 - i];
                            edgex[i] = edgex[1 - i] + dxi * tt;
                            edgey[i] = edgey[1 - i] + dyi * tt;
                        }
                    }
                }
                dx = (edgex[1] - edgex[0]) / num_vert as f32;
                dy = (edgey[1] - edgey[0]) / num_vert as f32;
                let ang2 = dy.atan2(dx);
                let perp_dx = (ang2 + PI * 0.5).cos();
                let perp_dy = (ang2 + PI * 0.5).sin();

                if new_wave_mode == 6 {
                    positions.resize(num_vert, [0.0, 0.0]);
                    for i in 0..num_vert {
                        let s = wave_l[i + sample_offset];
                        positions[i] = [edgex[0] + dx * (i as f32) + perp_dx * 0.25 * s,
                                        edgey[0] + dy * (i as f32) + perp_dy * 0.25 * s];
                    }
                } else {
                    // MODE 7: dual line — L line + R line separated by sep.
                    let sep = (wave_pos_y * 0.5 + 0.5).powi(2);
                    positions.resize(num_vert, [0.0, 0.0]);
                    let mut p2: Vec<[f32; 2]> = vec![[0.0, 0.0]; num_vert];
                    for i in 0..num_vert {
                        let s = wave_l[i + sample_offset];
                        positions[i] = [edgex[0] + dx * (i as f32) + perp_dx * (0.25 * s + sep),
                                        edgey[0] + dy * (i as f32) + perp_dy * (0.25 * s + sep)];
                    }
                    for i in 0..num_vert {
                        let s = wave_r[i + sample_offset];
                        p2[i] = [edgex[0] + dx * (i as f32) + perp_dx * (0.25 * s - sep),
                                 edgey[0] + dy * (i as f32) + perp_dy * (0.25 * s - sep)];
                    }
                    positions2 = Some(p2);
                }
            }
            _ => {
                // Unreachable: rem_euclid(8) constrains new_wave_mode to 0..=7.
                return;
            }
        }

        // color (computed once, shared by both polylines for mode 7) from the LIVE
        // post-per-frame wave_r/g/b (butterchurn basicWaveform.js:446-448 reads
        // mdVSFrame.wave_*), mirroring live_wave_a; fall back to the parsed base when
        // per-frame never wrote them (idx 9096 colors its waveform in per-frame eqs).
        let lc = |k: &str, d: f32| self.eel_env.get(k).copied().map(|v| v as f32).unwrap_or(d);
        let mut cr = lc("wave_r", self.bw_r).clamp(0.0, 1.0);
        let mut cg = lc("wave_g", self.bw_g).clamp(0.0, 1.0);
        let mut cb = lc("wave_b", self.bw_b).clamp(0.0, 1.0);
        if self.bw_brighten {
            let maxc = cr.max(cg).max(cb);
            if maxc > 0.01 { cr /= maxc; cg /= maxc; cb /= maxc; }
        }
        let color = [cr, cg, cb, alpha];

        if alpha <= 0.0 { return; }

        // Shared tail: Y-flip (butterchurn negates pos.y before smoothing), smooth,
        // push verts, push a WaveDraw. Called once for modes 0-6, twice for mode 7.
        let dots = self.bw_dots;
        let additive = self.bw_additive;
        let thick = self.bw_thick || self.bw_dots;
        let mut emit = |pos: &mut Vec<[f32; 2]>| {
            for p in pos.iter_mut() { p[1] = -p[1]; }
            let smoothed = smooth_wave(pos);
            if smoothed.is_empty() { return; }
            let start = verts.len() as u32;
            for p in &smoothed { verts.push(WaveVert { pos: *p, color }); }
            draws.push(WaveDraw {
                start_vert: start,
                count: smoothed.len() as u32,
                points: dots,
                additive,
                thick,
            });
        };
        emit(&mut positions);
        if let Some(mut p2) = positions2 { emit(&mut p2); }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_custom_waves(
        &mut self,
        t: f32,
        bass: f64, mid: f64, treb: f64, vol: f64,
        bass_att: f64, mid_att: f64, treb_att: f64,
        inv_aspectx: f32, inv_aspecty: f32,
        time_l: &[f32], time_r: &[f32], freq: &[f32],
        verts: &mut Vec<WaveVert>, draws: &mut Vec<WaveDraw>,
    ) {
        let max_samples = time_l.len();
        if max_samples == 0 { return; }
        let wave_scale_base = self.bw_scale;
        let frame_idx = self.frame_idx;
        // q1..q32 from the main per-frame EEL — custom waveforms read these (ORB's
        // laser tubes are entirely q1-driven). Captured before the &mut waves loop.
        // Full q1..q32 (was capped at q8) to match MilkDrop/Butterchurn shape/wave semantics.
        let mut qv = [0.0f64; 32];
        for i in 0..32 {
            qv[i] = self.eel_env.get(format!("q{}", i + 1).as_str()).copied().unwrap_or(0.0);
        }

        for wv in self.waves.iter_mut() {
            if !wv.def.enabled { continue; }

            // ── per-frame run ────────────────────────────────────────────────
            let env = &mut wv.env;
            env.insert("time".into(), t as f64);
            env.insert("frame".into(), frame_idx as f64);
            env.insert("fps".into(), 60.0);
            env.insert("bass".into(), bass);  env.insert("bass_att".into(), bass_att);
            env.insert("mid".into(), mid);    env.insert("mid_att".into(), mid_att);
            env.insert("treb".into(), treb);  env.insert("treb_att".into(), treb_att);
            env.insert("vol".into(), vol);
            env.insert("aspectx".into(), inv_aspectx as f64);
            env.insert("aspecty".into(), inv_aspecty as f64);
            for (i, q) in qv.iter().enumerate() { env.insert(format!("q{}", i + 1), *q); }
            env.insert("samples".into(), wv.def.samples as f64);
            env.insert("sep".into(), wv.def.sep as f64);
            env.insert("scaling".into(), wv.def.scaling as f64);
            env.insert("smoothing".into(), wv.def.smoothing as f64);
            env.insert("spectrum".into(), if wv.def.spectrum { 1.0 } else { 0.0 });
            env.insert("r".into(), wv.def.r as f64);
            env.insert("g".into(), wv.def.g as f64);
            env.insert("b".into(), wv.def.b as f64);
            env.insert("a".into(), wv.def.a as f64);
            if let Some(p) = &wv.per_frame_prog { p.run_with(env, &mut wv.state); }
            let rd = |env: &Env, k: &str, d: f64| env.get(k).copied().unwrap_or(d);
            let pf_samples = rd(env, "samples", wv.def.samples as f64).floor().max(0.0) as usize;
            let pf_sep = rd(env, "sep", wv.def.sep as f64).floor() as i32;
            let pf_scaling = rd(env, "scaling", wv.def.scaling as f64) as f32;
            let pf_spectrum = rd(env, "spectrum", if wv.def.spectrum {1.0} else {0.0}) != 0.0;
            let pf_smoothing = rd(env, "smoothing", wv.def.smoothing as f64) as f32;
            let frame_r = rd(env, "r", wv.def.r as f64) as f32;
            let frame_g = rd(env, "g", wv.def.g as f64) as f32;
            let frame_b = rd(env, "b", wv.def.b as f64) as f32;
            let frame_a = rd(env, "a", wv.def.a as f64) as f32;

            // ── sample prep (generateWaveform) ───────────────────────────────
            let mut samples = pf_samples.min(max_samples);
            let sep = pf_sep.max(0) as usize;
            samples = samples.saturating_sub(sep);
            if !(samples >= 2 || (wv.def.use_dots && samples >= 1)) { continue; }

            // The *128 converts our normalized [-1,1] TIME samples to butterchurn's Int8
            // [-128,127] range that the 0.004 constant assumes. The SPECTRUM branch must
            // NOT get it: butterchurn's customWaveform.js:167 uses bare 0.15 for spectrum,
            // and our synth freq array is already ~[0,1] — *128 shoves every bin off-screen
            // (idx 8548 went black; silent-audio control renders it at luma 0.30). Keep the
            // *128 on the time branch only.
            let scale = (if pf_spectrum { 0.15 } else { 0.004 * 128.0 }) * pf_scaling * wave_scale_base;
            // bSpectrum waveforms read the FFT freqArray (butterchurn customWaveform.js:
            // pointsLeft = useSpectrum ? freqArrayL : timeArrayL). Our freq array is mono
            // (512 bins, Butterchurn-shaped + equalized) so both channels read it. When no
            // live freq is available (headless/synthetic), fall back to time data so those
            // deterministic renders are unchanged.
            let use_freq = pf_spectrum && freq.len() == max_samples;
            let (src_l, src_r): (&[f32], &[f32]) = if use_freq {
                (freq, freq)
            } else {
                (time_l, time_r)
            };
            let (j0, j1, step) = if pf_spectrum {
                (0usize, 0usize, ((max_samples.saturating_sub(sep)).max(1) / samples.max(1)).max(1))
            } else {
                let j0 = ((max_samples as f32 - samples as f32) / 2.0 - sep as f32 / 2.0).floor().max(0.0) as usize;
                let j1 = ((max_samples as f32 - samples as f32) / 2.0 + sep as f32 / 2.0).floor().max(0.0) as usize;
                (j0, j1, 1usize)
            };
            let mix1 = (pf_smoothing * 0.98).max(0.0).powf(0.5);
            let mix2 = 1.0 - mix1;

            let mut pts_l = vec![0.0f32; samples];
            let mut pts_r = vec![0.0f32; samples];
            pts_l[0] = *src_l.get(j0.min(max_samples - 1)).unwrap_or(&0.0);
            pts_r[0] = *src_r.get(j1.min(max_samples - 1)).unwrap_or(&0.0);
            for j in 1..samples {
                let il = (j * step + j0).min(max_samples - 1);
                let ir = (j * step + j1).min(max_samples - 1);
                pts_l[j] = src_l[il] * mix2 + pts_l[j - 1] * mix1;
                pts_r[j] = src_r[ir] * mix2 + pts_r[j - 1] * mix1;
            }
            for j in (0..samples - 1).rev() {
                pts_l[j] = pts_l[j] * mix2 + pts_l[j + 1] * mix1;
                pts_r[j] = pts_r[j] * mix2 + pts_r[j + 1] * mix1;
            }
            for j in 0..samples {
                pts_l[j] *= scale;
                pts_r[j] *= scale;
            }

            // ── per-point loop ───────────────────────────────────────────────
            let mut positions: Vec<[f32; 2]> = Vec::with_capacity(samples);
            let mut colors: Vec<[f32; 4]> = Vec::with_capacity(samples);
            for j in 0..samples {
                let value1 = pts_l[j];
                let value2 = pts_r[j];
                let sample_t = if samples <= 1 { 0.0 } else { j as f64 / (samples - 1) as f64 };
                let env = &mut wv.env;
                env.insert("sample".into(), sample_t);
                env.insert("value1".into(), value1 as f64);
                env.insert("value2".into(), value2 as f64);
                env.insert("x".into(), 0.5 + value1 as f64);
                env.insert("y".into(), 0.5 + value2 as f64);
                env.insert("r".into(), frame_r as f64);
                env.insert("g".into(), frame_g as f64);
                env.insert("b".into(), frame_b as f64);
                env.insert("a".into(), frame_a as f64);
                if let Some(p) = &wv.per_point_prog { p.run_with(env, &mut wv.state); }
                let px = (rd(env, "x", 0.5) * 2.0 - 1.0) * inv_aspectx as f64;
                let py = (rd(env, "y", 0.5) * -2.0 + 1.0) * inv_aspecty as f64;
                // Clamp per-point color/alpha and guard NaN/inf (audio-gated point
                // eqs can drive these out of range under synthetic audio → an alpha
                // of 0/neg/NaN makes the wave vanish).
                let fin = |v: f32, d: f32| if v.is_finite() { v } else { d };
                let cr = fin(rd(env, "r", frame_r as f64) as f32, frame_r).clamp(0.0, 1.0);
                let cg = fin(rd(env, "g", frame_g as f64) as f32, frame_g).clamp(0.0, 1.0);
                let cb = fin(rd(env, "b", frame_b as f64) as f32, frame_b).clamp(0.0, 1.0);
                let ca = fin(rd(env, "a", frame_a as f64) as f32, frame_a).clamp(0.0, 1.0);
                let (px, py) = (fin(px as f32, 0.0), fin(py as f32, 0.0));
                positions.push([px, py]);
                colors.push([cr, cg, cb, ca]);
            }

            if wv.def.use_dots {
                let start = verts.len() as u32;
                for (p, c) in positions.iter().zip(colors.iter()) {
                    verts.push(WaveVert { pos: *p, color: *c });
                }
                draws.push(WaveDraw {
                    start_vert: start, count: positions.len() as u32,
                    points: true, additive: wv.def.additive, thick: wv.def.draw_thick,
                });
            } else {
                let (sp, sc) = smooth_wave_and_color(&positions, &colors);
                let start = verts.len() as u32;
                for (p, c) in sp.iter().zip(sc.iter()) {
                    verts.push(WaveVert { pos: *p, color: *c });
                }
                draws.push(WaveDraw {
                    start_vert: start, count: sp.len() as u32,
                    points: false, additive: wv.def.additive, thick: wv.def.draw_thick,
                });
            }
        }
    }

    /// Compute the per-vertex warped UV + decay rgb for the warp mesh.
    /// Runs the per_pixel EEL program per vertex (if present) and composes the
    /// butterchurn warped sample coordinate. `aspectx`/`aspecty` are the
    /// NON-inverted geometry aspect (landscape: ax=1, ay=h/w).
    fn compute_warp_verts(&mut self, t: f32, aspectx: f32, aspecty: f32) -> Vec<WarpVert> {
        let b = self.base_warp;
        // Resolve per-FRAME warp params: base, overridden by per-frame EEL writes.
        let getf = |k: &str, def: f32| self.eel_env.get(k).copied().map(|v| v as f32).unwrap_or(def);
        let fzoom    = getf("zoom",    b.zoom);
        let fzoomexp = getf("zoomexp", b.zoomexp);
        let frot     = getf("rot",     b.rot);
        let fwarp    = getf("warp",    b.warp);
        let fcx      = getf("cx",      b.cx);
        let fcy      = getf("cy",      b.cy);
        let fdx      = getf("dx",      b.dx);
        let fdy      = getf("dy",      b.dy);
        let fsx      = getf("sx",      b.sx);
        let fsy      = getf("sy",      b.sy);
        let fdecay   = getf("decay",   b.decay);
        let wscale   = getf("warpscale",     b.warpscale).max(1e-6);
        let wanim    = getf("warpanimspeed", b.warpanimspeed);

        let warp_time_v   = t * wanim;
        let warp_scale_inv = 1.0_f32 / wscale;
        let warpf0 = 11.68 + 4.0 * (warp_time_v * 1.413 + 10.0).cos();
        let warpf1 =  8.77 + 3.0 * (warp_time_v * 1.113 +  7.0).cos();
        let warpf2 = 10.54 + 3.0 * (warp_time_v * 1.233 +  3.0).cos();
        let warpf3 = 11.49 + 4.0 * (warp_time_v * 0.933 +  5.0).cos();

        let ax = aspectx as f64;
        let ay = aspecty as f64;
        // EEL `aspectx`/`aspecty` are seeded INVERTED per butterchurn presetEquationRunner.
        let inv_ax = if aspectx != 0.0 { 1.0 / aspectx as f64 } else { 1.0 };
        let inv_ay = if aspecty != 0.0 { 1.0 / aspecty as f64 } else { 1.0 };

        let has_prog = self.per_pixel_prog.is_some();

        // Seed constant per-frame inputs ONCE into the reusable scratch env.
        if has_prog {
            let qget = |k: &str| self.eel_env.get(k).copied().unwrap_or(0.0);
            let consts: [(&str, f64); 21] = [
                ("time", t as f64),
                ("frame", self.frame_idx as f64),
                ("fps", 60.0),
                ("bass", self.eel_env.get("bass").copied().unwrap_or(0.0)),
                ("mid",  self.eel_env.get("mid").copied().unwrap_or(0.0)),
                ("treb", self.eel_env.get("treb").copied().unwrap_or(0.0)),
                ("vol",  self.eel_env.get("vol").copied().unwrap_or(0.0)),
                ("bass_att", self.eel_env.get("bass_att").copied().unwrap_or(0.0)),
                ("mid_att",  self.eel_env.get("mid_att").copied().unwrap_or(0.0)),
                ("treb_att", self.eel_env.get("treb_att").copied().unwrap_or(0.0)),
                ("vol_att",  self.eel_env.get("vol_att").copied().unwrap_or(0.0)),
                ("aspectx", inv_ax), ("aspecty", inv_ay),
                ("q1", qget("q1")), ("q2", qget("q2")), ("q3", qget("q3")), ("q4", qget("q4")),
                ("q5", qget("q5")), ("q6", qget("q6")), ("q7", qget("q7")), ("q8", qget("q8")),
            ];
            for (k, v) in consts { self.warp_env.insert(k.to_string(), v); }
        }

        let vw = GRID_W + 1;
        let vh = GRID_H + 1;
        let mut verts = Vec::with_capacity((vw * vh) as usize);

        for j in 0..vh {
            for i in 0..vw {
                let x = (i as f32 / GRID_W as f32) * 2.0 - 1.0;
                let y = (j as f32 / GRID_H as f32) * 2.0 - 1.0;
                let xf = x as f64;
                let yf = y as f64;
                let rad = (xf * xf * ax * ax + yf * yf * ay * ay).sqrt();

                // Defaults (per-frame values) in case there's no program.
                let (mut zoom, mut zoomexp, mut rot, mut warp) = (fzoom, fzoomexp, frot, fwarp);
                let (mut cx, mut cy, mut dx, mut dy, mut sx, mut sy) = (fcx, fcy, fdx, fdy, fsx, fsy);
                let (mut dr, mut dg, mut db) = (fdecay, fdecay, fdecay);

                if let Some(prog) = self.per_pixel_prog.as_ref() {
                    let ang = if j == GRID_H / 2 && i == GRID_W / 2 {
                        0.0
                    } else {
                        (yf * ay).atan2(xf * ax)
                    };
                    let env = &mut self.warp_env;
                    env.insert("x".into(),   xf * 0.5 * ax + 0.5);
                    env.insert("y".into(),   yf * -0.5 * ay + 0.5);
                    env.insert("rad".into(), rad);
                    env.insert("ang".into(), ang);
                    env.insert("zoom".into(),    fzoom as f64);
                    env.insert("zoomexp".into(), fzoomexp as f64);
                    env.insert("rot".into(),     frot as f64);
                    env.insert("warp".into(),    fwarp as f64);
                    env.insert("cx".into(),      fcx as f64);
                    env.insert("cy".into(),      fcy as f64);
                    env.insert("dx".into(),      fdx as f64);
                    env.insert("dy".into(),      fdy as f64);
                    env.insert("sx".into(),      fsx as f64);
                    env.insert("sy".into(),      fsy as f64);
                    env.insert("decay".into(),   fdecay as f64);
                    env.insert("decay_r".into(), fdecay as f64);
                    env.insert("decay_g".into(), fdecay as f64);
                    env.insert("decay_b".into(), fdecay as f64);
                    prog.run_with(env, &mut self.warp_state);
                    let g = |k: &str, d: f32| env.get(k).copied().map(|v| v as f32).unwrap_or(d);
                    zoom = g("zoom", fzoom); zoomexp = g("zoomexp", fzoomexp);
                    rot = g("rot", frot);    warp = g("warp", fwarp);
                    cx = g("cx", fcx); cy = g("cy", fcy);
                    dx = g("dx", fdx); dy = g("dy", fdy);
                    sx = g("sx", fsx); sy = g("sy", fsy);
                    dr = g("decay_r", fdecay); dg = g("decay_g", fdecay); db = g("decay_b", fdecay);
                }

                if zoom.abs() < 1e-6 { zoom = 1e-6; }
                if sx.abs() < 1e-6 { sx = 1e-6; }
                if sy.abs() < 1e-6 { sy = 1e-6; }

                // ── UV composition (butterchurn renderer.js runPixelEquations) ──
                let zoom2v = zoom.powf(zoomexp.powf(rad as f32 * 2.0 - 1.0));
                let zoom2inv = 1.0_f32 / zoom2v;
                let mut u = x * 0.5 * aspectx * zoom2inv + 0.5;
                let mut v = -y * 0.5 * aspecty * zoom2inv + 0.5;
                // scale about (cx,cy)
                u = (u - cx) / sx + cx;
                v = (v - cy) / sy + cy;
                // warp octaves
                if warp.abs() > 1e-9 {
                    u += warp * 0.0035 * (warp_time_v * 0.333 + warp_scale_inv * (x * warpf0 - y * warpf3)).sin();
                    v += warp * 0.0035 * (warp_time_v * 0.375 - warp_scale_inv * (x * warpf2 + y * warpf1)).cos();
                    u += warp * 0.0035 * (warp_time_v * 0.753 - warp_scale_inv * (x * warpf1 - y * warpf2)).cos();
                    v += warp * 0.0035 * (warp_time_v * 0.825 + warp_scale_inv * (x * warpf0 + y * warpf3)).sin();
                }
                // rotate about (cx,cy)
                let u2 = u - cx;
                let v2 = v - cy;
                let cr = rot.cos();
                let sr = rot.sin();
                u = u2 * cr - v2 * sr + cx;
                v = u2 * sr + v2 * cr + cy;
                // translate
                u -= dx;
                v -= dy;
                // undo aspect
                u = (u - 0.5) / aspectx + 0.5;
                v = (v - 0.5) / aspecty + 0.5;

                let px = x;
                let py = -y; // clip-space: top row (j=0, y=-1) -> py=+1
                verts.push(WarpVert { pos: [px, py], uv: [u, v], decay: [dr, dg, db, 1.0] });
            }
        }
        verts
    }

    pub fn render(&mut self, surface_view: &wgpu::TextureView) {
        let t = match self.time_per_frame {
            Some(dt) => self.frame_idx as f32 * dt,
            None => self.start.elapsed().as_secs_f32(),
        };
        let (w, h) = (self.width as f32, self.height as f32);
        let asp = w / h;

        // Audio reactivity: live mic features when supplied, else synthetic sine
        // waves at different frequencies so offscreen/headless renders still animate.
        let (bass, mid, treb, vol) = match self.audio {
            Some([b, m, tr, v]) => (b as f64, m as f64, tr as f64, v as f64),
            None => {
                let bass = (1.0 + (t * 1.3).sin()) as f64;
                let mid  = (1.0 + (t * 2.1).sin()) as f64;
                let treb = (1.0 + (t * 3.7).sin()) as f64;
                let vol  = (bass + mid + treb) / 3.0;
                (bass, mid, treb, vol)
            }
        };
        // Attenuated (smoothed) envelopes. When no live att was supplied
        // (headless/synthetic), mirror the non-att values so deterministic renders
        // are bit-identical to before — only the live path drives distinct *_att.
        let (bass_att, mid_att, treb_att, vol_att) = match self.audio_att {
            Some([b, m, tr, v]) => (b as f64, m as f64, tr as f64, v as f64),
            None => (bass, mid, treb, vol),
        };

        // Run EEL2 per-frame equations
        if let Some(prog) = &self.eel_program {
            let env = &mut self.eel_env;
            // Reset q1..q32 to their post-init values BEFORE per-frame runs, so
            // accumulator-q presets re-seed from init each frame instead of
            // carrying the previous frame's q's (Butterchurn mdVS reset). User
            // vars and regs are NOT reset (they persist). No-op when no init ran.
            for (k, v) in &self.q_init {
                env.insert(k.clone(), *v);
            }
            // Reset built-in WARP motion vars to their header baseVals each frame,
            // matching Butterchurn (mdVSFrame = mdVS baseVals + qInit + USER keys only;
            // built-in motion vars do NOT persist — only user vars/megabuf/regs do).
            // Without this, accumulator presets (`zoom = zoom + ...`) compound every
            // frame and run away: idx 313's zoom blows up by ~f70, the feedback then
            // samples a magnified central pixel, flattens, and collapses to black.
            let bw = self.base_warp;
            env.insert("zoom".into(),          bw.zoom as f64);
            env.insert("zoomexp".into(),       bw.zoomexp as f64);
            env.insert("rot".into(),           bw.rot as f64);
            env.insert("warp".into(),          bw.warp as f64);
            env.insert("cx".into(),            bw.cx as f64);
            env.insert("cy".into(),            bw.cy as f64);
            env.insert("dx".into(),            bw.dx as f64);
            env.insert("dy".into(),            bw.dy as f64);
            env.insert("sx".into(),            bw.sx as f64);
            env.insert("sy".into(),            bw.sy as f64);
            env.insert("warpscale".into(),     bw.warpscale as f64);
            env.insert("warpanimspeed".into(), bw.warpanimspeed as f64);
            env.insert("decay".into(),         bw.decay as f64);
            // Seed read-only inputs before each frame
            env.insert("time".into(),     t as f64);
            env.insert("fps".into(),      60.0);
            env.insert("frame".into(),    self.frame_idx as f64);
            env.insert("progress".into(), (t % 30.0) as f64 / 30.0);
            env.insert("bass".into(),     bass);
            env.insert("mid".into(),      mid);
            env.insert("treb".into(),     treb);
            env.insert("vol".into(),      vol);
            env.insert("bass_att".into(), bass_att);
            env.insert("mid_att".into(),  mid_att);
            env.insert("treb_att".into(), treb_att);
            env.insert("vol_att".into(),  vol_att);
            // MilkDrop pseudo-var `diff`: frame-to-frame volume delta. Not a standard
            // seeded EEL input, but presets like orb_waaa gate mv_a on above(diff,10),
            // so seed it from the volume change to honor the preset's intent.
            env.insert("diff".into(), (vol - self.vol_prev).abs());
            // Butterchurn seeds the per-frame EEL aspectx/aspecty as the INVERTED geometry
            // aspect (matching the shape/wave/warp envs) + mesh/pixel dims (presetEquation
            // Runner mdVSBase). These were never seeded into the per-frame env, so per-frame
            // eqs reading aspectx/aspecty silently got 0 (idx 2007/4005/8637).
            let (gax, gay) = if self.width >= self.height {
                (1.0, self.height as f64 / self.width.max(1) as f64)
            } else {
                (self.width as f64 / self.height.max(1) as f64, 1.0)
            };
            env.insert("aspectx".into(), if gax != 0.0 { 1.0 / gax } else { 1.0 });
            env.insert("aspecty".into(), if gay != 0.0 { 1.0 / gay } else { 1.0 });
            env.insert("meshx".into(),   GRID_W as f64);
            env.insert("meshy".into(),   GRID_H as f64);
            env.insert("pixelsx".into(), self.width as f64);
            env.insert("pixelsy".into(), self.height as f64);
            prog.run_with(env, &mut self.eel_state);
        }
        self.vol_prev = vol;

        // Helper: read f64 var from EEL env, default 0
        let eq = |k: &str| self.eel_env.get(k).copied().unwrap_or(0.0) as f32;
        // Helper: read f64 var from EEL env, default to `def` if missing
        let eqd = |k: &str, def: f64| self.eel_env.get(k).copied().unwrap_or(def) as f32;

        // Snapshot live motion-vector / border / darken values from the per-frame EEL
        // env NOW (while eqd's immutable borrow is valid), before any &mut self call
        // (build_wave_geometry / compute_warp_verts). Used later to build geometry.
        let live_mv_a  = eqd("mv_a",  self.mv_a as f64);
        let live_mv_x  = eqd("mv_x",  self.mv_x as f64);
        let live_mv_y  = eqd("mv_y",  self.mv_y as f64);
        let live_mv_dx = eqd("mv_dx", self.mv_dx as f64);
        let live_mv_dy = eqd("mv_dy", self.mv_dy as f64);
        let live_mv_l  = eqd("mv_l",  self.mv_l as f64);
        let live_mv_r  = eqd("mv_r",  self.mv_r as f64);
        let live_mv_g  = eqd("mv_g",  self.mv_g as f64);
        let live_mv_b  = eqd("mv_b",  self.mv_b as f64);
        let live_darken = eqd("darken_center", if self.darken_center { 1.0 } else { 0.0 }) != 0.0;
        let live_ob_size = eqd("ob_size", self.ob_size as f64);
        let live_ob_a    = eqd("ob_a",    self.ob_a as f64);
        let live_ib_size = eqd("ib_size", self.ib_size as f64);
        let live_ib_a    = eqd("ib_a",    self.ib_a as f64);
        let live_outer_color = [eqd("ob_r", self.ob_r as f64), eqd("ob_g", self.ob_g as f64),
                                eqd("ob_b", self.ob_b as f64), live_ob_a];
        let live_inner_color = [eqd("ib_r", self.ib_r as f64), eqd("ib_g", self.ib_g as f64),
                                eqd("ib_b", self.ib_b as f64), live_ib_a];

        // ── Blur min/max range remap (butterchurn getBlurValues + getScaleAndBias) ──
        // The blur shader normalizes each level into [0,1] (scale_n,bias_n); the
        // comp/warp GetBlurN helpers apply the inverse (scale1..3, bias1..3) to recover
        // the original range. At defaults (min 0, max 1) both halves are identity.
        let (blur_sb, comp_blur) = {
            let mut bmin = [eqd("b1n", self.b1n as f64), eqd("b2n", self.b2n as f64), eqd("b3n", self.b3n as f64)];
            let mut bmax = [eqd("b1x", self.b1x as f64), eqd("b2x", self.b2x as f64), eqd("b3x", self.b3x as f64)];
            let fmin_dist = 0.1f32;
            // Min-distance enforcement: when a level's [min,max] is narrower than
            // fmin_dist, WIDEN it to fmin_dist about the midpoint (min down, max UP).
            // Butterchurn's source sets BOTH to `a - fmin_dist*0.5` (a typo → max==min →
            // scale=1/0=Inf/NaN); MilkDrop's intent (and the references we score against)
            // is max = a + fmin_dist*0.5. Use PLUS for max to restore the 0.1 range.
            if bmax[0] - bmin[0] < fmin_dist { let a = (bmin[0] + bmax[0]) * 0.5; bmin[0] = a - fmin_dist * 0.5; bmax[0] = a + fmin_dist * 0.5; }
            bmax[1] = bmax[1].min(bmax[0]); bmin[1] = bmin[1].max(bmin[0]);
            if bmax[1] - bmin[1] < fmin_dist { let a = (bmin[1] + bmax[1]) * 0.5; bmin[1] = a - fmin_dist * 0.5; bmax[1] = a + fmin_dist * 0.5; }
            bmax[2] = bmax[2].min(bmax[1]); bmin[2] = bmin[2].max(bmin[1]);
            if bmax[2] - bmin[2] < fmin_dist { let a = (bmin[2] + bmax[2]) * 0.5; bmin[2] = a - fmin_dist * 0.5; bmax[2] = a + fmin_dist * 0.5; }
            // blur-shader scale/bias (normalize into [0,1]) — butterchurn getScaleAndBias.
            let mut scale = [1.0f32; 3];
            let mut bias  = [0.0f32; 3];
            scale[0] = 1.0 / (bmax[0] - bmin[0]);
            bias[0]  = -bmin[0] * scale[0];
            let t_min1 = (bmin[1] - bmin[0]) / (bmax[0] - bmin[0]);
            let t_max1 = (bmax[1] - bmin[0]) / (bmax[0] - bmin[0]);
            scale[1] = 1.0 / (t_max1 - t_min1);
            bias[1]  = -t_min1 * scale[1];
            let t_min2 = (bmin[2] - bmin[1]) / (bmax[1] - bmin[1]);
            let t_max2 = (bmax[2] - bmin[1]) / (bmax[1] - bmin[1]);
            scale[2] = 1.0 / (t_max2 - t_min2);
            bias[2]  = -t_min2 * scale[2];
            // comp/warp-side inverse (butterchurn comp.js): scaleN = maxN-minN, biasN = minN.
            // (level 2/3 use the level-1 base in butterchurn's comp; mirror its actual code.)
            (
                [scale, bias],
                // comp uniforms: blur1_min/max + scale1/2/3 + bias1/2/3
                (bmin, bmax,
                 [bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]],
                 [bmin[0], bmin[1], bmin[2]]),
            )
        };

        // Build and upload PerFrame UBO
        let pf = PerFrame {
            texsize:    [w, h, 1.0 / w, 1.0 / h],
            aspect:     [asp, 1.0 / asp, 1.0, 1.0],
            // Time-based roam oscillators in [0,1] (butterchurn warp.js:838-861 / comp.js).
            // Were zeroed (..Zeroable) → roam-using warp/comp collapsed to grayscale/dim.
            slow_roam_cos: [0.5 + 0.5 * (t * 0.005).cos(), 0.5 + 0.5 * (t * 0.008).cos(),
                            0.5 + 0.5 * (t * 0.013).cos(), 0.5 + 0.5 * (t * 0.022).cos()],
            roam_cos:      [0.5 + 0.5 * (t * 0.3).cos(),   0.5 + 0.5 * (t * 1.3).cos(),
                            0.5 + 0.5 * (t * 5.0).cos(),   0.5 + 0.5 * (t * 20.0).cos()],
            slow_roam_sin: [0.5 + 0.5 * (t * 0.005).sin(), 0.5 + 0.5 * (t * 0.008).sin(),
                            0.5 + 0.5 * (t * 0.013).sin(), 0.5 + 0.5 * (t * 0.022).sin()],
            roam_sin:      [0.5 + 0.5 * (t * 0.3).sin(),   0.5 + 0.5 * (t * 1.3).sin(),
                            0.5 + 0.5 * (t * 5.0).sin(),   0.5 + 0.5 * (t * 20.0).sin()],
            rand_frame: [pseudo_rand(self.frame_idx), pseudo_rand(self.frame_idx + 1),
                         pseudo_rand(self.frame_idx + 2), pseudo_rand(self.frame_idx + 3)],
            rand_preset: self.rand_preset,
            // q1-q32 mapped to _qa.._qh (slots already reserved in the UBO; the
            // q9-q32 #defines live in preprocess.rs milk_fs_preamble).
            _qa: [eq("q1"),  eq("q2"),  eq("q3"),  eq("q4")],
            _qb: [eq("q5"),  eq("q6"),  eq("q7"),  eq("q8")],
            _qc: [eq("q9"),  eq("q10"), eq("q11"), eq("q12")],
            _qd: [eq("q13"), eq("q14"), eq("q15"), eq("q16")],
            _qe: [eq("q17"), eq("q18"), eq("q19"), eq("q20")],
            _qf: [eq("q21"), eq("q22"), eq("q23"), eq("q24")],
            _qg: [eq("q25"), eq("q26"), eq("q27"), eq("q28")],
            _qh: [eq("q29"), eq("q30"), eq("q31"), eq("q32")],
            time:       t,
            fps:        60.0,
            frame:      self.frame_idx as f32,
            progress:   (t % 30.0) / 30.0,
            bass:     bass as f32,
            mid:      mid  as f32,
            treb:     treb as f32,
            vol:      vol  as f32,
            bass_att: bass_att as f32,
            mid_att:  mid_att  as f32,
            treb_att: treb_att as f32,
            vol_att:  vol_att  as f32,
            // EEL/Butterchurn per-frame var names are `gamma` and `fshader` (no
            // underscore); reading `gamma_adj`/`f_shader` always missed → gamma was a
            // no-op (1.0) and fshader stuck at 0 (so hue gating couldn't work).
            gamma_adj: eqd("gamma", 1.0),
            f_shader:   eq("fshader"),
            echo_zoom:  eqd("echo_zoom",  1.0),
            echo_alpha: eq("echo_alpha"),
            // EEL/Butterchurn var name is "echo_orient" (the UBO field is named
            // echo_orientation). render() previously read "echo_orientation", a var no
            // preset ever sets via EEL2, so echo orientation was always silently 0.
            echo_orientation: eq("echo_orient"),
            // comp_blur = (mins, maxs, scales, biases) from the per-level blur remap.
            blur1_min: comp_blur.0[0], blur1_max: comp_blur.1[0],
            blur2_min: comp_blur.0[1], blur2_max: comp_blur.1[1],
            blur3_min: comp_blur.0[2], blur3_max: comp_blur.1[2],
            scale1: comp_blur.2[0], scale2: comp_blur.2[1], scale3: comp_blur.2[2],
            bias1:  comp_blur.3[0], bias2:  comp_blur.3[1], bias3:  comp_blur.3[2],
            brighten: if self.comp_brighten { 1.0 } else { 0.0 },
            darken:   if self.comp_darken   { 1.0 } else { 0.0 },
            solarize: if self.comp_solarize { 1.0 } else { 0.0 },
            invert:   if self.comp_invert   { 1.0 } else { 0.0 },
            ..bytemuck::Zeroable::zeroed()
        };
        self.queue.write_buffer(&self.perframe_buf, 0, bytemuck::bytes_of(&pf));

        // ── Blur range-remap: write per-level sb (scale,bias) into the blur UBOs
        // at offset 32 (the third vec4). The blur shader normalizes into [0,1]. ──
        let scale = blur_sb[0];
        let bias  = blur_sb[1];
        for (ubo, lvl) in [(&self.blur1_ubo, 0usize), (&self.blur2_ubo, 1), (&self.blur3_ubo, 2)] {
            let sb = [scale[lvl], bias[lvl], 0.0f32, 0.0f32];
            self.queue.write_buffer(ubo, 32, bytemuck::cast_slice(&sb));
        }

        // ── Build shape + waveform geometry (BEFORE any render pass opens) ────
        // Shapes use aspecty (landscape: h/w) to keep discs round. Custom waves use
        // the inverse-aspect convention (butterchurn invAspectx/invAspecty).
        let (shape_aspectx, shape_aspecty) = if self.width > self.height {
            (1.0f32, self.height as f32 / self.width as f32)
        } else {
            (self.width as f32 / self.height as f32, 1.0f32)
        };
        // butterchurn: aspectx=texsizeX>texsizeY?1:..., invAspectx=1/aspectx. For
        // built-in waveform the formula multiplies x by aspecty and y by aspectx
        // (the round-keeping factors). For custom waves it divides by aspect.
        let inv_aspectx = if shape_aspectx != 0.0 { 1.0 / shape_aspectx } else { 1.0 };
        let inv_aspecty = if shape_aspecty != 0.0 { 1.0 / shape_aspecty } else { 1.0 };

        // q1..q32 snapshot for shape per-frame programs (MilkDrop/Butterchurn pass the
        // full q1..q32 from mdVSQAfterFrame to custom shapes; capping at q8 left q9..q32
        // = 0 in shape eqs, e.g. idx 7550's `a = floor(rand(floor(q30)))/5` → alpha 0).
        let mut qsnap = [0.0f64; 32];
        for i in 0..32 {
            qsnap[i] = self.eel_env.get(format!("q{}", i + 1).as_str()).copied().unwrap_or(0.0);
        }

        let (fill_verts, fill_draws, border_verts, border_draws) =
            self.build_shape_geometry(t, bass, mid, treb, vol,
                bass_att, mid_att, treb_att, shape_aspectx, shape_aspecty, &qsnap);

        let (wave_l, wave_r) = self.frame_waveform(t);
        // freqArray for bSpectrum custom waves (mono, 512 bins). Empty in the
        // headless/synthetic path → build_custom_waves falls back to time data.
        let freq = self.freq_spectrum.clone();
        let (wave_verts, wave_draws) = self.build_wave_geometry(
            t, bass, mid, treb, vol, bass_att, mid_att, treb_att,
            inv_aspectx, inv_aspecty, &wave_l, &wave_r, &freq);

        // Upload all geometry up-front (no write_buffer inside a render pass).
        if !fill_verts.is_empty() {
            let n = fill_verts.len().min(SHAPE_VERT_CAP);
            self.queue.write_buffer(&self.shape_vert_buf, 0, bytemuck::cast_slice(&fill_verts[..n]));
        }
        if !border_verts.is_empty() {
            let n = border_verts.len().min(BORDER_VERT_CAP);
            self.queue.write_buffer(&self.border_vert_buf, 0, bytemuck::cast_slice(&border_verts[..n]));
        }
        if !wave_verts.is_empty() {
            let n = wave_verts.len().min(WAVE_VERT_CAP);
            self.queue.write_buffer(&self.wave_vert_buf, 0, bytemuck::cast_slice(&wave_verts[..n]));
        }
        // ShapeU textured flag: untextured for now (jelly_space is untextured).
        self.queue.write_buffer(&self.shape_uniform_buf, 0,
            bytemuck::bytes_of(&ShapeU { textured: 0.0, _pad: [0.0; 3] }));

        // Thick-offset slots for the waveform dyn-offset UBO (4 slots @ 256B).
        let tsx = 2.0 / self.width as f32;
        let tsy = 2.0 / self.height as f32;
        let offsets = [[0.0f32, 0.0, 0.0, 0.0], [tsx, 0.0, 0.0, 0.0],
                       [0.0, tsy, 0.0, 0.0], [tsx, tsy, 0.0, 0.0]];
        let mut off_bytes = vec![0u8; 4 * 256];
        for (k, o) in offsets.iter().enumerate() {
            off_bytes[k * 256..k * 256 + 16].copy_from_slice(bytemuck::cast_slice(o));
        }
        self.queue.write_buffer(&self.wave_off_buf, 0, &off_bytes);

        // Border thick-offset + color slots: reuse the same 4 offsets but each slot
        // holds a full BorderU. We overwrite color per border draw via a single slot 0
        // path; for the up-to-4 thick passes we use slots 0..4 with offsets baked in.
        let border_color_for_buf = border_draws.first().map(|d| d.color).unwrap_or([0.0; 4]);
        let mut bu_bytes = vec![0u8; 4 * 256];
        for (k, o) in offsets.iter().enumerate() {
            let bu = BorderU { color: border_color_for_buf, offset: *o };
            bu_bytes[k * 256..k * 256 + std::mem::size_of::<BorderU>()]
                .copy_from_slice(bytemuck::bytes_of(&bu));
        }
        self.queue.write_buffer(&self.border_uniform_buf, 0, &bu_bytes);

        // --- WARP geometry: compute per-vertex warped UV + decay ONCE (both paths
        // now use the warped mesh). MUST upload before opening the render pass, and
        // BEFORE the immutable ping-pong borrows below (compute_warp_verts is &mut). ---
        let warp_verts = self.compute_warp_verts(t, shape_aspectx, shape_aspecty);
        self.queue.write_buffer(&self.warp_vert_buf, 0, bytemuck::cast_slice(&warp_verts));

        // ── MOTION VECTORS geometry (butterchurn MotionVectors.generateMotionVectors)
        // Reuses warp_verts as the flow field. Built for BOTH warp paths (compute_warp_verts
        // produces a UV grid in all cases). Live mv_* read from the per-frame EEL env.
        let mv_count: u32 = {
            let mv_on   = self.mv_on;
            let mv_a    = live_mv_a;
            let mv_x    = live_mv_x;
            let mv_y    = live_mv_y;
            let mv_dx   = live_mv_dx;
            let mv_dy   = live_mv_dy;
            let mv_l    = live_mv_l;
            let mv_r    = live_mv_r;
            let mv_g    = live_mv_g;
            let mv_b    = live_mv_b;
            let mut n_x = mv_x.floor() as i32;
            let mut n_y = mv_y.floor() as i32;
            if mv_on && mv_a > 0.001 && n_x > 0 && n_y > 0 {
                let mut dx = mv_x - n_x as f32;
                let mut dy = mv_y - n_y as f32;
                if n_x > 64 { n_x = 64; dx = 0.0; }
                if n_y > 48 { n_y = 48; dy = 0.0; }
                let dx2 = mv_dx;
                let dy2 = mv_dy;
                let len_mult = mv_l;
                let min_len = 1.0 / self.width as f32;

                // Bilinear sample of the warp UV field; returns (fx2, 1.0-fy2) (V flip,
                // matching butterchurn getMotionDir). Mesh = GRID_W x GRID_H.
                let mw = GRID_W as f32;
                let mh = GRID_H as f32;
                let grid_x1 = (GRID_W + 1) as usize;
                let sample = |fx: f32, fy: f32| -> (f32, f32) {
                    let mut x0 = (fx * mw).floor() as i32;
                    let mut y0 = (fy * mh).floor() as i32;
                    let ddx = fx * mw - x0 as f32;
                    let ddy = fy * mh - y0 as f32;
                    // clamp to valid vertex indices [0, GRID]
                    let gx = GRID_W as i32;
                    let gy = GRID_H as i32;
                    if x0 < 0 { x0 = 0; }
                    if y0 < 0 { y0 = 0; }
                    let x1 = (x0 + 1).min(gx);
                    let y1 = (y0 + 1).min(gy);
                    let x0 = x0.min(gx);
                    let y0 = y0.min(gy);
                    let uv = |col: i32, row: i32| -> (f32, f32) {
                        let idx = (row as usize) * grid_x1 + (col as usize);
                        let v = warp_verts[idx].uv;
                        (v[0], v[1])
                    };
                    let (u00, v00) = uv(x0, y0);
                    let (u10, v10) = uv(x1, y0);
                    let (u01, v01) = uv(x0, y1);
                    let (u11, v11) = uv(x1, y1);
                    let fx2 = u00 * (1.0 - ddx) * (1.0 - ddy) + u10 * ddx * (1.0 - ddy)
                            + u01 * (1.0 - ddx) * ddy        + u11 * ddx * ddy;
                    let fy2 = v00 * (1.0 - ddx) * (1.0 - ddy) + v10 * ddx * (1.0 - ddy)
                            + v01 * (1.0 - ddx) * ddy        + v11 * ddx * ddy;
                    (fx2, 1.0 - fy2)
                };

                let mut mv_verts: Vec<MVVert> = Vec::with_capacity(MV_VERT_CAP);
                for j in 0..n_y {
                    let mut fy = (j as f32 + 0.25) / (n_y as f32 + dy + 0.25 - 1.0);
                    fy -= dy2;
                    if fy > 0.0001 && fy < 0.9999 {
                        for i in 0..n_x {
                            let mut fx = (i as f32 + 0.25) / (n_x as f32 + dx + 0.25 - 1.0);
                            fx += dx2;
                            if fx > 0.0001 && fx < 0.9999 {
                                let (fx2s, fy2s) = sample(fx, fy);
                                let mut dxi = (fx2s - fx) * len_mult;
                                let mut dyi = (fy2s - fy) * len_mult;
                                let fdist = (dxi * dxi + dyi * dyi).sqrt();
                                if fdist < min_len && fdist > 0.00000001 {
                                    let g = min_len / fdist;
                                    dxi *= g;
                                    dyi *= g;
                                } else {
                                    // VERBATIM butterchurn bug (lines 6828-6829):
                                    // dxi = minLen twice; dyi is NOT reset (keeps its
                                    // scaled value). Replicated exactly for parity.
                                    #[allow(unused_assignments)]
                                    { dxi = min_len; }
                                    dxi = min_len;
                                }
                                let efx2 = fx + dxi;
                                let efy2 = fy + dyi;
                                // NDC: x = 2*fx-1; y = 1.0-2*fy (negated vs butterchurn to
                                // match our compute_warp_verts y-down→y-up mapping).
                                let vx1 = 2.0 * fx - 1.0;
                                let vy1 = 1.0 - 2.0 * fy;
                                let vx2 = 2.0 * efx2 - 1.0;
                                let vy2 = 1.0 - 2.0 * efy2;
                                mv_verts.push(MVVert { pos: [vx1, vy1] });
                                mv_verts.push(MVVert { pos: [vx2, vy2] });
                            }
                        }
                    }
                }
                let cnt = mv_verts.len().min(MV_VERT_CAP);
                if cnt > 0 {
                    self.queue.write_buffer(&self.mv_vert_buf, 0, bytemuck::cast_slice(&mv_verts[..cnt]));
                    let col = MVColor { color: [mv_r, mv_g, mv_b, mv_a] };
                    self.queue.write_buffer(&self.mv_color_buf, 0, bytemuck::bytes_of(&col));
                }
                cnt as u32
            } else {
                0
            }
        };

        // ── DARKEN-CENTER geometry (butterchurn DarkenCenter). Small triangle-fan
        // (expanded to a triangle list): center black @ alpha 3/32, perimeter @ 0.
        let darken_on = live_darken;
        if darken_on {
            let half = 0.05f32;
            let ax = shape_aspecty; // butterchurn applies aspecty to the x extents
            // fan verts: [center, p1, p2, p3, p4, p5] with p5 == p1 (closing).
            let center = ([0.0f32, 0.0f32], [0.0f32, 0.0, 0.0, 3.0 / 32.0]);
            let p1 = ([-half * ax,  0.0f32], [0.0f32, 0.0, 0.0, 0.0]);
            let p2 = ([ 0.0f32,    -half],   [0.0f32, 0.0, 0.0, 0.0]);
            let p3 = ([ half * ax,  0.0f32], [0.0f32, 0.0, 0.0, 0.0]);
            let p4 = ([ 0.0f32,     half],   [0.0f32, 0.0, 0.0, 0.0]);
            let p5 = ([-half * ax,  0.0f32], [0.0f32, 0.0, 0.0, 0.0]);
            // TRIANGLE_FAN(6 verts) → 4 triangles, expanded to a triangle list.
            let fan = [center, p1, p2, p3, p4, p5];
            let tris = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5)];
            let mut dv: Vec<DarkenVert> = Vec::with_capacity(12);
            for (a, b, c) in tris {
                for k in [a, b, c] {
                    let (pos, color) = fan[k];
                    dv.push(DarkenVert { pos, color });
                }
            }
            self.queue.write_buffer(&self.darken_vert_buf, 0, bytemuck::cast_slice(&dv));
        }

        // ── FRAME-BORDER geometry (butterchurn Border.generateBorder). Outer ring
        // (prevBorderSize 0) + inner ring (prevBorderSize = ob_size). NDC, no aspect.
        let ob_size = live_ob_size;
        let ob_a    = live_ob_a;
        let ib_size = live_ib_size;
        let ib_a    = live_ib_a;
        let outer_color = live_outer_color;
        let inner_color = live_inner_color;
        // generate_border(border_size, prev_border_size) → 24 NDC verts, or None.
        let gen_border = |border_size: f32, prev_border_size: f32, alpha: f32| -> Option<Vec<BorderVert>> {
            if !(border_size > 0.0 && alpha > 0.0) { return None; }
            let width = 2.0f32;
            let height = 2.0f32;
            let wh = width / 2.0;
            let hh = height / 2.0;
            let pbw = prev_border_size / 2.0;
            let bw = border_size / 2.0 + pbw;
            let pbww = pbw * width;
            let pbwh = pbw * height;
            let bww = bw * width;
            let bwh = bw * height;
            let mut v: Vec<BorderVert> = Vec::with_capacity(24);
            let mut tri = |p1: [f32; 2], p2: [f32; 2], p3: [f32; 2]| {
                v.push(BorderVert { pos: p1 });
                v.push(BorderVert { pos: p2 });
                v.push(BorderVert { pos: p3 });
            };
            // 1st side (left)
            let a1 = [-wh + pbww, -hh + bwh];
            let a2 = [-wh + pbww,  hh - bwh];
            let a3 = [-wh + bww,   hh - bwh];
            let a4 = [-wh + bww,  -hh + bwh];
            tri(a4, a2, a1); tri(a4, a3, a2);
            // 2nd side (right)
            let b1 = [ wh - pbww, -hh + bwh];
            let b2 = [ wh - pbww,  hh - bwh];
            let b3 = [ wh - bww,   hh - bwh];
            let b4 = [ wh - bww,  -hh + bwh];
            tri(b1, b2, b4); tri(b2, b3, b4);
            // Top
            let c1 = [-wh + pbww, -hh + pbwh];
            let c2 = [-wh + pbww,  bwh - hh];
            let c3 = [ wh - pbww,  bwh - hh];
            let c4 = [ wh - pbww, -hh + pbwh];
            tri(c4, c2, c1); tri(c4, c3, c2);
            // Bottom
            let d1 = [-wh + pbww,  hh - pbwh];
            let d2 = [-wh + pbww,  hh - bwh];
            let d3 = [ wh - pbww,  hh - bwh];
            let d4 = [ wh - pbww,  hh - pbwh];
            tri(d1, d2, d4); tri(d2, d3, d4);
            Some(v)
        };
        let outer_verts = gen_border(ob_size, 0.0, ob_a);
        let inner_verts = gen_border(ib_size, ob_size, ib_a);
        // Pack the two border draws contiguously; record (start_vert, color_slot).
        let mut border_draws_frame: Vec<(u32, u32)> = Vec::new(); // (start_vert, slot)
        {
            let mut all: Vec<BorderVert> = Vec::new();
            if let Some(ov) = &outer_verts {
                border_draws_frame.push((all.len() as u32, 0));
                all.extend_from_slice(ov);
            }
            if let Some(iv) = &inner_verts {
                border_draws_frame.push((all.len() as u32, 1));
                all.extend_from_slice(iv);
            }
            if !all.is_empty() {
                self.queue.write_buffer(&self.frame_border_vert_buf, 0, bytemuck::cast_slice(&all));
                // slot 0 = outer color, slot 1 = inner color (dyn-offset 256B each)
                let mut fb_bytes = vec![0u8; 2 * 256];
                let ou = BorderU { color: outer_color, offset: [0.0; 4] };
                let iu = BorderU { color: inner_color, offset: [0.0; 4] };
                fb_bytes[0..std::mem::size_of::<BorderU>()].copy_from_slice(bytemuck::bytes_of(&ou));
                fb_bytes[256..256 + std::mem::size_of::<BorderU>()].copy_from_slice(bytemuck::bytes_of(&iu));
                self.queue.write_buffer(&self.frame_border_uniform_buf, 0, &fb_bytes);
            }
        }

        // Ping-pong: write_to_a determines current target
        let (write_view, read_bg, comp_bg) = if self.write_to_a {
            (&self.view_a, &self.bg_read_b, &self.bg_read_a)
        } else {
            (&self.view_b, &self.bg_read_a, &self.bg_read_b)
        };

        // Recreate the blur1 horizontal-pass bind group each frame so it reads from the
        // CURRENT write target (the warp output, not the previous frame).
        let blur1_h_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.blur_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(write_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.clamp_samp) },
                wgpu::BindGroupEntry { binding: 2, resource: self.blur1_ubo.as_entire_binding() },
            ],
        });

        let mut enc = self.device.create_command_encoder(&Default::default());

        // --- WARP pass: read from prev, write to curr (Clear: regenerate frame). ---
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("warp"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: write_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if self.has_custom_warp {
                // Custom warp FS, driven by the warped mesh VS.
                rp.set_pipeline(&self.warp_custom_pipeline);
                rp.set_bind_group(0, read_bg, &[]);          // sampler set (prev frame)
                rp.set_bind_group(1, &self.perframe_bg, &[]);
            } else {
                // Default warp mesh: sample prev at warped UV, multiply per-vertex decay.
                let mesh_bg = if self.write_to_a { &self.warp_mesh_bg_b } else { &self.warp_mesh_bg_a };
                rp.set_pipeline(&self.warp_mesh_pipeline);
                rp.set_bind_group(0, mesh_bg, &[]);
            }
            rp.set_vertex_buffer(0, self.warp_vert_buf.slice(..));
            rp.set_index_buffer(self.warp_idx_buf.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..self.warp_idx_count, 0, 0..1);
        }

        // --- BLUR passes (separable wide Gaussian, progressive chain) ---
        // Each level: H (src → btemp) then V (btemp → level). blur1←warp, blur2←blur1, blur3←blur2.
        {
            let mut blur_pass = |label: &str, pipeline: &wgpu::RenderPipeline,
                                 bg: &wgpu::BindGroup, target: &wgpu::TextureView| {
                let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(label),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                rp.set_pipeline(pipeline);
                rp.set_bind_group(0, bg, &[]);
                rp.draw(0..3, 0..1);
            };
            blur_pass("blur1-h", &self.blur_h_pipeline, &blur1_h_bg,       &self.view_btemp1);
            blur_pass("blur1-v", &self.blur_v_pipeline, &self.blur1_v_bg, &self.view_blur1);
            blur_pass("blur2-h", &self.blur_h_pipeline, &self.blur2_h_bg, &self.view_btemp2);
            blur_pass("blur2-v", &self.blur_v_pipeline, &self.blur2_v_bg, &self.view_blur2);
            blur_pass("blur3-h", &self.blur_h_pipeline, &self.blur3_h_bg, &self.view_btemp3);
            blur_pass("blur3-v", &self.blur_v_pipeline, &self.blur3_v_bg, &self.view_blur3);
        }

        // --- MOTION VECTORS pass: drawn into the warped+blurred feedback target
        // (write_view, LoadOp::Load) BEFORE shapes/waves, matching butterchurn order. ---
        if mv_count > 0 {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("motion-vectors"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: write_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(&self.mv_pipeline);
            rp.set_bind_group(0, &self.mv_bg, &[]);
            rp.set_vertex_buffer(0, self.mv_vert_buf.slice(..));
            rp.draw(0..mv_count, 0..1);
        }

        // --- SHAPES + WAVES pass: composite over the warped+blurred frame ---
        // Drawn into write_view with Load/Store (NEVER Clear). comp reads write_view,
        // so shapes/waves appear AND feed back next frame (MilkDrop behavior).
        let has_shapes = !fill_draws.is_empty() || !border_draws.is_empty();
        let has_waves  = !wave_draws.is_empty();
        if has_shapes || has_waves {
            // prev-frame texture for textured shapes = the OTHER ping-pong side.
            let shape_read_bg = if self.write_to_a { &self.shape_bg_read_b } else { &self.shape_bg_read_a };

            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shapes-waves"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: write_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // 1) shape fills
            if !fill_draws.is_empty() {
                rp.set_vertex_buffer(0, self.shape_vert_buf.slice(..));
                rp.set_index_buffer(self.shape_idx_buf.slice(..), wgpu::IndexFormat::Uint32);
                rp.set_bind_group(0, shape_read_bg, &[]);
                for d in &fill_draws {
                    let pipe = if d.additive { &self.shapes_fill_pipeline_additive } else { &self.shapes_fill_pipeline_alpha };
                    rp.set_pipeline(pipe);
                    rp.set_bind_group(0, shape_read_bg, &[]);
                    rp.draw_indexed(0..(d.sides * 3), d.base_vertex, 0..1);
                }
            }

            // 2) shape borders (LineStrip; up-to-4 thick passes via dyn offset slots)
            if !border_draws.is_empty() {
                rp.set_pipeline(&self.shapes_border_pipeline);
                rp.set_vertex_buffer(0, self.border_vert_buf.slice(..));
                // Note: all borders share the first border color (baked into the UBO);
                // multi-color borders are a known limitation. jelly_space has no border.
                let d = &border_draws[0];
                let passes = if d.thick { 4 } else { 1 };
                for k in 0..passes {
                    rp.set_bind_group(0, &self.border_bg, &[(k * 256) as u32]);
                    rp.draw(d.start_vert..(d.start_vert + d.count), 0..1);
                }
            }

            // 3) waveforms
            if !wave_draws.is_empty() {
                rp.set_vertex_buffer(0, self.wave_vert_buf.slice(..));
                for d in &wave_draws {
                    let pipe = match (d.points, d.additive) {
                        (true,  true)  => &self.wave_pipeline_points_additive,
                        (true,  false) => &self.wave_pipeline_points_alpha,
                        (false, true)  => &self.wave_pipeline_lines_additive,
                        (false, false) => &self.wave_pipeline_lines_alpha,
                    };
                    rp.set_pipeline(pipe);
                    // Clamp to the uploaded vertex count so an over-cap wave never
                    // draws from a stale/zero buffer tail.
                    if d.start_vert >= WAVE_VERT_CAP as u32 { continue; }
                    let end = (d.start_vert + d.count).min(WAVE_VERT_CAP as u32);
                    let passes = if d.thick { 4 } else { 1 };
                    for k in 0..passes {
                        rp.set_bind_group(0, &self.wave_bg, &[(k * 256) as u32]);
                        rp.draw(d.start_vert..end, 0..1);
                    }
                }
            }
        }

        // --- DARKEN-CENTER + FRAME-BORDERS pass: into write_view (LoadOp::Load),
        // AFTER shapes/waves, BEFORE comp (butterchurn draw order). ---
        if darken_on || !border_draws_frame.is_empty() {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("darken-borders"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: write_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // 1) darken center (12 verts = 4 fan triangles)
            if darken_on {
                rp.set_pipeline(&self.darken_pipeline);
                rp.set_vertex_buffer(0, self.darken_vert_buf.slice(..));
                rp.draw(0..12, 0..1);
            }
            // 2) frame borders (outer then inner), each 24 verts, dyn-offset color slot
            if !border_draws_frame.is_empty() {
                rp.set_pipeline(&self.frame_border_pipeline);
                rp.set_vertex_buffer(0, self.frame_border_vert_buf.slice(..));
                for (start_vert, slot) in &border_draws_frame {
                    rp.set_bind_group(0, &self.frame_border_bg, &[(slot * 256) as u32]);
                    rp.draw(*start_vert..(*start_vert + 24), 0..1);
                }
            }
        }

        // --- COMP pass: read from curr, write to offscreen comp target ---
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("comp"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.comp_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None
            });
            rp.set_pipeline(&self.comp_pipeline);
            rp.set_bind_group(0, comp_bg, &[]);
            rp.set_bind_group(1, &self.perframe_bg, &[]);
            rp.draw(0..3, 0..1);
        }

        // --- OUTPUT pass: FXAA the offscreen comp result → swapchain ---
        // Fullscreen triangle covers 100% → LoadOp::Clear (no needless read).
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fxaa-output"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: surface_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None
            });
            rp.set_pipeline(&self.output_pipeline);
            rp.set_bind_group(0, &self.fxaa_bg, &[]);
            rp.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(enc.finish()));
        self.write_to_a = !self.write_to_a;
        self.frame_idx += 1;
    }
}

// WaveUtils.smoothWave — positions only (used by BasicWaveform). Catmull-Rom-ish.
// `pts` is a flat list of (x,y); returns interleaved smoothed list of (n*2-1).
fn smooth_wave(pts: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let n = pts.len();
    if n < 2 { return pts.to_vec(); }
    let c1 = -0.15f32; let c2 = 1.15f32; let c3 = 1.15f32; let c4 = -0.15f32;
    let inv_sum = 1.0 / (c1 + c2 + c3 + c4); // = 0.5
    let mut out = vec![[0.0f32; 2]; n * 2 - 1];
    let mut j = 0usize;
    let mut i_below = 0usize;
    let mut i_above2 = 1usize;
    for i in 0..n - 1 {
        let i_above = i_above2;
        i_above2 = (i + 2).min(n - 1);
        out[j] = pts[i];
        out[j + 1][0] = (c1 * pts[i_below][0] + c2 * pts[i][0] + c3 * pts[i_above][0] + c4 * pts[i_above2][0]) * inv_sum;
        out[j + 1][1] = (c1 * pts[i_below][1] + c2 * pts[i][1] + c3 * pts[i_above][1] + c4 * pts[i_above2][1]) * inv_sum;
        i_below = i;
        j += 2;
    }
    out[j] = pts[n - 1];
    out
}

// WaveUtils.smoothWaveAndColor — positions + held color. Returns (positions, colors).
fn smooth_wave_and_color(pts: &[[f32; 2]], cols: &[[f32; 4]]) -> (Vec<[f32; 2]>, Vec<[f32; 4]>) {
    let n = pts.len();
    if n < 2 { return (pts.to_vec(), cols.to_vec()); }
    let c1 = -0.15f32; let c2 = 1.15f32; let c3 = 1.15f32; let c4 = -0.15f32;
    let inv_sum = 1.0 / (c1 + c2 + c3 + c4);
    let mut out_p = vec![[0.0f32; 2]; n * 2 - 1];
    let mut out_c = vec![[0.0f32; 4]; n * 2 - 1];
    let mut j = 0usize;
    let mut i_below = 0usize;
    let mut i_above2 = 1usize;
    for i in 0..n - 1 {
        let i_above = i_above2;
        i_above2 = (i + 2).min(n - 1);
        out_p[j] = pts[i];
        out_p[j + 1][0] = (c1 * pts[i_below][0] + c2 * pts[i][0] + c3 * pts[i_above][0] + c4 * pts[i_above2][0]) * inv_sum;
        out_p[j + 1][1] = (c1 * pts[i_below][1] + c2 * pts[i][1] + c3 * pts[i_above][1] + c4 * pts[i_above2][1]) * inv_sum;
        out_c[j] = cols[i];
        out_c[j + 1] = cols[i];
        i_below = i;
        j += 2;
    }
    out_p[j] = pts[n - 1];
    out_c[j] = cols[n - 1];
    (out_p, out_c)
}

fn pseudo_rand(seed: u32) -> f32 {
    let mut x = seed.wrapping_mul(0x9e3779b9).wrapping_add(0x6c62272e);
    x ^= x >> 16; x = x.wrapping_mul(0x45d9f3b); x ^= x >> 16;
    (x as f32) / (u32::MAX as f32)
}
