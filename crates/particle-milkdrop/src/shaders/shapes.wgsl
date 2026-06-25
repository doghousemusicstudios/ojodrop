// Custom-shape fill + border shaders (hand-written WGSL — NOT routed through the
// HLSL/GLSL translator). Positions arrive already in NDC.

struct VSIn  {
    @location(0) pos:   vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv:    vec2<f32>,
};
struct VSOut {
    @builtin(position) pos:   vec4<f32>,
    @location(0)       color: vec4<f32>,
    @location(1)       uv:    vec2<f32>,
};

// group(0): prev-frame texture (for textured shapes) + sampler + ShapeU uniform.
// Keep this 16 bytes (matches Rust ShapeU) — avoid vec3 which would pad to 32.
struct ShapeU { textured: f32, _p0: f32, _p1: f32, _p2: f32 };
@group(0) @binding(0) var prev_tex: texture_2d<f32>;
@group(0) @binding(1) var prev_samp: sampler;
@group(0) @binding(2) var<uniform> su: ShapeU;

@vertex
fn vs_shape(in: VSIn) -> VSOut {
    var o: VSOut;
    o.pos   = vec4<f32>(in.pos, 0.0, 1.0); // already NDC
    o.color = in.color;
    o.uv    = in.uv;
    return o;
}

@fragment
fn fs_shape(in: VSOut) -> @location(0) vec4<f32> {
    // Per-shape textured flag is baked into the UV: untextured verts carry a
    // negative UV sentinel (uniforms can't vary per-shape within one render pass).
    if (in.uv.x < 0.0) {
        return in.color;
    }
    return textureSample(prev_tex, prev_samp, in.uv) * in.color;
}

// ── Border (separate bind group: single uniform, no texture) ──────────────────
struct BorderU { color: vec4<f32>, offset: vec4<f32> }; // offset.xy = thickOffset
@group(0) @binding(0) var<uniform> bu: BorderU;

@vertex
fn vs_border(@location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(pos + bu.offset.xy, 0.0, 1.0);
}

@fragment
fn fs_border() -> @location(0) vec4<f32> {
    return bu.color;
}
