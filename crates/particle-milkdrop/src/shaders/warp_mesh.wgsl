// Standard MilkDrop warp mesh pass (no custom warp shader).
// Vertex buffer carries NDC screen position + warped UV (sample coord into the
// previous frame) + per-vertex decay rgb. The fragment shader samples the
// previous frame at the warped UV and multiplies by the per-vertex decay.

@group(0) @binding(0) var prev_tex:  texture_2d<f32>;
@group(0) @binding(1) var prev_samp: sampler;

struct VIn {
    @location(0) pos:   vec2<f32>,
    @location(1) uv:    vec2<f32>,
    @location(2) decay: vec4<f32>,
}
struct VOut {
    @builtin(position) clip:  vec4<f32>,
    @location(0)       uv:    vec2<f32>,
    @location(1)       decay: vec4<f32>,
}

@vertex
fn vs_main(v: VIn) -> VOut {
    var o: VOut;
    o.clip  = vec4<f32>(v.pos, 0.0, 1.0);
    o.uv    = v.uv;
    o.decay = v.decay;
    return o;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let c = textureSample(prev_tex, prev_samp, in.uv);
    return vec4<f32>(c.rgb * in.decay.rgb, 1.0);
}
