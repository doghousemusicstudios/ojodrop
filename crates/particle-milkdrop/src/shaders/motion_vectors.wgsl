// Motion vectors (butterchurn MotionVectors.drawMotionVectors parity).
// LineList of arrows drawn into the warped+blurred feedback target with alpha
// blend, BEFORE custom shapes/waveforms. Vertices arrive already in NDC.
// Single flat color uniform (mv_r, mv_g, mv_b, mvA) for the whole draw.

struct U { color: vec4<f32> };
@group(0) @binding(0) var<uniform> u: U;

struct VIn  { @location(0) pos: vec2<f32> };
struct VOut { @builtin(position) clip: vec4<f32> };

@vertex
fn vs_main(v: VIn) -> VOut {
    var o: VOut;
    o.clip = vec4<f32>(v.pos, 0.0, 1.0);
    return o;
}

@fragment
fn fs_main(i: VOut) -> @location(0) vec4<f32> {
    return u.color;
}
