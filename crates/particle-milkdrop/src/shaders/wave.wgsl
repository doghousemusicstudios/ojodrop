// Waveform shader (built-in + custom). Positions arrive in NDC; a per-draw
// thick-offset uniform is added (used for the 4-pass thick line / dot expansion).
// Matches butterchurn's waveform vert/frag (gl_Position = aPos + thickOffset;
// fragColor = vColor).

struct Off { off: vec4<f32> };
@group(0) @binding(0) var<uniform> u: Off;

struct VIn  {
    @location(0) pos:   vec2<f32>,
    @location(1) color: vec4<f32>,
};
struct VOut {
    @builtin(position) clip:  vec4<f32>,
    @location(0)       color: vec4<f32>,
};

@vertex
fn vs_main(v: VIn) -> VOut {
    var o: VOut;
    o.clip  = vec4<f32>(v.pos + u.off.xy, 0.0, 1.0);
    o.color = v.color;
    return o;
}

@fragment
fn fs_main(i: VOut) -> @location(0) vec4<f32> {
    return i.color;
}
