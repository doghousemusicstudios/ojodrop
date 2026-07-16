// Waveform shader (built-in + custom). Positions arrive in NDC; thick line/dot
// expansion uses instance_index so 4/9 stamps are emitted by one draw call.
// Matches butterchurn's waveform vert/frag (gl_Position = aPos + thickOffset;
// fragColor = vColor).

struct Off { texel: vec4<f32> };
@group(0) @binding(0) var<uniform> u: Off;

struct VIn  {
    @location(0) pos:   vec2<f32>,
    @location(1) color: vec4<f32>,
};
struct VOut {
    @builtin(position) clip:  vec4<f32>,
    @location(0)       color: vec4<f32>,
};

fn thick_offset(instance: u32) -> vec2<f32> {
    let x = u.texel.x;
    let y = u.texel.y;
    switch instance {
        case 1u: { return vec2<f32>(x, 0.0); }
        case 2u: { return vec2<f32>(0.0, y); }
        case 3u: { return vec2<f32>(x, y); }
        case 4u: { return vec2<f32>(-x, 0.0); }
        case 5u: { return vec2<f32>(0.0, -y); }
        case 6u: { return vec2<f32>(-x, -y); }
        case 7u: { return vec2<f32>(x, -y); }
        case 8u: { return vec2<f32>(-x, y); }
        default: { return vec2<f32>(0.0); }
    }
}

@vertex
fn vs_main(v: VIn, @builtin(instance_index) instance: u32) -> VOut {
    var o: VOut;
    o.clip  = vec4<f32>(v.pos + thick_offset(instance), 0.0, 1.0);
    o.color = v.color;
    return o;
}

@fragment
fn fs_main(i: VOut) -> @location(0) vec4<f32> {
    return i.color;
}
