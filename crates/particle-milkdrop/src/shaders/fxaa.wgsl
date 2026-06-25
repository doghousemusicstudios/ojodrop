// FXAA output pass — faithful port of Butterchurn's glsl-fxaa (output.js).
// Reads the offscreen comp result and resolves edges into the swapchain.
//
// The comp result is stored with our flipped-uv convention (uv(0,0) = top-left,
// matching quad.wgsl / MilkDrop). This VS reuses the SAME flip so the image
// presents upright — do NOT switch to the un-flipped p*0.5+0.5 form.
//
// Corner taps and the final span use texsize.zw = (1/W, 1/H) for both axes — the
// mathematically-correct standard FXAA (Butterchurn's VS has the upstream .zx
// quirk where the y-offset is W; at our 1:1 render-to-screen ratio (1/W,1/H) is
// what's intended).

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
    var pos: array<vec2<f32>, 3>;
    pos[0] = vec2<f32>(-1.0, -1.0);
    pos[1] = vec2<f32>( 3.0, -1.0);
    pos[2] = vec2<f32>(-1.0,  3.0);
    let p = pos[vi];
    var out: VOut;
    out.clip = vec4<f32>(p, 0.0, 1.0);
    // Same flip as quad.wgsl so orientation matches comp's offscreen output.
    out.uv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
    return out;
}

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
struct Fxaa { texsize: vec4<f32> }; // (W, H, 1/W, 1/H)
@group(0) @binding(2) var<uniform> u: Fxaa;

const REDUCE_MIN: f32 = 1.0 / 128.0;
const REDUCE_MUL: f32 = 1.0 / 8.0;
const SPAN_MAX:   f32 = 8.0;

fn t(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(tex, samp, uv, 0.0).xyz;
}

@fragment
fn fs_main(i: VOut) -> @location(0) vec4<f32> {
    let inv = u.texsize.zw; // (1/W, 1/H)
    let m = i.uv;
    let nw = m + vec2<f32>(-inv.x, -inv.y);
    let ne = m + vec2<f32>( inv.x, -inv.y);
    let sw = m + vec2<f32>(-inv.x,  inv.y);
    let se = m + vec2<f32>( inv.x,  inv.y);

    let rgbNW = t(nw);
    let rgbNE = t(ne);
    let rgbSW = t(sw);
    let rgbSE = t(se);
    let rgbM  = t(m);

    let luma = vec3<f32>(0.299, 0.587, 0.114);
    let lNW = dot(rgbNW, luma);
    let lNE = dot(rgbNE, luma);
    let lSW = dot(rgbSW, luma);
    let lSE = dot(rgbSE, luma);
    let lM  = dot(rgbM,  luma);

    let lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
    let lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));

    var dir = vec2<f32>(
        -((lNW + lNE) - (lSW + lSE)),
         ((lNW + lSW) - (lNE + lSE)),
    );
    let dirReduce = max((lNW + lNE + lSW + lSE) * (0.25 * REDUCE_MUL), REDUCE_MIN);
    let rcp = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcp, vec2<f32>(-SPAN_MAX), vec2<f32>(SPAN_MAX)) * inv;

    let rgbA = 0.5 * (
        t(m + dir * (1.0 / 3.0 - 0.5)) +
        t(m + dir * (2.0 / 3.0 - 0.5))
    );
    let rgbB = rgbA * 0.5 + 0.25 * (
        t(m + dir * -0.5) +
        t(m + dir *  0.5)
    );
    let lB = dot(rgbB, luma);

    if (lB < lMin || lB > lMax) {
        return vec4<f32>(rgbA, 1.0);
    } else {
        return vec4<f32>(rgbB, 1.0);
    }
}
