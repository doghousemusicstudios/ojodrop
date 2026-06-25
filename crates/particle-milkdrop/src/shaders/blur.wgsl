// Separable wide-Gaussian blur — Butterchurn parity (BlurHorizontal/BlurVertical).
// Kernel weights w = [4.0, 3.8, 3.5, 2.9, 1.9, 1.2, 0.7, 0.3], collapsed into
// bilinear 2-tap groups so 16 logical samples cost 8 texture fetches per pass.
// Each blur level runs fs_blur_h (src -> temp) then fs_blur_v (temp -> level),
// reaching ~±6.6 texels — vastly wider than the old 9-tap box, which is what
// turns the warp/comp feedback into smooth flowing ridges instead of granular noise.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;
@group(0) @binding(2) var<uniform> bu: BlurU;

struct BlurU {
    texel: vec4<f32>, // x = 1/levelW, y = 1/levelH
    edge:  vec4<f32>, // x = ed1 (1-b1ed), y = ed2 (b1ed), z = ed3 (=5.0)
    sb:    vec4<f32>, // x = scale, y = bias (per-level blur range remap → [0,1])
};

// Horizontal: 4 bilinear taps. wH=(w0+w1, w2+w3, w4+w5, w6+w7); dH = bilinear offsets.
const W_H: vec4<f32> = vec4<f32>(7.8, 6.4, 3.1, 1.0);
const D_H: vec4<f32> = vec4<f32>(0.97435897, 2.90625, 4.77419355, 6.6);
const WDIV_H: f32 = 0.02732240437; // 0.5 / (7.8+6.4+3.1+1.0)

// Vertical: 2 bilinear taps. wV=(w0+w1+w2+w3, w4+w5+w6+w7); dV = bilinear offsets.
const W_V: vec2<f32> = vec2<f32>(14.2, 4.1);
const D_V: vec2<f32> = vec2<f32>(0.90140845, 2.48780488);
const WDIV_V: f32 = 0.02732240437; // 1.0 / ((14.2+4.1) * 2)

@fragment
fn fs_blur_h(@location(0) vUv: vec2<f32>) -> @location(0) vec4<f32> {
    let tx = bu.texel.x;
    var b = vec3<f32>(0.0);
    b += (textureSample(src, src_samp, vUv + vec2<f32>( D_H.x * tx, 0.0)).xyz
        + textureSample(src, src_samp, vUv + vec2<f32>(-D_H.x * tx, 0.0)).xyz) * W_H.x;
    b += (textureSample(src, src_samp, vUv + vec2<f32>( D_H.y * tx, 0.0)).xyz
        + textureSample(src, src_samp, vUv + vec2<f32>(-D_H.y * tx, 0.0)).xyz) * W_H.y;
    b += (textureSample(src, src_samp, vUv + vec2<f32>( D_H.z * tx, 0.0)).xyz
        + textureSample(src, src_samp, vUv + vec2<f32>(-D_H.z * tx, 0.0)).xyz) * W_H.z;
    b += (textureSample(src, src_samp, vUv + vec2<f32>( D_H.w * tx, 0.0)).xyz
        + textureSample(src, src_samp, vUv + vec2<f32>(-D_H.w * tx, 0.0)).xyz) * W_H.w;
    b *= WDIV_H;
    // Butterchurn applies the per-level range remap in the HORIZONTAL pass:
    // blur = blur*scale + bias (scale = 1/(max-min), bias = -min*scale → [0,1]).
    // The comp/warp GetBlurN helpers apply the inverse to recover the range.
    b = b * bu.sb.x + bu.sb.y;
    return vec4<f32>(b, 1.0);
}

@fragment
fn fs_blur_v(@location(0) vUv: vec2<f32>) -> @location(0) vec4<f32> {
    let ty = bu.texel.y;
    var b = vec3<f32>(0.0);
    b += (textureSample(src, src_samp, vUv + vec2<f32>(0.0,  D_V.x * ty)).xyz
        + textureSample(src, src_samp, vUv + vec2<f32>(0.0, -D_V.x * ty)).xyz) * W_V.x;
    b += (textureSample(src, src_samp, vUv + vec2<f32>(0.0,  D_V.y * ty)).xyz
        + textureSample(src, src_samp, vUv + vec2<f32>(0.0, -D_V.y * ty)).xyz) * W_V.y;
    b *= WDIV_V;
    // Edge decay (Butterchurn BlurVertical): fade the blur toward the borders.
    var t = min(min(vUv.x, vUv.y), 1.0 - max(vUv.x, vUv.y));
    t = sqrt(max(t, 0.0));
    t = bu.edge.x + bu.edge.y * clamp(t * bu.edge.z, 0.0, 1.0);
    b *= t;
    return vec4<f32>(b, 1.0);
}
