// Vertex shader that drives the CUSTOM warp fragment shader through the warped mesh.
// Emits, in this exact location order to match the naga-generated FS varyings:
//   vUv     (loc 0) — screen position 0..1 (DirectX-UV, v=0 top) → rad/ang
//   vWarpUv (loc 1) — CPU-computed warped sample coord
//   vDecay  (loc 2) — per-vertex decay rgb

struct VIn {
    @location(0) pos:   vec2<f32>,
    @location(1) uv:    vec2<f32>,
    @location(2) decay: vec4<f32>,
}
struct VOut {
    @builtin(position) clip:    vec4<f32>,
    @location(0)       vUv:     vec2<f32>,
    @location(1)       vWarpUv: vec2<f32>,
    @location(2)       vDecay:  vec4<f32>,
}

@vertex
fn vs_main(v: VIn) -> VOut {
    var o: VOut;
    o.clip = vec4<f32>(v.pos, 0.0, 1.0);
    // Screen 0..1 DirectX-UV (matches quad.wgsl): top row (pos.y=+1) -> v=0.
    o.vUv     = vec2<f32>((v.pos.x + 1.0) * 0.5, (1.0 - v.pos.y) * 0.5);
    o.vWarpUv = v.uv;      // CPU-computed warped sample coord
    o.vDecay  = v.decay;
    return o;
}
