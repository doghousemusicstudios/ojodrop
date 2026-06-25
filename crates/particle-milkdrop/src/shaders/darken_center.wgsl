// Darken-center (butterchurn DarkenCenter.drawDarkenCenter parity).
// A small soft dark blob: a triangle-fan (here expanded to a triangle list) with
// the center vertex black at alpha 3/32 and the perimeter verts at alpha 0, so it
// multiplicatively darkens the image center via SRC_ALPHA/ONE_MINUS_SRC_ALPHA.
// Vertices arrive already in NDC with per-vertex color.

struct VIn  { @location(0) pos: vec2<f32>, @location(1) color: vec4<f32> };
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) color: vec4<f32> };

@vertex
fn vs_main(v: VIn) -> VOut {
    var o: VOut;
    o.clip  = vec4<f32>(v.pos, 0.0, 1.0);
    o.color = v.color;
    return o;
}

@fragment
fn fs_main(i: VOut) -> @location(0) vec4<f32> {
    return i.color;
}
