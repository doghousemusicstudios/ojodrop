struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) vUv: vec2<f32>,
    @location(1) vColor: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
    var pos: array<vec2<f32>, 3>;
    pos[0] = vec2<f32>(-1.0, -1.0);
    pos[1] = vec2<f32>( 3.0, -1.0);
    pos[2] = vec2<f32>(-1.0,  3.0);
    let p = pos[vi];
    var out: VOut;
    out.clip = vec4<f32>(p, 0.0, 1.0);
    // UV (0,0) = top-left, (1,1) = bottom-right (matches DirectX/MilkDrop convention)
    out.vUv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
    out.vColor = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    return out;
}
