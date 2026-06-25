fn main() {
    // Test xtramartin (1) comp shader: ALL before as file_globals, real inner
    let before_raw = "#define sat saturate
static const float2 pix = texsize.zw;
float glow, tmp, lamp, dist, bdist, b2dist, dist_c;
float2 dz, uv1, uv3;
static const float3 CamPos = float3(q4,q5,q6);
static const float myzoom = q7;
static const float3x3 RotMat = float3x3(q20,q21,q22,q23,q24,q25,q26,q27,q28);
static const float3 col_struc = float3(1,.7,.3) + .2*(rand_preset.xyz-.5);
static float2 center = float2 (q1,q2);

float3 GetBlurX (float2 uvi, float x) {return lerp (GetPixel(uvi), GetBlur1(uvi), x);}
float GetDist(float2 uvi)   {return 1-GetPixel(uvi).b;}
float GetDistB(float2 uvi)  {return 1-GetBlur1(uvi).b;}
float GetDistB2(float2 uvi) {return 1-GetBlur2(uvi).b;}

float MinDistB (float2 uvi) {float tmp2; float4 nb;
  tmp2 = GetDist(uvi);
  tmp2 = min(tmp2,GetDistB2(uvi)*1) ;
  return tmp2;}
";
    // Strip sampler lines if any, then pass rest as file_globals
    let inner = r#"float2 uvo = 0.5 + (uv-0.5)*float2(1.1,0.81);
float2 factorA = uv-float2(1-0.5,0.5);
float2 factorB = float2(0,-1024);
float2 product = float2( factorA.x*factorB.x - factorA.y*factorB.y, factorA.x*factorB.y + factorA.y*factorB.x);
uv = product.yx*float2(-1,1)*100;
uv = lerp(0.5 + (uvo-0.5)*2,uv+0.5,0.5);
uv1 = (uv-center)*aspect.xy;
dist = MinDistB(uv);
float3 uv2 = mul(float3((uv-.5)*MinDistB(uv),MinDistB(uv))/myzoom,RotMat)+CamPos;
float focus = sat(abs(GetDistB2(uv)-dist_c)*1+.2);
float struc2 = GetBlurX(uv,focus).r;
ret = float3(struc2, focus, dist);"#;
    
    println!("All-before _ex (comp shader):");
    match particle_milkdrop_converter_sys::convert_milk_shader_ex(before_raw, inner, true) {
        Ok(g) => println!("OK ({} chars):\n{}", g.len(), &g[..g.len().min(300)]),
        Err(e) => println!("FAIL: {}", e.lines().take(5).collect::<Vec<_>>().join("\n")),
    }
}
