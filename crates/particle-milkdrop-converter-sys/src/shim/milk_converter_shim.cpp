// C-ABI wrapper around hlsl2glslfork + glsl-optimizer for MilkDrop HLSL conversion.
// This is the logic from milkdrop-shader-converter/src/main.cpp minus nan/V8.
//
// Thread safety: a mutex serialises the Hlsl2Glsl_Initialize / compile / shutdown
// sequence (the library keeps global state).  glslopt_ctx is per-call and needs no
// external lock.
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

#include "hlsl2glsl.h"
#include "glsl_optimizer.h"

// MilkDrop HLSL preamble — all uniforms, helper macros, and the outer
// fragment-shader wrapper that prepareShader() (milkdrop-preset-utils) builds.
// The user's shader body is spliced at the [BODY] marker below.
static const char MILK_HLSL_PREFIX[] =
    "#define  M_PI   3.14159265359\n"
    "   #define  M_PI_2 6.28318530718\n"
    "   #define  M_INV_PI_2  0.159154943091895\n"
    "\n"
    "   uniform sampler2D sampler_main;\n"
    "   uniform sampler2D sampler_fw_main;\n"
    "   uniform sampler2D sampler_pw_main;\n"
    "   uniform sampler2D sampler_fc_main;\n"
    "   uniform sampler2D sampler_pc_main;\n"
    "\n"
    "   uniform sampler2D sampler_noise_lq;\n"
    "   uniform sampler2D sampler_noise_lq_lite;\n"
    "   uniform sampler2D sampler_noise_mq;\n"
    "   uniform sampler2D sampler_noise_hq;\n"
    "   uniform sampler3D sampler_noisevol_lq;\n"
    "   uniform sampler3D sampler_noisevol_hq;\n"
    "\n"
    "   uniform sampler2D sampler_pw_noise_lq;\n"
    "\n"
    "   uniform sampler2D sampler_blur1;\n"
    "   uniform sampler2D sampler_blur2;\n"
    "   uniform sampler2D sampler_blur3;\n"
    "\n"
    "   float4 texsize_noise_lq;\n"
    "   float4 texsize_noise_mq;\n"
    "   float4 texsize_noise_hq;\n"
    "   float4 texsize_noise_lq_lite;\n"
    "   float4 texsize_noisevol_lq;\n"
    "   float4 texsize_noisevol_hq;\n"
    "\n"
    "   float4 _qa;\n"
    "   float4 _qb;\n"
    "   float4 _qc;\n"
    "   float4 _qd;\n"
    "   float4 _qe;\n"
    "   float4 _qf;\n"
    "   float4 _qg;\n"
    "   float4 _qh;\n"
    "\n"
    "   float q1;\n"
    "   float q2;\n"
    "   float q3;\n"
    "   float q4;\n"
    "   float q5;\n"
    "   float q6;\n"
    "   float q7;\n"
    "   float q8;\n"
    "   float q9;\n"
    "   float q10;\n"
    "   float q11;\n"
    "   float q12;\n"
    "   float q13;\n"
    "   float q14;\n"
    "   float q15;\n"
    "   float q16;\n"
    "   float q17;\n"
    "   float q18;\n"
    "   float q19;\n"
    "   float q20;\n"
    "   float q21;\n"
    "   float q22;\n"
    "   float q23;\n"
    "   float q24;\n"
    "   float q25;\n"
    "   float q26;\n"
    "   float q27;\n"
    "   float q28;\n"
    "   float q29;\n"
    "   float q30;\n"
    "   float q31;\n"
    "   float q32;\n"
    "\n"
    "   float blur1_min;\n"
    "   float blur1_max;\n"
    "   float blur2_min;\n"
    "   float blur2_max;\n"
    "   float blur3_min;\n"
    "   float blur3_max;\n"
    "\n"
    "   float scale1;\n"
    "   float scale2;\n"
    "   float scale3;\n"
    "   float bias1;\n"
    "   float bias2;\n"
    "   float bias3;\n"
    "\n"
    "   float4 slow_roam_cos;\n"
    "   float4 roam_cos;\n"
    "   float4 slow_roam_sin;\n"
    "   float4 roam_sin;\n"
    "\n"
    "   float3 hue_shader;\n"
    "\n"
    "   float time;\n"
    "   float4 rand_preset;\n"
    "   float4 rand_frame;\n"
    "   float  progress;\n"
    "   float  frame;\n"
    "   float  fps;\n"
    "   float  decay;\n"
    "   float  bass;\n"
    "   float  mid;\n"
    "   float  treb;\n"
    "   float  vol;\n"
    "   float  bass_att;\n"
    "   float  mid_att;\n"
    "   float  treb_att;\n"
    "   float  vol_att;\n"
    "   float4 texsize;\n"
    "   float4 aspect;\n"
    "\n"
    "   float rad;\n"
    "   float ang;\n"
    "   float2 uv_orig;\n"
    "\n"
    "   #define GetMain(uv) (tex2D(sampler_main,uv).xyz)\n"
    "   #define GetPixel(uv) (tex2D(sampler_main,uv).xyz)\n"
    "   #define GetBlur1(uv) (tex2D(sampler_blur1,uv).xyz*scale1 + bias1)\n"
    "   #define GetBlur2(uv) (tex2D(sampler_blur2,uv).xyz*scale2 + bias2)\n"
    "   #define GetBlur3(uv) (tex2D(sampler_blur3,uv).xyz*scale3 + bias3)\n"
    "\n"
    "   #define lum(x) (dot(x,float3(0.32,0.49,0.29)))\n"
    "   #define tex2d tex2D\n"
    "   #define tex3d tex3D\n"
    "\n";

// The outer function wrapper that surrounds the user body.
static const char MILK_HLSL_WRAPPER_BEGIN[] =
    "   float4 shader_body (float2 uv : TEXCOORD0) : COLOR0\n"
    "   {\n"
    "       float3 ret;\n\n";

static const char MILK_HLSL_WRAPPER_END[] =
    "\n       return float4(ret, 1.0);\n"
    "   }\n";

// ES 300 compat preamble that ConvertString() prepends (from main.cpp).
static const char ES300_PREAMBLE[] =
    "#version 300 es\n"
    "#define lowp\n"
    "#define mediump\n"
    "#define highp\n"
    "#define gl_Vertex _glesVertex\n"
    "#define gl_Normal _glesNormal\n"
    "#define gl_Color _glesColor\n"
    "#define gl_MultiTexCoord0 _glesMultiTexCoord0\n"
    "#define gl_MultiTexCoord1 _glesMultiTexCoord1\n"
    "#define gl_MultiTexCoord2 _glesMultiTexCoord2\n"
    "#define gl_MultiTexCoord3 _glesMultiTexCoord3\n"
    "in highp vec4 _glesVertex;\n"
    "in highp vec3 _glesNormal;\n"
    "in lowp vec4 _glesColor;\n"
    "in highp vec4 _glesMultiTexCoord0;\n"
    "in highp vec4 _glesMultiTexCoord1;\n"
    "in highp vec4 _glesMultiTexCoord2;\n"
    "in highp vec4 _glesMultiTexCoord3;\n"
    "#define gl_FragData _glesFragData\n"
    "out lowp vec4 _glesFragData[4];\n";

static std::mutex s_hlsl_mutex;

// Run hlsl2glslfork on a full HLSL program string.
// Returns the GLSL ES 300 body (with preamble prepended) or throws on failure.
static std::string hlsl_to_glsl_es300(const std::string& hlsl_program)
{
    std::lock_guard<std::mutex> guard(s_hlsl_mutex);

    Hlsl2Glsl_Initialize();

    ShHandle parser = Hlsl2Glsl_ConstructCompiler(EShLangFragment);
    std::string result;

    try {
        int parse_ok = Hlsl2Glsl_Parse(parser, hlsl_program.c_str(),
                                        ETargetGLSL_ES_300, nullptr,
                                        ETranslateOpNone);
        if (!parse_ok) {
            const char* log = Hlsl2Glsl_GetInfoLog(parser);
            throw std::runtime_error(std::string("HLSL parse error: ") + (log ? log : "(null)"));
        }

        static const EAttribSemantic kSem[] = { EAttrSemTangent };
        static const char* kSemStr[] = { "TANGENT" };
        Hlsl2Glsl_SetUserAttributeNames(parser, kSem, kSemStr, 1);

        int translate_ok = Hlsl2Glsl_Translate(parser, "shader_body",
                                                ETargetGLSL_ES_300,
                                                ETranslateOpNone);
        if (!translate_ok) {
            const char* log = Hlsl2Glsl_GetInfoLog(parser);
            throw std::runtime_error(std::string("HLSL translate error: ") + (log ? log : "(null)"));
        }

        std::string glsl = Hlsl2Glsl_GetShader(parser);

        // Check for non-ASCII (same guard as original main.cpp).
        for (char c : glsl) {
            if (!isascii(c))
                throw std::runtime_error("HLSL output contains non-ASCII character");
        }

        result = std::string(ES300_PREAMBLE) + glsl;
    } catch (...) {
        Hlsl2Glsl_DestructCompiler(parser);
        Hlsl2Glsl_Shutdown();
        throw;
    }

    Hlsl2Glsl_DestructCompiler(parser);
    Hlsl2Glsl_Shutdown();
    return result;
}

static int run_full_hlsl(const std::string& full_hlsl, int optimize, char** out_glsl)
{
    try {
        std::string glsl_es300 = hlsl_to_glsl_es300(full_hlsl);

        if (optimize) {
            glslopt_ctx* ctx = glslopt_initialize(kGlslTargetOpenGLES30);
            glslopt_shader* shader = glslopt_optimize(
                ctx, kGlslOptShaderFragment, glsl_es300.c_str(), 0);

            if (glslopt_get_status(shader)) {
                const char* out = glslopt_get_output(shader);
                *out_glsl = strdup(out ? out : "");
                glslopt_shader_delete(shader);
                glslopt_cleanup(ctx);
                return 0;
            } else {
                const char* log = glslopt_get_log(shader);
                std::string err = std::string("glslopt error: ") + (log ? log : "(null)");
                *out_glsl = strdup(err.c_str());
                glslopt_shader_delete(shader);
                glslopt_cleanup(ctx);
                return 2;
            }
        } else {
            *out_glsl = strdup(glsl_es300.c_str());
            return 0;
        }
    } catch (const std::exception& e) {
        *out_glsl = strdup(e.what());
        return 3;
    } catch (...) {
        *out_glsl = strdup("unknown error in milk_convert_shader");
        return 4;
    }
}

extern "C" {

// Convert a raw MilkDrop HLSL shader body (without the shader_body{} wrapper)
// to optimized GLSL ES 3.00.
//
// Returns 0 on success: *out_glsl is a malloc'd C string; free with milk_convert_free.
// Returns nonzero on failure: *out_glsl is a malloc'd error string.
int milk_convert_shader(const char* hlsl_body, int optimize, char** out_glsl)
{
    if (!hlsl_body || !out_glsl) return 1;
    std::string full_hlsl =
        std::string(MILK_HLSL_PREFIX) +
        std::string(MILK_HLSL_WRAPPER_BEGIN) +
        std::string(hlsl_body) +
        std::string(MILK_HLSL_WRAPPER_END);
    return run_full_hlsl(full_hlsl, optimize, out_glsl);
}

// Like milk_convert_shader but accepts pre-shader_body file-scope globals
// (variable declarations, static initialisers, etc.) separately.  They are
// inserted between the MilkDrop HLSL prefix and the shader_body() wrapper so
// the HLSL compiler sees them at file scope — matching MilkDrop's original
// layout and avoiding redeclaration errors when the body re-uses a name.
// file_globals may be NULL or empty.
int milk_convert_shader_ex(const char* file_globals, const char* hlsl_body,
                            int optimize, char** out_glsl)
{
    if (!hlsl_body || !out_glsl) return 1;
    std::string full_hlsl =
        std::string(MILK_HLSL_PREFIX) +
        (file_globals ? std::string(file_globals) + "\n" : std::string()) +
        std::string(MILK_HLSL_WRAPPER_BEGIN) +
        std::string(hlsl_body) +
        std::string(MILK_HLSL_WRAPPER_END);
    return run_full_hlsl(full_hlsl, optimize, out_glsl);
}

void milk_convert_free(char* p)
{
    free(p);
}

} // extern "C"
