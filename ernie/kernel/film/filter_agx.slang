// SPDX-License-Identifier: BSD-3-Clause
// AgX: a filmic display transform in place of `exposure`'s clip. The exposed linear colour (the
// camera's exposure, `exposure` stops more) is encoded logarithmically over 16.5 stops about middle
// grey in a wider gamut, through a sigmoid, and back, so highlights roll off and bright saturated
// colours desaturate towards white rather than clipping to a hue. On display colour, after
// `exposure`, from the linear colour it kept.
//
// References:
// - T. Sobotka, AgX (https://github.com/sobotka/AgX), 2022, as Blender 4.0's view transform.
// - Google, Filament's AgX (ToneMapper.cpp), 2023, and three.js's AgXToneMapping (MIT), its inset
//   and outset matrices and sigmoid fit here, as three-geospatial's stories show their clouds.
import sensor;

// Linear sRGB to linear Rec. 2020 and back, and AgX's inset and outset (GLSL's columns, here rows,
// applied as mul(colour, m)).
static const float3x3 kSrgbToRec2020 = float3x3(0.6274, 0.0691, 0.0164,
                                                0.3293, 0.9195, 0.0880,
                                                0.0433, 0.0113, 0.8956);
static const float3x3 kRec2020ToSrgb = float3x3(1.6605, -0.1246, -0.0182,
                                                -0.5876, 1.1329, -0.1006,
                                                -0.0728, -0.0083, 1.1187);
static const float3x3 kInset = float3x3(0.856627153315983, 0.137318972929847, 0.11189821299995,
                                        0.0951212405381588, 0.761241990602591, 0.0767994186031903,
                                        0.0482516061458583, 0.101439036467562, 0.811302368396859);
static const float3x3 kOutset =
    float3x3(1.1271005818144368, -0.1413297634984383, -0.14132976349843826,
             -0.11060664309660323, 1.157823702216272, -0.11060664309660294,
             -0.016493938717834573, -0.016493938717834257, 1.2519364065950405);
// log2 of middle grey 10 stops below and 6.5 above.
static const float kMinEv = -12.47393;
static const float kMaxEv = 4.026069;

// AgX's sigmoid, a polynomial fit.
float3 contrast(float3 x) {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 +
           0.1191 * x - 0.00232;
}

// Exposed linear sRGB to displayed linear sRGB, 0 to 1.
float3 agx(float3 colour) {
    var c = mul(mul(colour, kSrgbToRec2020), kInset);
    c = saturate((log2(max(c, 1e-10)) - kMinEv) / (kMaxEv - kMinEv));
    c = mul(contrast(c), kOutset);
    c = pow(max(c, 0.0), 2.2);
    return saturate(mul(c, kRec2020ToSrgb));
}

float encode(float linear) {
    return linear <= 0.0031308 ? 12.92 * linear : 1.055 * pow(linear, 1.0 / 2.4) - 0.055;
}

struct AgX : IPixelFilter {
    struct Params {
        float exposure;
    }

    static void apply(inout SensorPixel value, PixelContext pixel, Params params) {
        let cam = pixel.uniforms();
        let shown = agx(value.rgb * cam.lens.z * exp2(params.exposure));
        // As `exposure`: the clear colour where nothing covers the pixel, unless it shows the sky.
        let linear = cam.clear.w > 0.5 && cam.clear.w < 1.5
                         ? shown
                         : lerp(cam.clear.rgb, shown, value.coverage);
        value.display = float4(encode(linear.r), encode(linear.g), encode(linear.b), 1.0);
    }
}
