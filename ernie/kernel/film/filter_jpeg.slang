// JPEG compression: the display image through baseline JPEG and back, as a camera that stores
// JPEGs would deliver it: 8-bit sRGB to YCbCr (JFIF), level-shifted 8 x 8 blocks through the
// discrete cosine transform, each coefficient quantised by the standard tables (ITU T.81 Annex K)
// scaled to `quality` as the IJG library scales them, and back; chroma is kept at full resolution
// (4:4:4). Blocking and ringing grow as `quality` falls. On display colour, after `exposure`.
// Each pixel transforms its own block and reconstructs itself.
//
// References:
// - ITU-T Recommendation T.81, "Information technology - Digital compression and coding of
//   continuous-tone still images", 1992 (the codec and its example quantisation tables).
// - E. Hamilton, "JPEG File Interchange Format", version 1.02, 1992 (YCbCr from RGB).
// - Independent JPEG Group, libjpeg's `jpeg_quality_scaling`.
import sensor;

static const float kLuma[64] = {
    16, 11, 10, 16, 24,  40,  51,  61,  12, 12, 14, 19, 26,  58,  60,  55,
    14, 13, 16, 24, 40,  57,  69,  56,  14, 17, 22, 29, 51,  87,  80,  62,
    18, 22, 37, 56, 68,  109, 103, 77,  24, 35, 55, 64, 81,  104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99};
static const float kChroma[64] = {
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99};

// A quantiser of the standard table entry `base` at IJG quality `quality`, 1 to 100.
float quantiser(float base, float quality) {
    let q = clamp(quality, 1.0, 100.0);
    let scale = q < 50.0 ? 5000.0 / q : 200.0 - 2.0 * q;
    return clamp(floor((base * scale + 50.0) / 100.0), 1.0, 255.0);
}

float3 toYCbCr(float3 rgb) {
    return float3(0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b,
                  -0.168736 * rgb.r - 0.331264 * rgb.g + 0.5 * rgb.b + 128.0,
                  0.5 * rgb.r - 0.418688 * rgb.g - 0.081312 * rgb.b + 128.0);
}

float3 toRgb(float3 ycc) {
    let cb = ycc.y - 128.0;
    let cr = ycc.z - 128.0;
    return float3(ycc.x + 1.402 * cr, ycc.x - 0.344136 * cb - 0.714136 * cr, ycc.x + 1.772 * cb);
}

struct Jpeg : ISpatialPixelFilter {
    struct Params {
        float quality;  // 1 to 100, as encoders take it
    }

    static void apply(inout SensorPixel value, PixelContext pixel, SpatialImage input,
                      Params params) {
        let size = input.size(pixel);
        let at = int2(pixel.id.xy);
        let origin = (at / 8) * 8;
        // cos((2x + 1) u pi / 16), with C(0) = 1 / sqrt(2) folded in.
        float basis[8][8];
        for (int u = 0; u < 8; ++u) {
            for (int x = 0; x < 8; ++x) {
                basis[u][x] = cos(float((2 * x + 1) * u) * 3.14159265 / 16.0) *
                              (u == 0 ? 0.70710678 : 1.0);
            }
        }
        // The block, level-shifted, edge pixels repeated past the image as encoders pad.
        float3 block[64];
        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                let p = clamp(origin + int2(x, y), int2(0), size - 1);
                let rgb = floor(saturate(input.values(pixel, p).rgb) * 255.0 + 0.5);
                block[y * 8 + x] = toYCbCr(rgb) - 128.0;
            }
        }
        let local = at - origin;
        var result = float3(0.0);
        for (int v = 0; v < 8; ++v) {
            for (int u = 0; u < 8; ++u) {
                var coefficient = float3(0.0);
                for (int y = 0; y < 8; ++y) {
                    for (int x = 0; x < 8; ++x) {
                        coefficient += block[y * 8 + x] * (basis[u][x] * basis[v][y]);
                    }
                }
                coefficient *= 0.25;
                let index = v * 8 + u;
                let step = float3(quantiser(kLuma[index], params.quality),
                                  quantiser(kChroma[index], params.quality),
                                  quantiser(kChroma[index], params.quality));
                coefficient = round(coefficient / step) * step;
                result += coefficient * (basis[u][local.x] * basis[v][local.y]);
            }
        }
        let rgb = clamp(toRgb(0.25 * result + 128.0), float3(0.0), float3(255.0));
        value.display = float4(floor(rgb + 0.5) / 255.0, input.values(pixel, at).a);
    }
}
