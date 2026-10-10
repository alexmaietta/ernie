// SPDX-License-Identifier: BSD-3-Clause
// Glare: light a lens and the eye scatter far from where it lands, a wide halo whose point-spread
// falls as a power of the distance (about r^-3 for lens scatter), around every bright source:
// out = (1 - strength) c + strength sum_k w_k blur_k / sum_k w_k, blur_k the image's pyramid level
// k (a blur about 2^k pixels wide) and w_k = 2^(-k (falloff - 2)), the share of a r^-falloff
// point-spread between radii 2^(k-1) and 2^k. Light is kept: the halo takes `strength` of it.
// On linear colour, after `cie`, reading its pyramid.
//
// References:
// - G. Spencer, P. Shirley, K. Zimmerman and D. P. Greenberg, "Physically-Based Glare Effects for
//   Digital Images", SIGGRAPH 1995, 325-334. https://doi.org/10.1145/218380.218466
// - L. Williams, "Pyramidal Parametrics", SIGGRAPH 1983 (the pyramid).
import sensor;

struct Glare : ISpatialPixelFilter {
    struct Params {
        float strength;  // fraction of the light scattered into the halo, 0 to 1
        float falloff;   // the point-spread's power of distance, above 2
        float levels;    // pyramid levels summed, the halo reaching about 2^levels pixels
    }

    static void apply(inout SensorPixel value, PixelContext pixel, SpatialImage input,
                      Params params) {
        let size = float2(input.size(pixel));
        let uv = (float2(pixel.id.xy) + 0.5) / size;
        let own = input.values(pixel, int2(pixel.id.xy)).xyz;
        let top = min(uint(max(params.levels, 1.0)), input.levels() - 1);
        var halo = float3(0.0);
        var weights = 0.0;
        for (uint k = 1; k <= top; ++k) {
            let w = exp2(-float(k) * max(params.falloff - 2.0, 0.01));
            halo += w * input.valuesAt(pixel, uv, float(k)).xyz;
            weights += w;
        }
        let s = weights > 0.0 ? saturate(params.strength) : 0.0;
        value.rgb = (1.0 - s) * own + s * halo / max(weights, 1e-12);
    }
}
