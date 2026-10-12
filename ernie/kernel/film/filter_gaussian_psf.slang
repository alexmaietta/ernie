// SPDX-License-Identifier: BSD-3-Clause
// Multi-Gaussian point spread: light scattered by the optics as a weighted sum of Gaussians, PSF(r)
// = sum_k w_k G(r; sigma_k) / sum_k w_k, the narrow terms the core and the wide ones the scatter
// halo, as measured PSFs are commonly fitted; a sigma below a quarter pixel is the unscattered
// image itself. Light is kept. Each term reads the image's pyramid at the coarsest level whose
// texels are no wider than its sigma, with a 7 x 7 Gaussian of the sigma left over once the
// level's box (variance (4^l - 1) / 12 pixels squared) is taken out, so wide halos cost no more
// than narrow ones. On electrons, after `qe` and before `noise`.
//
// References:
// - STAR-Dundee, "PANGU v7 Datasheet", 2024 (a multi-weighted Gaussian PSF for scattered light).
// - L. Williams, "Pyramidal Parametrics", SIGGRAPH 1983 (the pyramid).
import sensor;

struct GaussianPsf : ISpatialPixelFilter {
    struct Params {
        float terms;        // Gaussians used, 1 to 8
        float weights[8];   // each Gaussian's share of the light
        float sigmas[8];    // each Gaussian's sigma, pixels
    }

    static void apply(inout SensorPixel value, PixelContext pixel, SpatialImage input,
                      Params params) {
        let own = input.values(pixel, int2(pixel.id.xy));
        let size = float2(input.size(pixel));
        let uv = (float2(pixel.id.xy) + 0.5) / size;
        let top = int(input.levels()) - 1;
        let count = clamp(int(params.terms), 0, 8);
        var sum = float4(0.0);
        var total = 0.0;
        for (int k = 0; k < count; ++k) {
            let w = max(params.weights[k], 0.0);
            if (w == 0.0) {
                continue;
            }
            let sigma = params.sigmas[k];
            total += w;
            if (sigma < 0.25) {
                sum += w * own;
                continue;
            }
            let level = clamp(int(ceil(log2(sigma))), 0, top);
            let texel = exp2(float(level));
            let left = max(sigma * sigma - (texel * texel - 1.0) / 12.0, 0.0);
            let s = max(sqrt(left) / texel, 0.25);
            let reach = min(int(ceil(3.0 * s)), 3);
            var blurred = float4(0.0);
            var weights = 0.0;
            for (int j = -reach; j <= reach; ++j) {
                for (int i = -reach; i <= reach; ++i) {
                    let g = exp(-float(i * i + j * j) / (2.0 * s * s));
                    let tap = level == 0
                                  ? input.values(pixel, int2(pixel.id.xy) + int2(i, j))
                                  : input.valuesAt(pixel, uv + float2(i, j) * texel / size,
                                                   float(level));
                    blurred += g * tap;
                    weights += g;
                }
            }
            sum += w * blurred / weights;
        }
        value.electrons = total > 0.0 ? sum / total : own;
    }
}
