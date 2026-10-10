// SPDX-License-Identifier: BSD-3-Clause
// Temporal average: an exponential moving average of linear colour over the product's frames,
// out_k = alpha c_k + (1 - alpha) out_(k-1), the first frame as it is: noise averages down by
// about sqrt(alpha / (2 - alpha)) on a still scene, and moving objects smear, as in temporal
// denoisers, long-exposure stacking or a slow display's persistence. On linear colour, after
// `cie` (temporal: it keeps each pixel's average between frames).
//
// Reference: S. W. Roberts, "Control chart tests based on geometric moving averages",
// Technometrics 1(3):239-250, 1959 (the exponentially weighted moving average).
import sensor;

struct TemporalAverage : IPixelFilter {
    struct Params {
        float alpha;  // weight of the new frame, 0 to 1
    }

    static void apply(inout SensorPixel value, PixelContext pixel, Params params) {
        let a = saturate(params.alpha);
        let average = hasHistory(pixel) ? lerp(readState(pixel, 0).xyz, value.rgb, a) : value.rgb;
        value.rgb = average;
        writeState(pixel, 0, float4(average, 1.0));
    }
}
