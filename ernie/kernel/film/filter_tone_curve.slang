// SPDX-License-Identifier: BSD-3-Clause
// Tone curve: each display channel (sRGB-encoded, 0 to 1) through `curve`, 33 samples from
// input 0 to 1 (identity by default), as a camera's picture style shapes contrast after its
// encoding; a SensorFilter prim sets it as a float array of any length, resampled. On display
// colour, after `exposure`.
//
// Reference: M. S. Brown, "Understanding the In-Camera Image Processing Pipeline for Computer
// Vision", CVPR 2016 tutorial (the tone curve).
import sensor;

struct ToneCurve : IPixelFilter {
    struct Params {
        float curve[33];
    }

    static void apply(inout SensorPixel value, PixelContext pixel, Params params) {
        value.display = float4(curveAt(params.curve, value.display.r),
                               curveAt(params.curve, value.display.g),
                               curveAt(params.curve, value.display.b), value.display.a);
    }
}
