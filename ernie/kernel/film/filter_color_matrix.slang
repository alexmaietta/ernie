// SPDX-License-Identifier: BSD-3-Clause
// Colour correction matrix: linear colour times a 3 x 3 matrix, row by row (m00 m01 m02 make the
// red output from the red, green and blue inputs), as a camera's pipeline maps its channels'
// colour space to the output's; the identity by default. On linear colour, after `cie` and
// before `exposure`.
//
// Reference: M. S. Brown, "Understanding the In-Camera Image Processing Pipeline for Computer
// Vision", CVPR 2016 tutorial (white balance, the colour correction matrix and tone curve).
import sensor;

struct ColorMatrix : IPixelFilter {
    struct Params {
        float m00;
        float m01;
        float m02;
        float m10;
        float m11;
        float m12;
        float m20;
        float m21;
        float m22;
    }

    static void apply(inout SensorPixel value, PixelContext pixel, Params params) {
        let c = value.rgb;
        value.rgb = float3(params.m00 * c.r + params.m01 * c.g + params.m02 * c.b,
                           params.m10 * c.r + params.m11 * c.g + params.m12 * c.b,
                           params.m20 * c.r + params.m21 * c.g + params.m22 * c.b);
    }
}
