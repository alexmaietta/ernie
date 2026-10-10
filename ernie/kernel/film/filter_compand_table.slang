// SPDX-License-Identifier: BSD-3-Clause
// Piecewise-linear companding: digital numbers compressed through up to five linear segments, as
// LROC's cameras encode their 12-bit samples into 8 bits. Segment i covers inputs from
// `starts[i]` up to the next segment's start, and maps an input x to floor(slopes[i] x +
// offsets[i]) (the encoders truncate). The defaults are the NAC's nominal table, code 0:
//   0-31 -> x / 2, 32-135 -> x / 4 + 8, 136-542 -> x / 8 + 25, 543-2206 -> x / 16 + 59,
//   2207-4095 -> x / 32 + 128.
// On digital numbers, after `adc`.
//
// Reference: M. S. Robinson et al., "Lunar Reconnaissance Orbiter Camera (LROC) Instrument
// Overview", Space Science Reviews 150:81-124, 2010, Appendix A (NAC bit-compression tables).
import sensor;

struct CompandTable : IPixelFilter {
    struct Params {
        float segments;    // segments used, 1 to 5
        float starts[5];   // each segment's first input DN, increasing
        float slopes[5];   // each segment's output DN per input DN
        float offsets[5];  // each segment's output DN at input 0
    }

    static void apply(inout SensorPixel value, PixelContext pixel, Params params) {
        let count = uint(clamp(params.segments, 1.0, 5.0));
        var result = uint4(0);
        for (uint c = 0; c < 4; ++c) {
            let x = float(value.digital[c]);
            uint segment = 0;
            for (uint i = 1; i < count; ++i) {
                if (x >= params.starts[i]) {
                    segment = i;
                }
            }
            result[c] = uint(max(floor(params.slopes[segment] * x + params.offsets[segment]),
                                 0.0));
        }
        value.digital = result;
    }
}
