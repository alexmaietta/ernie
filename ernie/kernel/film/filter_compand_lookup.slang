// SPDX-License-Identifier: BSD-3-Clause
// Lookup-table companding: digital numbers encoded to the output codes of a stored table, as
// LROC's Wide Angle Camera encodes its 11-bit samples into 8 bits. The table is each output
// code's last input DN (`uppers`, non-decreasing over the first `codes`): an input x encodes to
// the first code whose bound reaches it, the last code above the table. A code no input reaches
// (the WAC's 3 and 6) repeats the bound of the code before it. The default table is the
// identity; demos/cameras/lroc sets the WAC's. On digital numbers, after `adc`.
//
// Reference: LROC EDR/CDR Data Product Software Interface Specification, version 1.18, 2010
// (LROCSIS.PDF in the LROC PDS volumes' DOCUMENT directories), section 3.3
// (LRO:LOOKUP_CONVERSION_TABLE) and Appendix B (the WAC square-root companding table, 11-bit to
// 8-bit).
import sensor;

struct CompandLookup : IPixelFilter {
    struct Params {
        float codes;        // output codes used, 1 to 256
        float uppers[256];  // each code's last input DN, non-decreasing
    }

    static void apply(inout SensorPixel value, PixelContext pixel, Params params) {
        let count = uint(clamp(params.codes, 1.0, 256.0));
        var result = uint4(0);
        for (uint c = 0; c < 4; ++c) {
            // The first code whose bound is not below x, by binary search.
            let x = float(value.digital[c]);
            uint low = 0;
            uint high = count;
            while (low < high) {
                let middle = (low + high) / 2;
                if (params.uppers[middle] < x) {
                    low = middle + 1;
                } else {
                    high = middle;
                }
            }
            result[c] = min(low, count - 1);
        }
        value.digital = result;
    }
}
