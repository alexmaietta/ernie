// SPDX-License-Identifier: BSD-3-Clause
// Readout table: a line scanner's measured readout per detector (column), as a calibration pipeline
// such as ISIS's lronaccal inverts it. Each detector's linear number x is scaled by its flatfield
// gain, then reads y with y - 1 / (a b^y + c) = x where y is below `limit` (a low-signal
// nonlinearity, solved by bisection) and y = x above, and is sent as y - offset + dark. On digital
// numbers, after `adc`; the result stays within the ADC's range.
//
// References:
// - D. C. Humm et al., "Flight Calibration of the LROC Narrow Angle Camera", Space Science Reviews
//   200:431-473, 2016 (the nonlinearity, offsets, darks and flatfield).
// - USGS Astrogeology, ISIS's lronaccal (`CorrectNonlinearity`, `CorrectFlatfield`).
import sensor;

struct ReadoutTable : IPixelFilter {
    struct Params {
        float gain;    // table: each detector's flatfield gain; 0 for no data (left as is)
        float dark;    // table: each detector's dark number, DN
        float offset;  // table: each detector's linearization offset, DN
        float a;       // table: each detector's nonlinearity coefficients
        float b;
        float c;
        float limit;   // DN below which the nonlinearity applies; 0 for none
    }

    static void apply(inout SensorPixel value, PixelContext pixel, Params params) {
        let detector = pixel.id.x;
        let gain = paramTable(params.gain, detector);
        let dark = paramTable(params.dark, detector);
        let offset = paramTable(params.offset, detector);
        let a = paramTable(params.a, detector);
        let b = paramTable(params.b, detector);
        let c = paramTable(params.c, detector);
        let largest = pixel.uniforms().adc.w;
        let channels = pixel.uniforms().size.w;
        var dn = float4(value.digital);
        for (uint k = 0; k < channels; ++k) {
            let x = gain > 0.0 ? dn[k] * gain : dn[k];
            var y = x;
            if (params.limit > 0.0) {
                // y - h(y) = x, h(y) = 1 / (a b^y + c) falling in y, so y lies in [x, x + h(x)].
                let h = 1.0 / max(a * pow(b, x) + c, 1e-6);
                var low = x;
                var high = x + h;
                for (int i = 0; i < 32; ++i) {
                    let mid = 0.5 * (low + high);
                    let f = mid - 1.0 / max(a * pow(b, mid) + c, 1e-6) - x;
                    if (f < 0.0) {
                        low = mid;
                    } else {
                        high = mid;
                    }
                }
                let solved = 0.5 * (low + high);
                y = solved < params.limit ? solved : x;
            }
            dn[k] = y - offset + dark;
        }
        value.digital = uint4(clamp(floor(dn + 0.5), float4(0.0), float4(largest)));
    }
}
