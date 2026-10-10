// SPDX-License-Identifier: BSD-3-Clause
// Detector pattern: a fixed gain and offset per detector of a line scanner, the same down its
// whole column (or row, with `axis` 1), so a pushbroom image shows the along-track stripes of its
// detectors' differing responses and dark levels; the per-pixel PRNU and DSNU of `noise` stand for
// what varies within a column. On electrons, after `qe` and before `noise` (the gain scales the
// signal that shot noise then follows).
//
// References:
// - H. Carfantan and J. Idier, "Statistical Linear Destriping of Satellite-Based Pushbroom-Type
//   Images", IEEE Transactions on Geoscience and Remote Sensing 48(4), 2010.
// - EMVA Standard 1288 Release 4.0 Linear, 2021 (PRNU and DSNU, and their row and column parts).
import sensor;

struct DetectorPattern : IPixelFilter {
    struct Params {
        float gain;    // standard deviation of each detector's relative gain
        float offset;  // standard deviation of each detector's offset, electrons
        float axis;    // 0: a detector per column (lines are rows), 1: a detector per row
    }

    static void apply(inout SensorPixel value, PixelContext pixel, Params params) {
        let detector = params.axis < 0.5 ? pixel.id.x : pixel.id.y;
        let channels = pixel.uniforms().size.w;
        for (uint c = 0; c < channels; ++c) {
            let gain = 1.0 + params.gain * pixel.fixedNormal(uint2(detector, c), 0x51u);
            let offset = params.offset * pixel.fixedNormal(uint2(detector, c), 0x52u);
            value.electrons[c] = max(value.electrons[c] * gain + offset, 0.0);
        }
    }
}
