// SPDX-License-Identifier: BSD-3-Clause
// Image lag: a fraction `fraction` of the charge each frame collects is left in the pixel and
// read out with the next frame, as incomplete charge transfer leaves it in CMOS pinned photodiodes
// and some CCDs, so moving bright objects trail a ghost of where they were: frame k reads
// (1 - fraction) e_k + fraction e_(k-1). The first frame reads (1 - fraction) e_0. On electrons,
// after `qe` (temporal: it keeps each pixel's last charge between frames).
//
// Reference: J. R. Janesick, "Photon Transfer", SPIE Press, 2007 (image lag).
import sensor;

struct Lag : IPixelFilter {
    struct Params {
        float fraction;  // of each frame's charge read out with the next, 0 to 1
    }

    static void apply(inout SensorPixel value, PixelContext pixel, Params params) {
        let f = saturate(params.fraction);
        let collected = value.electrons;
        value.electrons = (1.0 - f) * collected + f * readState(pixel, 0);
        writeState(pixel, 0, collected);
    }
}
