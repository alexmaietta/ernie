// SPDX-License-Identifier: BSD-3-Clause
// Bayer colour filter array: each pixel of a three-channel (RGB) sensor keeps the electrons of the
// one channel its filter passes, in a 2 x 2 tile of `pattern` 0 RGGB, 1 BGGR, 2 GRBG or 3 GBRG
// (the top-left pixel first, row by row), so the raw image is a mosaic, as real colour sensors
// record it; `demosaic` interpolates it back. Between `noise` and `adc` the other channels hold no
// electrons at all, as on a real mosaic; before `noise` they gain its read noise. The photographic
// chain, from the spectrum through `cie`, is not mosaicked. Single-channel sensors are untouched.
//
// Reference: B. E. Bayer, "Color imaging array", US patent 3,971,065, 1976.
import sensor;

// The channel (0 red, 1 green, 2 blue) of the filter over pixel `at` in `pattern`.
uint bayerChannel(int2 at, uint pattern) {
    let x = uint(at.x) & 1u;
    let y = uint(at.y) & 1u;
    // RGGB, BGGR, GRBG and GBRG, each as its tile's (0,0), (1,0), (0,1), (1,1) channels.
    let tiles = uint4[4](uint4(0, 1, 1, 2), uint4(2, 1, 1, 0), uint4(1, 0, 2, 1), uint4(1, 2, 0, 1));
    return tiles[min(pattern, 3u)][y * 2u + x];
}

struct Bayer : IPixelFilter {
    struct Params {
        float pattern;  // 0 RGGB, 1 BGGR, 2 GRBG, 3 GBRG
    }

    static void apply(inout SensorPixel value, PixelContext pixel, Params params) {
        if (pixel.uniforms().size.w != 3) {
            return;
        }
        let keep = bayerChannel(int2(pixel.id.xy), uint(params.pattern));
        for (uint c = 0; c < 3; ++c) {
            if (c != keep) {
                value.electrons[c] = 0.0;
            }
        }
    }
}
