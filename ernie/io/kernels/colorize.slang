// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;
import ernie.io.kernels.encoding;

#include "ernie/io/kernels/colorize.h"

using namespace ernie;

[shader("compute")]
[numthreads(kColorizeThreads, 1, 1)]
void ScalarRange(uniform ColorizeParams p) {
    const uint pixel = LinearThreadIndex();
    if (pixel >= p.count) {
        return;
    }
    const float value = asfloat(p.input[pixel * p.channels + p.channel]);
    if (ScalarShown(value, p.flags)) {
        InterlockedMin(p.range[0], OrderedBits(value));
        InterlockedMax(p.range[1], OrderedBits(value));
    }
}

[shader("compute")]
[numthreads(kColorizeThreads, 1, 1)]
void ColorizeScalars(uniform ColorizeParams p) {
    const uint pixel = LinearThreadIndex();
    if (pixel >= p.count) {
        return;
    }
    float low = asfloat(0x7F800000u);
    float high = asfloat(0xFF800000u);
    const uint fixed = kFixedMinimum | kFixedMaximum;
    if ((p.flags & fixed) != fixed && p.range[0] != kNoRange) {
        low = OrderedFloat(p.range[0]);
        high = OrderedFloat(p.range[1]);
    }
    const ScalarBounds bounds = ResolveBounds(low, high, p.minimum, p.maximum, p.flags);
    if (pixel == 0 && (p.flags & kWriteBounds) != 0) {
        p.bounds[0] = bounds.low;
        p.bounds[1] = bounds.high;
    }
    const float value = asfloat(p.input[pixel * p.channels + p.channel]);
    p.output[pixel] = ScalarRgba(value, bounds, p.flags, p.colormap);
}

[shader("compute")]
[numthreads(kColorizeThreads, 1, 1)]
void ColorizeHdr(uniform ColorizeParams p) {
    const uint pixel = LinearThreadIndex();
    if (pixel >= p.count) {
        return;
    }
    const uint base = pixel * p.channels;
    const float alpha = p.channels == 4 ? asfloat(p.input[base + 3]) : 1.0f;
    p.output[pixel] = HdrRgba(asfloat(p.input[base]), asfloat(p.input[base + 1]),
                              asfloat(p.input[base + 2]), alpha, p.scale);
}

[shader("compute")]
[numthreads(kColorizeThreads, 1, 1)]
void ColorizeNormals(uniform ColorizeParams p) {
    const uint pixel = LinearThreadIndex();
    if (pixel >= p.count) {
        return;
    }
    const uint base = pixel * p.channels;
    p.output[pixel] = NormalRgba(asfloat(p.input[base]), asfloat(p.input[base + 1]),
                                 asfloat(p.input[base + 2]));
}

[shader("compute")]
[numthreads(kColorizeThreads, 1, 1)]
void ColorizeIds(uniform ColorizeParams p) {
    const uint pixel = LinearThreadIndex();
    if (pixel >= p.count) {
        return;
    }
    const uint base = pixel * p.channels;
    uint key = p.input[base];
    bool empty = key == 0;
    for (uint c = 1; c < p.channels; ++c) {
        const uint word = p.input[base + c];
        key = MixIdWord(key, word);
        empty = empty && word == 0;
    }
    p.output[pixel] = empty ? 0 : IdRgba(key);
}
