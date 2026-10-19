// SPDX-License-Identifier: BSD-3-Clause
/// @file instance_colours.slang
/// @brief Instances' colours from a device column of a bucket's display colours, as CUDA or a
/// kernel wrote them: one thread a bucket row, each copying its colour to its instance's row. The
/// parameters are shared with C++ (instances.cpp).
#ifdef __cplusplus
#pragma once
#include "ernie/shaders/shared.h"
namespace ernie::shaders {
#else
import buffer_ref;
#endif

/// What `targets` holds for a bucket row that draws no instance.
static const uint kNoInstanceRow = 0xffffffff;

struct ColourParameters {
    /// Three floats a bucket row.
    BufferRef<float> from;
    /// Each bucket row's instance row, or `kNoInstanceRow`.
    BufferRef<uint> targets;
    /// Three floats an instance row.
    BufferRef<float> to;
    uint count;
};

#ifdef __cplusplus
}  // namespace ernie::shaders
#else

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uniform ColourParameters parameters) {
    uint row = id.x;
    if (row >= parameters.count) {
        return;
    }
    uint target = parameters.targets[row];
    if (target == kNoInstanceRow) {
        return;
    }
    for (uint channel = 0; channel < 3; ++channel) {
        parameters.to[3 * target + channel] = parameters.from[3 * row + channel];
    }
}

#endif
