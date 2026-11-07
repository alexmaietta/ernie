// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;

#include "ernie/scene/kernels/readback.h"

using namespace ernie;

[shader("compute")]
[numthreads(kReadbackGroup, 1, 1)]
void GatherRows(uniform GatherParams p) {
    const uint index = LinearThreadIndex();
    if (index >= p.count) {
        return;
    }
    const uint row = p.rows[index];
    for (uint word = 0; word < p.words; ++word) {
        p.out[index * p.words + word] = row == kNoRow ? 0 : p.source[row * p.words + word];
    }
}
