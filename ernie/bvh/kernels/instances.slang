// SPDX-License-Identifier: BSD-3-Clause

import ernie.bvh.kernels.common;

using namespace ernie;

[shader("compute")]
[numthreads(kMaskInstancesThreads, 1, 1)]
void MaskInstances(uniform MaskInstancesParams p) {
    const uint i = LinearThreadIndex();
    if (i >= p.total) {
        return;
    }
    const uint world = FindOffset(p.offsets, p.worlds, i);
    InstanceRecord record = p.source[i];
    if (i - p.offsets[world] >= p.counts[world]) {
        record.custom &= kInstanceCustomMask;
    }
    p.destination[i] = record;
}

[shader("compute")]
[numthreads(kGatherWorldsThreads, 1, 1)]
void GatherWorlds(uniform GatherWorldsParams p) {
    const uint i = LinearThreadIndex();
    if (i < p.selected && p.counts != nullptr) {
        p.gatheredCounts[i] = p.counts[p.worlds[i]];
    }
    if (i >= p.total) {
        return;
    }
    const uint world = FindOffset(p.offsets, p.selected, i);
    p.destination[i] = p.source[p.sourceOffsets[world] + i - p.offsets[world]];
}
