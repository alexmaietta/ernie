// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;
import ernie.algorithms.kernels.group;

#include "ernie/algorithms/kernels/scan.h"

using namespace ernie;

groupshared uint s_tile[kScanTileItems];

uint LiveCount(ScanParams p) {
    return min(p.deviceCount[0], p.count);
}

[shader("compute")]
[numthreads(kScanThreads, 1, 1)]
void SumTiles(uniform ScanParams p) {
    const uint tile = LinearGroupIndex();
    const uint count = LiveCount(p);
    if (tile * kScanTileItems >= count) {
        return;
    }

    const uint thread = WaveOrderedThreadIndex();
    uint sum = 0;
    for (uint i = 0; i < kScanItems; ++i) {
        const uint item = tile * kScanTileItems + i * kScanThreads + thread;
        sum += item < count ? p.input[item] : 0;
    }

    const uint total = GroupCombine<SumCombine>(thread, kScanThreads, sum);
    if (thread == 0) {
        p.sums[tile] = total;
    }
}

[shader("compute")]
[numthreads(kScanThreads, 1, 1)]
void ScanTiles(uniform ScanParams p) {
    const uint tile = LinearGroupIndex();
    const uint start = tile * kScanTileItems;
    const uint count = LiveCount(p);
    if (start >= count) {
        return;
    }

    const uint thread = WaveOrderedThreadIndex();
    for (uint i = 0; i < kScanItems; ++i) {
        const uint place = i * kScanThreads + thread;
        s_tile[place] = start + place < count ? p.input[start + place] : 0;
    }
    GroupMemoryBarrierWithGroupSync();

    uint values[kScanItems];
    uint sum = 0;
    for (uint i = 0; i < kScanItems; ++i) {
        values[i] = s_tile[thread * kScanItems + i];
        sum += values[i];
    }

    uint total;
    uint prefix = GroupExclusiveSum(thread, kScanThreads, sum, total);
    if (p.hasSums != 0) {
        prefix += p.sums[tile];
    }
    for (uint i = 0; i < kScanItems; ++i) {
        s_tile[thread * kScanItems + i] = prefix;
        prefix += values[i];
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint i = 0; i < kScanItems; ++i) {
        const uint place = i * kScanThreads + thread;
        if (start + place < count) {
            p.output[start + place] = s_tile[place];
        }
    }
}
