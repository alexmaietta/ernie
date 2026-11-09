// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;

#include "ernie/stage/columns/kernels/changes.h"

using namespace ernie;

[shader("compute")]
[numthreads(kChangesGroup, 1, 1)]
void StampRows(uniform StampParams p) {
    const uint item = LinearThreadIndex();
    if (item >= p.count || !Selected(p.mask, item)) {
        return;
    }
    const uint row = ViewRow(p.rows, p.firstRow, item);
    if (row != kNoRow) {
        StampRow(p.target, row);
    }
}

[shader("compute")]
[numthreads(kChangesGroup, 1, 1)]
void CompactChanged(uniform CompactParams p) {
    const uint row = LinearThreadIndex();
    if (row >= p.rows) {
        return;
    }
    const Paged<uint64_t> stamps = { p.stamps };
    const Paged<uint> prims = { p.tablePrims };
    const uint64_t stamp = stamps[row];
    if (stamp <= p.since || stamp > p.until || prims[row] == kNoPrim) {
        return;
    }
    uint slot;
    InterlockedAdd(p.out[0], 1u, slot);
    p.out[kChangedRowsOffset + slot] = row;
}

[shader("compute")]
[numthreads(kChangesGroup, 1, 1)]
void CompactWorlds(uniform WorldsParams p) {
    const uint slot = LinearThreadIndex();
    AuthoredData* scene = p.scene;
    if (slot > scene->worldCount) {
        return;
    }
    const uint64_t stamp = scene->worldStamps[slot];
    if (stamp <= p.since || stamp > p.until) {
        return;
    }
    uint index;
    InterlockedAdd(p.out[0], 1u, index);
    p.out[kChangedRowsOffset + index] = SlotWorld(slot);
}
