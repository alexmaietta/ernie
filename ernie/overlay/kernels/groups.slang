// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.overlay.kernels.common;

using namespace ernie;

[shader("compute")]
[numthreads(kOverlayGroup, 1, 1)]
void KeyOverlayRows(uniform OverlayGroupParams g) {
    const uint row = LinearThreadIndex();
    if (row >= g.rows) {
        return;
    }
    const uint world = g.worlds[row * g.worldStride];
    uint bucket = world == kGlobalWorld ? 0 : (world < g.groups ? world + 1 : g.groups + 1);
    if (row >= min(g.itemCounts[0], g.rows)) {
        bucket = g.groups + 1;
    }
    g.keys[row] = bucket;
    g.order[row] = row;
}

[shader("compute")]
[numthreads(kOverlayGroup, 1, 1)]
void StartOverlayGroups(uniform OverlayGroupParams g) {
    const uint bucket = LinearThreadIndex();
    if (bucket > g.groups + 2) {
        return;
    }
    uint low = 0;
    uint high = g.rows;
    while (low < high) {
        const uint middle = (low + high) / 2;
        if (g.keys[middle] < bucket) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    g.starts[bucket] = low;
}

[shader("compute")]
[numthreads(kOverlayGroup, 1, 1)]
void MeasureOverlayGroups(uniform OverlayGroupParams g) {
    const uint world = LinearThreadIndex();
    if (world >= max(g.groups, 1u)) {
        return;
    }
    const uint global = g.starts[1] - g.starts[0];
    const uint own = g.groups > 0 ? g.starts[world + 2] - g.starts[world + 1] : 0;
    InterlockedMax(g.needs[g.need], global + own);
}
