// SPDX-License-Identifier: BSD-3-Clause
// Overlay batches sorted by world (`OverlayPass`): the items of a batch with worlds are counted per
// world, the counts summed into offsets, and the items scattered into world order, so each view of
// a batched camera draws only the items of its own world and the shared ones (world -1), through
// indirect draws, rather than every item of every world.
//
// A batch's slots: 0 the shared items, 1 + w world w's, then one for worlds no view shows. Its sort
// data: counts per slot, cursors per slot, then its items in slot order (`Batch.extras.z` in
// overlay.slang). Its draw arguments: per slot but the last, per draw of the batch (2), a
// VkDrawIndirectCommand.
//
// Bindings: 0 batches' arrays, 1 batches, 2 sort data, 3 draw arguments.

// overlay.slang's Batch.
struct Batch {
    uint4 shape;
    uint4 arrays;
    uint4 extras;
    uint4 style;
    float4 origin;
    float4 params;
}

struct SortConstants {
    // x the batch, y its slots, z its first word of sort data, w its first draw arguments.
    uint4 batch;
    // Vertices per item of the batch's first and second draws (0 for none).
    uint4 vertices;
}

[[vk::binding(0, 0)]] StructuredBuffer<float> batchData;
[[vk::binding(1, 0)]] StructuredBuffer<Batch> batches;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> sortData;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint4> drawArgs;
[[vk::push_constant]] ConstantBuffer<SortConstants> constants;

// Item `i`'s slot.
uint slotOf(Batch b, uint i) {
    let world = asint(batchData[b.extras.y + i]);
    let slots = constants.batch.y;
    if (world < 0) {
        return 0;
    }
    return min(uint(world) + 1, slots - 1);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sortClearMain(uint3 id : SV_DispatchThreadID) {
    if (id.x < constants.batch.y) {
        sortData[constants.batch.z + id.x] = 0;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sortCountMain(uint3 id : SV_DispatchThreadID) {
    let b = batches[constants.batch.x];
    if (id.x < b.shape.y) {
        InterlockedAdd(sortData[constants.batch.z + slotOf(b, id.x)], 1);
    }
}

// One thread: each slot's first item, and its draws' arguments. The draws start at vertex 0 (the
// vertex shaders' SV_VertexID leaves out a first vertex): they find the slot's first item as its
// cursor less its count, once the items are scattered.
[shader("compute")]
[numthreads(1, 1, 1)]
void sortScanMain() {
    let slots = constants.batch.y;
    let counts = constants.batch.z;
    let cursors = counts + slots;
    uint first = 0;
    for (uint s = 0; s < slots; ++s) {
        let count = sortData[counts + s];
        sortData[cursors + s] = first;
        if (s + 1 < slots) {
            for (uint d = 0; d < 2; ++d) {
                drawArgs[constants.batch.w + 2 * s + d] =
                    uint4(count * constants.vertices[d], 1, 0, 0);
            }
        }
        first += count;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sortScatterMain(uint3 id : SV_DispatchThreadID) {
    let b = batches[constants.batch.x];
    if (id.x >= b.shape.y) {
        return;
    }
    let slots = constants.batch.y;
    uint at;
    InterlockedAdd(sortData[constants.batch.z + slots + slotOf(b, id.x)], 1, at);
    sortData[b.extras.z + at] = id.x;
}
