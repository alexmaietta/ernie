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
    uint4 text;
    float4 place;
    // x half heights or heads, y 1 + pose indices' first float, z 1 + its count's (0 for none),
    // w points per trail.
    uint4 more;
}

struct SortConstants {
    // x the batch, y its slots, z its first word of sort data, w its first draw arguments. For
    // `posesMain`: x the batch, y its first anchor row, z its items, w its poses' first float.
    uint4 batch;
    // Vertices per item of the batch's first and second draws (0 for none). For `posesMain`: x
    // 1 + its pose indices' first float, or 0.
    uint4 vertices;
    // `posesMain`: the poses' frame, its top three rows, relative to the frame's origin.
    float4 frame[3];
}

[[vk::binding(0, 0)]] StructuredBuffer<float> batchData;
[[vk::binding(1, 0)]] StructuredBuffer<Batch> batches;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> sortData;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint4> drawArgs;
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> anchors;
[[vk::push_constant]] ConstantBuffer<SortConstants> constants;

// The items of batch `b` in use: its rows, or as many as its count says.
uint itemsInUse(Batch b) {
    if (b.more.z == 0) {
        return b.shape.y;
    }
    return min(uint(max(asint(batchData[b.more.z - 1]), 0)), b.shape.y);
}

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
    if (id.x < itemsInUse(b)) {
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
    if (id.x >= itemsInUse(b)) {
        return;
    }
    let slots = constants.batch.y;
    uint at;
    InterlockedAdd(sortData[constants.batch.z + slots + slotOf(b, id.x)], 1, at);
    sortData[b.extras.z + at] = id.x;
}

// One thread: the draws' arguments of a batch drawn by its count, not sorted by world, from its
// first items.
[shader("compute")]
[numthreads(1, 1, 1)]
void countArgsMain() {
    let b = batches[constants.batch.x];
    let items = itemsInUse(b);
    for (uint d = 0; d < 2; ++d) {
        drawArgs[constants.batch.w + d] = uint4(items * constants.vertices[d], 1, 0, 0);
    }
}

// The anchor rows of a batch posed under a pose array (`OverlayBatch::poses`): item i's the
// frame times its pose (position, then quaternion x, y, z, w), row-major as overlay.slang reads
// the anchor table.
[shader("compute")]
[numthreads(64, 1, 1)]
void posesMain(uint3 id : SV_DispatchThreadID) {
    let i = id.x;
    if (i >= constants.batch.z) {
        return;
    }
    let indices = constants.vertices.x;
    let pose = indices != 0 ? uint(max(asint(batchData[indices - 1 + i]), 0)) : i;
    let at = constants.batch.w + 7 * pose;
    let t = float3(batchData[at], batchData[at + 1], batchData[at + 2]);
    let q = normalize(float4(batchData[at + 3], batchData[at + 4], batchData[at + 5],
                             batchData[at + 6]));
    let x = q.x;
    let y = q.y;
    let z = q.z;
    let w = q.w;
    let local = float4x4(
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), t.x,
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), t.y,
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), t.z,
        0, 0, 0, 1);
    let frame = float4x4(constants.frame[0], constants.frame[1], constants.frame[2],
                         float4(0, 0, 0, 1));
    let m = mul(frame, local);
    let row = 4 * (constants.batch.y + i);
    for (uint r = 0; r < 4; ++r) {
        anchors[row + r] = m[r];
    }
}
