// SPDX-License-Identifier: BSD-3-Clause
// Indirect draws of meshes' triangles in use (`MeshDrawable::RecordActiveDraws`): for each slot
// of `activeCounts`, a VkDrawIndirectCommand drawing the whole triangles of its indices in use, at
// most the triangles the mesh holds, so triangles past the count never reach the vertex shader.
//
// Bindings: 0 the meshes' points and triangle indices in use (int32 pairs by slot), 1 the draws, 8
// uints per slot: the command (vertices, instances 1, 0, 0), then the triangles the mesh holds.

[[vk::binding(0, 0)]] StructuredBuffer<int> activeCounts;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> draws;
// x: the slots.
[[vk::push_constant]] ConstantBuffer<uint4> slots;

[shader("compute")]
[numthreads(64, 1, 1)]
void activeDrawsMain(uint3 id: SV_DispatchThreadID) {
    let slot = id.x;
    if (slot >= slots.x) {
        return;
    }
    let indices = uint(max(activeCounts[2 * slot + 1], 0));
    draws[8 * slot] = 3 * min(indices / 3, draws[8 * slot + 4]);
}
