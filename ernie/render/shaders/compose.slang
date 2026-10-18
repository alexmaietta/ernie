// SPDX-License-Identifier: BSD-3-Clause
/// @file compose.slang
/// @brief One level of the transform graph: each node's world from its parent's and its local, in
/// double-float.
import affine;
import double_float;
// The op kinds and program layout, generated from transform_graph.cpp's constants.
import ops;

struct Parameters { uint* slots; uint* parents; float4* locals; float4* worlds; uint count; };
[[vk::push_constant]] ConstantBuffer<Parameters> parameters;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= parameters.count) {
        return;
    }
    uint slot = parameters.slots[id.x];
    uint parent = parameters.parents[slot];
    DoubleAffine local = LoadDouble(parameters.locals, slot);
    StoreDouble(parameters.worlds, slot,
                parent == kNone ? local : Mul(LoadDouble(parameters.worlds, parent), local));
}
