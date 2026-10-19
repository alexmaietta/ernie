// SPDX-License-Identifier: BSD-3-Clause
/// @file write_instances.slang
/// @brief Every batch world's top-level records from the transform graph's worlds, one thread a
/// record; shared instances have one in every world.
/// The records and parameters are shared with C++ (rt/top_levels.cpp).
#ifdef __cplusplus
#pragma once
#include "ernie/shaders/shared.h"
namespace ernie::shaders {
#else
import affine;
import buffer_ref;
import transform;
#endif

/// A top level's instance, as Vulkan's VkAccelerationStructureInstanceKHR lays it out.
struct Instance {
    /// The instance's transform: the geometry's frame to the scene's.
    float4 rows[3];
    /// The custom index (the geometry id) in the low 24 bits, the visibility mask in the high 8.
    uint indexAndMask;
    /// The hit-group offset in the low 24 bits, the instance flags in the high 8.
    uint offsetAndFlags;
    /// The bottom level's device address.
    uint64_t bottom;
};

struct WriteParameters {
    /// The transform graph's worlds, by slot.
    BufferRef<float4> worlds;
    /// Each instance row's transform-graph slot, geometry id, and shape transform.
    BufferRef<uint> nodes;
    BufferRef<uint> geometries;
    BufferRef<float4> shapes;
    /// Each geometry's bottom-level address, 0 until it is built.
    BufferRef<uint64_t> bottoms;
    /// Each record's instance row, and the records, every world's.
    BufferRef<uint> sources;
    BufferRef<Instance> instances;
    /// A built bottom level, which instances whose own is not built point at, switched off.
    uint64_t placeholder;
    uint count;
};

#ifdef __cplusplus
}  // namespace ernie::shaders
#else

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uniform WriteParameters parameters) {
    uint record = id.x;
    if (record >= parameters.count) {
        return;
    }
    uint row = parameters.sources[record];
    uint geometry = parameters.geometries[row];
    Affine m = Mul(LoadWorld(parameters.worlds, parameters.nodes[row]),
                   Load(parameters.shapes, row));
    uint64_t bottom = parameters.bottoms[geometry];
    Instance instance;
    instance.rows = m.rows;
    // An instance whose bottom level is not built yet is switched off: lavapipe faults on a null
    // reference even then, so it points at another level.
    instance.indexAndMask = (bottom == 0 ? 0u : 0xff000000u) | (geometry & 0xffffffu);
    instance.offsetAndFlags = 0;
    instance.bottom = bottom == 0 ? parameters.placeholder : bottom;
    parameters.instances[record] = instance;
}

#endif
