// SPDX-License-Identifier: BSD-3-Clause
/// @file write_instances.slang
/// @brief The top level's instance records from the transform graph's worlds, one thread a row.
/// The records and parameters are shared with C++ (rt/top_levels.cpp).
#ifdef __cplusplus
#pragma once
#include "ernie/shaders/shared.h"
namespace ernie::shaders {
#else
import affine;
import ref;
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
    Ref<float4> worlds;
    /// Each instance row's transform-graph slot, geometry id, and shape transform.
    Ref<uint> nodes;
    Ref<uint> geometries;
    Ref<float4> shapes;
    /// Each geometry's bottom-level address, 0 until it is built.
    Ref<uint64_t> bottoms;
    Ref<Instance> instances;
    uint count;
};

#ifdef __cplusplus
}  // namespace ernie::shaders
#else

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uniform WriteParameters parameters) {
    uint row = id.x;
    if (row >= parameters.count) {
        return;
    }
    uint geometry = parameters.geometries[row];
    Affine m = Mul(LoadWorld(parameters.worlds, parameters.nodes[row]),
                   Load(parameters.shapes, row));
    uint64_t bottom = parameters.bottoms[geometry];
    Instance instance;
    instance.rows = m.rows;
    // An instance whose bottom level is not built yet is switched off.
    instance.indexAndMask = (bottom == 0 ? 0u : 0xff000000u) | (geometry & 0xffffffu);
    instance.offsetAndFlags = 0;
    instance.bottom = bottom;
    parameters.instances[row] = instance;
}

#endif
