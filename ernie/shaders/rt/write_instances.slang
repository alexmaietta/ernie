// SPDX-License-Identifier: BSD-3-Clause
/// @file write_instances.slang
/// @brief The top level's instance records from the transform graph's worlds, one thread a row.
import affine;
import transform;

/// VkAccelerationStructureInstanceKHR.
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

/// As top_levels.cpp's `WriteParameters`.
struct Parameters {
    /// The transform graph's worlds, by slot.
    float4* worlds;
    /// Each instance row's transform-graph slot, geometry id, and shape transform.
    uint* nodes;
    uint* geometries;
    float4* shapes;
    /// Each geometry's bottom-level address, 0 until it is built.
    uint64_t* bottoms;
    Instance* instances;
    uint count;
};
[[vk::push_constant]] ConstantBuffer<Parameters> parameters;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
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
