// SPDX-License-Identifier: BSD-3-Clause
/// @file write_instances.slang
/// @brief The top level's instance records from the transform graph's worlds, one thread a row.
import affine;
import transform;

/// VkAccelerationStructureInstanceKHR.
struct Instance { float4 rows[3]; uint indexAndMask; uint offsetAndFlags; uint64_t bottom; };
struct Parameters {
    float4* worlds; uint* nodes; uint* geometries; float4* shapes; uint64_t* bottoms;
    Instance* instances; uint count;
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
