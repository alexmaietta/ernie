// SPDX-License-Identifier: BSD-3-Clause
/// @file pose_step.h
/// @brief A step of many instances' poses on device columns, for the test that it moves no bulk
/// bytes through the host and the benchmark that times it: a kernel writes each instance's pose
/// in place, as a simulation in Warp or PyTorch would through DLPack, and an instance pass turns
/// the poses into the renderer's instance transforms.
#pragma once

#include "ernie/device/column.h"
#include "ernie/device/kernel.h"
#include "ernie/device/shader_compiler.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace ernie::test {

const ShaderModule kSimulate{"simulate", R"(
struct Parameters { float4* poses; uint instances; uint step; };
[[vk::push_constant]] ConstantBuffer<Parameters> parameters;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint instance = id.x;
    if (instance >= parameters.instances) {
        return;
    }
    float angle = 0.01 * float(instance % 628) + 0.05 * float(parameters.step);
    parameters.poses[2 * instance] =
        float4(float(instance % 128), float(instance / 128), 0.1 * float(parameters.step), 0);
    parameters.poses[2 * instance + 1] = float4(0, 0, sin(angle / 2), cos(angle / 2));
}
)"};

const ShaderModule kInstances{"instances", R"(
struct Parameters { float4* poses; float4* transforms; uint instances; };
[[vk::push_constant]] ConstantBuffer<Parameters> parameters;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint instance = id.x;
    if (instance >= parameters.instances) {
        return;
    }
    float3 t = parameters.poses[2 * instance].xyz;
    float4 q = parameters.poses[2 * instance + 1];
    // Rows of a 3x4 matrix, as VkTransformMatrixKHR lays them out.
    parameters.transforms[3 * instance] = float4(1 - 2 * (q.y * q.y + q.z * q.z),
        2 * (q.x * q.y - q.z * q.w), 2 * (q.x * q.z + q.y * q.w), t.x);
    parameters.transforms[3 * instance + 1] = float4(2 * (q.x * q.y + q.z * q.w),
        1 - 2 * (q.x * q.x + q.z * q.z), 2 * (q.y * q.z - q.x * q.w), t.y);
    parameters.transforms[3 * instance + 2] = float4(2 * (q.x * q.z - q.y * q.w),
        2 * (q.y * q.z + q.x * q.w), 1 - 2 * (q.x * q.x + q.y * q.y), t.z);
}
)"};

struct SimulateParameters {
    VkDeviceAddress poses;
    std::uint32_t instances, step;
};
struct InstanceParameters {
    VkDeviceAddress poses, transforms;
    std::uint32_t instances;
};

/// @brief Per-instance poses (position, then orientation as x, y, z, w) and instance transforms in
/// device columns, stepped wholly on the device.
class PoseStep {
  public:
    PoseStep(DeviceContext& context, const ShaderCompiler& compiler, std::uint32_t instances)
        : context_(context), count_(instances),
          arenas_(context, MemoryUse::Device, "pose step", std::uint64_t{16} << 20),
          poses_(arenas_, "poses", DType::Float32, {2, 4}, instances),
          transforms_(arenas_, "transforms", DType::Float32, {3, 4}, instances),
          simulate_(context, compiler.Spirv(std::span(&kSimulate, 1), "main"), "simulate",
                    sizeof(SimulateParameters)),
          instances_(context, compiler.Spirv(std::span(&kInstances, 1), "main"), "instances",
                     sizeof(InstanceParameters)) {}

    /// @brief Writes step `step`'s poses, then their transforms, after `waits`.
    Ticket Step(std::uint32_t step, std::span<const Ticket> waits = {}) {
        const glm::uvec3 groups{(count_ + 63) / 64, 1, 1};
        return context_.Main().Submit(
            [&](VkCommandBuffer commands) {
                // After the last step's reads of the poses.
                ComputeBarrier(commands);
                simulate_.Dispatch(commands, groups,
                                   SimulateParameters{poses_.Address(), count_, step});
                ComputeBarrier(commands);
                instances_.Dispatch(
                    commands, groups,
                    InstanceParameters{poses_.Address(), transforms_.Address(), count_});
            },
            waits);
    }

    const DeviceColumn& Transforms() const {
        return transforms_;
    }

    /// @return Instance `instance`'s transform after step `step`, worked out on the host.
    static std::array<float, 12> Expected(std::uint32_t instance, std::uint32_t step) {
        const float angle = 0.01f * float(instance % 628) + 0.05f * float(step);
        const float c = std::cos(angle);
        const float s = std::sin(angle);
        return {c, -s, 0, float(instance % 128), s, c, 0, float(instance / 128),
                0, 0,  1, 0.1f * float(step)};
    }

  private:
    DeviceContext& context_;
    std::uint32_t count_;
    Arenas arenas_;
    DeviceColumn poses_;
    DeviceColumn transforms_;
    ComputeKernel simulate_;
    ComputeKernel instances_;
};

}  // namespace ernie::test
