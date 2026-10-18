// SPDX-License-Identifier: BSD-3-Clause
/// @file kernel.h
/// @brief A compute kernel: its pipeline, and the per-dispatch bytes it reads.
#pragma once

#include "ernie/device/context.h"

#include <glm/vec3.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace ernie {

/// @brief A compute pipeline from SPIR-V. Each dispatch passes a few bytes, the indices and
/// addresses it works on, as push constants on Vulkan (a small uniform on WebGPU, which has
/// none). Destroying one the GPU may still run is an error: retire it instead.
class ComputeKernel {
  public:
    /// @param parameterBytes The bytes each dispatch passes, at most the device's push-constant
    /// limit (128 at least).
    /// @throws DeviceError if the pipeline cannot be created or the bytes exceed the limit.
    ComputeKernel(DeviceContext& context, std::span<const std::uint32_t> spirv,
                  std::string_view name, std::uint32_t parameterBytes = 0,
                  std::string_view entryPoint = "main");
    ~ComputeKernel();
    ComputeKernel(const ComputeKernel&) = delete;
    ComputeKernel& operator=(const ComputeKernel&) = delete;

    /// @brief Records a dispatch of `groups` workgroups, passing `parameters`.
    void Dispatch(VkCommandBuffer commands, glm::uvec3 groups,
                  std::span<const std::byte> parameters = {}) const;
    template <typename T>
    void Dispatch(VkCommandBuffer commands, glm::uvec3 groups, const T& parameters) const {
        Dispatch(commands, groups, std::as_bytes(std::span(&parameters, 1)));
    }

    const std::string& Name() const {
        return name_;
    }

  private:
    VkDevice device_;
    std::string name_;
    std::uint32_t parameterBytes_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

}  // namespace ernie
