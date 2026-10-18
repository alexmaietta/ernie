// SPDX-License-Identifier: BSD-3-Clause
/// @file kernel.cpp
/// @brief Creating compute pipelines and recording dispatches.
#include "ernie/device/kernel.h"

#include <stdexcept>

namespace ernie {

ComputeKernel::ComputeKernel(DeviceContext& context, std::span<const std::uint32_t> spirv,
                             std::string_view name, std::uint32_t parameterBytes,
                             const BindlessHeap* heap, std::string_view entryPoint)
    : device_(context.Device()), name_(name), parameterBytes_(parameterBytes),
      heap_(heap != nullptr ? heap->Set() : VK_NULL_HANDLE) {
    if (parameterBytes > context.Limits().maxPushConstantsSize) {
        throw DeviceError(DeviceErrorKind::Unsupported,
                          name_ + " passes more bytes per dispatch than the device allows");
    }
    const VkPushConstantRange range{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                    .size = parameterBytes};
    const VkDescriptorSetLayout heapLayout = heap != nullptr ? heap->Layout() : VK_NULL_HANDLE;
    const VkPipelineLayoutCreateInfo layout{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                            .setLayoutCount = heap != nullptr ? 1u : 0u,
                                            .pSetLayouts = &heapLayout,
                                            .pushConstantRangeCount = parameterBytes > 0 ? 1u : 0u,
                                            .pPushConstantRanges = &range};
    CheckVk(vkCreatePipelineLayout(device_, &layout, nullptr, &layout_),
            "Creating " + name_ + "'s layout");

    // maintenance5 would take the code inline; a module is portable to every Vulkan 1.3 driver.
    const VkShaderModuleCreateInfo code{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                        .codeSize = spirv.size_bytes(),
                                        .pCode = spirv.data()};
    VkShaderModule module = VK_NULL_HANDLE;
    CheckVk(vkCreateShaderModule(device_, &code, nullptr, &module), "Loading " + name_);
    const std::string entry(entryPoint);
    const VkComputePipelineCreateInfo pipeline{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = module,
                  .pName = entry.c_str()},
        .layout = layout_};
    const VkResult result =
        vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline, nullptr, &pipeline_);
    vkDestroyShaderModule(device_, module, nullptr);
    CheckVk(result, "Creating " + name_);
    context.Name(VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<std::uint64_t>(pipeline_), name_);
}

ComputeKernel::~ComputeKernel() {
    vkDestroyPipeline(device_, pipeline_, nullptr);
    vkDestroyPipelineLayout(device_, layout_, nullptr);
}

void ComputeKernel::Dispatch(VkCommandBuffer commands, glm::uvec3 groups,
                             std::span<const std::byte> parameters) const {
    if (parameters.size() != parameterBytes_) {
        throw std::invalid_argument(name_ + " takes " + std::to_string(parameterBytes_) +
                                    " bytes per dispatch, not " +
                                    std::to_string(parameters.size()));
    }
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    if (heap_ != VK_NULL_HANDLE) {
        vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &heap_, 0,
                                nullptr);
    }
    if (parameterBytes_ > 0) {
        vkCmdPushConstants(commands, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, parameterBytes_,
                           parameters.data());
    }
    vkCmdDispatch(commands, groups.x, groups.y, groups.z);
}

void ComputeBarrier(VkCommandBuffer commands) {
    const VkMemoryBarrier2 barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                                   .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                                   .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   .dstAccessMask =
                                       VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT};
    const VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                      .memoryBarrierCount = 1,
                                      .pMemoryBarriers = &barrier};
    vkCmdPipelineBarrier2(commands, &dependency);
}

}  // namespace ernie
