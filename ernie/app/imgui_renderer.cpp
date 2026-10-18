// SPDX-License-Identifier: BSD-3-Clause
/// @file imgui_renderer.cpp
/// @brief Setting up ImGui's Vulkan backend and recording its draws.
#include "ernie/app/imgui_renderer.h"

#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>

namespace ernie::app {
namespace {

void TransitionImage(VkCommandBuffer commands, VkImage image, VkImageLayout from, VkImageLayout to,
                     VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                     VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
    const VkImageMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = srcStage,
        .srcAccessMask = srcAccess,
        .dstStageMask = dstStage,
        .dstAccessMask = dstAccess,
        .oldLayout = from,
        .newLayout = to,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    const VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                      .imageMemoryBarrierCount = 1,
                                      .pImageMemoryBarriers = &barrier};
    vkCmdPipelineBarrier2(commands, &dependency);
}

}  // namespace

ImGuiRenderer::ImGuiRenderer(DeviceContext& context, VkFormat format, std::uint32_t imageCount)
    : context_(context) {
    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion = VK_API_VERSION_1_3;
    info.Instance = context.Instance();
    info.PhysicalDevice = context.Physical();
    info.Device = context.Device();
    info.QueueFamily = context.Main().Family();
    info.Queue = context.Main().Handle();
    info.DescriptorPoolSize = 64;
    info.MinImageCount = std::max(imageCount, 2u);
    info.ImageCount = std::max(imageCount, 2u);
    info.UseDynamicRendering = true;
    info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    info.PipelineInfoMain.PipelineRenderingCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &format};
    info.CheckVkResultFn = [](VkResult result) { CheckVk(result, "ImGui's Vulkan backend"); };
    if (!ImGui_ImplVulkan_Init(&info)) {
        throw DeviceError(DeviceErrorKind::Failed, "Could not set up ImGui's Vulkan backend");
    }
}

ImGuiRenderer::~ImGuiRenderer() {
    Quietly([this] { context_.Main().Last().Wait(); });
    ImGui_ImplVulkan_Shutdown();
}

void ImGuiRenderer::NewFrame() {
    ImGui_ImplVulkan_NewFrame();
}

void ImGuiRenderer::UpdateTextures(ImDrawData& data) {
    if (data.Textures == nullptr) {
        return;
    }
    for (ImTextureData* texture : *data.Textures) {
        if (texture->Status != ImTextureStatus_OK) {
            ImGui_ImplVulkan_UpdateTexture(texture);
        }
    }
}

void ImGuiRenderer::Record(VkCommandBuffer commands, ImDrawData& data, const FrameTarget& target,
                           const glm::vec4& clear) {
    // Whatever the image held is replaced; earlier work on it, a previous frame's, finishes first.
    TransitionImage(commands, target.image, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                    VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT);
    const VkRenderingAttachmentInfo attachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = target.view,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = {.color = {.float32 = {clear.r, clear.g, clear.b, clear.a}}}};
    const VkRenderingInfo rendering{.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                                    .renderArea = {{0, 0}, {target.extent.x, target.extent.y}},
                                    .layerCount = 1,
                                    .colorAttachmentCount = 1,
                                    .pColorAttachments = &attachment};
    vkCmdBeginRendering(commands, &rendering);
    ImGui_ImplVulkan_RenderDrawData(&data, commands);
    vkCmdEndRendering(commands);
    TransitionImage(commands, target.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    target.finalLayout, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                    kAnyAccess);
}

}  // namespace ernie::app
