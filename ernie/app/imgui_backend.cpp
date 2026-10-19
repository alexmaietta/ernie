// SPDX-License-Identifier: BSD-3-Clause
/// @file imgui_backend.cpp
/// @brief Creating the ImGui context, setting up its Vulkan backend and recording its draws.
#include "ernie/app/imgui_backend.h"

#include "ernie/app/icons.h"
#include "ernie/util/profiler.h"

#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include <cstddef>
#include <cstdio>

// The typefaces compiled in by cmake/third_party/fonts.cmake.
extern "C" {
extern const unsigned char kErnieFont_inter[];
extern const std::size_t kErnieFont_inter_size;
extern const unsigned char kErnieFont_lucide[];
extern const std::size_t kErnieFont_lucide_size;
}

namespace ernie::app {
namespace {

void Transition(VkCommandBuffer commands, VkImage image, VkImageLayout from, VkImageLayout to,
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

ImGuiSession::ImGuiSession(const Theme& theme) {
    ERNIE_ZONE("Set up ImGui");
    IMGUI_CHECKVERSION();
    context_ = ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    // Layouts are the viewer's to save; nothing is written behind its back.
    io.IniFilename = nullptr;
    io.LogFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable | ImGuiConfigFlags_NavEnableKeyboard;
    LoadFonts(theme);
    ApplyTheme(theme);
}

void ImGuiSession::LoadFonts(const Theme& theme) {
    ImGuiIO& io = ImGui::GetIO();
    static const ImWchar kIcons[] = {static_cast<ImWchar>(icons::kFirst),
                                     static_cast<ImWchar>(icons::kLast), 0};
    ImFontConfig text;
    text.FontDataOwnedByAtlas = false;
    // Inter has glyphs of its own in the private use area, which would hide the icons'.
    text.GlyphExcludeRanges = kIcons;
    std::snprintf(text.Name, sizeof(text.Name), "Inter");
    io.Fonts->AddFontFromMemoryTTF(const_cast<unsigned char*>(kErnieFont_inter),
                                   static_cast<int>(kErnieFont_inter_size), theme.fontSize, &text);
    ImFontConfig icon;
    icon.FontDataOwnedByAtlas = false;
    icon.MergeMode = true;
    icon.GlyphMinAdvanceX = theme.fontSize;
    icon.GlyphOffset = ImVec2(0.0f, 0.15f * theme.fontSize);
    std::snprintf(icon.Name, sizeof(icon.Name), "Lucide");
    io.Fonts->AddFontFromMemoryTTF(const_cast<unsigned char*>(kErnieFont_lucide),
                                   static_cast<int>(kErnieFont_lucide_size), theme.fontSize, &icon,
                                   kIcons);
}

ImGuiSession::~ImGuiSession() {
    ImGui::DestroyContext(context_);
}

ImGuiRenderer::ImGuiRenderer(DeviceContext& context, VkFormat format, std::uint32_t imageCount)
    : context_(context) {
    ERNIE_ZONE("Set up ImGui's Vulkan backend");
    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion = VK_API_VERSION_1_3;
    info.Instance = context.Instance();
    info.PhysicalDevice = context.Physical();
    info.Device = context.Device();
    info.QueueFamily = context.Main().Family();
    // Used only by texture uploads, which borrow the queue (`UpdateTextures`).
    info.Queue = context.Main().Handle();
    info.DescriptorPoolSize = 64;
    info.MinImageCount = imageCount;
    info.ImageCount = imageCount;
    info.UseDynamicRendering = true;
    info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    info.PipelineInfoMain.PipelineRenderingCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &format};
    info.PipelineCache = context.PipelineCache();
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

ImTextureID ImGuiRenderer::AddTexture(VkImageView view, VkSampler sampler) {
    return reinterpret_cast<ImTextureID>(
        ImGui_ImplVulkan_AddTexture(sampler, view, VK_IMAGE_LAYOUT_GENERAL));
}

void ImGuiRenderer::RemoveTexture(ImTextureID texture) {
    ImGui_ImplVulkan_RemoveTexture(reinterpret_cast<VkDescriptorSet>(texture));
}

void ImGuiRenderer::UpdateTextures(ImDrawData& data) {
    if (data.Textures == nullptr) {
        return;
    }
    for (ImTextureData* texture : *data.Textures) {
        if (texture->Status != ImTextureStatus_OK) {
            // The font atlas, on the first frame and when it grows; the backend submits the
            // upload and waits for it itself.
            ERNIE_ZONE("Upload a UI texture");
            context_.Main().Borrow([&](VkQueue) { ImGui_ImplVulkan_UpdateTexture(texture); });
        }
    }
}

void ImGuiRenderer::Record(VkCommandBuffer commands, ImDrawData& data, const FrameTarget& target,
                           const glm::vec4& clear) {
    // Whatever the image held is replaced; earlier work on it, a previous frame's, finishes first.
    Transition(commands, target.image, VK_IMAGE_LAYOUT_UNDEFINED,
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
    Transition(commands, target.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, target.finalLayout,
               VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
               VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
               kAnyAccess);
}

}  // namespace ernie::app
