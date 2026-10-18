// SPDX-License-Identifier: BSD-3-Clause
/// @file imgui_backend.h
/// @brief Dear ImGui's plumbing: its context as the viewer sets it up, and its Vulkan backend.
#pragma once

#include "ernie/app/ui.h"
#include "ernie/device/context.h"

#include <glm/vec4.hpp>

struct ImDrawData;
struct ImGuiContext;

namespace ernie::app {

struct Theme;

/// @brief Makes and destroys an ImGui context with docking and keyboard navigation, the theme
/// applied and no settings file, current while it lives. One at a time.
class ImGuiSession {
  public:
    explicit ImGuiSession(const Theme& theme);
    ~ImGuiSession();
    ImGuiSession(const ImGuiSession&) = delete;
    ImGuiSession& operator=(const ImGuiSession&) = delete;

  private:
    ImGuiContext* context_;
};

/// @brief ImGui's Vulkan backend on a device context's main queue, drawing into images of one
/// format with dynamic rendering. Needs a current ImGui context for its whole life.
class ImGuiRenderer {
  public:
    ImGuiRenderer(DeviceContext& context, VkFormat format, std::uint32_t imageCount);
    ~ImGuiRenderer();
    ImGuiRenderer(const ImGuiRenderer&) = delete;
    ImGuiRenderer& operator=(const ImGuiRenderer&) = delete;

    /// @brief Starts the backend's part of a UI frame.
    void NewFrame();
    /// @brief Uploads the textures ImGui asks for, such as its font atlas. The backend submits and
    /// waits for these copies itself, so this runs before recording the frame.
    void UpdateTextures(ImDrawData& data);
    /// @brief Records clearing the target to `clear`, drawing the UI over it, and leaving it in
    /// its final layout.
    void Record(VkCommandBuffer commands, ImDrawData& data, const FrameTarget& target,
                const glm::vec4& clear);
    /// @return A texture the UI draws an image with in place, sampled in the general layout.
    ImTextureID AddTexture(VkImageView view, VkSampler sampler);
    /// @brief Forgets a texture `AddTexture` made, once no frame in flight draws with it.
    void RemoveTexture(ImTextureID texture);

  private:
    DeviceContext& context_;
};

}  // namespace ernie::app
