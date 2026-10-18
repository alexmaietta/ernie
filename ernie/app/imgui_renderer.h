// SPDX-License-Identifier: BSD-3-Clause
/// @file imgui_renderer.h
/// @brief Drawing ImGui's draw data with Vulkan.
#pragma once

#include "ernie/app/presenter.h"
#include "ernie/device/context.h"

#include <glm/vec4.hpp>

struct ImDrawData;

namespace ernie::app {

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

  private:
    DeviceContext& context_;
};

}  // namespace ernie::app
