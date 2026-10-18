// SPDX-License-Identifier: BSD-3-Clause
/// @file offscreen_presenter.cpp
/// @brief The offscreen image and the UI's display size.
#include "ernie/app/offscreen_presenter.h"

#include <imgui.h>

#include <algorithm>

namespace ernie::app {

OffscreenPresenter::OffscreenPresenter(DeviceContext& context, glm::uvec2 extent)
    : image_(context,
             {.format = ImageFormat::Rgba8Unorm,
              .extent = {extent, 1},
              .usage = ImageUsage::Attachment | ImageUsage::Sampled},
             "Viewer frame"),
      drawn_(image_.Ready()) {}

void OffscreenPresenter::BeginUiFrame(double delta) {
    ImGuiIO& io = ImGui::GetIO();
    const glm::uvec3 extent = image_.Description().extent;
    io.DisplaySize = {static_cast<float>(extent.x), static_cast<float>(extent.y)};
    io.DeltaTime = static_cast<float>(std::max(delta, 1e-4));
}

std::optional<FrameTarget> OffscreenPresenter::Acquire() {
    // The previous frame's drawing is ordered before this one's on the same queue.
    return FrameTarget{image_.Handle(), image_.View(), glm::uvec2(image_.Description().extent),
                       VK_IMAGE_LAYOUT_GENERAL};
}

}  // namespace ernie::app
