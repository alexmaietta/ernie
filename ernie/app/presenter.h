// SPDX-License-Identifier: BSD-3-Clause
/// @file presenter.h
/// @brief Where the viewer's frames go: a window's swapchain, or an image offscreen.
#pragma once

#include "ernie/device/queue.h"

#include <glm/vec2.hpp>

#include <optional>

namespace ernie::app {

/// @brief An image to draw one frame into, and the layout to leave it in.
struct FrameTarget {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    glm::uvec2 extent{0};
    VkImageLayout finalLayout = VK_IMAGE_LAYOUT_GENERAL;
};

/// @brief Supplies the images frames are drawn into and the UI's platform input, then shows
/// them.
class Presenter {
  public:
    virtual ~Presenter() = default;

    virtual VkFormat Format() const = 0;
    /// @return How many images it cycles through, at least 2.
    virtual std::uint32_t ImageCount() const = 0;
    /// @brief Takes the window system's events.
    virtual void PollEvents() = 0;
    /// @return Whether the user asked to close it.
    virtual bool CloseRequested() const = 0;
    /// @brief Starts the UI frame's platform part: display size, time step and input.
    virtual void BeginUiFrame(double delta) = 0;
    /// @return The next image, or none when there is nothing to draw into, as while minimised.
    virtual std::optional<FrameTarget> Acquire() = 0;
    /// @brief Shows the image last acquired once `drawn` is done.
    virtual void Present(Ticket drawn) = 0;
};

}  // namespace ernie::app
