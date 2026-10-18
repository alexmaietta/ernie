// SPDX-License-Identifier: BSD-3-Clause
/// @file swapchain.h
/// @brief A window surface's swapchain, presented on the device's main queue.
#pragma once

#include "ernie/device/context.h"

#include <glm/vec2.hpp>

#include <optional>
#include <vector>

namespace ernie::app {

/// @brief A swapchain of a surface it owns. The host waits for each acquired image and each drawn
/// frame, so presentation needs no semaphores beside the queue's timeline.
class Swapchain {
  public:
    /// @param surface Taken: destroyed with the swapchain.
    /// @throws DeviceError when the device cannot present to the surface.
    Swapchain(DeviceContext& context, VkSurfaceKHR surface, glm::uvec2 extent, bool vsync);
    ~Swapchain();
    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    struct Image {
        std::uint32_t index = 0;
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
    };

    /// @return The next image, or none when the swapchain no longer matches the surface: call
    /// `Resize`, then try again.
    std::optional<Image> Acquire();
    /// @brief Presents an image once `drawn` is done.
    /// @return Whether it still matches the surface.
    bool Present(const Image& image, const Ticket& drawn);
    /// @brief Makes the swapchain again at `extent`, keeping the surface.
    void Resize(glm::uvec2 extent);

    VkFormat Format() const {
        return format_.format;
    }
    glm::uvec2 Extent() const {
        return {extent_.width, extent_.height};
    }
    std::uint32_t ImageCount() const {
        return static_cast<std::uint32_t>(images_.size());
    }

  private:
    void Create(glm::uvec2 extent);
    void DestroyViews();

    DeviceContext& context_;
    VkSurfaceKHR surface_;
    bool vsync_;
    VkSurfaceFormatKHR format_{};
    VkExtent2D extent_{};
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    std::vector<VkImage> images_;
    std::vector<VkImageView> views_;
    VkFence acquired_ = VK_NULL_HANDLE;
};

}  // namespace ernie::app
