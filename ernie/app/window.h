// SPDX-License-Identifier: BSD-3-Clause
/// @file window.h
/// @brief A desktop window through GLFW, for Vulkan to present to.
#pragma once

#include "ernie/device/vulkan.h"

#include <glm/vec2.hpp>

#include <string>
#include <vector>

struct GLFWwindow;

namespace ernie::app {

/// @brief A resizable window with no graphics API of its own. GLFW is initialised by the first
/// window and terminated with the last.
class Window {
  public:
    /// @throws std::runtime_error when there is no display or the window cannot be made.
    Window(const std::string& title, glm::uvec2 size);
    ~Window();
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    /// @return Whether a window could open: GLFW initialises, finding a display, and Vulkan can
    /// present to it.
    static bool Available();
    /// @return The instance extensions presenting to a window needs.
    static std::vector<std::string> InstanceExtensions();

    /// @return A surface of the window on `instance`, which the caller destroys.
    VkSurfaceKHR CreateSurface(VkInstance instance) const;
    /// @return The drawable size in pixels; zero while minimised.
    glm::uvec2 FramebufferSize() const;
    /// @return The display's scale over 96 dpi, at least 1.
    float ContentScale() const;
    bool CloseRequested() const;
    void PollEvents() const;
    GLFWwindow* Handle() const {
        return window_;
    }

  private:
    GLFWwindow* window_ = nullptr;
};

}  // namespace ernie::app
