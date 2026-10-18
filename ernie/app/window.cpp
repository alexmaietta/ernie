// SPDX-License-Identifier: BSD-3-Clause
/// @file window.cpp
/// @brief GLFW's windows, routed through volk's Vulkan loader.
#include "ernie/app/window.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ernie::app {
namespace {

int users = 0;

/// @return Whether GLFW is initialised, initialising it for one more user.
bool Acquire() {
    if (users == 0) {
        glfwSetErrorCallback([](int code, const char* message) {
            log::Detail("GLFW error " + std::to_string(code) + ": " + message);
        });
        // GLFW finds Vulkan through the loader volk opened, as the device layer does.
        if (volkInitialize() != VK_SUCCESS) {
            return false;
        }
        glfwInitVulkanLoader(vkGetInstanceProcAddr);
        if (glfwInit() == GLFW_FALSE) {
            return false;
        }
    }
    ++users;
    return true;
}

void Release() {
    if (--users == 0) {
        glfwTerminate();
    }
}

}  // namespace

bool Window::Available() {
    if (!Acquire()) {
        return false;
    }
    const bool supported = glfwVulkanSupported() == GLFW_TRUE;
    Release();
    return supported;
}

std::vector<std::string> Window::InstanceExtensions() {
    if (!Acquire()) {
        return {};
    }
    std::uint32_t count = 0;
    const char** names = glfwGetRequiredInstanceExtensions(&count);
    std::vector<std::string> extensions(names, names + (names != nullptr ? count : 0));
    Release();
    return extensions;
}

Window::Window(const std::string& title, glm::uvec2 size) {
    if (!Acquire()) {
        throw std::runtime_error("No display to open a window on; --offscreen draws without one");
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window_ = glfwCreateWindow(static_cast<int>(size.x), static_cast<int>(size.y), title.c_str(),
                               nullptr, nullptr);
    if (window_ == nullptr) {
        Release();
        throw std::runtime_error("Could not open a window");
    }
}

Window::~Window() {
    glfwDestroyWindow(window_);
    Release();
}

VkSurfaceKHR Window::CreateSurface(VkInstance instance) const {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    CheckVk(glfwCreateWindowSurface(instance, window_, nullptr, &surface),
            "glfwCreateWindowSurface");
    return surface;
}

glm::uvec2 Window::FramebufferSize() const {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    return {static_cast<std::uint32_t>(std::max(width, 0)),
            static_cast<std::uint32_t>(std::max(height, 0))};
}

float Window::ContentScale() const {
    float x = 1.0f;
    float y = 1.0f;
    glfwGetWindowContentScale(window_, &x, &y);
    return std::max(x, 1.0f);
}

bool Window::CloseRequested() const {
    return glfwWindowShouldClose(window_) == GLFW_TRUE;
}

void Window::PollEvents() const {
    glfwPollEvents();
}

}  // namespace ernie::app
