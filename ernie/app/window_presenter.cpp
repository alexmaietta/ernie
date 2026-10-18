// SPDX-License-Identifier: BSD-3-Clause
/// @file window_presenter.cpp
/// @brief Presenting to a window and taking its input through ImGui's GLFW backend.
#include "ernie/app/window_presenter.h"

#include <imgui_impl_glfw.h>

namespace ernie::app {

WindowPresenter::WindowPresenter(DeviceContext& context, std::unique_ptr<Window> window, bool vsync)
    : window_(std::move(window)), swapchain_(context, window_->CreateSurface(context.Instance()),
                                             window_->FramebufferSize(), vsync) {
    ImGui_ImplGlfw_InitForVulkan(window_->Handle(), true);
}

WindowPresenter::~WindowPresenter() {
    ImGui_ImplGlfw_Shutdown();
}

void WindowPresenter::BeginUiFrame(double /*delta*/) {
    // The backend measures time steps and sizes itself.
    ImGui_ImplGlfw_NewFrame();
}

std::optional<FrameTarget> WindowPresenter::Acquire() {
    const glm::uvec2 size = window_->FramebufferSize();
    if (size.x == 0 || size.y == 0) {
        return std::nullopt;
    }
    if (stale_ || size != swapchain_.Extent()) {
        swapchain_.Resize(size);
        stale_ = false;
    }
    image_ = swapchain_.Acquire();
    if (!image_) {
        stale_ = true;
        return std::nullopt;
    }
    return FrameTarget{image_->image, image_->view, swapchain_.Extent(),
                       VK_IMAGE_LAYOUT_PRESENT_SRC_KHR};
}

void WindowPresenter::Present(Ticket drawn) {
    if (image_) {
        stale_ = !swapchain_.Present(*image_, drawn);
        image_.reset();
    }
}

}  // namespace ernie::app
