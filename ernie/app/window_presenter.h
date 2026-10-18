// SPDX-License-Identifier: BSD-3-Clause
/// @file window_presenter.h
/// @brief Frames shown in a window, with its input going to the UI.
#pragma once

#include "ernie/app/presenter.h"
#include "ernie/app/swapchain.h"
#include "ernie/app/window.h"

#include <memory>

namespace ernie::app {

/// @brief Presents to a window's swapchain, made again whenever the window's size changes, and
/// feeds the window's input to ImGui, whose context must outlive it.
class WindowPresenter final : public Presenter {
  public:
    /// @param window Made before the device context, which needs its instance extensions.
    WindowPresenter(DeviceContext& context, std::unique_ptr<Window> window, bool vsync);
    ~WindowPresenter() override;

    VkFormat Format() const override {
        return swapchain_.Format();
    }
    std::uint32_t ImageCount() const override {
        return swapchain_.ImageCount();
    }
    void PollEvents() override {
        window_->PollEvents();
    }
    bool CloseRequested() const override {
        return window_->CloseRequested();
    }
    void BeginUiFrame(double delta) override;
    std::optional<FrameTarget> Acquire() override;
    void Present(Ticket drawn) override;

  private:
    std::unique_ptr<Window> window_;
    Swapchain swapchain_;
    std::optional<Swapchain::Image> image_;
    bool stale_ = false;
};

}  // namespace ernie::app
