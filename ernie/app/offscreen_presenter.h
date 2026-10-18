// SPDX-License-Identifier: BSD-3-Clause
/// @file offscreen_presenter.h
/// @brief Frames drawn into an image without a window, for tests and headless runs.
#pragma once

#include "ernie/app/presenter.h"
#include "ernie/device/image.h"

namespace ernie::app {

/// @brief Draws every frame into one image of a fixed size; the UI gets no input.
class OffscreenPresenter final : public Presenter {
  public:
    OffscreenPresenter(DeviceContext& context, glm::uvec2 extent);

    VkFormat Format() const override {
        return ToVkFormat(image_.Description().format);
    }
    std::uint32_t ImageCount() const override {
        return 2;
    }
    void PollEvents() override {}
    bool CloseRequested() const override {
        return false;
    }
    void BeginUiFrame(double delta) override;
    std::optional<FrameTarget> Acquire() override;
    void Present(Ticket drawn) override {
        drawn_ = drawn;
    }

    /// @return The image frames are drawn into, in the general layout.
    const DeviceImage& Image() const {
        return image_;
    }
    /// @return The ticket of the last frame drawn.
    Ticket Drawn() const {
        return drawn_;
    }

  private:
    DeviceImage image_;
    Ticket drawn_;
};

}  // namespace ernie::app
