// SPDX-License-Identifier: BSD-3-Clause
/// @file viewer.h
/// @brief The interactive viewer: a window, or an offscreen image, showing the shell's panels
/// around the viewport.
#pragma once

#include "ernie/app/shell.h"
#include "ernie/device/queue.h"

#include <glm/vec2.hpp>

#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>

namespace ernie {
class DeviceContext;
class Scene;
}  // namespace ernie

namespace ernie::app {

class ImGuiRenderer;
class ImGuiSession;
class OffscreenPresenter;
class Presenter;

struct ViewerOptions {
    std::string title = "Ernie";
    /// The window's or the offscreen image's size, in pixels.
    glm::uvec2 size{1600, 900};
    /// Draw into an image without a window, for tests and machines without a display.
    bool offscreen = false;
    /// Frames to draw before `Run` returns; 0 runs until the window closes.
    std::uint64_t frames = 0;
    /// The device: its UUID in hex or part of its name; empty for the best.
    std::string device;
    /// The Khronos validation layer, when installed.
    bool validation = false;
    /// Wait for the display's refresh.
    bool vsync = true;
    ShellOptions shell;
};

/// @brief The viewer on a scene. Each frame takes input, draws the shell's UI, clears the frame
/// to the theme's viewport colour, draws the UI over it and presents it. Drive it on the thread
/// that made it.
class Viewer {
  public:
    /// @param scene Read by the panels; must outlive the viewer.
    /// @throws std::runtime_error without a display unless offscreen, DeviceError without a
    /// device.
    Viewer(const Scene& scene, const ViewerOptions& options = {});
    ~Viewer();
    Viewer(const Viewer&) = delete;
    Viewer& operator=(const Viewer&) = delete;

    /// @brief Draws frames until the window closes, the user quits or the frame count is reached.
    void Run();
    /// @brief Draws and presents one frame.
    /// @return Whether to keep going.
    bool Step();

    std::uint64_t Frames() const {
        return frames_;
    }
    Shell& GetShell() {
        return *shell_;
    }
    DeviceContext& Context() {
        return *context_;
    }
    /// @return The offscreen presenter, or null when presenting to a window.
    const OffscreenPresenter* Offscreen() const;

  private:
    ViewerOptions options_;
    // Made in this order, destroyed in reverse.
    std::unique_ptr<ImGuiSession> imgui_;
    std::unique_ptr<DeviceContext> context_;
    std::unique_ptr<Presenter> presenter_;
    std::unique_ptr<ImGuiRenderer> renderer_;
    std::unique_ptr<Shell> shell_;
    /// The frames the device may still be drawing, oldest first.
    std::deque<Ticket> inFlight_;
    std::uint64_t frames_ = 0;
    std::chrono::steady_clock::time_point last_ = std::chrono::steady_clock::now();
};

}  // namespace ernie::app
