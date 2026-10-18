// SPDX-License-Identifier: BSD-3-Clause
/// @file viewer.h
/// @brief The interactive viewer: a window, or an offscreen image, showing the shell's panels
/// around the viewport, and the command line that opens it.
#pragma once

#include "ernie/app/imgui.h"
#include "ernie/app/ui.h"

#include <glm/vec2.hpp>

#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace ernie::app {

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
    /// Wait for the display's refresh; off by default, so frames show what they cost.
    bool vsync = false;
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
    const OffscreenPresenter* Offscreen() const {
        return dynamic_cast<const OffscreenPresenter*>(presenter_.get());
    }

  private:
    /// @return Whether to keep going.
    bool Step();

    ViewerOptions options_;
    // Made in this order, destroyed in reverse.
    ImGuiSession imgui_;
    std::unique_ptr<DeviceContext> context_;
    std::unique_ptr<Presenter> presenter_;
    std::unique_ptr<ImGuiRenderer> renderer_;
    std::unique_ptr<Shell> shell_;
    /// The frames the device may still be drawing, oldest first.
    std::deque<Ticket> inFlight_;
    std::uint64_t frames_ = 0;
    std::chrono::steady_clock::time_point last_ = std::chrono::steady_clock::now();
};

/// @brief What the viewer's command line, `ernie [view] [SCENE.json] [options]`, asks for.
struct ViewCommand {
    /// The scene file to open; none for an empty scene.
    std::optional<std::filesystem::path> scene;
    ViewerOptions viewer;
};

/// @return The options the viewer's command line usage lists.
std::string_view ViewUsage();

/// @brief Reads the viewer's arguments: an optional scene file and options.
/// @throws std::invalid_argument for an unknown option, a missing value or a second scene.
ViewCommand ParseViewCommand(std::span<const std::string_view> arguments);

/// @brief Opens the viewer on the command's scene and runs it until closed.
/// @return The process's exit status.
int RunView(const ViewCommand& command);

}  // namespace ernie::app
