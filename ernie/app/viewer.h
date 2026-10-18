// SPDX-License-Identifier: BSD-3-Clause
/// @file viewer.h
/// @brief The interactive viewer: a window, or an offscreen image, showing the shell's panels
/// around the viewport, and the command line that opens it.
#pragma once

#include "ernie/app/imgui_backend.h"
#include "ernie/app/ui.h"
#include "ernie/device/bindless.h"
#include "ernie/render/device_scene.h"
#include "ernie/render/rt/camera_pass.h"

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

/// @brief The viewer on a scene. Each frame takes input, draws the shell's UI, traces the orbit
/// camera into the viewport (or clears it to the theme's colour on devices without ray queries),
/// draws the UI over it and presents it. Drive it on the thread that made it.
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
    /// @brief Sizes the viewport's image to the dock space's central node and draws it there,
    /// behind the panels.
    void ShowViewport();
    /// @brief Makes the viewport's image `pixels` in size, in the UI and the heap.
    void ResizeViewport(glm::uvec2 pixels);
    /// @brief Adds the pass filling the viewport: traced, or the theme's colour without ray
    /// queries.
    /// @return The viewport, filled.
    graph::Image AddViewportPass(graph::Graph& graph);
    /// @brief Adds the frame's own passes: rendering the viewport, then the UI over the target.
    void AddPasses(graph::Graph& graph, ImDrawData& data, const FrameTarget& target);

    ViewerOptions options_;
    // Made in this order, destroyed in reverse.
    ImGuiSession imgui_;
    std::unique_ptr<DeviceContext> context_;
    std::unique_ptr<Presenter> presenter_;
    std::unique_ptr<ImGuiRenderer> renderer_;
    std::unique_ptr<Shell> shell_;
    std::unique_ptr<Arenas> arenas_;
    std::unique_ptr<DeviceScene> deviceScene_;
    /// The heap and pass tracing the viewport; null without ray queries.
    std::unique_ptr<BindlessHeap> heap_;
    std::unique_ptr<CameraPass> camera_;
    /// The image the viewport is rendered into, the UI's texture of it, and its storage slot.
    std::unique_ptr<Sampler> sampler_;
    std::unique_ptr<DeviceImage> viewport_;
    ImTextureID viewportTexture_ = 0;
    BindlessIndex viewportImage_;
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
