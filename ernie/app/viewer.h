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
    /// Where `RunView` saves the last frame drawn offscreen as a PNG; empty for nowhere.
    std::filesystem::path screenshot;
    ShellOptions shell;
};

/// @brief The viewer on a scene. Each frame takes input, draws the shell's UI, traces the orbit
/// camera into the viewport (or clears it to the theme's colour on devices without ray queries),
/// draws the UI over it and presents it. Drive it on the thread that made it.
class Viewer {
  public:
    /// @param scene Shown by the panels and edited through the menus, updated every frame; must
    /// outlive the viewer.
    /// @throws std::runtime_error without a display unless offscreen, DeviceError without a
    /// device.
    Viewer(Scene& scene, const ViewerOptions& options = {});
    ~Viewer();
    Viewer(const Viewer&) = delete;
    Viewer& operator=(const Viewer&) = delete;

    /// @brief Draws frames until the window closes, the user quits or the frame count is reached.
    void Run();

    std::uint64_t Frames() const {
        return frames_;
    }
    /// @return Whether eco mode skipped rendering the last frame.
    bool EcoPaused() const {
        return ecoPaused_;
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
    /// @brief Saves the last frame drawn offscreen as a PNG.
    /// @throws std::logic_error when presenting to a window; std::runtime_error as `WritePng`.
    void SaveScreenshot(const std::filesystem::path& path);

  private:
    /// @return Whether to keep going.
    bool Step();
    /// @brief Sizes the viewport's image to the dock space's central node and draws it there,
    /// behind the panels.
    void ShowViewport();
    /// @brief Makes the viewport's image `pixels` in size, in the UI and the heap, with its prims.
    void ResizeViewport(glm::uvec2 pixels);
    /// @brief Selects the prim the frame `drawn` shows under a click.
    void Pick(const Shell::Pick& pick, const Ticket& drawn);
    /// @return Whether eco mode skips rendering this frame: nothing it renders from has changed
    /// for the frames the shell's display asks.
    bool EcoPause();
    /// @brief Adds the pass filling the viewport: traced, or the theme's colour without ray
    /// queries.
    /// @return The viewport, filled.
    graph::Image AddViewportPass(graph::Graph& graph);
    /// @brief Adds the frame's own passes: rendering the viewport, then the UI over the target.
    void AddPasses(graph::Graph& graph, ImDrawData& data, const FrameTarget& target);

    ViewerOptions options_;
    Scene& scene_;
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
    /// Each viewport pixel's prim slot, which picking reads; null without ray queries.
    std::unique_ptr<DeviceColumn> prims_;
    /// What a frame renders from; eco mode counts the frames it stays the same.
    struct ViewState {
        std::uint64_t version = 0;
        glm::dmat4 camera{1.0};
        double fov = 0.0;
        glm::uvec2 size{0};
        bool operator==(const ViewState&) const = default;
    };
    std::optional<ViewState> lastView_;
    std::uint64_t stillFrames_ = 0;
    bool ecoPaused_ = false;
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
