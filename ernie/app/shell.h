// SPDX-License-Identifier: BSD-3-Clause
/// @file shell.h
/// @brief The viewer's UI: the menu bar, the docked panels around the viewport, the heads-up
/// display and the camera's controls. It draws ImGui windows only, so it runs without a device or
/// a window.
#pragma once

#include "ernie/app/dock_layout.h"
#include "ernie/app/hud.h"
#include "ernie/app/menu_bar.h"
#include "ernie/app/message_log.h"
#include "ernie/app/navigation.h"
#include "ernie/app/orbit_camera.h"
#include "ernie/app/selection.h"
#include "ernie/app/theme.h"

#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace ernie {
class DeviceContext;
class Scene;
}  // namespace ernie

namespace ernie::app {

struct ShellOptions {
    Theme theme;
    DockSizes docks;
    /// Record profiler zones every frame, not only while the Profiler panel shows.
    bool profile = false;
};

/// @brief The panels and controls over a scene. Panels read the scene; its owner keeps it updated
/// (`Scene::Update`) between frames.
class Shell {
  public:
    /// @param device The device in use, for the Devices panel; null for none.
    Shell(const Scene& scene, const DeviceContext* device, const ShellOptions& options = {});

    /// @brief Draws this frame's UI, between `ImGui::NewFrame` and `ImGui::Render`.
    void Draw();
    /// @return Whether the viewer should keep running: false once the user quits.
    bool Open() const {
        return open_;
    }
    void Close() {
        open_ = false;
    }
    /// @brief Aims the camera at the selection, or at the whole scene when nothing is selected.
    void Frame();

    /// @return The viewport's position and size on the screen: the dock space's central node.
    std::pair<ImVec2, ImVec2> Viewport() const {
        return layout_.Viewport();
    }
    std::span<const std::unique_ptr<Panel>> Panels() const {
        return panels_;
    }
    /// @return The panel titled `title`, or null.
    Panel* FindPanel(std::string_view title) const;
    MessageLog& Log() {
        return log_;
    }
    Selection& Selected() {
        return selection_;
    }
    OrbitCamera& Camera() {
        return camera_;
    }
    const Theme& GetTheme() const {
        return theme_;
    }

  private:
    void DrawPanels();
    void HandleShortcuts();
    void Navigate();
    void Apply(MenuChoices& choices);

    const Scene& scene_;
    const DeviceContext* device_;
    Theme theme_;
    MessageLog log_;
    Selection selection_;
    OrbitCamera camera_;
    Navigation navigation_;
    Hud hud_;
    DockLayout layout_;
    MenuChoices menu_;
    std::vector<std::unique_ptr<Panel>> panels_;
    bool open_ = true;
};

}  // namespace ernie::app
