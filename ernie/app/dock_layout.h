// SPDX-License-Identifier: BSD-3-Clause
/// @file dock_layout.h
/// @brief The viewer's default docking layout: panels around a central viewport.
#pragma once

#include "ernie/app/panel.h"

#include <imgui.h>

#include <map>
#include <memory>
#include <span>
#include <utility>

namespace ernie::app {

/// @brief The default layout's dock sizes, as fractions of the space the docks placed before them
/// leave (left, then the right column, then the bottom).
struct DockSizes {
    float left = 0.20f;
    float right = 0.26f;
    /// `Dock::RightLower`'s share of the right column's height.
    float rightLower = 0.45f;
    float bottom = 0.25f;
};

/// @brief Docks panels around the dock space's central node, which the viewport shows through.
/// The default layout is built on first use and after `Reset`; panels shown later dock where they
/// ask.
class DockLayout {
  public:
    explicit DockLayout(const DockSizes& sizes = {}) : sizes_(sizes) {}

    /// @brief Draws the dock space over the main viewport, building the default layout first when
    /// due. Call before drawing the panels.
    void Draw(std::span<const std::unique_ptr<Panel>> panels);
    /// @brief Docks the next window where `panel` docks, when it first appears.
    void DockNext(const Panel& panel) const;
    /// @brief Builds the default layout again on the next `Draw`.
    void Reset() {
        built_ = false;
    }

    /// @return The central node's position and size on the screen, where the viewport shows.
    std::pair<ImVec2, ImVec2> Viewport() const;
    ImGuiID DockSpace() const;

  private:
    void Build(ImGuiID dockSpace, std::span<const std::unique_ptr<Panel>> panels);

    DockSizes sizes_;
    bool built_ = false;
    std::map<Dock, ImGuiID> nodes_;
};

}  // namespace ernie::app
