// SPDX-License-Identifier: BSD-3-Clause
/// @file panel.h
/// @brief Dockable viewer panels.
#pragma once

#include <cstdint>
#include <string>
#include <utility>

namespace ernie::app {

/// @brief Where a panel docks in the default layout.
enum class Dock : std::uint8_t {
    Left,
    /// The right column's upper part; all of it when nothing docks below.
    Right,
    /// The right column's lower part.
    RightLower,
    Bottom,
    /// Not docked: a free window.
    Floating,
};

/// @brief A dockable window. Panels take what they show and edit in their constructors, and draw
/// with Dear ImGui in `Draw`, between the window's `Begin` and `End`.
class Panel {
  public:
    /// @param title Its title, also its id in saved layouts.
    Panel(std::string title, Dock dock, bool visible = true)
        : title_(std::move(title)), dock_(dock), visible_(visible) {}
    virtual ~Panel() = default;
    Panel(const Panel&) = delete;
    Panel& operator=(const Panel&) = delete;

    const std::string& Title() const {
        return title_;
    }
    /// @return Where it docks in the default layout; a saved layout overrides it.
    Dock DefaultDock() const {
        return dock_;
    }
    bool Visible() const {
        return visible_;
    }
    void SetVisible(bool visible) {
        visible_ = visible;
    }

    virtual void Draw() = 0;
    /// @brief Called on frames it is not drawn: closed, or a tab behind another.
    virtual void Undrawn() {}

  private:
    std::string title_;
    Dock dock_;
    bool visible_;
};

}  // namespace ernie::app
