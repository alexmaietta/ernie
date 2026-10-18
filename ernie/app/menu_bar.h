// SPDX-License-Identifier: BSD-3-Clause
/// @file menu_bar.h
/// @brief The viewer's main menu bar.
#pragma once

#include "ernie/app/panel.h"

#include <memory>
#include <span>

namespace ernie::app {

/// @brief What the menus ask for this frame, and the help windows they show.
struct MenuChoices {
    bool quit = false;
    bool resetLayout = false;
    bool frame = false;
    bool clearSelection = false;
    bool controlsShown = false;
    bool aboutShown = false;
    bool demoShown = false;

    /// @brief Forgets this frame's requests, keeping which windows are shown.
    void ClearRequests() {
        quit = resetLayout = frame = clearSelection = false;
    }
};

/// @brief Draws File, Edit, Window and Help, showing and hiding `panels` from Window.
void DrawMenuBar(std::span<const std::unique_ptr<Panel>> panels, bool selected,
                 MenuChoices& choices);

}  // namespace ernie::app
