// SPDX-License-Identifier: BSD-3-Clause
/// @file help_windows.h
/// @brief The Help menu's windows: the controls and About Ernie.
#pragma once

#include <span>
#include <string_view>

namespace ernie::app {

/// @brief A control and what it does, as the Keyboard and Mouse window lists it.
struct Control {
    const char* input;
    const char* action;
};

/// @return The viewer's mouse and keyboard controls.
std::span<const Control> Controls();

/// @brief Draws the Keyboard and Mouse window while `open`; its close button clears it.
void DrawControlsWindow(bool& open);

/// @brief Draws the About Ernie window while `open`; its close button clears it.
/// @param device The name of the device in use; empty for none.
void DrawAboutWindow(bool& open, std::string_view device);

}  // namespace ernie::app
