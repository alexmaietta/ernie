// SPDX-License-Identifier: BSD-3-Clause
/// @file menu.h
/// @brief Menu items drawn as Omniverse Kit draws them, inside an open menu, popup or combo: a
/// left gutter holds markers (a bar on the chosen option, a check on a toggle) in the accent colour
/// (`ImGuiCol_TextLink`), and shortcuts sit muted at the right.
#pragma once

namespace ernie::app::menu {

/// @brief A section title: muted, with a rule to the right edge.
void Section(const char* label);

/// @brief One choice among several, barred in the gutter if chosen.
/// @param shortcut A shortcut to show, or null.
/// @return Whether it was clicked.
bool Option(const char* label, bool chosen, const char* shortcut = nullptr, bool enabled = true);

/// @brief A toggle, checked in the gutter when on; a click flips `on`.
/// @return Whether it was clicked.
bool Check(const char* label, bool* on, const char* shortcut = nullptr, bool enabled = true);

/// @brief A plain action, aligned with the marked items.
/// @return Whether it was clicked.
bool Action(const char* label, const char* shortcut = nullptr, bool enabled = true);

/// @brief A submenu item, at least `MinWidth` wide when open.
/// @return Whether it is open: draw it, then call `ImGui::EndMenu`.
bool BeginSubmenu(const char* label, bool enabled = true);

/// @return The least width Kit's menus open to: about 17 lines of the current font.
float MinWidth();

/// @brief Makes the next menu, popup or combo open at least `MinWidth` wide; call just before
/// `BeginMenu` or `BeginCombo`.
void NextMenuWide();

}  // namespace ernie::app::menu
