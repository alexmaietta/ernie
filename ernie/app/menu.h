// SPDX-License-Identifier: BSD-3-Clause
#pragma once

namespace ernie::app::menu {

/// Menu items as Omniverse Kit draws them, inside an open menu, popup
/// or combo: a gutter at the left holds each item's marker (a bar on
/// the chosen one of several, a checkmark on a toggle that is on),
/// shortcuts sit muted at the right, and sections open with a muted
/// title and a rule across the menu. Markers take the accent, the
/// style's `ImGuiCol_TextLink`.

/// A section's title: muted, with a rule to the menu's right edge.
void Section(const char* label);

/// One choice among several, marked with a bar in the gutter when
/// `chosen`. Returns whether it was clicked.
bool Option(const char* label, bool chosen,
            const char* shortcut = nullptr, bool enabled = true);

/// A toggle, marked with a checkmark in the gutter when `*on`, which
/// a click flips. Returns whether it was clicked.
bool Check(const char* label, bool* on,
           const char* shortcut = nullptr, bool enabled = true);

/// A plain action, aligned with the marked items. Returns whether it
/// was clicked.
bool Action(const char* label, const char* shortcut = nullptr,
            bool enabled = true);

/// A submenu's item, aligned with the marked items; when it returns
/// true, draw the submenu and call `ImGui::EndMenu`.
bool BeginSubmenu(const char* label, bool enabled = true);

}  // namespace ernie::app::menu
