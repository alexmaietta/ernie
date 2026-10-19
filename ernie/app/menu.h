// SPDX-License-Identifier: BSD-3-Clause
/// @file menu.h
/// @brief Menu items as Omniverse Kit draws them: a left gutter holds a check on toggles in the
/// accent colour (`ImGuiCol_TextLink`), shortcuts sit muted at the right, and menus open at least
/// 17 lines wide.
#pragma once

#include <imgui.h>

#include <cfloat>
#include <string>

namespace ernie::app {

/// @return Whether the plain item was clicked.
bool MenuAction(const char* label, const char* shortcut = nullptr, bool enabled = true);

/// @brief A toggle, checked in the gutter when on; a click flips `on`.
void MenuCheck(const char* label, bool& on);

/// @brief A section heading inside a menu, muted, before its items.
void MenuSection(const char* label);

/// @brief Draws a menu titled `name` whose items `items` draws, while it is open.
template <typename Items> void Menu(const char* name, Items&& items) {
    // A menu that stays closed leaves the constraint unused.
    ImGui::SetNextWindowSizeConstraints(ImVec2(17.0f * ImGui::GetFontSize(), 0.0f),
                                        ImVec2(FLT_MAX, FLT_MAX));
    if (ImGui::BeginMenu(name)) {
        items();
        ImGui::EndMenu();
    }
}

/// @brief Opens `target`, a URL or a folder, in the desktop's own application, in the background.
/// @return Whether that could start.
bool OpenExternally(const std::string& target);

}  // namespace ernie::app
