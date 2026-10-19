// SPDX-License-Identifier: BSD-3-Clause
/// @file menu.cpp
/// @brief The viewer's menu items, and opening links in the desktop's applications.
#include "ernie/app/menu.h"

#include <imgui_internal.h>

#include <cstdlib>

namespace ernie::app {
namespace {

/// Spaces before every label, where the markers go.
constexpr const char* kGutter = "    ";

}  // namespace

bool MenuAction(const char* label, const char* shortcut, bool enabled) {
    return ImGui::MenuItem((std::string(kGutter) + label).c_str(), shortcut, false, enabled);
}

void MenuCheck(const char* label, bool& on) {
    if (MenuAction(label)) {
        on = !on;
    }
    if (on) {
        const ImVec2 low = ImGui::GetItemRectMin();
        const ImVec2 high = ImGui::GetItemRectMax();
        const float size = 0.7f * ImGui::GetTextLineHeight();
        const ImVec2 centre(low.x + 0.5f * ImGui::CalcTextSize(kGutter).x, 0.5f * (low.y + high.y));
        ImGui::RenderCheckMark(ImGui::GetWindowDrawList(),
                               ImVec2(centre.x - 0.5f * size, centre.y - 0.5f * size),
                               ImGui::GetColorU32(ImGuiCol_TextLink), size);
    }
}

void MenuSection(const char* label) {
    ImGui::TextDisabled("%s", label);
}

bool OpenExternally(const std::string& target) {
#if defined(_WIN32)
    const std::string command = "start \"\" \"" + target + "\"";
#elif defined(__APPLE__)
    const std::string command = "open '" + target + "' &";
#else
    const std::string command = "xdg-open '" + target + "' >/dev/null 2>&1 &";
#endif
    return std::system(command.c_str()) == 0;
}

}  // namespace ernie::app
