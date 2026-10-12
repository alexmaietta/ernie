// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/menu.h"

#include <imgui.h>
#include <imgui_internal.h>

#include <algorithm>
#include <string>

namespace ernie::app::menu {

namespace {

/// Spaces before every label, where the markers go.
constexpr const char* kGutter = "    ";

std::string Indented(const char* label) {
    return std::string(kGutter) + label;
}

/// Where the last item's marker goes: the middle of its gutter.
ImVec2 MarkerCentre() {
    const ImVec2 low = ImGui::GetItemRectMin();
    const ImVec2 high = ImGui::GetItemRectMax();
    const float gutter = ImGui::CalcTextSize(kGutter).x;
    return {low.x + 0.5f * gutter, 0.5f * (low.y + high.y)};
}

}  // namespace

void Section(const char* label) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(
                                             ImGuiCol_TextDisabled));
    ImGui::SeparatorText(label);
    ImGui::PopStyleColor();
}

bool Option(const char* label, bool chosen, const char* shortcut,
            bool enabled) {
    const bool clicked = ImGui::MenuItem(Indented(label).c_str(),
                                         shortcut, false, enabled);
    if (chosen) {
        // Kit's bar: a short accent stroke down the gutter.
        const ImVec2 centre = MarkerCentre();
        const float half = 0.3f * ImGui::GetTextLineHeight();
        const float width = std::max(2.0f, 0.12f * half * 2.0f);
        ImGui::GetWindowDrawList()->AddRectFilled(
            ImVec2(centre.x - 0.5f * width, centre.y - half),
            ImVec2(centre.x + 0.5f * width, centre.y + half),
            ImGui::GetColorU32(ImGuiCol_TextLink));
    }
    return clicked;
}

bool Check(const char* label, bool* on, const char* shortcut,
           bool enabled) {
    const bool clicked = ImGui::MenuItem(Indented(label).c_str(),
                                         shortcut, false, enabled);
    if (clicked) {
        *on = !*on;
    }
    if (*on) {
        const ImVec2 centre = MarkerCentre();
        const float size = 0.7f * ImGui::GetTextLineHeight();
        ImGui::RenderCheckMark(
            ImGui::GetWindowDrawList(),
            ImVec2(centre.x - 0.5f * size, centre.y - 0.5f * size),
            ImGui::GetColorU32(ImGuiCol_TextLink), size);
    }
    return clicked;
}

bool Action(const char* label, const char* shortcut, bool enabled) {
    return ImGui::MenuItem(Indented(label).c_str(), shortcut, false,
                           enabled);
}

bool BeginSubmenu(const char* label, bool enabled) {
    NextMenuWide();
    return ImGui::BeginMenu(Indented(label).c_str(), enabled);
}

float MinWidth() {
    return 17.0f * ImGui::GetFontSize();
}

void NextMenuWide() {
    // A closed menu consumes the constraint unused, as Begin would.
    ImGui::SetNextWindowSizeConstraints(ImVec2(MinWidth(), 0.0f),
                                        ImVec2(FLT_MAX, FLT_MAX));
}

}  // namespace ernie::app::menu
