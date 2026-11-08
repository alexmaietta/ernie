// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/viewer/widgets/widgets.h"

#include <algorithm>
#include <cctype>

#include <fmt/format.h>
#include <imgui.h>
#include <imgui_internal.h>

namespace ernie::app {

std::string FormatBytes(uint64_t bytes) {
    const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = double(bytes);
    int unit = 0;
    while (value >= 1024.0 && unit < 4) {
        value /= 1024.0;
        ++unit;
    }
    return unit == 0 ? fmt::format("{:.0f} {}", value, units[unit])
                     : fmt::format("{:.1f} {}", value, units[unit]);
}

std::string Lowercase(std::string_view text) {
    std::string lower(text);
    std::ranges::transform(lower, lower.begin(), [](unsigned char c) { return std::tolower(c); });
    return lower;
}

void Row(const char* label, const std::string& value) {
    ImGui::TextDisabled("%s", label);
    ImGui::SameLine(9.0f * ImGui::GetFontSize());
    ImGui::TextUnformatted(value.c_str());
}

bool RowButton(const char* icon, const char* tip) {
    const ImVec2 low = ImGui::GetItemRectMin();
    const ImVec2 high = ImGui::GetItemRectMax();
    const float size = high.y - low.y;
    const ImVec2 after = ImGui::GetCursorScreenPos();
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    const ImVec2 extent = window->DC.CursorMaxPos;
    ImGui::SetCursorScreenPos({high.x - size, low.y});
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {0, 0});
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
    const bool clicked = ImGui::Button(icon, {size, size});
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
    ImGui::SetItemTooltip("%s", tip);
    window->DC.CursorMaxPos = extent;
    window->DC.CursorPos = after;
    window->DC.IsSetPos = false;
    if (clicked) {
        ImGui::CloseCurrentPopup();
    }
    return clicked;
}

}
