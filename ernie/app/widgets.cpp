// SPDX-License-Identifier: BSD-3-Clause
/// @file widgets.cpp
/// @brief Shared ImGui helpers.
#include "ernie/app/widgets.h"

#include <array>
#include <cstdio>
#include <string>

namespace ernie::app {

void Tooltip(std::string_view text) {
    if (text.empty() ||
        !ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip | ImGuiHoveredFlags_AllowWhenDisabled)) {
        return;
    }
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28.0f);
    ImGui::TextUnformatted(text.data(), text.data() + text.size());
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
}

void LabelValue(std::string_view label, std::string_view value) {
    ImGui::TextUnformatted(label.data(), label.data() + label.size());
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::TextUnformatted(value.data(), value.data() + value.size());
    ImGui::PopStyleColor();
}

std::string FormatBytes(std::uint64_t bytes) {
    constexpr std::array kUnits{"B", "KiB", "MiB", "GiB", "TiB"};
    auto value = static_cast<double>(bytes);
    std::size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < kUnits.size()) {
        value /= 1024.0;
        ++unit;
    }
    std::array<char, 32> text{};
    std::snprintf(text.data(), text.size(), unit == 0 ? "%.0f %s" : "%.1f %s", value, kUnits[unit]);
    return text.data();
}

}  // namespace ernie::app
