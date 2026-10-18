// SPDX-License-Identifier: BSD-3-Clause
/// @file hud.cpp
/// @brief Formatting and drawing the heads-up display.
#include "ernie/app/hud.h"

#include <array>
#include <cstdio>

namespace ernie::app {
namespace {

/// Seconds between refreshes of the figures.
constexpr double kRefresh = 0.25;

template <typename... Values> std::string Format(const char* format, Values... values) {
    std::array<char, 128> text{};
    std::snprintf(text.data(), text.size(), format, values...);
    return text.data();
}

}  // namespace

void Hud::Refresh(ImVec2 size, const OrbitCamera& camera) {
    const float fps = ImGui::GetIO().Framerate;
    const glm::dvec3 eye = camera.Eye();
    lines_ = {
        Format("%.0f fps, %.2f ms", static_cast<double>(fps),
               fps > 0.0f ? 1000.0 / static_cast<double>(fps) : 0.0),
        Format("%.0f x %.0f", static_cast<double>(size.x), static_cast<double>(size.y)),
        Format("Eye %.3g, %.3g, %.3g m", eye.x, eye.y, eye.z),
        Format("Target %.3g, %.3g, %.3g m", camera.target.x, camera.target.y, camera.target.z),
    };
}

void Hud::Draw(ImVec2 origin, ImVec2 size, const OrbitCamera& camera) {
    const double now = ImGui::GetTime();
    if (refreshed_ < 0.0 || now - refreshed_ >= kRefresh) {
        Refresh(size, camera);
        refreshed_ = now;
    }
    ImDrawList* draw = ImGui::GetBackgroundDrawList();
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec2 at(origin.x + style.WindowPadding.x, origin.y + style.WindowPadding.y);
    for (const std::string& line : lines_) {
        draw->AddText(at, IM_COL32(230, 232, 228, 255), line.c_str());
        at.y += ImGui::GetTextLineHeightWithSpacing();
    }
}

}  // namespace ernie::app
