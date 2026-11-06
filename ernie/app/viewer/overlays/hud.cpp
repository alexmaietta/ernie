// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/viewer/overlays/hud.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <optional>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace ernie::app {
namespace {

constexpr double kRefreshSeconds = 0.25;

}

std::optional<uint64_t> ResidentBytes() {
#if defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    uint64_t size = 0;
    uint64_t resident = 0;
    if (statm >> size >> resident) {
        return resident * uint64_t(sysconf(_SC_PAGESIZE));
    }
#endif
    return std::nullopt;
}

float Hud::Draw(ImDrawList& list, ImVec2 topRight, const Theme& theme,
                const std::function<std::vector<std::string>()>& lines) {
    const double now = ImGui::GetTime();
    if (m_refreshed < 0 || now - m_refreshed >= kRefreshSeconds) {
        m_refreshed = now;
        m_lines = lines();
    }
    if (m_lines.empty()) {
        return topRight.x;
    }

    const float scale = ImGui::GetStyle().FontScaleDpi;
    const float pad = 6.0f * scale;
    const float line = ImGui::GetTextLineHeight();
    float width = 0;
    for (const std::string& text : m_lines) {
        width = std::max(width, ImGui::CalcTextSize(text.c_str()).x);
    }
    const ImVec2 corner(topRight.x - width - 2 * pad, topRight.y);
    list.AddRectFilled(corner, {topRight.x, corner.y + line * float(m_lines.size()) + 2 * pad},
                       ImGui::GetColorU32(WithAlpha(theme.overlay, theme.overlay.w * 0.85f)),
                       4.0f * scale);
    float y = corner.y + pad;
    for (const std::string& text : m_lines) {
        const size_t colon = text.find(": ");
        const std::string label = text.substr(0, colon == std::string::npos ? 0 : colon + 2);
        const float x = topRight.x - pad - ImGui::CalcTextSize(text.c_str()).x;
        list.AddText({x, y}, ImGui::GetColorU32(theme.textMuted), label.c_str());
        list.AddText({x + ImGui::CalcTextSize(label.c_str()).x, y}, ImGui::GetColorU32(theme.text),
                     text.c_str() + label.size());
        y += line;
    }
    return corner.x;
}

}
