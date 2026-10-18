// SPDX-License-Identifier: BSD-3-Clause
/// @file widgets.h
/// @brief Small ImGui helpers the viewer's panels share.
#pragma once

#include <glm/vec4.hpp>
#include <imgui.h>

#include <cstdint>
#include <string>
#include <string_view>

namespace ernie::app {

inline ImVec4 ToImVec4(const glm::vec4& colour) {
    return {colour.r, colour.g, colour.b, colour.a};
}

/// @brief Shows `text` as the last item's tooltip while it is hovered, disabled items included,
/// wrapped to a narrow width; nothing when `text` is empty.
void Tooltip(std::string_view text);

/// @brief Draws a label and a value on one line, the value muted.
void LabelValue(std::string_view label, std::string_view value);

/// @return Bytes as a short size with a binary unit, such as "1.5 GiB".
std::string FormatBytes(std::uint64_t bytes);

}  // namespace ernie::app
