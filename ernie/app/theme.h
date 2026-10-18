// SPDX-License-Identifier: BSD-3-Clause
/// @file theme.h
/// @brief The viewer UI's colours and metrics.
#pragma once

#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

struct ImGuiStyle;

namespace ernie::app {

/// @brief Colours and metrics of the viewer's UI: a classic scientific desktop look, after PANGU
/// v7's GUI, with grey-green windows, white sunken fields, one-pixel edges, square corners and a
/// navy accent. Colours are sRGB-encoded RGBA.
struct Theme {
    /// Tab bar, menu bar, behind the docks.
    glm::vec4 background{0.788f, 0.800f, 0.773f, 1.0f};
    /// Panel bodies.
    glm::vec4 panel{0.831f, 0.843f, 0.816f, 1.0f};
    /// Section bars, table headers.
    glm::vec4 header{0.776f, 0.788f, 0.761f, 1.0f};
    /// Input fields, combos.
    glm::vec4 field{1.0f, 1.0f, 1.0f, 1.0f};
    glm::vec4 fieldHovered{0.957f, 0.965f, 0.945f, 1.0f};
    glm::vec4 button{0.863f, 0.875f, 0.847f, 1.0f};
    /// A pressed or chosen button.
    glm::vec4 buttonLit{0.698f, 0.761f, 0.835f, 1.0f};
    /// Lines between areas.
    glm::vec4 separator{0.667f, 0.678f, 0.651f, 1.0f};
    /// Frames', tabs' and popups' edges.
    glm::vec4 border{0.553f, 0.565f, 0.537f, 1.0f};
    glm::vec4 text{0.0f, 0.0f, 0.0f, 1.0f};
    /// Labels, types and disabled text.
    glm::vec4 textMuted{0.333f, 0.345f, 0.325f, 1.0f};
    /// Selected and hovered rows.
    glm::vec4 selection{0.627f, 0.741f, 0.863f, 1.0f};
    /// Chosen menu items, active tools.
    glm::vec4 accent{0.122f, 0.290f, 0.541f, 1.0f};
    glm::vec4 warning{0.910f, 0.698f, 0.235f, 1.0f};
    glm::vec4 error{0.886f, 0.341f, 0.318f, 1.0f};
    /// Menus, popups and the viewport's heads-up display.
    glm::vec4 overlay{0.831f, 0.843f, 0.816f, 0.92f};
    glm::vec4 checkBox{1.0f, 1.0f, 1.0f, 1.0f};
    glm::vec4 checkMark{0.0f, 0.0f, 0.0f, 1.0f};
    /// The viewport's colour where nothing is rendered.
    glm::vec4 viewport{0.18f, 0.19f, 0.20f, 1.0f};
    /// Text size, in pixels at a content scale of 1.
    float fontSize = 14.0f;
    /// Corner radius of frames, tabs and popups, in pixels.
    float rounding = 0.0f;
    /// Fields' and buttons' edge width, in pixels at a content scale of 1.
    float frameBorder = 1.0f;
    /// Padding inside frames, in pixels.
    glm::vec2 framePadding{6.0f, 3.0f};
    /// Spacing between items, in pixels.
    glm::vec2 itemSpacing{6.0f, 4.0f};
};

/// @brief Resets an ImGui style and writes a theme into it.
/// @param contentScale Scales sizes and fonts, for high-density displays.
void ApplyTheme(const Theme& theme, float contentScale, ImGuiStyle& style);

}  // namespace ernie::app
