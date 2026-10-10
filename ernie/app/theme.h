// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <glm/glm.hpp>

struct ImGuiStyle;

namespace ernie::app {

/// Colours (sRGB-encoded with alpha, as written to the screen) and metrics of the viewer's UI. The
/// defaults follow NVIDIA Omniverse Kit's dark look: a near-black tab bar and docking area,
/// charcoal panels, darker input fields, grey labels with light values, section headers as
/// lighter bars, a steel-blue accent for selection and changed values, and dark translucent pills
/// over the viewport.
struct Theme {
    glm::vec4 background{0.090f, 0.094f, 0.098f, 1.0f};  ///< Tab bar, menu bar, behind the docks.
    glm::vec4 panel{0.122f, 0.129f, 0.137f, 1.0f};       ///< Panel bodies (Kit's #1F2123).
    glm::vec4 header{0.196f, 0.204f, 0.204f, 1.0f};      ///< Section bars, table headers.
    glm::vec4 field{0.165f, 0.173f, 0.184f, 1.0f};       ///< Input fields, combos, sliders.
    glm::vec4 fieldHovered{0.224f, 0.235f, 0.247f, 1.0f};
    glm::vec4 separator{0.071f, 0.075f, 0.078f, 1.0f};  ///< One-pixel lines between areas.
    glm::vec4 text{0.800f, 0.804f, 0.808f, 1.0f};
    glm::vec4 textMuted{0.557f, 0.573f, 0.580f, 1.0f};  ///< Labels, types and disabled text.
    glm::vec4 selection{0.176f, 0.220f, 0.247f, 1.0f};  ///< Selected rows (Kit's subtle blue-grey).
    glm::vec4 accent{0.290f, 0.522f, 0.678f, 1.0f};     ///< Changed values, active tools, sliders.
    glm::vec4 outline{1.0f, 0.62f, 0.16f, 1.0f};        ///< The viewport's selection outline.
    glm::vec4 warning{0.910f, 0.698f, 0.235f, 1.0f};
    glm::vec4 error{0.886f, 0.341f, 0.318f, 1.0f};
    glm::vec4 overlay{0.141f, 0.157f, 0.169f, 0.88f};  ///< Viewport toolbar pills and HUD.

    float fontSize = 14.0f;     ///< Text size in pixels at a content scale of 1.
    float iconSize = 15.0f;     ///< Icon size in pixels at a content scale of 1.
    float rounding = 3.0f;      ///< Corner radius of frames, tabs and popups.
    float pillRounding = 6.0f;  ///< Corner radius of the viewport's toolbar pills.
    glm::vec2 framePadding{6.0f, 3.0f};
    glm::vec2 itemSpacing{6.0f, 4.0f};
    float propertyLabelWidth = 130.0f;  ///< Width of the label column in the Property panel.
};

/// The default dark theme.
Theme DarkTheme();

/// A light theme with the same metrics and accent.
Theme LightTheme();

/// Resets `style` and writes `theme`'s colours and metrics into it, with sizes and fonts scaled by
/// `contentScale`.
void ApplyTheme(const Theme& theme, float contentScale, ImGuiStyle& style);

}  // namespace ernie::app
