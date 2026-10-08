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
    glm::vec4 background{0.102f, 0.102f, 0.106f, 1.0f};  ///< Tab bar, menu bar, behind the docks.
    glm::vec4 panel{0.180f, 0.188f, 0.192f, 1.0f};       ///< Panel bodies.
    glm::vec4 header{0.235f, 0.243f, 0.259f, 1.0f};      ///< Section headers, inactive tabs.
    glm::vec4 field{0.137f, 0.141f, 0.149f, 1.0f};       ///< Input fields, combos, sliders.
    glm::vec4 fieldHovered{0.200f, 0.208f, 0.220f, 1.0f};
    glm::vec4 separator{0.102f, 0.102f, 0.106f, 1.0f};  ///< One-pixel lines between areas.
    glm::vec4 text{0.820f, 0.820f, 0.820f, 1.0f};
    glm::vec4 textMuted{0.600f, 0.608f, 0.604f, 1.0f};  ///< Labels and disabled text.
    glm::vec4 accent{0.290f, 0.522f, 0.678f, 1.0f};     ///< Selection, changed values, sliders.
    glm::vec4 warning{0.910f, 0.698f, 0.235f, 1.0f};
    glm::vec4 error{0.886f, 0.341f, 0.318f, 1.0f};
    glm::vec4 overlay{0.141f, 0.157f, 0.169f, 0.88f};  ///< Viewport toolbar pills and HUD.

    float fontSize = 14.0f;  ///< Text size in pixels at a content scale of 1.
    float iconSize = 15.0f;  ///< Icon size in pixels at a content scale of 1.
    float rounding = 3.0f;   ///< Corner radius of frames, tabs and popups.
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
