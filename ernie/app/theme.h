// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <glm/glm.hpp>

struct ImGuiStyle;

namespace ernie::app {

/// Colours (sRGB-encoded with alpha, as written to the screen) and
/// metrics of the viewer's UI: a classic scientific desktop look, as
/// Qt's and GTK's of the 2000s that tools such as PANGU wear (colours
/// read from PANGU v7's renderer GUI): light grey-green windows
/// (#D4D7D0), white sunken fields and raised buttons with one-pixel
/// edges, square corners, black text, a navy accent and pale-blue
/// selection.
struct Theme {
    glm::vec4 background{
        0.788f, 0.800f, 0.773f,
        1.0f};  ///< Tab bar, menu bar, behind the docks.
    glm::vec4 panel{0.831f, 0.843f, 0.816f,
                    1.0f};  ///< Panel bodies (#D4D7D0).
    glm::vec4 header{0.776f, 0.788f, 0.761f,
                     1.0f};  ///< Section bars, table headers.
    glm::vec4 field{1.0f, 1.0f, 1.0f,
                    1.0f};  ///< Input fields, combos.
    glm::vec4 fieldHovered{0.957f, 0.965f, 0.945f, 1.0f};
    glm::vec4 button{0.863f, 0.875f, 0.847f, 1.0f};  ///< Buttons.
    glm::vec4 buttonLit{0.698f, 0.761f, 0.835f,
                        1.0f};  ///< A pressed or chosen button.
    glm::vec4 separator{0.667f, 0.678f, 0.651f,
                        1.0f};  ///< Lines between areas (#AAADA6).
    glm::vec4 border{0.553f, 0.565f, 0.537f,
                     1.0f};  ///< Frames', tabs' and popups' edges.
    glm::vec4 text{0.0f, 0.0f, 0.0f, 1.0f};
    glm::vec4 textMuted{0.333f, 0.345f, 0.325f,
                        1.0f};  ///< Labels, types and disabled text.
    glm::vec4 selection{0.627f, 0.741f, 0.863f,
                        1.0f};  ///< Selected and hovered rows.
    glm::vec4 accent{0.122f, 0.290f, 0.541f,
                     1.0f};  ///< Chosen menu items, active tools.
    glm::vec4 outline{1.0f, 0.62f, 0.16f,
                      1.0f};  ///< The viewport's selection outline.
    glm::vec4 warning{0.910f, 0.698f, 0.235f, 1.0f};
    glm::vec4 error{0.886f, 0.341f, 0.318f, 1.0f};
    glm::vec4 overlay{
        0.831f, 0.843f, 0.816f,
        0.92f};  ///< The viewport's toolbar, menus, HUD.
    glm::vec4 overlayBorder{0.553f, 0.565f, 0.537f,
                            1.0f};  ///< The toolbar buttons' edge.
    glm::vec4 checkBox{1.0f, 1.0f, 1.0f, 1.0f};  ///< A checked box.
    glm::vec4 checkMark{0.0f, 0.0f, 0.0f,
                        1.0f};  ///< The check in it.

    float fontSize =
        14.0f;  ///< Text size in pixels at a content scale of 1.
    float iconSize =
        15.0f;  ///< Icon size in pixels at a content scale of 1.
    float rounding =
        0.0f;  ///< Corner radius of frames, tabs, popups.
    float pillRounding =
        0.0f;  ///< Corner radius of the viewport's toolbar buttons.
    float frameBorder = 1.0f;  ///< Fields' and buttons' edge width,
                               ///< in pixels at a content scale of 1.
    glm::vec2 framePadding{6.0f, 3.0f};
    glm::vec2 itemSpacing{6.0f, 4.0f};
    float propertyLabelWidth =
        130.0f;  ///< Width of the label column in the Property panel.
};

/// Resets `style` and writes `theme`'s colours and metrics into it,
/// with sizes and fonts scaled by `contentScale`.
void ApplyTheme(const Theme& theme, float contentScale,
                ImGuiStyle& style);

}  // namespace ernie::app
