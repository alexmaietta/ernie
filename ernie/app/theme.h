// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <glm/glm.hpp>

struct ImGuiStyle;

namespace ernie::app {

/// Colours (sRGB-encoded with alpha, as written to the screen) and
/// metrics of the viewer's UI. The defaults follow NVIDIA Omniverse
/// Kit's dark look (colours read from USD Composer), a shade darker
/// behind the docks: near-black tab bars and docking area, charcoal
/// panels and fields (Kit's #1F2123), section headers as lighter bars
/// (#323434), menus and the viewport's pills in #25282A, a blue-teal
/// hover (#284D5B) and Kit's bright blue (#33C2F9) for the chosen and
/// checked items, and segmented buttons lit in grey (#777777).
struct Theme {
    glm::vec4 background{
        0.071f, 0.075f, 0.078f,
        1.0f};  ///< Tab bar, menu bar, behind the docks.
    glm::vec4 panel{0.122f, 0.129f, 0.137f,
                    1.0f};  ///< Panel bodies (Kit's #1F2123).
    glm::vec4 header{0.196f, 0.204f, 0.204f,
                     1.0f};  ///< Section bars, table headers.
    glm::vec4 field{0.090f, 0.098f, 0.102f,
                    1.0f};  ///< Input fields and combos.
    glm::vec4 fieldHovered{0.169f, 0.180f, 0.188f, 1.0f};
    glm::vec4 button{0.173f, 0.180f, 0.188f,
                     1.0f};  ///< Buttons, a step off the panels.
    glm::vec4 buttonLit{0.467f, 0.467f, 0.467f,
                        1.0f};  ///< A chosen segment (#777777).
    glm::vec4 separator{0.055f, 0.059f, 0.063f,
                        1.0f};  ///< One-pixel lines between areas.
    glm::vec4 text{0.835f, 0.835f, 0.835f, 1.0f};
    glm::vec4 textMuted{0.580f, 0.592f, 0.600f,
                        1.0f};  ///< Labels, types and disabled text.
    glm::vec4 selection{
        0.157f, 0.302f, 0.357f,
        1.0f};  ///< Selected and hovered rows (Kit's #284D5B).
    glm::vec4 accent{
        0.200f, 0.761f, 0.976f,
        1.0f};  ///< Chosen items, checks, active tools (#33C2F9).
    glm::vec4 outline{1.0f, 0.62f, 0.16f,
                      1.0f};  ///< The viewport's selection outline.
    glm::vec4 warning{0.910f, 0.698f, 0.235f, 1.0f};
    glm::vec4 error{0.886f, 0.341f, 0.318f, 1.0f};
    glm::vec4 overlay{
        0.145f, 0.157f, 0.165f,
        0.94f};  ///< Viewport pills, menus, HUD (#25282A).
    glm::vec4 overlayBorder{0.247f, 0.255f, 0.259f,
                            1.0f};  ///< The pills' edge (#3F4142).
    glm::vec4 checkBox{0.580f, 0.592f, 0.600f,
                       1.0f};  ///< A checked box (Kit's grey).
    glm::vec4 checkMark{0.122f, 0.129f, 0.137f,
                        1.0f};  ///< The check in it.
    glm::vec4 border{0.055f, 0.059f, 0.063f,
                     1.0f};  ///< Frames' and popups' edges.

    float fontSize =
        14.0f;  ///< Text size in pixels at a content scale of 1.
    float iconSize =
        15.0f;  ///< Icon size in pixels at a content scale of 1.
    float rounding =
        3.0f;  ///< Corner radius of frames, tabs and popups.
    float pillRounding =
        6.0f;  ///< Corner radius of the viewport's toolbar pills.
    float frameBorder = 0.0f;  ///< Fields' and buttons' edge width,
                               ///< in pixels at a content scale of 1.
    glm::vec2 framePadding{6.0f, 3.0f};
    glm::vec2 itemSpacing{6.0f, 4.0f};
    float propertyLabelWidth =
        130.0f;  ///< Width of the label column in the Property panel.
};

/// Omniverse Kit's dark theme.
Theme DarkTheme();

/// A light theme with the dark theme's metrics and accent.
Theme LightTheme();

/// The default: a classic scientific desktop look, as Qt's and GTK's
/// of the 2000s that tools such as PANGU wear: light grey-green
/// windows (#D4D7D0), white sunken fields and raised buttons with
/// one-pixel edges, square corners, black text, a navy accent and
/// pale-blue selection.
Theme ClassicTheme();

/// Resets `style` and writes `theme`'s colours and metrics into it,
/// with sizes and fonts scaled by `contentScale`.
void ApplyTheme(const Theme& theme, float contentScale,
                ImGuiStyle& style);

}  // namespace ernie::app
