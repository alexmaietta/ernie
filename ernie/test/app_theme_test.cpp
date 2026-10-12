// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/theme.h"
#include "ernie/app/viewer.h"

#include <gtest/gtest.h>
#include <imgui.h>

namespace ernie::app {
namespace {}  // namespace
}  // namespace ernie::app

using namespace ernie;
using namespace ernie::app;

TEST(ThemeTest, DarkThemeHasDarkOpaquePanelsAndBrighterText) {
    const Theme theme = DarkTheme();
    for (const glm::vec4& colour :
         {theme.background, theme.panel, theme.header, theme.field}) {
        EXPECT_FLOAT_EQ(colour.a, 1.0f);
        EXPECT_LT(colour.r + colour.g + colour.b, 1.0f);
    }
    EXPECT_GT(theme.text.r, theme.panel.r);
    EXPECT_LT(theme.overlay.a, 1.0f);
}

TEST(ThemeTest, LightThemeInvertsContrastAndKeepsAccentAndMetrics) {
    const Theme dark = DarkTheme();
    const Theme light = LightTheme();
    EXPECT_GT(light.panel.r, 0.5f);
    EXPECT_LT(light.text.r, light.panel.r);
    EXPECT_EQ(light.accent, dark.accent);
    EXPECT_EQ(light.fontSize, dark.fontSize);
    EXPECT_EQ(light.framePadding, dark.framePadding);
}

TEST(ThemeTest, ClassicThemeIsLightAndFramedWithSquareCorners) {
    const Theme theme = ClassicTheme();
    EXPECT_GT(theme.panel.g, 0.8f);
    EXPECT_EQ(theme.text, glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    EXPECT_EQ(theme.field, glm::vec4(1.0f));
    EXPECT_EQ(theme.rounding, 0.0f);
    ImGuiStyle style;
    ApplyTheme(theme, 1.0f, style);
    EXPECT_FLOAT_EQ(style.FrameBorderSize, 1.0f);
    EXPECT_FLOAT_EQ(style.TabBorderSize, 1.0f);
    EXPECT_FLOAT_EQ(style.Colors[ImGuiCol_CheckMark].x, 0.0f)
        << "a black check in a white box";
}

TEST(ThemeTest, ApplyThemeWritesColoursAndScalesMetrics) {
    Theme theme = DarkTheme();
    theme.accent = {0.1f, 0.2f, 0.3f, 1.0f};
    ImGuiStyle style;
    ApplyTheme(theme, 2.0f, style);
    EXPECT_FLOAT_EQ(style.Colors[ImGuiCol_WindowBg].x, theme.panel.r);
    EXPECT_FLOAT_EQ(style.Colors[ImGuiCol_TextLink].z, 0.3f)
        << "menus mark their items in the accent";
    EXPECT_FLOAT_EQ(style.Colors[ImGuiCol_Text].y, theme.text.g);
    EXPECT_FLOAT_EQ(style.FramePadding.x,
                    theme.framePadding.x * 2.0f);
    EXPECT_FLOAT_EQ(style.FrameRounding, theme.rounding * 2.0f);
    EXPECT_FLOAT_EQ(style.FontSizeBase, theme.fontSize);
    EXPECT_FLOAT_EQ(style.FontScaleDpi, 2.0f);
}

TEST(ThemeTest, ApplyThemeResetsEarlierStyleChanges) {
    ImGuiStyle style;
    style.Alpha = 0.25f;
    ApplyTheme(DarkTheme(), 1.0f, style);
    EXPECT_FLOAT_EQ(style.Alpha, 1.0f);
}

TEST(ThemeTest,
     ViewerOptionsDefaultToAVisibleWindowWithTheClassicTheme) {
    const ViewerOptions options;
    EXPECT_FALSE(options.hidden);
    EXPECT_TRUE(options.layout.empty());
    EXPECT_TRUE(options.settings.IsEmpty());
    EXPECT_EQ(options.theme.panel, ClassicTheme().panel);
}
