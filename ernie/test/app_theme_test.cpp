// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/theme.h"
#include "ernie/app/viewer.h"

#include <gtest/gtest.h>
#include <imgui.h>

using namespace ernie;
using namespace ernie::app;

TEST(ThemeTest, IsLightOpaqueAndFramedWithSquareCorners) {
    const Theme theme;
    for (const glm::vec4& colour :
         {theme.background, theme.panel, theme.header, theme.field}) {
        EXPECT_FLOAT_EQ(colour.a, 1.0f);
        EXPECT_GT(colour.g, 0.75f);
    }
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
    Theme theme;
    theme.accent = {0.1f, 0.2f, 0.3f, 1.0f};
    theme.rounding = 2.0f;
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
    ApplyTheme(Theme{}, 1.0f, style);
    EXPECT_FLOAT_EQ(style.Alpha, 1.0f);
}

TEST(ThemeTest, ViewerOptionsDefaultToAVisibleWindowWithTheTheme) {
    const ViewerOptions options;
    EXPECT_FALSE(options.hidden);
    EXPECT_TRUE(options.layout.empty());
    EXPECT_TRUE(options.settings.IsEmpty());
    EXPECT_EQ(options.theme.panel, Theme{}.panel);
}
