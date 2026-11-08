// SPDX-License-Identifier: BSD-3-Clause
#include <gtest/gtest.h>

#include "ernie/app/viewer/viewport/viewport.h"

namespace ernie::app {
namespace {

void ExpectExtent(Extent actual, uint32_t width, uint32_t height) {
    EXPECT_EQ(actual.width, width);
    EXPECT_EQ(actual.height, height);
}

TEST(ViewportTest, FitsSceneCamerasInsideTheViewport) {
    ExpectExtent(FitAspect({1600, 900}, 1.0), 900, 900);
    ExpectExtent(FitAspect({800, 1000}, 16.0 / 9), 800, 450);
    ExpectExtent(FitAspect({1280, 720}, 16.0 / 9), 1280, 720);
    ExpectExtent(FitAspect({640, 480}, 0), 640, 480);
}

TEST(ViewportTest, FindsTheAovsItShows) {
    const AovInfo* color = FindAov("LdrColor");
    ASSERT_NE(color, nullptr);
    EXPECT_EQ(color->kind, AovKind::Color);
    ASSERT_NE(FindAov("DistanceToCamera"), nullptr);
    EXPECT_STREQ(FindAov("DistanceToCamera")->units, "m");
    EXPECT_EQ(FindAov("ldrcolor"), nullptr);
    EXPECT_EQ(FindAov(""), nullptr);
    for (const AovInfo& aov : kAovs) {
        EXPECT_EQ(FindAov(aov.name), &aov);
    }
}

}
}
