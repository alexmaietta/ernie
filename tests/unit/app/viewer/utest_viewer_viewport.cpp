// SPDX-License-Identifier: BSD-3-Clause
#include <gtest/gtest.h>

#include "ernie/app/viewer/viewport/viewport.h"

namespace ernie::app {
namespace {

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
