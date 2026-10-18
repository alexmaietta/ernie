// SPDX-License-Identifier: BSD-3-Clause
/// @file bounds_test.cpp
/// @brief Tests of axis-aligned boxes.
#include "ernie/geometry/bounds.h"

#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>

namespace ernie {
namespace {

TEST(Box, StartsEmptyAndGrowsToHoldWhatItIsGiven) {
    Box box;
    EXPECT_TRUE(box.Empty());
    EXPECT_EQ(box.Size(), glm::dvec3(0.0));
    box.Extend(glm::dvec3(1.0, 2.0, 3.0)).Extend(glm::dvec3(-1.0, 0.0, 5.0));
    EXPECT_FALSE(box.Empty());
    EXPECT_EQ(box.min, glm::dvec3(-1.0, 0.0, 3.0));
    EXPECT_EQ(box.max, glm::dvec3(1.0, 2.0, 5.0));
    EXPECT_EQ(box.Center(), glm::dvec3(0.0, 1.0, 4.0));
    EXPECT_EQ(Box(box).Extend(Box()), box) << "an empty box adds nothing";
    EXPECT_EQ(box.Grown(1.0).Size(), glm::dvec3(4.0, 4.0, 4.0));
}

TEST(Box, TransformedBoxesHoldEveryTransformedCorner) {
    const Box box = Box::Around(glm::dvec3(0.0), glm::dvec3(1.0, 2.0, 3.0));
    const glm::dmat4 transform =
        glm::translate(glm::dmat4(1.0), glm::dvec3(10.0, 0.0, 0.0)) *
        glm::rotate(glm::dmat4(1.0), glm::radians(90.0), glm::dvec3(0.0, 0.0, 1.0));
    const Box moved = box.Transformed(transform);
    EXPECT_NEAR(moved.min.x, 8.0, 1e-12);
    EXPECT_NEAR(moved.max.x, 12.0, 1e-12);
    EXPECT_NEAR(moved.min.y, -1.0, 1e-12);
    EXPECT_NEAR(moved.max.z, 3.0, 1e-12);
    EXPECT_TRUE(Box().Transformed(transform).Empty());
}

}  // namespace
}  // namespace ernie
