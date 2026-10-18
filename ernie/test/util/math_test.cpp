// SPDX-License-Identifier: BSD-3-Clause
/// @file math_test.cpp
/// @brief Tests of the shared scalar functions.
#include "ernie/util/math.h"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

namespace ernie {
namespace {

TEST(Math, FractAndModWrapNegativeNumbersUpward) {
    EXPECT_DOUBLE_EQ(Fract(2.25), 0.25);
    EXPECT_DOUBLE_EQ(Fract(-0.25), 0.75);
    EXPECT_DOUBLE_EQ(Mod(7.0, 3.0), 1.0);
    EXPECT_DOUBLE_EQ(Mod(-1.0, 3.0), 2.0);
}

TEST(Math, SmoothStepIsClampedAndSymmetric) {
    EXPECT_EQ(SmoothStep(1.0, 2.0, 0.0), 0.0);
    EXPECT_EQ(SmoothStep(1.0, 2.0, 3.0), 1.0);
    EXPECT_DOUBLE_EQ(SmoothStep(1.0, 2.0, 1.5), 0.5);
    EXPECT_NEAR(SmoothStep(0.0, 1.0, 0.2) + SmoothStep(0.0, 1.0, 0.8), 1.0, 1e-15);
}

TEST(Math, InterpolateIsLinearBetweenSamplesAndConstantBeyond) {
    const std::vector<double> xs{1.0, 2.0, 4.0};
    const std::vector<double> ys{10.0, 20.0, 0.0};
    EXPECT_DOUBLE_EQ(Interpolate<double>(xs, ys, 1.5), 15.0);
    EXPECT_DOUBLE_EQ(Interpolate<double>(xs, ys, 3.0), 10.0);
    EXPECT_DOUBLE_EQ(Interpolate<double>(xs, ys, 0.0), 10.0);
    EXPECT_DOUBLE_EQ(Interpolate<double>(xs, ys, 9.0), 0.0);
    EXPECT_DOUBLE_EQ(Interpolate<double>({}, {}, 1.0), 0.0);
}

TEST(Math, BracketFindsTheIntervalAndClampsBeyondTheEnds) {
    const std::vector<double> xs{1.0, 2.0, 2.0, 4.0};
    EXPECT_EQ(Bracket(xs, 1.25), std::pair(std::size_t{0}, 0.25));
    EXPECT_EQ(Bracket(xs, 3.0), std::pair(std::size_t{2}, 0.5));
    EXPECT_EQ(Bracket(xs, 0.0), std::pair(std::size_t{0}, 0.0));
    EXPECT_EQ(Bracket(xs, 9.0), std::pair(std::size_t{2}, 1.0));
    EXPECT_EQ(Bracket(std::vector<double>{1.0, 1.0}, 1.0), std::pair(std::size_t{0}, 0.0))
        << "an interval of no width";
}

TEST(Math, CosineSpacingSpansTheUnitIntervalDenselyAtItsEnds) {
    EXPECT_DOUBLE_EQ(CosineSpacing(0, 9), 0.0);
    EXPECT_DOUBLE_EQ(CosineSpacing(8, 9), 1.0);
    EXPECT_NEAR(CosineSpacing(4, 9), 0.5, 1e-15);
    EXPECT_LT(CosineSpacing(1, 9) - CosineSpacing(0, 9), CosineSpacing(5, 9) - CosineSpacing(4, 9));
}

TEST(Math, TrilinearMatchesCornersAndIsLinearAlongEdges) {
    std::array<double, 8> corners{};
    for (int i = 0; i < 8; ++i) {
        corners[static_cast<std::size_t>(i)] = i;  // = x + 2y + 4z at the corners.
    }
    EXPECT_DOUBLE_EQ(Trilinear(corners, glm::dvec3(0.0)), 0.0);
    EXPECT_DOUBLE_EQ(Trilinear(corners, glm::dvec3(1.0)), 7.0);
    EXPECT_DOUBLE_EQ(Trilinear(corners, glm::dvec3(0.25, 0.5, 0.75)), 0.25 + 1.0 + 3.0);
}

}  // namespace
}  // namespace ernie
