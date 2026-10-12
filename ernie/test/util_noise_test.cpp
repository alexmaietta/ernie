// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/noise.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

namespace ernie {
namespace {

TEST(Noise, IsZeroOnTheLatticeAndBoundedBetween) {
    const GradientNoise noise(7);
    EXPECT_EQ(noise(glm::dvec3(3.0, -2.0, 5.0)), 0.0);
    double low = 0.0;
    double high = 0.0;
    for (int i = 0; i < 4000; ++i) {
        const glm::dvec3 p(0.173 * i, 0.311 * i - 40.0, 0.057 * i);
        const double value = noise(p);
        low = std::min(low, value);
        high = std::max(high, value);
    }
    EXPECT_GE(low, -1.1);
    EXPECT_LE(high, 1.1);
    EXPECT_LT(low, -0.3) << "it varies";
    EXPECT_GT(high, 0.3);
}

TEST(Noise, IsSmoothAndFixedBySeed) {
    const GradientNoise noise(1);
    const glm::dvec3 p(1.3, 2.7, -0.4);
    EXPECT_NEAR(noise(p), noise(p + glm::dvec3(1e-6)), 1e-5);
    EXPECT_EQ(noise(p), GradientNoise(1)(p));
    EXPECT_NE(noise(p), GradientNoise(2)(p));
}

TEST(Noise, FbmAddsOctavesWithinTheSameRange) {
    const GradientNoise noise(3);
    const glm::dvec3 p(0.4, 1.9, 2.2);
    EXPECT_EQ(noise.Fbm(p, 1), noise(p));
    EXPECT_NEAR(noise.Fbm(p, 2, 2.0, 0.5),
                (noise(p) + 0.5 * noise(2.0 * p)) / 1.5, 1e-15);
    for (int i = 0; i < 1000; ++i) {
        const double value =
            noise.Fbm(glm::dvec3(0.37 * i, 0.11 * i, 1.0));
        EXPECT_LE(std::abs(value), 1.1);
    }
}

}  // namespace
}  // namespace ernie
