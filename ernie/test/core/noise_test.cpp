// SPDX-License-Identifier: BSD-3-Clause
/// @file noise_test.cpp
/// @brief Tests of the seeded gradient noise.
#include "ernie/core/noise.h"

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
    EXPECT_NEAR(noise.Fbm(p, 2, 2.0, 0.5), (noise(p) + 0.5 * noise(2.0 * p)) / 1.5, 1e-15);
    for (int i = 0; i < 1000; ++i) {
        const double value = noise.Fbm(glm::dvec3(0.37 * i, 0.11 * i, 1.0));
        EXPECT_LE(std::abs(value), 1.1);
    }
}

// One period on in any axis gives the same noise, so textures built from it repeat seamlessly.
TEST(Noise, WorleyAndPerlinTileAcrossTheirPeriod) {
    for (const double cells : {4.0, 8.0, 32.0}) {
        for (const double t : {0.013, 0.37, 0.81}) {
            const glm::dvec3 p(t, 0.21, 0.6);
            const double here = TileableWorley(p, cells);
            EXPECT_NEAR(TileableWorley(p + glm::dvec3(1.0, 0.0, 0.0), cells), here, 1e-9);
            EXPECT_NEAR(TileableWorley(p + glm::dvec3(0.0, 1.0, 0.0), cells), here, 1e-9);
            EXPECT_NEAR(TileableWorley(p + glm::dvec3(0.0, 0.0, 1.0), cells), here, 1e-9);
            EXPECT_GE(here, 0.0);
            EXPECT_LE(here, 1.0);
        }
    }
    const glm::dvec3 frequency(6.0, 12.0, 1.0);
    for (const double t : {0.013, 0.37, 0.81}) {
        const glm::dvec3 p(t, 0.4, 0.0);
        const double here = TileablePerlin(p, frequency, 4);
        EXPECT_NEAR(TileablePerlin(p + glm::dvec3(1.0, 0.0, 0.0), frequency, 4), here, 1e-5);
        EXPECT_NEAR(TileablePerlin(p + glm::dvec3(0.0, 1.0, 0.0), frequency, 4), here, 1e-5);
    }
}

TEST(Noise, ValueNoiseAndWorleyFbmStayInRange) {
    for (int i = 0; i < 1000; ++i) {
        const glm::dvec3 p(0.173 * i, 0.311 * i - 40.0, 0.057 * i);
        EXPECT_GE(ValueNoise(p), 0.0);
        EXPECT_LE(ValueNoise(p), 1.0);
        const double fbm = WorleyFbm(glm::fract(p), 2.0);
        EXPECT_GE(fbm, 0.0);
        EXPECT_LE(fbm, 1.0);
    }
}

}  // namespace
}  // namespace ernie
