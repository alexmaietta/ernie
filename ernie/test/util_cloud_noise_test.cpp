// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/cloud_noise.h"

#include <gtest/gtest.h>

#include <algorithm>

TEST(CloudNoise, WorleyAndPerlinTileAcrossTheirPeriod) {
    // One period on in any axis, the same noise: the textures repeat
    // seamlessly.
    for (const double cells : {4.0, 8.0, 32.0}) {
        for (const double t : {0.013, 0.37, 0.81}) {
            const double here =
                ernie::TileableWorley(t, 0.21, 0.6, cells);
            EXPECT_NEAR(
                ernie::TileableWorley(t + 1.0, 0.21, 0.6, cells),
                here, 1e-9);
            EXPECT_NEAR(ernie::TileableWorley(t, 1.21, 0.6, cells),
                        here, 1e-9);
            EXPECT_NEAR(ernie::TileableWorley(t, 0.21, 1.6, cells),
                        here, 1e-9);
            EXPECT_GE(here, 0.0);
            EXPECT_LE(here, 1.0);
        }
    }
    for (const double t : {0.013, 0.37, 0.81}) {
        const double here =
            ernie::TileablePerlin(t, 0.4, 0.0, 6.0, 12.0, 1.0, 4);
        EXPECT_NEAR(ernie::TileablePerlin(t + 1.0, 0.4, 0.0, 6.0,
                                          12.0, 1.0, 4),
                    here, 1e-5);
        EXPECT_NEAR(
            ernie::TileablePerlin(t, 1.4, 0.0, 6.0, 12.0, 1.0, 4),
            here, 1e-5);
    }
}

TEST(CloudNoise, TexturesHoldCloudsShapesAndWeather) {
    const ernie::CloudNoise noise = ernie::CachedCloudNoise();
    ASSERT_EQ(noise.shape.size(), 128u * 128u * 128u);
    ASSERT_EQ(noise.detail.size(), 32u * 32u * 32u);
    ASSERT_EQ(noise.weather.size(), 512u * 512u * 4u);
    ASSERT_EQ(noise.turbulence.size(), 128u * 128u * 3u);
    const auto [low, high] =
        std::minmax_element(noise.shape.begin(), noise.shape.end());
    // Perlin-Worley remapped: billows over most of the range.
    EXPECT_GE(*low, 0.0f);
    EXPECT_LE(*high, 1.0f);
    EXPECT_GT(*high - *low, 0.5f);
    // Low and mid clouds cover part of the weather map, not all.
    double low_ = 0.0;
    double mid = 0.0;
    for (std::size_t i = 0; i < noise.weather.size(); i += 4) {
        low_ += double(noise.weather[i]);
        mid += double(noise.weather[i + 1]);
    }
    const double texels = double(noise.weather.size() / 4);
    EXPECT_GT(low_ / texels, 0.02);
    EXPECT_LT(low_ / texels, 0.8);
    EXPECT_GT(mid / texels, 0.02);
    EXPECT_LT(mid / texels, 0.8);
}
