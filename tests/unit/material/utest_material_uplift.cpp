// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/material/uplift.h"

#include <array>
#include <filesystem>

#include <gtest/gtest.h>

#include "support/support.h"

namespace ernie {
namespace {

TEST(UpliftTest, GreysFitFlatSpectraAndColoursReproduceTheirRgb) {
    const std::array<float, 3> grey = FitSigmoid({0.5f, 0.5f, 0.5f}, {});
    for (const float nanometers : {400.0f, 550.0f, 700.0f, 905.0f}) {
        EXPECT_NEAR(SigmoidValue(grey, nanometers), 0.5f, 0.02f) << nanometers;
    }
    for (const std::array<float, 3> rgb :
         {std::array<float, 3>{0.8f, 0.1f, 0.1f}, std::array<float, 3>{0.1f, 0.6f, 0.2f},
          std::array<float, 3>{0.05f, 0.1f, 0.7f}}) {
        const std::array<float, 3> fitted = SigmoidRgb(FitSigmoid(rgb, {}));
        for (int c = 0; c < 3; ++c) {
            EXPECT_NEAR(fitted[c], rgb[c], 0.01f) << c;
        }
    }
}

TEST(UpliftTest, TheTableCoversEveryColour) {
    const SigmoidTable& table = Sigmoids();
    ASSERT_EQ(table.scales.size(), kSigmoidResolution);
    ASSERT_EQ(table.coefficients.size(),
              size_t(3) * kSigmoidResolution * kSigmoidResolution * kSigmoidResolution * 3);
    EXPECT_EQ(table.scales.front(), 0);
    EXPECT_EQ(table.scales.back(), 1);
}

TEST(UpliftTest, TheCachedTableMatchesTheFittedOne) {
    const test::TemporaryDirectory directory;
    const DiskCache cache(directory.Path());
    const SigmoidTable fitted = CachedSigmoids(cache);
    EXPECT_FALSE(std::filesystem::is_empty(directory.Path()));
    const SigmoidTable loaded = CachedSigmoids(cache);
    EXPECT_EQ(loaded.scales, fitted.scales);
    EXPECT_EQ(loaded.coefficients, fitted.coefficients);
}

}
}
