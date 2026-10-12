// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/regolith.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

// Diviner's equatorial temperatures: about 390 K at noon, 100 K at
// midnight and 95 K before dawn (Vasavada et al. 2012, figure 4;
// Hayne et al. 2017, figure A2), the afternoon warmer than the
// morning at the same sun for the heat the regolith stored.
TEST(Regolith, TheMoonsEquatorMatchesDiviner) {
    const ernie::DiurnalTable& table = ernie::LunarDiurnalTable();
    ASSERT_EQ(table.latitudes, 19);
    ASSERT_EQ(table.times, 96);
    const auto at = [&](int row, double hour) {
        return static_cast<double>(
            table.kelvin[static_cast<std::size_t>(row * table.times) +
                         static_cast<std::size_t>(hour * 4.0)]);
    };
    EXPECT_NEAR(at(0, 12.0), 390.0, 10.0);
    EXPECT_NEAR(at(0, 0.0), 102.0, 10.0);
    EXPECT_NEAR(at(0, 5.75), 95.0, 8.0);
    EXPECT_LT(at(0, 5.75), at(0, 0.0));
    EXPECT_GT(at(0, 15.0), at(0, 9.0) + 0.5);
    // Away from the equator the noon sun is lower: at 60 degrees the
    // equator's noon times the quarter power of cos 60 and of the
    // absorptance at 60 degrees' incidence (albedo 0.272) over that at
    // normal incidence (0.12), colder still at 85.
    EXPECT_NEAR(at(12, 12.0),
                at(0, 12.0) * std::pow(0.5 * 0.728 / 0.88, 0.25), 6.0);
    EXPECT_LT(at(17, 12.0), 250.0);
    const auto [low, high] =
        std::minmax_element(table.kelvin.begin(), table.kelvin.end());
    EXPECT_GT(*low, 20.0f);
    EXPECT_LT(*high, 400.0f);
}

TEST(Regolith, LessSunlightCoolsTheDay) {
    ernie::RegolithProperties far;
    far.solarIrradiance = 1361.0 / (1.52 * 1.52);
    const auto moon = ernie::DiurnalSurfaceTemperatures({}, 0.0, 24);
    const auto mars = ernie::DiurnalSurfaceTemperatures(far, 0.0, 24);
    EXPECT_NEAR(mars[12] / moon[12], std::pow(1.0 / (1.52 * 1.52), 0.25),
                0.02);
    EXPECT_THROW(ernie::DiurnalSurfaceTemperatures({}, 0.0, 1),
                 std::invalid_argument);
}
