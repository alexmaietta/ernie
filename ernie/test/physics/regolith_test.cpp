// SPDX-License-Identifier: BSD-3-Clause
/// @file regolith_test.cpp
/// @brief Tests of the regolith thermal model.
#include "ernie/physics/regolith.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

// Diviner's equatorial temperatures: about 390 K at noon, 100 K at
// midnight and 95 K before dawn (Vasavada et al. 2012, figure 4;
// Hayne et al. 2017, figure A2), the afternoon warmer than the
// morning at the same sun for the heat the regolith stored.
TEST(Regolith, TheMoonsEquatorMatchesDiviner) {
    const ernie::DiurnalTable& table = ernie::LunarDiurnalTable();
    ASSERT_EQ(table.latitudes, 19);
    ASSERT_EQ(table.times, 96);
    const auto at = [&](int row, double hour) {
        return static_cast<double>(table.kelvin[static_cast<std::size_t>(row * table.times) +
                                                static_cast<std::size_t>(hour * 4.0)]);
    };
    EXPECT_NEAR(at(0, 12.0), 390.0, 10.0);
    EXPECT_NEAR(at(0, 0.0), 102.0, 10.0);
    EXPECT_NEAR(at(0, 5.75), 95.0, 8.0);
    EXPECT_LT(at(0, 5.75), at(0, 0.0));
    EXPECT_GT(at(0, 15.0), at(0, 9.0) + 0.5);
    // Away from the equator the noon sun is lower: at 60 degrees the
    // equator's noon times the quarter power of cos 60 and of the
    // absorptance at 60 degrees' incidence (albedo 0.272) over that
    // at normal incidence (0.12), colder still at 85.
    EXPECT_NEAR(at(12, 12.0), at(0, 12.0) * std::pow(0.5 * 0.728 / 0.88, 0.25), 6.0);
    EXPECT_LT(at(17, 12.0), 250.0);
    const auto [low, high] = std::minmax_element(table.kelvin.begin(), table.kelvin.end());
    EXPECT_GT(*low, 20.0f);
    EXPECT_LT(*high, 400.0f);
}

TEST(Regolith, LessSunlightCoolsTheDay) {
    ernie::RegolithProperties far;
    far.solarIrradiance = 1361.0 / (1.52 * 1.52);
    const auto moon = ernie::DiurnalSurfaceTemperatures({}, 0.0, 24);
    const auto mars = ernie::DiurnalSurfaceTemperatures(far, 0.0, 24);
    EXPECT_NEAR(mars[12] / moon[12], std::pow(1.0 / (1.52 * 1.52), 0.25), 0.02);
    EXPECT_THROW(ernie::DiurnalSurfaceTemperatures({}, 0.0, 1), std::invalid_argument);
}

TEST(Regolith, ThermalInertiaScalesTheConductivities) {
    const ernie::RegolithProperties moon;
    EXPECT_NEAR(ernie::ThermalInertia(moon), 64.0, 4.0);
    const ernie::RegolithProperties rock = ernie::WithThermalInertia(moon, 2000.0);
    EXPECT_NEAR(ernie::ThermalInertia(rock), 2000.0, 1e-6);
    EXPECT_EQ(rock.deepDensity, moon.deepDensity);
    EXPECT_THROW(ernie::WithThermalInertia(moon, 0.0), std::invalid_argument);
}

// Bandfield et al. 2011: rocks hold the day's heat, so at night they
// stay far warmer than the fines around them; at noon both near the
// equilibrium the sunlight sets, darker ground warmer.
TEST(Regolith, TablesSpanThermalInertiaAndAlbedo) {
    const ernie::DiurnalTable table =
        ernie::DiurnalTemperatureTable({}, 2, 24, {50.0f, 2000.0f}, {0.07f, 0.2f});
    ASSERT_EQ(table.kelvin.size(), 2u * 24u * 2u * 2u);
    const auto at = [&](int hour, int inertia, int albedo) {
        return table.kelvin[table.At(0, hour, inertia, albedo)];
    };
    EXPECT_GT(at(23, 1, 0), at(23, 0, 0) + 60.0f) << "rocks warmer before dawn";
    EXPECT_GT(at(12, 0, 0), at(12, 0, 1) + 5.0f) << "darker ground hotter at noon";
    EXPECT_LT(at(12, 1, 0), at(12, 0, 0)) << "rocks cooler at noon";
    // A bright regolith, its albedo past 1 at grazing sun but for the
    // clamp, stays physical.
    const ernie::DiurnalTable bright = ernie::DiurnalTemperatureTable({}, 2, 24, {50.0f}, {0.6f});
    const auto [low, high] = std::minmax_element(bright.kelvin.begin(), bright.kelvin.end());
    EXPECT_GT(*low, 20.0f) << "the pole's";
    EXPECT_LT(*high, 400.0f);
    // The regolith's own inertia and albedo without axes.
    const ernie::DiurnalTable own = ernie::DiurnalTemperatureTable({}, 2, 4);
    EXPECT_EQ(own.inertias, std::vector<float>{static_cast<float>(ernie::ThermalInertia({}))});
    EXPECT_EQ(own.albedos, std::vector<float>{0.12f});
    EXPECT_THROW(ernie::DiurnalTemperatureTable({}, 2, 4, {100.0f, 50.0f}), std::invalid_argument);
    EXPECT_THROW(ernie::DiurnalTemperatureTable({}, 2, 4, {}, {1.0f}), std::invalid_argument);
}

TEST(Regolith, TablesRoundTripThroughModelData) {
    ernie::DiurnalTable table;
    table.latitudes = 2;
    table.times = 3;
    table.inertias = {50.0f, 500.0f};
    table.albedos = {0.1f};
    for (int i = 0; i < 12; ++i) {
        table.kelvin.push_back(100.0f + static_cast<float>(i));
    }
    const std::vector<float> data = table.ModelData();
    ASSERT_EQ(data.size(), 5u + 3u + 12u);
    EXPECT_EQ(data[0], ernie::DiurnalTable::kModelDataTag);
    EXPECT_EQ(data[5 + 3 + table.At(1, 2, 1)], 111.0f);
    const ernie::DiurnalTable back = ernie::DiurnalTable::FromModelData(data);
    EXPECT_EQ(back.inertias, table.inertias);
    EXPECT_EQ(back.kelvin, table.kelvin);
    // The older layout, a cut table, a bad count.
    EXPECT_THROW(ernie::DiurnalTable::FromModelData({2.0f, 3.0f}), std::invalid_argument);
    std::vector<float> cut = data;
    cut.pop_back();
    EXPECT_THROW(ernie::DiurnalTable::FromModelData(cut), std::invalid_argument);
    std::vector<float> fractional = data;
    fractional[1] = 2.5f;
    EXPECT_THROW(ernie::DiurnalTable::FromModelData(fractional), std::invalid_argument);
}
