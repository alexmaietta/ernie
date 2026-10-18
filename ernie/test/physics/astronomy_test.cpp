// SPDX-License-Identifier: BSD-3-Clause
/// @file astronomy_test.cpp
/// @brief Tests of time, sidereal time and solar-system positions.
#include "ernie/physics/astronomy.h"
#include "ernie/physics/spectrum.h"

#include <glm/gtc/constants.hpp>
#include <gtest/gtest.h>

#include <cmath>

namespace {

glm::dvec3 Equatorial(double rightAscension, double declination) {
    const double ra = glm::radians(rightAscension);
    const double dec = glm::radians(declination);
    return {std::cos(dec) * std::cos(ra), std::cos(dec) * std::sin(ra), std::sin(dec)};
}

}  // namespace

TEST(Astronomy, JulianDatesAndSiderealTimeMatchMeeus) {
    EXPECT_DOUBLE_EQ(ernie::JulianDate(2000, 1, 1, 12.0), 2451545.0) << "J2000.0";
    EXPECT_DOUBLE_EQ(ernie::JulianDate(1987, 4, 10), 2446895.5) << "Meeus example 12.a";
    // Meeus example 12.a: 13h 10m 46.3668s.
    EXPECT_NEAR(ernie::GreenwichMeanSiderealTime(2446895.5),
                (13.0 + 10.0 / 60.0 + 46.3668 / 3600.0) * 15.0, 1e-4);
}

TEST(Astronomy, TheCelestialPoleStandsAtTheLatitudeDueNorth) {
    const double latitude = 40.0;
    const double jd = ernie::JulianDate(2024, 3, 20, 3.5);
    const glm::dmat3 toLocal(ernie::CelestialToLocal(latitude, -105.0, jd));
    const glm::dvec3 pole = toLocal * glm::dvec3(0.0, 0.0, 1.0);
    EXPECT_NEAR(pole.x, 0.0, 1e-12) << "due north: no east component";
    EXPECT_NEAR(std::asin(pole.z), glm::radians(latitude), 1e-12) << "altitude = latitude";

    // A star on the meridian at declination = latitude passes
    // overhead; one six hours of right ascension later (east of it)
    // is on the eastern horizon.
    const double local = ernie::GreenwichMeanSiderealTime(jd) - 105.0;
    const glm::dvec3 zenith = toLocal * Equatorial(local, latitude);
    EXPECT_NEAR(zenith.z, 1.0, 1e-12);
    const glm::dvec3 east = toLocal * Equatorial(local + 90.0, 0.0);
    EXPECT_NEAR(east.x, 1.0, 1e-12);
}

TEST(Astronomy, MagnitudesAndColoursGiveIlluminanceAndTemperature) {
    EXPECT_DOUBLE_EQ(ernie::StarIlluminance(0.0), 2.54e-6);
    EXPECT_NEAR(ernie::StarIlluminance(5.0) / ernie::StarIlluminance(0.0), 0.01, 1e-12)
        << "five magnitudes are a factor of 100";
    EXPECT_NEAR(ernie::ColorIndexToTemperature(0.65), 5778.0, 5.0) << "the Sun's B-V";
    EXPECT_GT(ernie::ColorIndexToTemperature(-0.3), 15000.0) << "blue stars are hot";
    EXPECT_LT(ernie::ColorIndexToTemperature(1.8), 3500.0) << "red stars are cool";

    // The spectrum integrates back to the illuminance through the
    // photopic curve.
    for (const double kelvin : {2500.0, 5772.0, 9600.0, 30000.0}) {
        double lux = 0.0;
        for (double nm = 360.5; nm < 830.0; nm += 1.0) {
            lux += ernie::kLuminousEfficacy * ernie::BlackbodySpectralIrradiance(nm, kelvin, 2.0) *
                   ernie::CieXyz(nm).y;
        }
        EXPECT_NEAR(lux, 2.0, 0.01) << kelvin << " K";
    }
}
