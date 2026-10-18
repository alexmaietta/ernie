// SPDX-License-Identifier: BSD-3-Clause
/// @file solar_test.cpp
/// @brief Tests of the solar spectrum.
#include "ernie/physics/solar.h"

#include <gtest/gtest.h>

#include <array>

// The total of the TSIS-1 HSRS version 2 full spectrum extension's
// 1 nm bins, 1361.4892 W/m^2 (tools/solar_spectrum.py).
TEST(Solar, IntegratesToTheTotalSolarIrradiance) {
    EXPECT_NEAR(ernie::TotalSolarIrradiance() / 1361.4892, 1.0, 5e-4);
    const ernie::Spectrum& sun = ernie::SolarSpectralIrradiance();
    EXPECT_NEAR(sun.Wavelengths().front(), 115.5, 1e-9);
    EXPECT_GT(sun.Wavelengths().back(), 199000.0);
}

// Band means of the HSRS at its 0.1 nm resolution (LISIRD's
// tsis1_hsrs_p1nm, integrated by the trapezoid rule): the 1 nm table
// keeps them to 10^-3, interpolating between bin means blurring each
// band's edges by half a nanometre (most where Fraunhofer lines crowd
// the violet), well within the spectrum's own 0.3-1.3%.
TEST(Solar, KeepsBandMeansOfTheFullResolutionSpectrum) {
    struct Band {
        double low;
        double high;
        double mean;
    };
    const std::array<Band, 4> bands = {{{380.0, 780.0, 1.65312},
                                        {400.0, 700.0, 1.78658},
                                        {380.0, 405.0, 1.21074},
                                        {550.0, 575.0, 1.86297}}};
    const ernie::Spectrum& sun = ernie::SolarSpectralIrradiance();
    for (const Band& band : bands) {
        EXPECT_NEAR(sun.Average(band.low, band.high) / band.mean, 1.0, 1e-3)
            << band.low << "-" << band.high << " nm";
    }
}

// Fraunhofer's H and K lines of ionised calcium at 397 and 393 nm,
// which a black body lacks.
TEST(Solar, ShowsTheCalciumLines) {
    const ernie::Spectrum& sun = ernie::SolarSpectralIrradiance();
    EXPECT_LT(sun(393.5), 0.35 * sun(410.5));
    EXPECT_LT(sun(396.5), 0.6 * sun(410.5));
}
