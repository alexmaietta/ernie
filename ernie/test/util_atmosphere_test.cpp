// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/atmosphere.h"

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr double kPi = std::numbers::pi;

const ernie::Constituent&
Find(const ernie::AtmosphereModel& atmosphere,
     std::string_view name) {
    for (const ernie::Constituent& constituent :
         atmosphere.constituents) {
        if (constituent.name == name) {
            return constituent;
        }
    }
    throw std::runtime_error("no constituent " + std::string(name));
}

double Depth(const ernie::AtmosphereModel& atmosphere,
             std::string_view name, double nm) {
    const ernie::Constituent& constituent = Find(atmosphere, name);
    const ernie::ParticleOptics optics = constituent.optics(nm, {});
    return constituent.numberDensity *
           (optics.scattering + optics.absorption) *
           constituent.density.Column(atmosphere.topAltitude);
}

/// Integrates a phase table at `PhaseCosines(count)` over the sphere:
/// theta = pi u^2, so d(omega) = 2 pi sin(theta) 2 pi u du.
double SphereIntegral(std::span<const float> phase) {
    const int count = static_cast<int>(phase.size());
    double sum = 0.0;
    for (int i = 0; i < count; ++i) {
        const double u = static_cast<double>(i) / (count - 1);
        const double weight = (i == 0 || i == count - 1) ? 0.5 : 1.0;
        sum +=
            weight *
            static_cast<double>(phase[static_cast<std::size_t>(i)]) *
            std::sin(kPi * u * u) * 2.0 * kPi * u;
    }
    return 2.0 * kPi * sum / (count - 1);
}

}  // namespace

TEST(DensityProfile, ColumnsMatchClosedForms) {
    const ernie::DensityProfile exponential =
        ernie::DensityProfile::Exponential(8000.0);
    EXPECT_DOUBLE_EQ(exponential(0.0), 1.0);
    EXPECT_NEAR(exponential.Column(100e3),
                8000.0 * (1.0 - std::exp(-100e3 / 8000.0)), 1e-6);

    const ernie::DensityProfile tent =
        ernie::DensityProfile::Tent(25e3, 15e3);
    EXPECT_DOUBLE_EQ(tent(25e3), 1.0);
    EXPECT_DOUBLE_EQ(tent(10e3), 0.0);
    EXPECT_NEAR(tent.Column(100e3), 15e3, 1e-3);

    ernie::DensityProfile dust;
    dust.layers = {{ernie::DensityLayer::Shape::DoubleExponential,
                    4000.0, 0.0, 3.0},
                   {ernie::DensityLayer::Shape::DoubleExponential,
                    20000.0, 0.0, 1.0}};
    EXPECT_DOUBLE_EQ(dust(0.0), 1.0);
    EXPECT_NEAR(dust(4000.0),
                (3.0 * std::exp(1.0 - std::exp(1.0)) +
                 std::exp(1.0 - std::exp(0.2))) /
                    4.0,
                1e-15);
    EXPECT_DOUBLE_EQ(ernie::DensityProfile{}(0.0), 0.0);
}

// The hydrostatic column p N_A / (M g) of the U.S. Standard
// Atmosphere, against Bodhaine et al. (1999)'s 0.097069 at 550 nm,
// sea level and 45 degrees latitude, whose column takes g at the
// column's mass-weighted height (978.92 cm s^-2) instead of the
// standard 980.665.
TEST(EarthAtmosphere, RayleighDepthMatchesBodhaine) {
    const ernie::AtmosphereModel earth = ernie::EarthAtmosphere(
        6371e3, {.ozoneColumn = 0.0, .aerosolOpticalDepth = 0.0});
    EXPECT_NEAR(Depth(earth, "molecules", 550.0) /
                    (0.097069 * 978.916 / 980.665),
                1.0, 2e-3);
    // Surface density p / (k T), and the U.S. Standard Atmosphere's
    // density at 11 km (its table 4: 0.36392 of 1.2250 kg m^-3; the
    // table's heights are geopotential, 19 m below these at 11 km).
    const ernie::Constituent& molecules = Find(earth, "molecules");
    EXPECT_NEAR(molecules.numberDensity, 2.5469e25, 1e21);
    EXPECT_NEAR(molecules.density(11000.0) / (0.36392 / 1.2250), 1.0,
                3e-3);
    // Blue sky: scattering grows a little faster than lambda^-4, by
    // the dispersion of air (16.97 by Bodhaine et al.'s eq. 29).
    EXPECT_NEAR(Depth(earth, "molecules", 400.0) /
                    Depth(earth, "molecules", 800.0),
                16.97, 0.02);
}

TEST(EarthAtmosphere, OzoneHoldsItsColumn) {
    const ernie::AtmosphereModel earth =
        ernie::EarthAtmosphere(6371e3, {.ozoneColumn = 300.0});
    const ernie::Constituent& ozone = Find(earth, "ozone");
    EXPECT_NEAR(ozone.numberDensity *
                    ozone.density.Column(earth.topAltitude) /
                    2.6867e20,
                300.0, 1e-3);
    // 300 DU absorbs a few percent in the Chappuis band.
    EXPECT_NEAR(Depth(earth, "ozone", 605.0),
                300.0 * 2.6867e20 * 5.03275e-25, 1e-6);
}

TEST(EarthAtmosphere, AerosolsHaveTheRequestedDepth) {
    const ernie::AtmosphereModel earth =
        ernie::EarthAtmosphere(6371e3, {.aerosolOpticalDepth = 0.15});
    EXPECT_NEAR(Depth(earth, "aerosols", 550.0), 0.15, 1e-9);
    EXPECT_NEAR(earth.VerticalOpticalDepth(550.0),
                Depth(earth, "molecules", 550.0) +
                    Depth(earth, "ozone", 550.0) + 0.15,
                1e-9);
    // Haze absorbs a little (k = 0.015) and is less
    // wavelength-dependent than molecules.
    const ernie::ParticleOptics haze =
        Find(earth, "aerosols").optics(550.0, {});
    EXPECT_GT(haze.scattering / (haze.scattering + haze.absorption),
              0.85);
    EXPECT_LT(Depth(earth, "aerosols", 400.0) /
                  Depth(earth, "aerosols", 800.0),
              4.0);
}

TEST(MarsAtmosphere, ThinCarbonDioxideAndDust) {
    const ernie::AtmosphereModel mars =
        ernie::MarsAtmosphere(3389.5e3, {.dustOpticalDepth = 0.5});
    const ernie::Constituent& molecules = Find(mars, "molecules");
    EXPECT_NEAR(molecules.numberDensity,
                610.0 / (1.380649e-23 * 210.0), 1e18);
    EXPECT_NEAR(molecules.density.layers[0].height, 10.84e3, 0.02e3);
    // About 1/170 of Earth's column, scattering about 2.4 times more
    // per molecule.
    EXPECT_GT(Depth(mars, "molecules", 550.0), 0.001);
    EXPECT_LT(Depth(mars, "molecules", 550.0), 0.003);
    EXPECT_NEAR(Depth(mars, "dust", 880.0), 0.5, 1e-9);
}

// Martian dust absorbs blue light (hematite) and forward-scatters
// strongly, the cause of blue sunsets (Ehlers et al. 2014; Schneegans
// et al. 2024).
TEST(MarsAtmosphere, DustAbsorbsBlueAndScattersForward) {
    const ernie::AtmosphereModel mars =
        ernie::MarsAtmosphere(3389.5e3);
    const ernie::Constituent& dust = Find(mars, "dust");
    const auto albedo = [&dust](double nm) {
        const ernie::ParticleOptics optics = dust.optics(nm, {});
        return optics.scattering /
               (optics.scattering + optics.absorption);
    };
    EXPECT_LT(albedo(450.0), albedo(700.0));
    EXPECT_GT(albedo(880.0), 0.9);
    const double cosines[] = {1.0, -1.0};
    const ernie::ParticleOptics optics = dust.optics(450.0, cosines);
    EXPECT_GT(optics.phase[0] / optics.phase[1], 100.0);
}

TEST(PhaseCosines, DenseNearForward) {
    const std::vector<double> cosines = ernie::PhaseCosines(5);
    EXPECT_DOUBLE_EQ(cosines.front(), 1.0);
    EXPECT_DOUBLE_EQ(cosines.back(), -1.0);
    EXPECT_NEAR(cosines[2], std::cos(kPi / 4.0), 1e-15);
    EXPECT_THROW(ernie::PhaseCosines(1), std::invalid_argument);
}

TEST(BandAverage, AveragesCoefficientsAndNormalisedPhases) {
    const ernie::AtmosphereModel earth =
        ernie::EarthAtmosphere(6371e3);
    const ernie::SpectralBands bands{380.0, 780.0, 16};
    const ernie::BandedAtmosphere banded =
        ernie::BandAverage(earth, bands, 256);
    ASSERT_EQ(banded.constituents.size(), earth.constituents.size());
    EXPECT_EQ(banded.phaseAngles, 256);
    for (std::size_t c = 0; c < banded.constituents.size(); ++c) {
        const ernie::BandedConstituent& constituent =
            banded.constituents[c];
        ASSERT_EQ(constituent.scattering.size(), 16u);
        ASSERT_EQ(constituent.phase.size(), 16u * 256u);
        for (int band = 0; band < bands.count; ++band) {
            const std::span<const float> phase(
                constituent.phase.data() + band * 256, 256);
            EXPECT_NEAR(SphereIntegral(phase), 1.0, 5e-3)
                << constituent.name << " band " << band;
        }
    }
    // Molecules: the band's coefficient is the mean of its 25 nm
    // (three-sample) average.
    const ernie::Constituent& molecules = earth.constituents[0];
    double mean = 0.0;
    for (int s = 0; s < 3; ++s) {
        mean += molecules.optics(380.0 + (s + 0.5) * 25.0 / 3.0, {})
                    .scattering /
                3.0;
    }
    EXPECT_NEAR(
        static_cast<double>(banded.constituents[0].scattering[0]) /
            (molecules.numberDensity * mean),
        1.0, 1e-6);
    EXPECT_GT(banded.constituents[0].scattering[0],
              banded.constituents[0].scattering[15]);
    // Ozone only absorbs.
    EXPECT_FLOAT_EQ(banded.constituents[1].scattering[8], 0.0f);
    EXPECT_GT(banded.constituents[1].absorption[8], 0.0f);
}

// Refraction, against the two closed forms of an exponential
// atmosphere: away from the horizon the plane-parallel (n0 - 1) tan
// z, which no profile changes; at the horizon (n0 - 1) sqrt(pi r' /
// 2H), Laplace's relation of refraction to air mass with Chapman's
// grazing function, integrating the bending |dn/dr| = (n0 - 1) / H
// exp(-x^2 / 2r'H) along the horizontal ray, which curves down with
// k = r (n0 - 1) / H of the planet's curvature and so meets the air
// as over a planet of radius r' = r / (1 - k) (geodesy's effective
// radius): about 36 arcminutes for an isothermal Earth. Then the
// U.S. Standard Atmosphere's falling temperature, against Bennett's
// formula for the observed refraction at the horizon, 34.5
// arcminutes (at 10 C and 1010 hPa; the standard's 15 C makes some
// 1.7% less).
//
// References:
// - A. T. Young, "Air mass and refraction", Applied Optics
//   33(6):1108-1110, 1994. https://doi.org/10.1364/AO.33.001108
// - G. G. Bennett, "The Calculation of Astronomical Refraction in
//   Marine Navigation", Journal of Navigation 35(2):255-259, 1982.
//   https://doi.org/10.1017/S0373463300022037
TEST(BandedAtmosphere, RefractsAsTheClosedForms) {
    const ernie::AtmosphereModel earth = ernie::EarthAtmosphere(
        6371e3, {.ozoneColumn = 0.0, .aerosolOpticalDepth = 0.0});
    ernie::BandedAtmosphere banded =
        ernie::BandAverage(earth, {540.0, 560.0, 1}, 16);
    const double n0 = banded.RefractiveIndex(0.0, 0);
    EXPECT_NEAR((n0 - 1.0) / 2.78e-4, 1.0, 0.01);
    const double r = banded.bottomRadius;

    // The standard atmosphere, against Bennett.
    constexpr double kArcminute = kPi / (180.0 * 60.0);
    const double standard = banded.Refract(r + 1.0, 0.0, 0).deviation;
    EXPECT_NEAR(standard / kArcminute, 34.5 * (1.0 - 0.017), 1.0);

    // An isothermal column at the surface's 288.15 K, its scale
    // height R T / (M g), against the closed forms.
    const double height =
        8.314462618 * 288.15 / (28.9644e-3 * 9.80665);
    banded.constituents[0].density =
        ernie::DensityProfile::Exponential(height);
    EXPECT_NEAR(banded.Refract(r, 1.0, 0).deviation, 0.0, 1e-9);
    const double mu45 = std::sqrt(0.5);
    const ernie::BandedAtmosphere::Refraction slant =
        banded.Refract(r, mu45, 0);
    EXPECT_FALSE(slant.ground);
    EXPECT_NEAR(slant.deviation / (n0 - 1.0), 1.0, 0.01);

    const double k = r * (n0 - 1.0) / height;
    const ernie::BandedAtmosphere::Refraction horizontal =
        banded.Refract(r + 1.0, 0.0, 0);
    EXPECT_FALSE(horizontal.ground) << "refraction lifts it clear";
    EXPECT_NEAR(horizontal.deviation /
                    ((n0 - 1.0) *
                     std::sqrt(kPi * r / (2.0 * height * (1.0 - k)))),
                1.0, 0.03);
    EXPECT_GT(horizontal.deviation, standard)
        << "an isothermal column bends more";
    EXPECT_TRUE(banded.Refract(r + 1.0, -0.02, 0).ground);

    // Mars: a hundredth of the air, and CO2's refractivity.
    const ernie::AtmosphereModel mars =
        ernie::MarsAtmosphere(3389.5e3, {.dustOpticalDepth = 0.0});
    const ernie::BandedAtmosphere thin =
        ernie::BandAverage(mars, {540.0, 560.0, 1}, 16);
    const double m0 = thin.RefractiveIndex(0.0, 0);
    EXPECT_GT(m0 - 1.0, 3e-6);
    EXPECT_LT(m0 - 1.0, 4.5e-6);
    EXPECT_NEAR(thin.Refract(thin.bottomRadius, mu45, 0).deviation /
                    (m0 - 1.0),
                1.0, 0.01);
}
