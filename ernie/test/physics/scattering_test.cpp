// SPDX-License-Identifier: BSD-3-Clause
/// @file scattering_test.cpp
/// @brief Tests of Rayleigh and Mie scattering.
#include "ernie/physics/scattering.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

namespace {

constexpr double kPi = std::numbers::pi;

/// Integrates a phase function over the sphere with `count` trapezoid
/// steps in theta.
template <typename Phase> double SphereIntegral(Phase phase, int count = 20000) {
    double sum = 0.0;
    for (int i = 0; i <= count; ++i) {
        const double theta = kPi * i / count;
        const double weight = (i == 0 || i == count) ? 0.5 : 1.0;
        sum += weight * phase(std::cos(theta)) * std::sin(theta);
    }
    return 2.0 * kPi * sum * kPi / count;
}

}  // namespace

// Bodhaine et al. (1999), table 3: cross sections per molecule of dry
// air with 360 ppm CO2.
TEST(Rayleigh, AirCrossSectionsMatchBodhaine) {
    struct Row {
        double nm;
        double crossSectionCm2;
    };
    const std::array<Row, 6> table = {{{400.0, 1.6738e-26},
                                       {450.0, 1.0274e-26},
                                       {500.0, 6.6614e-27},
                                       {550.0, 4.5105e-27},
                                       {725.0, 1.4684e-27},
                                       {775.0, 1.1214e-27}}};
    for (const Row& row : table) {
        const double sigma = ernie::RayleighCrossSection(row.nm, ernie::AirRefractivity(row.nm),
                                                         ernie::AirKingFactor(row.nm));
        EXPECT_NEAR(sigma / (row.crossSectionCm2 * 1e-4), 1.0, 2e-4) << row.nm << " nm";
    }
    // The table's King factor column.
    EXPECT_NEAR(ernie::AirKingFactor(550.0), 1.04882, 1e-5);
    EXPECT_NEAR(ernie::AirKingFactor(400.0), 1.05126, 1e-5);
}

// Bodhaine et al. (1999), eq. (29): their fit to table 3, accurate
// to 0.01% over 250-850 nm.
TEST(Rayleigh, AirCrossSectionsMatchBodhaineFit) {
    for (double nm = 380.0; nm <= 850.0; nm += 10.0) {
        const double s = 1.0 / (nm * 1e-3 * nm * 1e-3);
        const double um2 = nm * 1e-3 * nm * 1e-3;
        const double fitCm2 = 1e-28 * (1.0455996 - 341.29061 * s - 0.90230850 * um2) /
                              (1.0 + 0.0027059889 * s - 85.968563 * um2);
        const double sigma =
            ernie::RayleighCrossSection(nm, ernie::AirRefractivity(nm), ernie::AirKingFactor(nm));
        EXPECT_NEAR(sigma / (fitCm2 * 1e-4), 1.0, 3e-4) << nm << " nm";
    }
}

TEST(Rayleigh, CarbonDioxideScattersMoreThanAir) {
    // Owens's CO2 refractivity is about 1.5 times air's, and
    // scattering goes as its square.
    const double ratio = ernie::CarbonDioxideRefractivity(550.0) / ernie::AirRefractivity(550.0);
    EXPECT_NEAR(ratio, 1.54, 0.02);
    const double co2 = ernie::RayleighCrossSection(550.0, ernie::CarbonDioxideRefractivity(550.0),
                                                   ernie::kCarbonDioxideKingFactor);
    const double air = ernie::RayleighCrossSection(550.0, ernie::AirRefractivity(550.0),
                                                   ernie::AirKingFactor(550.0));
    EXPECT_GT(co2 / air, 2.3);
    EXPECT_LT(co2 / air, 2.7);
}

TEST(Rayleigh, PhaseFunctionIsNormalisedAndSymmetric) {
    for (const double king : {1.0, 1.04882, 1.15}) {
        EXPECT_NEAR(SphereIntegral([king](double mu) { return ernie::RayleighPhase(mu, king); }),
                    1.0, 1e-6);
        EXPECT_DOUBLE_EQ(ernie::RayleighPhase(0.3, king), ernie::RayleighPhase(-0.3, king));
    }
    EXPECT_NEAR(ernie::RayleighPhase(1.0, 1.0), 3.0 / (8.0 * kPi), 1e-15);
    // Depolarisation fills in the minimum at 90 degrees.
    EXPECT_GT(ernie::RayleighPhase(0.0, 1.15) / ernie::RayleighPhase(1.0, 1.15),
              ernie::RayleighPhase(0.0, 1.0) / ernie::RayleighPhase(1.0, 1.0));
}

// Reference values from miepython 3.3.0 (S. Prahl), which is
// validated against Wiscombe's MIEV0 test cases; miepython writes
// absorbing indices as n - ik, Ernie as n + ik.
TEST(Mie, MatchesReferenceImplementation) {
    struct Case {
        std::complex<double> m;
        double x;
        double qext;
        double qsca;
        double g;
        std::array<double, 4> phase;  // at cosines 1, 0.5, 0, -1
    };
    const std::array<Case, 5> cases = {{
        {{1.5, 0.0},
         10.0,
         2.88199895,
         2.88199895,
         0.74291290,
         {5.75272921, 0.0377253, 0.0101338, 0.04680393}},
        {{1.33, 0.0},
         100.0,
         2.10108955,
         2.10108955,
         0.86831486,
         {4.18243214e+02, 1.23725788e-02, 1.17356931e-03, 8.48727308e-02}},
        {{1.5, 0.1},
         2.0,
         1.94147843,
         1.28616797,
         0.65708302,
         {0.44942538, 0.11067371, 0.0237639, 0.00791472}},
        {{1.55, 0.0},
         5.213,
         3.10499592,
         3.10499592,
         0.63310442,
         {1.95622689, 0.06293128, 0.02438255, 0.07494411}},
        {{1.4, 0.015},
         0.5,
         0.02747808,
         0.00970398,
         0.04682583,
         {0.13313607, 0.07916222, 0.05955171, 0.10630475}},
    }};
    const std::array<double, 4> cosines = {1.0, 0.5, 0.0, -1.0};
    for (const Case& c : cases) {
        std::array<double, 4> phase{};
        const ernie::MieEfficiencies q = ernie::MieSphere(c.x, c.m, cosines, phase);
        EXPECT_NEAR(q.extinction / c.qext, 1.0, 1e-6) << "x " << c.x;
        EXPECT_NEAR(q.scattering / c.qsca, 1.0, 1e-6) << "x " << c.x;
        EXPECT_NEAR(q.asymmetry / c.g, 1.0, 1e-6) << "x " << c.x;
        for (std::size_t i = 0; i < cosines.size(); ++i) {
            EXPECT_NEAR(phase[i] / c.phase[i], 1.0, 1e-6) << "x " << c.x << " mu " << cosines[i];
        }
    }
}

TEST(Mie, SmallSpheresScatterLikeRayleigh) {
    const std::complex<double> m(1.5, 0.0);
    const double x = 0.01;
    const ernie::MieEfficiencies q = ernie::MieSphere(x, m, {}, {});
    const double polarisability = std::norm((m * m - 1.0) / (m * m + 2.0));
    EXPECT_NEAR(q.scattering / (8.0 / 3.0 * std::pow(x, 4) * polarisability), 1.0, 1e-3);
    EXPECT_NEAR(q.asymmetry, 0.0, 1e-3);
}

// The extinction paradox: large spheres remove twice their area,
// approached as about 2 + 2 x^(-2/3) from the edge rays (van de
// Hulst); 2.010254493801417 by miepython 3.3.0.
TEST(Mie, LargeSpheresExtinguishTwiceTheirArea) {
    const ernie::MieEfficiencies q = ernie::MieSphere(2000.0, {1.33, 0.0}, {}, {});
    EXPECT_NEAR(q.extinction / 2.010254493801417, 1.0, 1e-9);
    EXPECT_NEAR(q.extinction, 2.0, 2.0 * std::pow(2000.0, -2.0 / 3.0));
}

TEST(Mie, PhaseFunctionIsNormalised) {
    for (const double x : {0.5, 5.0, 30.0}) {
        const double integral = SphereIntegral(
            [x](double mu) {
                const double cosine[] = {mu};
                double phase[1];
                ernie::MieSphere(x, {1.5, 0.01}, cosine, phase);
                return phase[0];
            },
            4000);
        EXPECT_NEAR(integral, 1.0, 1e-3) << "x " << x;
    }
}

TEST(Mie, LogNormalEffectiveParametersRoundTrip) {
    const ernie::LogNormalMode mode = ernie::LogNormalMode::FromEffective(1.4e-6, 0.35, 2.0);
    EXPECT_NEAR(mode.EffectiveRadius(), 1.4e-6, 1e-18);
    EXPECT_NEAR(mode.EffectiveVariance(), 0.35, 1e-12);
    EXPECT_DOUBLE_EQ(mode.relativeNumber, 2.0);
    EXPECT_NEAR(mode.sigma, std::sqrt(std::log(1.35)), 1e-12);
}

TEST(Mie, MonodisperseParticlesAreOneSphere) {
    const double radius = 0.5e-6;
    const double nm = 550.0;
    const std::array<double, 3> cosines = {1.0, 0.0, -1.0};
    const ernie::LogNormalMode modes[] = {{radius, 0.0, 1.0}};
    const ernie::ParticleOptics optics = ernie::MieParticles(modes, {1.5, 0.01}, nm, cosines);
    std::array<double, 3> phase{};
    const double x = 2.0 * kPi * radius / (nm * 1e-9);
    const ernie::MieEfficiencies q = ernie::MieSphere(x, {1.5, 0.01}, cosines, phase);
    const double area = kPi * radius * radius;
    EXPECT_NEAR(optics.scattering / (q.scattering * area), 1.0, 1e-12);
    EXPECT_NEAR(optics.absorption / (q.Absorption() * area), 1.0, 1e-12);
    for (std::size_t i = 0; i < cosines.size(); ++i) {
        EXPECT_NEAR(optics.phase[i] / phase[i], 1.0, 1e-12);
    }
}

TEST(Mie, DistributionPhaseIsNormalisedAndModesMix) {
    const ernie::LogNormalMode modes[] = {{0.1e-6, 0.6, 1000.0}, {1.0e-6, 0.6, 1.0}};
    std::vector<double> cosines;
    constexpr int kAngles = 2001;
    for (int i = 0; i < kAngles; ++i) {
        cosines.push_back(std::cos(kPi * i / (kAngles - 1)));
    }
    const ernie::ParticleOptics optics = ernie::MieParticles(modes, {1.4, 0.0}, 550.0, cosines);
    double integral = 0.0;
    for (int i = 0; i < kAngles; ++i) {
        const double theta = kPi * i / (kAngles - 1);
        const double weight = (i == 0 || i == kAngles - 1) ? 0.5 : 1.0;
        integral += weight * optics.phase[static_cast<std::size_t>(i)] * std::sin(theta);
    }
    EXPECT_NEAR(2.0 * kPi * integral * kPi / (kAngles - 1), 1.0, 2e-3);
    EXPECT_LT(optics.absorption, 1e-12 * optics.scattering);

    // Mixing by relative number: the mean of each mode alone,
    // weighted 1000:1.
    const ernie::ParticleOptics small =
        ernie::MieParticles(std::span(modes, 1), {1.4, 0.0}, 550.0, {});
    const ernie::ParticleOptics large =
        ernie::MieParticles(std::span(modes + 1, 1), {1.4, 0.0}, 550.0, {});
    EXPECT_NEAR(optics.scattering / ((1000.0 * small.scattering + large.scattering) / 1001.0), 1.0,
                1e-12);
}

TEST(MaxwellGarnett, BlendsBetweenMatrixAndInclusion) {
    const std::complex<double> matrix(1.5, 0.0);
    const std::complex<double> inclusion(2.9, 0.4);
    EXPECT_NEAR(std::abs(ernie::MaxwellGarnett(matrix, inclusion, 0.0) - matrix), 0.0, 1e-15);
    EXPECT_NEAR(std::abs(ernie::MaxwellGarnett(matrix, inclusion, 1.0) - inclusion), 0.0, 1e-14);
    const std::complex<double> mixed = ernie::MaxwellGarnett(matrix, inclusion, 0.03);
    EXPECT_GT(mixed.real(), matrix.real());
    EXPECT_GT(mixed.imag(), 0.0);
    EXPECT_LT(mixed.imag(), 0.03 * inclusion.imag() * 2.0);
}

TEST(RefractiveIndex, InterpolatesAndClamps) {
    const ernie::RefractiveIndex index({400.0, 500.0}, {{1.0, 0.2}, {2.0, 0.0}});
    EXPECT_NEAR(std::abs(index(450.0) - std::complex<double>(1.5, 0.1)), 0.0, 1e-15);
    EXPECT_NEAR(std::abs(index(300.0) - std::complex<double>(1.0, 0.2)), 0.0, 1e-15);
    EXPECT_NEAR(std::abs(index(900.0) - std::complex<double>(2.0, 0.0)), 0.0, 1e-15);
    EXPECT_THROW(ernie::RefractiveIndex({500.0, 400.0}, {{1.0, 0.0}, {1.0, 0.0}}),
                 std::invalid_argument);
    // Hematite absorbs blue far more than red.
    const ernie::RefractiveIndex& hematite = ernie::HematiteRefractiveIndex();
    EXPECT_NEAR(hematite(550.0).imag(), 0.460, 1e-12);
    EXPECT_GT(hematite(450.0).imag(), 5.0 * hematite(700.0).imag());
}

TEST(Ozone, ChappuisBandPeaksNear600Nm) {
    // The 600-610 nm bin's average at its centre.
    EXPECT_DOUBLE_EQ(ernie::OzoneCrossSection(605.0), 5.03275e-25);
    EXPECT_GT(ernie::OzoneCrossSection(600.0), 10.0 * ernie::OzoneCrossSection(450.0));
    EXPECT_GT(ernie::OzoneCrossSection(600.0), 10.0 * ernie::OzoneCrossSection(800.0));
    EXPECT_DOUBLE_EQ(ernie::OzoneCrossSection(300.0), 0.0);
    EXPECT_DOUBLE_EQ(ernie::OzoneCrossSection(900.0), 0.0);
}
