// SPDX-License-Identifier: BSD-3-Clause
/// @file cloud_test.cpp
/// @brief Tests of cloud droplet phase functions, slab transport and noise textures.
#include "ernie/physics/cloud.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>

namespace ernie {
namespace {

using std::numbers::pi;

/// The integral of a table row over the sphere, by the trapezoid rule.
double SphereIntegral(const DropletPhaseTable& table, std::size_t row) {
    const float* values = table.values.data() + row * table.angles;
    double total = 0.0;
    for (std::size_t i = 0; i + 1 < table.angles; ++i) {
        const double a = table.Angle(i);
        const double b = table.Angle(i + 1);
        total += 0.5 * (double(values[i]) * std::sin(a) + double(values[i + 1]) * std::sin(b)) *
                 (b - a) * 2.0 * pi;
    }
    return total;
}

/// Row `row`'s value nearest scattering angle `degrees`.
double NearestValue(const DropletPhaseTable& table, std::size_t row, double degrees) {
    const double theta = degrees * pi / 180.0;
    std::size_t best = 0;
    for (std::size_t i = 0; i < table.angles; ++i) {
        if (std::abs(table.Angle(i) - theta) < std::abs(table.Angle(best) - theta)) {
            best = i;
        }
    }
    return double(table.values[row * table.angles + best]);
}

/// The droplets' folded phase function at scattering cosine `mu`, interpolated.
double FoldedPhase(const DropletPhaseTable& phase, double mu) {
    const double theta = std::acos(std::clamp(mu, -1.0, 1.0));
    const double u = std::acos(1.0 - 2.0 * theta / pi) / pi;
    const double at = u * static_cast<double>(phase.angles - 1);
    const std::size_t i = std::min(static_cast<std::size_t>(at), phase.angles - 2);
    const float* mean = phase.values.data() + phase.values.size() - phase.angles;
    return double(mean[i]) + double(mean[i + 1] - mean[i]) * (at - static_cast<double>(i));
}

/// Small enough to simulate in a test: the first `thicknesses` slabs.
CloudTransportTable Simulate(const DropletPhaseTable& phase, std::size_t thicknesses,
                             std::size_t photons) {
    CloudTransportOptions options;
    options.shape.thicknesses = thicknesses;
    options.photons = photons;
    return SimulateCloudTransport(phase, options);
}

TEST(CloudPhase, DropletsScatterAsBouthorsEtAlFindOverTheSphere) {
    // Stratiform cloud droplets (Bouthors et al. 2006: an effective radius of 7 um, gamma 2), in
    // four bands.
    const SpectralBands bands{400.0, 700.0, 4};
    const DropletPhaseTable table = DropletPhase(7.0, 2.0, bands, 1024);
    ASSERT_EQ(table.values.size(), 5u * 1024u);
    // About half the light in the narrow forward peak (theirs 51%).
    EXPECT_GT(table.forwardFraction, 0.4);
    EXPECT_LT(table.forwardFraction, 0.6);
    const auto rows = static_cast<std::size_t>(bands.count);
    for (std::size_t row = 0; row <= rows; ++row) {
        const bool mean = row == rows;
        // The bands' mean, its peak folded away, integrates to 1 over the sphere; each band's, its
        // peak kept, to 1 over the extinction's share left.
        const double expected = mean ? 1.0 : 1.0 / (1.0 - table.forwardFraction);
        EXPECT_NEAR(SphereIntegral(table, row), expected, (mean ? 0.002 : 0.05) * expected)
            << "row " << row;
        // The forward peak, kept or folded away; the wide forward lobe, the fogbow's rise beyond
        // 120 degrees and the glory at 180.
        if (mean) {
            EXPECT_NEAR(NearestValue(table, row, 2.0), NearestValue(table, row, 4.0),
                        1e-6 * NearestValue(table, row, 4.0));
        } else {
            EXPECT_GT(NearestValue(table, row, 2.0), NearestValue(table, row, 10.0))
                << "row " << row;
        }
        EXPECT_GT(NearestValue(table, row, 10.0), 10.0 * NearestValue(table, row, 90.0))
            << "row " << row;
        EXPECT_GT(NearestValue(table, row, 140.0), NearestValue(table, row, 120.0))
            << "row " << row;
        EXPECT_GT(NearestValue(table, row, 180.0), NearestValue(table, row, 170.0))
            << "row " << row;
    }
}

TEST(CloudTransport, SlabsScatterOnceAsTheyShould) {
    const DropletPhaseTable phase = DropletPhase(7.0, 2.0, {400.0, 700.0, 4}, 1024);
    // Thin slabs, to an optical depth of 1, with photons enough for light scattered once alone.
    const CloudTransportTable table = Simulate(phase, 3, 1 << 20);
    const CloudTransportShape& shape = table.shape;
    // Light scattered once leaving the lit face, against its analytic radiance
    // p mu_L / (mu_L + mu) (1 - e^-tau (1 / mu_L + 1 / mu)), averaged over each bin.
    const std::size_t light = 3;
    const std::size_t thickness = 2;
    const double muL = shape.LightCosine(light);
    const double tau = shape.Thickness(thickness);
    const auto bins = [](std::size_t count) { return static_cast<double>(count); };
    for (std::size_t c = 0; c < shape.viewCosines / 2 - 1; ++c) {
        // Summed over each cosine's azimuths, as the bins' photons are too few alone.
        double expected = 0.0;
        double got = 0.0;
        for (std::size_t a = 0; a < shape.azimuths; ++a) {
            constexpr int kSub = 32;
            for (int i = 0; i < kSub; ++i) {
                for (int j = 0; j < kSub; ++j) {
                    const double mu =
                        -1.0 + (bins(c) + (i + 0.5) / kSub) * 2.0 / bins(shape.viewCosines);
                    const double psi = (bins(a) + (j + 0.5) / kSub) * pi / bins(shape.azimuths);
                    const double up = -mu;
                    const double s = std::sqrt(1.0 - mu * mu);
                    const double sL = std::sqrt(1.0 - muL * muL);
                    const double cosine = s * sL * std::cos(psi) + mu * muL;
                    expected += FoldedPhase(phase, cosine) * muL / (muL + up) *
                                (1.0 - std::exp(-tau * (1.0 / muL + 1.0 / up))) / (kSub * kSub);
                }
            }
            got += double(table.single[shape.Cell(light, thickness, 0, c, a)]);
        }
        EXPECT_NEAR(got, expected, 0.05 * expected) << "cosine bin " << c;
    }
}

TEST(CloudTransport, SlabsKeepTheirLightAndCollectorsSpreadWithOrder) {
    const DropletPhaseTable phase = DropletPhase(7.0, 2.0, {400.0, 700.0, 4}, 1024);
    // Slabs to an optical depth of 8.
    const CloudTransportTable table = Simulate(phase, 6, 1 << 17);
    const CloudTransportShape& shape = table.shape;
    const double solidAngle = 2.0 / double(shape.viewCosines) * 2.0 * pi / double(shape.azimuths);
    // What enters each slab leaves it: scattered back up through the lit face, scattered on down
    // through the other, or unscattered.
    for (std::size_t l = 0; l < shape.lightCosines; ++l) {
        for (std::size_t t = 0; t < shape.thicknesses; ++t) {
            const double muLight = shape.LightCosine(l);
            double out = muLight * std::exp(-shape.Thickness(t) / muLight);
            for (std::size_t c = 0; c < shape.viewCosines; ++c) {
                const bool up = shape.ViewCosine(c) < 0.0;
                const std::size_t depth = up ? 0 : shape.depths - 1;
                for (std::size_t a = 0; a < shape.azimuths; ++a) {
                    const std::size_t cell = shape.Cell(l, t, depth, c, a);
                    double radiance = double(table.single[cell]);
                    for (std::size_t set = 0; set < kCloudOrderSets; ++set) {
                        radiance += double(table.At(set, cell, 0));
                    }
                    out += radiance * std::abs(shape.ViewCosine(c)) * solidAngle;
                }
            }
            EXPECT_NEAR(out, muLight, 0.04 * muLight)
                << "light " << muLight << " thickness " << shape.Thickness(t);
        }
    }
    // Looking straight down on the lit face of the thickest slab, lit obliquely: light leaving
    // there entered upstream of the light, the more spread the more it scattered.
    const std::size_t cell = shape.Cell(3, shape.thicknesses - 1, 0, 0, shape.azimuths / 2);
    EXPECT_LT(table.At(0, cell, 1), 0.0f);
    float spread = 0.0f;
    for (std::size_t set = 0; set < kCloudOrderSets; ++set) {
        EXPECT_GT(table.At(set, cell, 0), 0.0f) << "set " << set;
        EXPECT_GT(table.At(set, cell, 3), spread) << "set " << set;
        spread = table.At(set, cell, 3);
    }
}

// Small textures: the statistics hold at any size, and the full ones take seconds to build.
TEST(CloudNoise, TexturesHoldCloudsShapesAndWeather) {
    const CloudNoise noise = MakeCloudNoise(32, 16, 128, 32);
    ASSERT_EQ(noise.shape.size(), 32u * 32u * 32u);
    ASSERT_EQ(noise.detail.size(), 16u * 16u * 16u);
    ASSERT_EQ(noise.weather.size(), 128u * 128u * 4u);
    ASSERT_EQ(noise.turbulence.size(), 32u * 32u * 3u);
    const auto [low, high] = std::minmax_element(noise.shape.begin(), noise.shape.end());
    // Perlin-Worley remapped: billows over most of the range.
    EXPECT_GE(*low, 0.0f);
    EXPECT_LE(*high, 1.0f);
    EXPECT_GT(*high - *low, 0.5f);
    // Low and mid clouds cover part of the weather map, not all.
    double lowCover = 0.0;
    double midCover = 0.0;
    for (std::size_t i = 0; i < noise.weather.size(); i += 4) {
        lowCover += double(noise.weather[i]);
        midCover += double(noise.weather[i + 1]);
    }
    const double texels = double(noise.weather.size() / 4);
    EXPECT_GT(lowCover / texels, 0.02);
    EXPECT_LT(lowCover / texels, 0.8);
    EXPECT_GT(midCover / texels, 0.02);
    EXPECT_LT(midCover / texels, 0.8);
}

}  // namespace
}  // namespace ernie
