// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/cloud_transport.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numbers>

namespace {

using ernie::CloudTransportOptions;
using ernie::CloudTransportShape;
using ernie::CloudTransportTable;
using ernie::DropletPhaseTable;
using ernie::kCloudOrderSets;

constexpr double kPi = std::numbers::pi;

// The droplets' folded phase function at scattering cosine `mu`.
double PhaseAt(const DropletPhaseTable& phase, double mu) {
    const double theta = std::acos(std::clamp(mu, -1.0, 1.0));
    const double u = std::acos(1.0 - 2.0 * theta / kPi) / kPi;
    const double at = u * (phase.angles - 1);
    const int i =
        std::clamp(static_cast<int>(at), 0, phase.angles - 2);
    const float* mean =
        phase.values.data() + phase.values.size() - phase.angles;
    return mean[i] + (mean[i + 1] - mean[i]) * (at - i);
}

// Small enough to simulate in a test: the first `thicknesses` slabs.
CloudTransportTable Simulate(const DropletPhaseTable& phase,
                             int thicknesses, int photons) {
    CloudTransportOptions options;
    options.shape.thicknesses = thicknesses;
    options.photons = photons;
    return ernie::SimulateCloudTransport(phase, options);
}

}  // namespace

TEST(CloudTransport, SlabsScatterOnceAsTheyShould) {
    const DropletPhaseTable phase =
        ernie::DropletPhase(7.0, 2.0, {400.0, 700.0, 4}, 1024);
    // Thin slabs, to an optical depth of 1, with photons enough for
    // light scattered once alone.
    const CloudTransportTable table = Simulate(phase, 3, 1 << 20);
    const CloudTransportShape& shape = table.shape;
    // Light scattered once leaving the lit face, against its analytic
    // radiance p mu_L / (mu_L + mu) (1 - e^-tau (1 / mu_L + 1 / mu)),
    // averaged over each bin.
    const int light = 3;
    const int thickness = 2;
    const double muL = shape.LightCosine(light);
    const double tau = shape.Thickness(thickness);
    for (int c = 0; c < shape.viewCosines / 2 - 1; ++c) {
        // Over each cosine's azimuths, the bins' photons too few
        // alone.
        double expected = 0.0;
        double got = 0.0;
        for (int a = 0; a < shape.azimuths; ++a) {
            constexpr int kSub = 32;
            for (int i = 0; i < kSub; ++i) {
                for (int j = 0; j < kSub; ++j) {
                    const double mu = -1.0 + (c + (i + 0.5) / kSub) *
                                                 2.0 /
                                                 shape.viewCosines;
                    const double psi =
                        (a + (j + 0.5) / kSub) * kPi / shape.azimuths;
                    const double up = -mu;
                    const double s = std::sqrt(1.0 - mu * mu);
                    const double sL = std::sqrt(1.0 - muL * muL);
                    const double cosine =
                        s * sL * std::cos(psi) + mu * muL;
                    expected +=
                        PhaseAt(phase, cosine) * muL / (muL + up) *
                        (1.0 -
                         std::exp(-tau * (1.0 / muL + 1.0 / up))) /
                        (kSub * kSub);
                }
            }
            got +=
                table.single[shape.Cell(light, thickness, 0, c, a)];
        }
        EXPECT_NEAR(got, expected, 0.05 * expected)
            << "cosine bin " << c;
    }
}

TEST(CloudTransport,
     SlabsKeepTheirLightAndCollectorsSpreadWithOrder) {
    const DropletPhaseTable phase =
        ernie::DropletPhase(7.0, 2.0, {400.0, 700.0, 4}, 1024);
    // Slabs to an optical depth of 8.
    const CloudTransportTable table = Simulate(phase, 6, 1 << 17);
    const CloudTransportShape& shape = table.shape;
    const double solidAngle =
        2.0 / shape.viewCosines * 2.0 * kPi / shape.azimuths;
    // What enters each slab leaves it, scattered back up through the
    // lit face, scattered on down through the other, or unscattered.
    for (int l = 0; l < shape.lightCosines; ++l) {
        for (int t = 0; t < shape.thicknesses; ++t) {
            const double muLight = shape.LightCosine(l);
            double out =
                muLight * std::exp(-shape.Thickness(t) / muLight);
            for (int c = 0; c < shape.viewCosines; ++c) {
                const bool up = shape.ViewCosine(c) < 0.0;
                const int depth = up ? 0 : shape.depths - 1;
                for (int a = 0; a < shape.azimuths; ++a) {
                    const int cell = shape.Cell(l, t, depth, c, a);
                    double radiance = table.single[cell];
                    for (int set = 0; set < kCloudOrderSets; ++set) {
                        radiance += table.At(set, cell, 0);
                    }
                    out += radiance * std::abs(shape.ViewCosine(c)) *
                           solidAngle;
                }
            }
            EXPECT_NEAR(out, muLight, 0.04 * muLight)
                << "light " << muLight << " thickness "
                << shape.Thickness(t);
        }
    }
    // Looking straight down on the lit face of the thickest slab, lit
    // obliquely: light leaving there entered upstream of the light,
    // the more spread the more it scattered.
    const int cell = shape.Cell(3, shape.thicknesses - 1, 0, 0,
                                shape.azimuths / 2);
    EXPECT_LT(table.At(0, cell, 1), 0.0);
    double spread = 0.0;
    for (int set = 0; set < kCloudOrderSets; ++set) {
        EXPECT_GT(table.At(set, cell, 0), 0.0) << "set " << set;
        EXPECT_GT(table.At(set, cell, 3), spread) << "set " << set;
        spread = table.At(set, cell, 3);
    }
}
