// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/cloud_phase.h"

#include "ernie/util/scattering.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <future>
#include <numbers>
#include <span>

namespace ernie {

namespace {

// Liquid water's refractive index across the visible, its absorption
// negligible there.
constexpr std::complex<double> kWater(1.333, 0.0);

}  // namespace

DropletPhaseTable DropletPhase(double effectiveRadiusUm,
                               double spread,
                               const SpectralBands& bands,
                               int angles) {
    constexpr double kPi = std::numbers::pi;
    DropletPhaseTable table;
    table.angles = std::max(angles, 2);
    std::vector<double> thetas(table.angles);
    std::vector<double> cosines(table.angles);
    for (int i = 0; i < table.angles; ++i) {
        const double u = static_cast<double>(i) / (table.angles - 1);
        thetas[i] = 0.5 * kPi * (1.0 - std::cos(kPi * u));
        cosines[i] = std::cos(thetas[i]);
    }
    const LogNormalMode mode = LogNormalMode::FromEffective(
        std::max(effectiveRadiusUm, 0.5) * 1e-6,
        1.0 / (std::max(spread, 0.1) + 2.0));
    // Each band's phase function, its forward peak folded away.
    const double cut = kDropletForwardDegrees * kPi / 180.0;
    std::vector<std::future<std::pair<std::vector<double>, double>>>
        jobs;
    for (int b = 0; b < bands.count; ++b) {
        jobs.push_back(std::async(std::launch::async, [&, b] {
            ParticleOptics optics =
                MieParticles(std::span(&mode, 1), kWater,
                             bands.Centre(b), cosines, 48);
            std::vector<double>& phase = optics.phase;
            // The peak's share, by the trapezoid rule over the
            // sphere.
            double peak = 0.0;
            for (int i = 0; i + 1 < table.angles && thetas[i] < cut;
                 ++i) {
                const double to = std::min(thetas[i + 1], cut);
                const double along =
                    (to - thetas[i]) / (thetas[i + 1] - thetas[i]);
                const double end =
                    phase[i] + (phase[i + 1] - phase[i]) * along;
                peak += 0.5 *
                        (phase[i] * std::sin(thetas[i]) +
                         end * std::sin(to)) *
                        (to - thetas[i]) * 2.0 * kPi;
            }
            peak = std::clamp(peak, 0.0, 0.95);
            // Against the extinction less the peak's share: the peak
            // kept for single scattering (its silver lining, about
            // the sun), folded away from the rest.
            for (double& value : phase) {
                value /= 1.0 - peak;
            }
            return std::make_pair(std::move(phase), peak);
        }));
    }
    table.values.assign(static_cast<std::size_t>(bands.count + 1) *
                            table.angles,
                        0.0f);
    float* mean =
        table.values.data() +
        static_cast<std::size_t>(bands.count) * table.angles;
    for (int b = 0; b < bands.count; ++b) {
        const auto [phase, peak] = jobs[b].get();
        table.forwardFraction += peak / bands.count;
        for (int i = 0; i < table.angles; ++i) {
            table.values[static_cast<std::size_t>(b) * table.angles +
                         i] = static_cast<float>(phase[i]);
            if (thetas[i] >= cut) {
                mean[i] += static_cast<float>(phase[i] / bands.count);
            }
        }
    }
    // Within the peak, the mean held at its edge's value, for light
    // scattered many times shows no hole about the sun; integrating
    // to 1 again.
    int edge = 0;
    while (edge + 1 < table.angles && thetas[edge] < cut) {
        ++edge;
    }
    std::fill(mean, mean + edge, mean[edge]);
    double total = 0.0;
    for (int i = 0; i + 1 < table.angles; ++i) {
        total += 0.5 *
                 (mean[i] * std::sin(thetas[i]) +
                  mean[i + 1] * std::sin(thetas[i + 1])) *
                 (thetas[i + 1] - thetas[i]) * 2.0 * kPi;
    }
    for (int i = 0; i < table.angles; ++i) {
        mean[i] = static_cast<float>(mean[i] / total);
    }
    return table;
}

}  // namespace ernie
