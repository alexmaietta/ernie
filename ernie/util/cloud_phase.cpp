// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/cloud_phase.h"

#include "ernie/util/scattering.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
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
                               std::size_t angles) {
    constexpr double kPi = std::numbers::pi;
    DropletPhaseTable table;
    table.angles = std::max<std::size_t>(angles, 2);
    const std::size_t n = table.angles;
    const auto count =
        static_cast<std::size_t>(std::max(bands.count, 0));
    std::vector<double> thetas(n);
    std::vector<double> cosines(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double u =
            static_cast<double>(i) / static_cast<double>(n - 1);
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
            for (std::size_t i = 0; i + 1 < n && thetas[i] < cut;
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
    // The bands' mean, summed in doubles.
    table.values.assign((count + 1) * n, 0.0f);
    std::vector<double> mean(n, 0.0);
    for (std::size_t b = 0; b < count; ++b) {
        const auto [phase, peak] = jobs[b].get();
        table.forwardFraction += peak / static_cast<double>(count);
        for (std::size_t i = 0; i < n; ++i) {
            table.values[b * n + i] = static_cast<float>(phase[i]);
            if (thetas[i] >= cut) {
                mean[i] += phase[i] / static_cast<double>(count);
            }
        }
    }
    // Within the peak, the mean held at its edge's value, for light
    // scattered many times shows no hole about the sun; integrating
    // to 1 again.
    std::size_t edge = 0;
    while (edge + 1 < n && thetas[edge] < cut) {
        ++edge;
    }
    std::fill(mean.begin(),
              mean.begin() + static_cast<std::ptrdiff_t>(edge),
              mean[edge]);
    double total = 0.0;
    for (std::size_t i = 0; i + 1 < n; ++i) {
        total += 0.5 *
                 (mean[i] * std::sin(thetas[i]) +
                  mean[i + 1] * std::sin(thetas[i + 1])) *
                 (thetas[i + 1] - thetas[i]) * 2.0 * kPi;
    }
    for (std::size_t i = 0; i < n; ++i) {
        table.values[count * n + i] =
            static_cast<float>(mean[i] / total);
    }
    return table;
}

}  // namespace ernie
