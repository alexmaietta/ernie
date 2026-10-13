// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/cloud_phase.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <numbers>

namespace {

using ernie::DropletPhase;
using ernie::DropletPhaseTable;
using ernie::SpectralBands;

constexpr double kPi = std::numbers::pi;

double ThetaOf(const DropletPhaseTable& table, std::size_t i) {
    const double u = static_cast<double>(i) /
                     static_cast<double>(table.angles - 1);
    return 0.5 * kPi * (1.0 - std::cos(kPi * u));
}

// Row `row`'s value nearest scattering angle `degrees`.
double At(const DropletPhaseTable& table, std::size_t row,
          double degrees) {
    const double theta = degrees * kPi / 180.0;
    std::size_t best = 0;
    for (std::size_t i = 0; i < table.angles; ++i) {
        if (std::abs(ThetaOf(table, i) - theta) <
            std::abs(ThetaOf(table, best) - theta)) {
            best = i;
        }
    }
    return double(table.values[row * table.angles + best]);
}

}  // namespace

TEST(CloudPhase, DropletsScatterAsBouthorsEtAlFindOverTheSphere) {
    // Stratiform cloud droplets (Bouthors et al. 2006: an effective
    // radius of 7 um, gamma 2), in four bands.
    const SpectralBands bands{400.0, 700.0, 4};
    const DropletPhaseTable table =
        DropletPhase(7.0, 2.0, bands, 1024);
    ASSERT_EQ(table.values.size(), 5u * 1024u);
    // About half the light in the narrow forward peak (theirs 51%).
    EXPECT_GT(table.forwardFraction, 0.4);
    EXPECT_LT(table.forwardFraction, 0.6);
    const auto rows = static_cast<std::size_t>(bands.count);
    for (std::size_t row = 0; row <= rows; ++row) {
        const bool mean = row == rows;
        // The bands' mean, its peak folded away, integrates to 1 over
        // the sphere; each band's, its peak kept, to 1 over the
        // extinction's share left (its own peak's about the mean's).
        double total = 0.0;
        for (std::size_t i = 0; i + 1 < table.angles; ++i) {
            const double a = ThetaOf(table, i);
            const double b = ThetaOf(table, i + 1);
            const float* values =
                table.values.data() + row * table.angles;
            total += 0.5 *
                     (double(values[i]) * std::sin(a) +
                      double(values[i + 1]) * std::sin(b)) *
                     (b - a) * 2.0 * kPi;
        }
        const double expected =
            mean ? 1.0 : 1.0 / (1.0 - table.forwardFraction);
        EXPECT_NEAR(total, expected, (mean ? 0.002 : 0.05) * expected)
            << "row " << row;
        // The forward peak, kept or folded away; the wide forward
        // lobe, the fogbow's rise beyond 120 degrees and the glory at
        // 180.
        if (mean) {
            // Flat within the peak, held at its edge's value.
            EXPECT_NEAR(At(table, row, 2.0), At(table, row, 4.0),
                        1e-6 * At(table, row, 4.0));
        } else {
            EXPECT_GT(At(table, row, 2.0), At(table, row, 10.0))
                << "row " << row;
        }
        EXPECT_GT(At(table, row, 10.0), 10.0 * At(table, row, 90.0))
            << "row " << row;
        EXPECT_GT(At(table, row, 140.0), At(table, row, 120.0))
            << "row " << row;
        EXPECT_GT(At(table, row, 180.0), At(table, row, 170.0))
            << "row " << row;
    }
}
