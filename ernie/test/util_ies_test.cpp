// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/ies.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

namespace {

/// A downlight in LM-63-2002 form: rotationally symmetric, 100 cd at the nadir falling to 0 at
/// 90 degrees as cos, nothing above; candela multiplier 2.
constexpr const char* kDownlight = R"(IESNA:LM-63-2002
[TEST] Ernie's
[MANUFAC] none
TILT=NONE
1 -1 2.0 4 1 1 2 0.1 0.1 0.0
1.0 1.0 10
0 30 60 90
0
50 43.30127 25 0
)";

}  // namespace

TEST(Ies, ReadsARotationallySymmetricProfile) {
    const ernie::IesProfile profile = ernie::ParseIes(kDownlight);
    ASSERT_EQ(profile.vertical.size(), 4u);
    ASSERT_EQ(profile.horizontal.size(), 1u);
    EXPECT_DOUBLE_EQ(profile.Sample(0.0, 0.0), 100.0) << "times the multiplier";
    EXPECT_NEAR(profile.Sample(30.0, 123.0), 100.0 * std::cos(M_PI / 6.0), 1e-4);
    EXPECT_NEAR(profile.Sample(45.0, 0.0), 0.5 * (86.60254 + 50.0), 1e-4) << "linear between";
    EXPECT_EQ(profile.Sample(120.0, 0.0), 0.0) << "nothing above the horizon";
    // A cos lobe of 100 cd gives pi 100 lumens; the 30 degree steps lose a little of it.
    EXPECT_NEAR(profile.Flux(), M_PI * 100.0, 0.03 * M_PI * 100.0);
}

TEST(Ies, MirrorsTheHorizontalPlanesBySymmetry) {
    // Quadrant symmetry: 0 to 90 degrees, 10 cd along phi 0 and 30 along phi 90, every theta.
    const char* quadrant = R"(IESNA91
TILT=INCLUDE
1
2
0 90
1 1
1 1000 1 2 2 1 1 0 0 0
1 1 100
0 180
0 90
10 10
30 30
)";
    const ernie::IesProfile profile = ernie::ParseIes(quadrant);
    EXPECT_DOUBLE_EQ(profile.Sample(60.0, 0.0), 10.0);
    EXPECT_DOUBLE_EQ(profile.Sample(60.0, 90.0), 30.0);
    EXPECT_DOUBLE_EQ(profile.Sample(60.0, 45.0), 20.0);
    EXPECT_DOUBLE_EQ(profile.Sample(60.0, 180.0), 10.0) << "mirrored";
    EXPECT_DOUBLE_EQ(profile.Sample(60.0, 270.0), 30.0);
    EXPECT_DOUBLE_EQ(profile.Sample(60.0, 315.0), 20.0);
    EXPECT_DOUBLE_EQ(profile.Sample(60.0, -45.0), 20.0);
}

TEST(Ies, RejectsWhatItCannotRead) {
    EXPECT_THROW(ernie::ParseIes("not a photometric file"), std::invalid_argument);
    EXPECT_THROW(ernie::ParseIes("TILT=NONE\n1 -1 1 2 1 2 2 0 0 0\n1 1 1\n0 90\n0\n1 1\n"),
                 std::invalid_argument)
        << "type B";
    EXPECT_THROW(ernie::ParseIes("TILT=NONE\n1 -1 1 2 1 1 2 0 0 0\n1 1 1\n90 0\n0\n1 1\n"),
                 std::invalid_argument)
        << "decreasing angles";
    EXPECT_THROW(ernie::ParseIes("TILT=NONE\n1 -1 1 2 1 1 2 0 0 0\n1 1 1\n0 90\n0\n1\n"),
                 std::invalid_argument)
        << "too few values";
    EXPECT_THROW(ernie::ReadIes("no-such-directory/lamp.ies"), std::runtime_error);
}
