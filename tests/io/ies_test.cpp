// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/io/ies.h"

#include "tests/common.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

/// A downlight in LM-63-2002 form: rotationally symmetric, 100 cd at
/// the nadir falling to 0 at 90 degrees as cos, nothing above;
/// candela multiplier 2.
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

/// A profile of type `type` (2 B, 3 A) giving one quarter: vertical
/// and horizontal angles 0, 30, 60, 90, the candela 100 + V + H / 3.
std::string Quarter(int type) {
    return "IESNA:LM-63-2002\nTILT=NONE\n1 -1 1 4 4 " + std::to_string(type) +
           " 2 0 0 0\n1 1 10\n0 30 60 90\n0 30 60 90\n"
           "100 130 160 190\n110 140 170 200\n120 150 180 210\n130 "
           "160 190 220\n";
}

/// Theta (from the nadir, -z) and phi (from x towards y) in degrees
/// of unit direction `e`.
std::pair<double, double> Angles(const glm::dvec3& e) {
    return {glm::degrees(std::acos(-e.z)), glm::degrees(std::atan2(e.y, e.x))};
}

}  // namespace

TEST(Ies, ReadsTypeBAboutTheWidth) {
    const ernie::IesProfile profile = ernie::ParseIes(Quarter(2));
    EXPECT_EQ(profile.type, ernie::IesType::B);
    const auto at = [&](const glm::dvec3& e) {
        const auto [theta, phi] = Angles(glm::normalize(e));
        return profile.Sample(theta, phi);
    };
    const double s = std::sin(glm::radians(30.0));
    const double c = std::cos(glm::radians(30.0));
    EXPECT_NEAR(at({0.0, 0.0, -1.0}), 100.0, 1e-9) << "the beam's axis";
    EXPECT_NEAR(at({s, 0.0, -c}), 130.0, 1e-9) << "V 30: turned about y towards x";
    EXPECT_NEAR(at({-s, 0.0, -c}), 130.0, 1e-9) << "V -30, mirrored";
    EXPECT_NEAR(at({0.0, -s, -c}), 110.0, 1e-9) << "H 30: out of the plane towards -y";
    EXPECT_NEAR(at({0.0, s, -c}), 110.0, 1e-9) << "H -30, mirrored";
    // V 30 and H 30: the plane turned 30 degrees, then 30 degrees out
    // of it.
    EXPECT_NEAR(at({c * s, -s, -c * c}), 140.0, 1e-9);
    EXPECT_EQ(at({c, 0.0, s}), 0.0) << "V 120 is beyond the file's angles";
}

TEST(Ies, ReadsTypeAAboutTheLength) {
    const ernie::IesProfile profile = ernie::ParseIes(Quarter(3));
    EXPECT_EQ(profile.type, ernie::IesType::A);
    const auto at = [&](const glm::dvec3& e) {
        const auto [theta, phi] = Angles(glm::normalize(e));
        return profile.Sample(theta, phi);
    };
    const double s = std::sin(glm::radians(30.0));
    const double c = std::cos(glm::radians(30.0));
    EXPECT_NEAR(at({0.0, s, -c}), 130.0, 1e-9) << "V 30: turned about x towards y";
    EXPECT_NEAR(at({s, 0.0, -c}), 110.0, 1e-9) << "H 30: out of the plane towards x";
    EXPECT_NEAR(at({-s, 0.0, -c}), 110.0, 1e-9) << "H -30, mirrored";
}

TEST(Ies, TiltsScaleTheLampByTheLuminairesTilt) {
    // A vertical lamp losing a fifth of its output at 45 degrees and
    // half lying down.
    const char* included = R"(IESNA:LM-63-2002
TILT=INCLUDE
1
3
0 45 90
1 0.8 0.5
1 -1 1 2 1 1 2 0 0 0
1 1 10
0 90
0
100 0
)";
    const ernie::IesProfile profile = ernie::ParseIes(included);
    EXPECT_EQ(profile.tilt.geometry, 1);
    EXPECT_DOUBLE_EQ(profile.tilt.Factor(0.0), 1.0);
    EXPECT_DOUBLE_EQ(profile.tilt.Factor(22.5), 0.9);
    EXPECT_DOUBLE_EQ(profile.tilt.Factor(135.0), 0.8) << "mirrored about 90 degrees";
    EXPECT_DOUBLE_EQ(profile.Sample(0.0, 0.0), 100.0) << "the candela themselves untilted";
    // Turned 45 degrees about x: its up axis 45 degrees from
    // vertical.
    const double h = std::sqrt(0.5);
    EXPECT_NEAR(profile.tilt.Angle({1, 0, 0}, {0, h, h}, {0, -h, h}), 45.0, 1e-9);
    // A horizontal lamp along x: its elevation, whichever way the
    // luminaire turns about it.
    ernie::IesTilt horizontal = profile.tilt;
    horizontal.geometry = 2;
    EXPECT_NEAR(horizontal.Angle({1, 0, 0}, {0, h, h}, {0, -h, h}), 0.0, 1e-9);
    EXPECT_NEAR(horizontal.Angle({h, 0, h}, {0, 1, 0}, {-h, 0, h}), 45.0, 1e-9);
    EXPECT_DOUBLE_EQ(ernie::IesTilt{}.Factor(30.0), 1.0) << "no tilt data";

    // The same data from a tilt file beside the profile.
    const std::filesystem::path directory = std::filesystem::temp_directory_path();
    std::ofstream(directory / "ernie_util_ies_lamp.tlt") << "1\n3\n0 45 90\n1 0.8 0.5\n";
    std::ofstream(directory / "ernie_util_ies_lamp.ies")
        << "IESNA:LM-63-2002\nTILT=ernie_util_ies_lamp.tlt\n1 -1 1 2 "
           "1 1 2 0 0 0\n1 1 10\n"
           "0 90\n0\n100 0\n";
    const ernie::IesProfile filed = ernie::ReadIes(directory / "ernie_util_ies_lamp.ies");
    EXPECT_EQ(filed.tilt.factors, (std::vector<double>{1.0, 0.8, 0.5}));
    std::filesystem::remove(directory / "ernie_util_ies_lamp.tlt");
    std::filesystem::remove(directory / "ernie_util_ies_lamp.ies");
}

TEST(Ies, ReadsARotationallySymmetricProfile) {
    const ernie::IesProfile profile = ernie::ParseIes(kDownlight);
    ASSERT_EQ(profile.vertical.size(), 4u);
    ASSERT_EQ(profile.horizontal.size(), 1u);
    EXPECT_DOUBLE_EQ(profile.Sample(0.0, 0.0), 100.0) << "times the multiplier";
    EXPECT_NEAR(profile.Sample(30.0, 123.0), 100.0 * std::cos(glm::radians(30.0)), 1e-4);
    EXPECT_NEAR(profile.Sample(45.0, 0.0), 0.5 * (86.60254 + 50.0), 1e-4) << "linear between";
    EXPECT_EQ(profile.Sample(120.0, 0.0), 0.0) << "nothing above the horizon";
    // A cos lobe of 100 cd gives pi 100 lumens; the 30 degree steps
    // lose a little of it.
    EXPECT_NEAR(profile.Flux(), std::numbers::pi * 100.0, 0.03 * std::numbers::pi * 100.0);
}

TEST(Ies, MirrorsTheHorizontalPlanesBySymmetry) {
    // Quadrant symmetry: 0 to 90 degrees, 10 cd along phi 0 and 30
    // along phi 90, every theta.
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
    EXPECT_THROW(ernie::ParseIes("TILT=NONE\n1 -1 1 2 1 4 2 0 0 0\n1 1 1\n0 90\n0\n1 1\n"),
                 std::invalid_argument)
        << "no type 4";
    EXPECT_THROW(ernie::ParseIes("TILT=NONE\n1 -1 1 2 1 2 2 0 0 0\n1 "
                                 "1 1\n0 120\n0\n1 1\n"),
                 std::invalid_argument)
        << "type B's angles reach 90 degrees";
    EXPECT_THROW(ernie::ParseIes("TILT=lamp.tlt\n1 -1 1 2 1 1 2 0 0 "
                                 "0\n1 1 1\n0 90\n0\n1 1\n",
                                 "no-such-directory"),
                 std::runtime_error)
        << "no tilt file";
    EXPECT_THROW(ernie::ParseIes("TILT=NONE\n1 -1 1 2 1 1 2 0 0 0\n1 1 1\n90 0\n0\n1 1\n"),
                 std::invalid_argument)
        << "decreasing angles";
    EXPECT_THROW(ernie::ParseIes("TILT=NONE\n1 -1 1 2 1 1 2 0 0 0\n1 1 1\n0 90\n0\n1\n"),
                 std::invalid_argument)
        << "too few values";
    EXPECT_THROW(ernie::ReadIes("no-such-directory/lamp.ies"), std::runtime_error);
}

TEST(Ies, ReadsAQuadrantFileWithItsTiltFileBesideIt) {
    const ernie::IesProfile profile =
        ernie::ReadIes(ernie::test::FixturePath("ies/downlight_quadrant.ies"));
    EXPECT_EQ(profile.type, ernie::IesType::C);
    EXPECT_NEAR(profile.Sample(0.0, 0.0), 500.0, 1e-9) << "ballast factor 0.5";
    EXPECT_NEAR(profile.Sample(45.0, 45.0), 350.0, 1e-9);
    EXPECT_NEAR(profile.Sample(45.0, 135.0), 350.0, 1e-9) << "mirrored into the quadrant";
    EXPECT_NEAR(profile.Sample(22.5, 270.0), 375.0, 1e-9);
    EXPECT_NEAR(profile.Sample(30.0, 0.0), 450.0, 1e-9) << "linear between vertical angles";
    EXPECT_NEAR(profile.Sample(120.0, 0.0), 0.0, 1e-9) << "nothing above the horizon";

    EXPECT_EQ(profile.tilt.geometry, 1);
    EXPECT_NEAR(profile.tilt.Factor(0.0), 1.0, 1e-12);
    EXPECT_NEAR(profile.tilt.Factor(22.5), 0.95, 1e-12);
    EXPECT_NEAR(profile.tilt.Factor(135.0), 0.9, 1e-12) << "mirrored about 90 degrees";
}

TEST(Ies, ReadsACommaSeparatedTypeBFileWithItsTiltIncluded) {
    const ernie::IesProfile profile =
        ernie::ReadIes(ernie::test::FixturePath("ies/floodlight_type_b.ies"));
    EXPECT_EQ(profile.type, ernie::IesType::B);
    EXPECT_NEAR(profile.Sample(0.0, 0.0), 1200.0, 1e-9) << "candela multiplier 2";
    EXPECT_NEAR(profile.Sample(10.0, 0.0), 1000.0, 1e-9);
    EXPECT_NEAR(profile.Sample(10.0, 180.0), 1000.0, 1e-9) << "mirrored about the vertical plane";
    EXPECT_NEAR(profile.Sample(5.0, 0.0), 1100.0, 1e-9);
    EXPECT_NEAR(profile.Sample(20.0, -90.0), 1000.0, 1e-9);
    EXPECT_NEAR(profile.Sample(20.0, 90.0), 1000.0, 1e-9);

    const double h = std::sqrt(0.5);
    EXPECT_EQ(profile.tilt.geometry, 2);
    EXPECT_NEAR(profile.tilt.Angle({h, 0, h}, {0, 1, 0}, {-h, 0, h}), 45.0, 1e-9);
    EXPECT_NEAR(profile.tilt.Factor(45.0), 0.9, 1e-12);
}
