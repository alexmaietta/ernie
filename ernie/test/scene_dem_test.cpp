// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/dem.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

/// A LOLA LDEM-style label and 16-bit image of `heights` (metres,
/// `lines` rows from the northern) in a directory of its own, half-
/// metre steps above a 1737.4 km sphere, 4 samples a degree from 40 S
/// and 340 E. Returns the label's path.
std::string WriteDem(const std::string& name,
                     const std::vector<double>& heights, int lines,
                     int samples) {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        ("ernie_scene_dem_test_" + name);
    std::filesystem::create_directories(directory);
    std::ofstream image(directory / "TEST_DEM.IMG", std::ios::binary);
    for (const double height : heights) {
        const auto dn =
            static_cast<std::int16_t>(std::lround(height / 0.5));
        image.write(reinterpret_cast<const char*>(&dn), 2);
    }
    std::ofstream label(directory / "TEST_DEM.LBL");
    label << "PDS_VERSION_ID = PDS3\n"
          << "^IMAGE = \"TEST_DEM.IMG\"\n"
          << "/* A comment\n   over two lines. */\n"
          << "OBJECT = IMAGE\n"
          << "  LINES = " << lines << "\n"
          << "  LINE_SAMPLES = " << samples << "\n"
          << "  SAMPLE_TYPE = LSB_INTEGER\n"
          << "  SAMPLE_BITS = 16\n"
          << "  SCALING_FACTOR = 0.5\n"
          << "  OFFSET = 1737400.\n"
          << "END_OBJECT = IMAGE\n"
          << "OBJECT = IMAGE_MAP_PROJECTION\n"
          << "  MAP_PROJECTION_TYPE = \"SIMPLE CYLINDRICAL\"\n"
          << "  MAP_RESOLUTION = 4 <pix/deg>\n"
          << "  A_AXIS_RADIUS = 1737.4 <km>\n"
          << "  MAXIMUM_LATITUDE = -40 <deg>\n"
          << "  WESTERNMOST_LONGITUDE = 340 <deg>\n"
          << "END_OBJECT = IMAGE_MAP_PROJECTION\n"
          << "END\n";
    return (directory / "TEST_DEM.LBL").string();
}

}  // namespace

TEST(Pds3Label, ReadsKeywordsObjectsUnitsAndContinuations) {
    const ernie::Pds3Label label =
        ernie::ParsePds3Label("PDS_VERSION_ID = PDS3\n"
                              "/* skipped = 1 */\n"
                              "NOTE = \"two\n  lines\"\n"
                              "OBJECT = IMAGE\n"
                              "  LINES = 23040\n"
                              "END_OBJECT = IMAGE\n"
                              "OBJECT = IMAGE_MAP_PROJECTION\n"
                              "  MAP_SCALE = 0.0592252938 <km/pix>\n"
                              "END_OBJECT\n"
                              "END\n");
    EXPECT_EQ(label.Text("PDS_VERSION_ID"), "PDS3");
    EXPECT_FALSE(label.Get("skipped"));
    EXPECT_EQ(label.Text("NOTE"), "two lines");
    EXPECT_EQ(label.Number("IMAGE.LINES"), 23040.0);
    EXPECT_DOUBLE_EQ(label.Number("IMAGE_MAP_PROJECTION.MAP_SCALE"),
                     0.0592252938);
    EXPECT_THROW(label.Number("LINES"), std::invalid_argument);
    EXPECT_THROW(label.Number("NOTE"), std::invalid_argument);
}

TEST(CylindricalDem, ReadsLolasLdemLabelsAndSamplesBetweenCells) {
    // Heights rising 10 m a line southwards and 1 m a sample
    // eastwards.
    constexpr int kLines = 8;
    constexpr int kSamples = 12;
    std::vector<double> heights;
    for (int l = 0; l < kLines; ++l) {
        for (int s = 0; s < kSamples; ++s) {
            heights.push_back(10.0 * l + s);
        }
    }
    const ernie::CylindricalDem dem = ernie::ReadCylindricalDem(
        WriteDem("rising", heights, kLines, kSamples));
    EXPECT_EQ(dem.lines, 8u);
    EXPECT_EQ(dem.samples, 12u);
    EXPECT_EQ(dem.referenceRadius, 1737400.0);
    // Line 2, sample 3 is centred at 40.625 S, 340.875 E; longitudes
    // wrap; between cells the heights are bilinear.
    EXPECT_DOUBLE_EQ(dem.HeightAt(-40.625, 340.875), 23.0);
    EXPECT_DOUBLE_EQ(dem.HeightAt(-40.625, -19.125), 23.0);
    EXPECT_DOUBLE_EQ(dem.HeightAt(-40.75, 340.875), 28.0);
    EXPECT_THROW(ernie::ReadCylindricalDem("/no/such/dem.lbl"),
                 std::runtime_error);
}

TEST(LocalTerrain, HoldsThePlanetsCurvatureAndFramesThePoint) {
    // A flat sphere about 41 S, 341 E: the heightfield drops as the
    // sphere does, and the frame takes the point to its origin.
    const ernie::CylindricalDem dem = ernie::ReadCylindricalDem(
        WriteDem("flat", std::vector<double>(8 * 12, 0.0), 8, 12));
    const ernie::LocalTerrain terrain =
        ernie::LocalTerrainOf(dem, -41.0, 341.0, 21, 1000.0);
    ASSERT_EQ(terrain.heights.Shape()[0], 21);
    const float* heights = terrain.heights.As<float>();
    constexpr double kRadius = 1737400.0;
    EXPECT_NEAR(heights[10 * 21 + 10], 0.0, 1e-3);
    const double drop =
        std::sqrt(kRadius * kRadius - 1e4 * 1e4) - kRadius;
    EXPECT_NEAR(heights[10 * 21 + 20], drop, 1e-2);
    EXPECT_EQ(terrain.origin, glm::dvec2(-10000.0, 10000.0));
    const glm::dvec3 point =
        kRadius * ernie::BodyDirection(-41.0, 341.0);
    const glm::dvec3 local(terrain.bodyToLocal *
                           glm::dvec4(point * 1.001, 1.0));
    EXPECT_NEAR(local.x, 0.0, 1e-6);
    EXPECT_NEAR(local.y, 0.0, 1e-6);
    EXPECT_NEAR(local.z, 0.001 * kRadius, 1e-6);
    EXPECT_THROW(ernie::LocalTerrainOf(dem, 0.0, 0.0, 1, 1.0),
                 std::invalid_argument);
}
