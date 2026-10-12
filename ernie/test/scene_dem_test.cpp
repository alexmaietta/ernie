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

/// The map projection of `WriteDem`'s labels by default: 4 samples a
/// degree from 40 S and 340 E.
constexpr const char* kCylindrical =
    "  MAP_PROJECTION_TYPE = \"SIMPLE CYLINDRICAL\"\n"
    "  MAP_RESOLUTION = 4 <pix/deg>\n"
    "  A_AXIS_RADIUS = 1737.4 <km>\n"
    "  MAXIMUM_LATITUDE = -40 <deg>\n"
    "  WESTERNMOST_LONGITUDE = 340 <deg>\n";

/// A LOLA LDEM-style label and 16-bit image of `heights` (metres,
/// `lines` rows from the northern) in a directory of its own, half-
/// metre steps above a 1737.4 km sphere, on map projection
/// `projection` (the IMAGE_MAP_PROJECTION object's lines). Returns
/// the label's path.
std::string WriteDem(const std::string& name,
                     const std::vector<double>& heights, int lines,
                     int samples,
                     const char* projection = kCylindrical) {
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
          << projection << "END_OBJECT = IMAGE_MAP_PROJECTION\n"
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

TEST(PolarStereographicDem, ReadsLolasPolarLabelsAndProjectsThePole) {
    // A south polar grid of 10 m pixels, the pole at the centre of 21
    // x 21; heights rising 10 m a line and 1 m a sample.
    constexpr int kSize = 21;
    std::vector<double> heights;
    for (int l = 0; l < kSize; ++l) {
        for (int s = 0; s < kSize; ++s) {
            heights.push_back(10.0 * l + s);
        }
    }
    const std::string label =
        WriteDem("polar", heights, kSize, kSize,
                 "  MAP_PROJECTION_TYPE = \"POLAR STEREOGRAPHIC\"\n"
                 "  A_AXIS_RADIUS = 1737.4 <km>\n"
                 "  CENTER_LATITUDE = -90 <deg>\n"
                 "  CENTER_LONGITUDE = 0 <deg>\n"
                 "  MAP_SCALE = 10 <m/pix>\n"
                 "  LINE_PROJECTION_OFFSET = 10.0 <pix>\n"
                 "  SAMPLE_PROJECTION_OFFSET = 10.0 <pix>\n");
    const ernie::PolarStereographicDem dem =
        ernie::ReadPolarStereographicDem(label);
    EXPECT_EQ(dem.lines, 21u);
    EXPECT_EQ(dem.mapScale, 10.0);
    EXPECT_EQ(dem.referenceRadius, 1737400.0);
    const glm::dvec2 pole = dem.ToGrid(-90.0, 123.0);
    EXPECT_NEAR(pole.x, 10.0, 1e-9);
    EXPECT_NEAR(pole.y, 10.0, 1e-9);
    // 20 m from the pole: along 90 E two samples right (x), along the
    // prime meridian two lines up (y), as LOLA's grids lie.
    constexpr double kRadius = 1737400.0;
    const double latitude =
        (2.0 * std::atan(20.0 / (2.0 * kRadius)) - 0.5 * M_PI) *
        180.0 / M_PI;
    EXPECT_NEAR(dem.HeightAt(latitude, 90.0), 10.0 * 10 + 12, 1e-6);
    EXPECT_NEAR(dem.HeightAt(latitude, 0.0), 10.0 * 8 + 10, 1e-6);
    EXPECT_NEAR(dem.HeightAt(latitude, 180.0), 10.0 * 12 + 10, 1e-6);
    EXPECT_THROW(ernie::ReadCylindricalDem(label),
                 std::invalid_argument);
    EXPECT_THROW(ernie::ReadPolarStereographicDem(WriteDem(
                     "notpolar", std::vector<double>(4, 0.0), 2, 2)),
                 std::invalid_argument);
    // A local terrain near the pole: flat ground about the point.
    const ernie::LocalTerrain terrain =
        ernie::LocalTerrainOf(dem, -89.9999, 0.0, 3, 1.0);
    EXPECT_EQ(terrain.heights.Shape()[0], 3);
}

TEST(CylindricalDem, ReadsNacDtmsTrueAtTheirParallelWithNullCells) {
    // A 2 x 4 grid of 32-bit reals, as LROC's NAC DTMs hold theirs: 4
    // pixels a degree along the meridians, true at 60 S so 2 along
    // the parallels, its last sample PDS's null.
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        "ernie_scene_dem_test_dtm";
    std::filesystem::create_directories(directory);
    {
        std::ofstream image(directory / "DTM.IMG", std::ios::binary);
        const float values[8] = {1.0f, 2.0f, 3.0f, -3.4028227e38f,
                                 5.0f, 6.0f, 7.0f, 8.0f};
        image.write(reinterpret_cast<const char*>(values),
                    sizeof(values));
    }
    std::ofstream(directory / "DTM.LBL")
        << "PDS_VERSION_ID = PDS3\n"
           "^IMAGE = \"DTM.IMG\"\n"
           "OBJECT = IMAGE\n"
           "  LINES = 2\n"
           "  LINE_SAMPLES = 4\n"
           "  SAMPLE_TYPE = PC_REAL\n"
           "  SAMPLE_BITS = 32\n"
           "END_OBJECT = IMAGE\n"
           "OBJECT = IMAGE_MAP_PROJECTION\n"
           "  MAP_PROJECTION_TYPE = \"EQUIRECTANGULAR\"\n"
           "  A_AXIS_RADIUS = 1737.4 <KM>\n"
           "  CENTER_LATITUDE = -60.0 <DEG>\n"
           "  MAP_RESOLUTION = 4 <PIX/DEG>\n"
           "  MAXIMUM_LATITUDE = -59.5 <DEG>\n"
           "  WESTERNMOST_LONGITUDE = 10.0 <DEG>\n"
           "END_OBJECT = IMAGE_MAP_PROJECTION\n"
           "END\n";
    const ernie::CylindricalDem dtm =
        ernie::ReadCylindricalDem((directory / "DTM.LBL").string());
    EXPECT_DOUBLE_EQ(dtm.standardParallelDeg, -60.0);
    EXPECT_EQ(dtm.referenceRadius, 1737400.0);
    // Line 1, sample 2 is centred at 59.875 S, 11.25 E (two samples a
    // degree of longitude).
    EXPECT_NEAR(dtm.HeightAt(-59.875, 11.25), 7.0, 1e-9);
    EXPECT_TRUE(std::isnan(dtm.HeightAt(-59.625, 11.75)));
    // Where it has no data a fallback stands in.
    ernie::CylindricalDem fallback;
    fallback.referenceRadius = 1737400.0;
    fallback.lines = fallback.samples = 1;
    fallback.pixelsPerDegree = 1.0 / 360.0;
    fallback.maxLatitudeDeg = 90.0;
    fallback.heights = {100.0f};
    const ernie::LocalTerrain terrain =
        ernie::LocalTerrainOf(dtm, fallback, -59.625, 11.75, 3, 1.0);
    EXPECT_NEAR(terrain.heights.As<float>()[4], 100.0, 1e-3);
}
