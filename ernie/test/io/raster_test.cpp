// SPDX-License-Identifier: BSD-3-Clause
/// @file raster_test.cpp
/// @brief Tests of raster reading: PDS3 labels and images, TIFFs, opening by extension and DEMs.
#include "ernie/io/raster.h"

#include "ernie/util/bytes.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace ernie {
namespace {

namespace fs = std::filesystem;

fs::path Temp(const std::string& name) {
    return fs::path(testing::TempDir()) / name;
}

fs::path Fixture(const std::string& name) {
    return fs::path(ERNIE_TEST_FIXTURES) / "tiff" / name;
}

void WriteText(const fs::path& path, const std::string& text) {
    std::ofstream(path, std::ios::binary) << text;
}

/// Sample `i` of `bytes` as T.
template <class T> T At(const std::vector<std::byte>& bytes, std::size_t i) {
    return LoadAs<T>(bytes.data() + i * sizeof(T));
}

// ----------------------------------------------------------------------------------------------
// PDS3 labels

TEST(Pds3Label, ReadsKeywordsObjectsUnitsAndContinuations) {
    const Pds3Label label = ParsePds3Label("PDS_VERSION_ID = PDS3\n"
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
    EXPECT_DOUBLE_EQ(label.Number("IMAGE_MAP_PROJECTION.MAP_SCALE"), 0.0592252938);
    EXPECT_THROW(label.Number("LINES"), std::invalid_argument);
    EXPECT_THROW(label.Number("NOTE"), std::invalid_argument);
}

TEST(Pds3Label, KeepsObjectNestingAndFindsKeysInAnyObjectAndCase) {
    const Pds3Label label = ParsePds3Label("^IMAGE = (\"A.IMG\", 2)\n"
                                           "OBJECT = IMAGE\n"
                                           "  Lines = 4 /* a comment\n"
                                           "  that runs on */\n"
                                           "  OBJECT = TABLE\n"
                                           "    LINES = 7\n"
                                           "  END_OBJECT = TABLE\n"
                                           "  SAMPLE_TYPE = \"MSB_INTEGER\"\n"
                                           "END_OBJECT = IMAGE\n"
                                           "SET = {A,\n  B}\n"
                                           "END\n"
                                           "AFTER = 1\n");
    ASSERT_EQ(label.keywords.size(), 5u);
    EXPECT_EQ(label.keywords[1].object, "IMAGE.");
    EXPECT_EQ(label.keywords[2].object, "IMAGE.TABLE.");
    EXPECT_EQ(label.Text("IMAGE.Lines"), "4");
    EXPECT_FALSE(label.Get("IMAGE.LINES"));
    EXPECT_EQ(label.Text("IMAGE.TABLE.LINES"), "7");
    EXPECT_EQ(label.Text("SET"), "{A, B}");
    // Find: the first in the label, any case, as written.
    EXPECT_EQ(label.Find("lines"), "4");
    EXPECT_EQ(label.Find("sample_type"), "\"MSB_INTEGER\"");
    EXPECT_EQ(label.Find("^image"), "(\"A.IMG\", 2)");
    EXPECT_FALSE(label.Find("AFTER"));
    EXPECT_FALSE(label.Find("OBJECT"));
}

TEST(Pds3Label, CleanValuesLoseUnitsAndQuotes) {
    EXPECT_EQ(CleanPds3Value(" 10 <m/pix> "), "10");
    EXPECT_EQ(CleanPds3Value("\"SIMPLE CYLINDRICAL\""), "SIMPLE CYLINDRICAL");
    EXPECT_EQ(CleanPds3Value("'A'"), "A");
    EXPECT_EQ(CleanPds3Value("(\"A.IMG\", 2 <BYTES>)"), "(\"A.IMG\", 2 <BYTES>)")
        << "a pointer keeps its units";
}

// ----------------------------------------------------------------------------------------------
// PDS images

TEST(PdsImage, DetachedLabelOfLolaHeights) {
    // A 6 x 4 grid as LOLA's LDEM files hold it: little-endian int16 100x - 300y in a .img,
    // scaled by 0.5 about a 1737.4 km sphere, described by a .lbl in capitals beside it, the whole
    // Moon from 0 degrees east; -32768 missing.
    const fs::path data = Temp("ldem_test.img");
    {
        std::ofstream file(data, std::ios::binary);
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 6; ++x) {
                const auto v = static_cast<std::int16_t>(100 * x - 300 * y);
                file.write(reinterpret_cast<const char*>(&v), 2);
            }
        }
    }
    WriteText(Temp("LDEM_TEST.LBL"), "PDS_VERSION_ID = \"PDS3\"\n"
                                     "/* A comment */\n"
                                     "DESCRIPTION = \"Heights, over\n  two lines\"\n"
                                     "RECORD_BYTES = 12\n"
                                     "^IMAGE = \"LDEM_TEST.IMG\"\n"
                                     "OBJECT = IMAGE\n"
                                     "  LINES = 4\n  LINE_SAMPLES = 6\n"
                                     "  SAMPLE_TYPE = LSB_INTEGER\n  SAMPLE_BITS = 16\n"
                                     "  SCALING_FACTOR = 0.5\n  OFFSET = 1737400.\n"
                                     "  MISSING_CONSTANT = -32768\n"
                                     "END_OBJECT = IMAGE\n"
                                     "OBJECT = IMAGE_MAP_PROJECTION\n"
                                     "  MAP_PROJECTION_TYPE = \"SIMPLE CYLINDRICAL\"\n"
                                     "  MAP_RESOLUTION = 0.0166666666666666667 <pix/deg>\n"
                                     "  CENTER_LATITUDE = 0 <deg>\n"
                                     "  MAXIMUM_LATITUDE = 90 <deg>\n"
                                     "  WESTERNMOST_LONGITUDE = 0 <deg>\n"
                                     "END_OBJECT = IMAGE_MAP_PROJECTION\n"
                                     "END\n");
    const PdsImage image(data);
    EXPECT_EQ(image.Width(), 6u);
    EXPECT_EQ(image.Height(), 4u);
    EXPECT_EQ(image.Type(), GridSampleType::Int16);
    EXPECT_EQ(image.Scale(), 0.5);
    EXPECT_EQ(image.Offset(), 1737400.0);
    EXPECT_EQ(image.NoData(), -32768.0);
    EXPECT_EQ(image.Label("DESCRIPTION"), "Heights, over two lines");
    EXPECT_EQ(image.Keywords().Text("IMAGE.LINES"), "4");
    ASSERT_TRUE(image.Grid().has_value());
    EXPECT_EQ(image.Grid()->west, 0.0);
    EXPECT_EQ(image.Grid()->north, 90.0);
    EXPECT_NEAR(image.Grid()->degreesAcross, 60.0, 1e-9);
    std::vector<std::byte> part(3 * 2 * 2);
    image.Read(2, 1, 3, 2, part);
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 3; ++x) {
            EXPECT_EQ(At<std::int16_t>(part, static_cast<std::size_t>(y * 3 + x)),
                      100 * (x + 2) - 300 * (y + 1));
        }
    }
    EXPECT_THROW(image.Read(4, 0, 3, 1, part), std::out_of_range);
}

TEST(PdsImage, AttachedLabelOfBigEndianFloats) {
    // A label of 128 bytes ahead of 3 x 2 big-endian floats x + 10y, the image at its second
    // record.
    std::string label = "PDS_VERSION_ID = PDS3\nRECORD_BYTES = 128\n^IMAGE = 2\n"
                        "LINES = 2\nLINE_SAMPLES = 3\nSAMPLE_TYPE = IEEE_REAL\n"
                        "SAMPLE_BITS = 32\nEND\n";
    ASSERT_LE(label.size(), 128u);
    label.resize(128, ' ');
    std::string bytes = label;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 3; ++x) {
            const float v = static_cast<float>(x + 10 * y);
            std::uint32_t u;
            std::memcpy(&u, &v, 4);
            for (int b = 3; b >= 0; --b) {
                bytes.push_back(static_cast<char>((u >> (8 * b)) & 0xFF));
            }
        }
    }
    WriteText(Temp("attached.img"), bytes);
    const PdsImage image(Temp("attached.img"));
    EXPECT_EQ(image.Type(), GridSampleType::Float32);
    const std::vector<std::byte> all = image.ReadAll();
    for (std::size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(At<float>(all, i), static_cast<float>(i % 3 + 10 * (i / 3)));
    }
    EXPECT_FALSE(image.Grid().has_value());
}

// ----------------------------------------------------------------------------------------------
// TIFF

constexpr std::uint32_t kWidth = 37;
constexpr std::uint32_t kHeight = 23;

void ExpectRgb(const TiffImage& image) {
    ASSERT_EQ(image.Width(), kWidth);
    ASSERT_EQ(image.Height(), kHeight);
    ASSERT_EQ(image.Channels(), 3u);
    ASSERT_EQ(image.Type(), GridSampleType::UInt8);
    const std::vector<std::byte> all = image.ReadAll();
    for (std::uint32_t y = 0; y < kHeight; ++y) {
        for (std::uint32_t x = 0; x < kWidth; ++x) {
            const std::size_t p = (std::size_t{y} * kWidth + x) * 3;
            ASSERT_EQ(At<std::uint8_t>(all, p), (x * 7) % 256);
            ASSERT_EQ(At<std::uint8_t>(all, p + 1), (y * 11) % 256);
            ASSERT_EQ(At<std::uint8_t>(all, p + 2), ((x + y) * 3) % 256);
        }
    }
    // A rectangle across strips or tiles reads as the whole's.
    std::vector<std::byte> part(5 * 13 * 3);
    image.Read(11, 6, 5, 13, part);
    for (std::uint32_t y = 0; y < 13; ++y) {
        for (std::uint32_t x = 0; x < 5; ++x) {
            const std::size_t p = (std::size_t{y} * 5 + x) * 3;
            EXPECT_EQ(At<std::uint8_t>(part, p), ((x + 11) * 7) % 256);
            EXPECT_EQ(At<std::uint8_t>(part, p + 1), ((y + 6) * 11) % 256);
        }
    }
}

void ExpectFloat(const TiffImage& image) {
    ASSERT_EQ(image.Channels(), 1u);
    ASSERT_EQ(image.Type(), GridSampleType::Float32);
    const std::vector<std::byte> all = image.ReadAll();
    for (std::uint32_t y = 0; y < kHeight; ++y) {
        for (std::uint32_t x = 0; x < kWidth; ++x) {
            ASSERT_EQ(At<float>(all, std::size_t{y} * kWidth + x),
                      static_cast<float>(x * 0.5 - y * 1.25 + 0.125))
                << x << ", " << y;
        }
    }
}

TEST(Tiff, LzwWithHorizontalPredictionInStrips) {
    // LROC's colour mosaics' encoding.
    ExpectRgb(TiffImage(Fixture("rgb_lzw_predictor2.tif")));
}

TEST(Tiff, DeflateInTiles) {
    ExpectRgb(TiffImage(Fixture("rgb_tiled_deflate.tif")));
}

TEST(Tiff, PlanesStoredApart) {
    // USGS's Viking colour mosaic of Mars keeps each channel's strips apart.
    ExpectRgb(TiffImage(Fixture("rgb_planar_raw.tif")));
    ExpectRgb(TiffImage(Fixture("rgb_planar_lzw.tif")));
}

TEST(Tiff, UncompressedFloats) {
    // LOLA's elevation models' encoding.
    ExpectFloat(TiffImage(Fixture("float_raw.tif")));
}

TEST(Tiff, DeflateWithFloatingPointPrediction) {
    ExpectFloat(TiffImage(Fixture("float_deflate_predictor3.tif")));
}

TEST(Tiff, BigEndianBigTiffWithGeoTiffTags) {
    // int16 100x - 300y, LZW, the whole Earth in degrees, no-data -32768.
    const TiffImage image(Fixture("int16_bigtiff_bigendian.tif"));
    ASSERT_EQ(image.Type(), GridSampleType::Int16);
    const std::vector<std::byte> all = image.ReadAll();
    for (std::uint32_t y = 0; y < kHeight; ++y) {
        for (std::uint32_t x = 0; x < kWidth; ++x) {
            ASSERT_EQ(At<std::int16_t>(all, std::size_t{y} * kWidth + x),
                      static_cast<std::int16_t>(x * 100 - y * 300));
        }
    }
    ASSERT_TRUE(image.PixelScale().has_value());
    EXPECT_DOUBLE_EQ((*image.PixelScale())[0], 360.0 / kWidth);
    EXPECT_DOUBLE_EQ((*image.PixelScale())[1], 180.0 / kHeight);
    ASSERT_TRUE(image.Tiepoint().has_value());
    EXPECT_DOUBLE_EQ((*image.Tiepoint())[3], -180.0);
    EXPECT_DOUBLE_EQ((*image.Tiepoint())[4], 90.0);
    ASSERT_TRUE(image.NoData().has_value());
    EXPECT_EQ(*image.NoData(), -32768.0);
}

TEST(Tiff, RejectsWhatIsNotATiff) {
    EXPECT_THROW(TiffImage(Fixture("README.md")), std::runtime_error);
    const TiffImage image(Fixture("float_raw.tif"));
    std::vector<std::byte> out(4);
    EXPECT_THROW(image.Read(kWidth, 0, 1, 1, out), std::out_of_range);
    EXPECT_THROW(image.Read(0, 0, 2, 1, out), std::out_of_range) << "too little room";
}

// ----------------------------------------------------------------------------------------------
// Rasters

TEST(Raster, SamplesDecodeToTheirValues) {
    const std::int16_t negative = -300;
    const float real = 2.5f;
    EXPECT_EQ(DecodeSample(reinterpret_cast<const std::byte*>(&negative), GridSampleType::Int16),
              -300.0);
    EXPECT_EQ(DecodeSample(reinterpret_cast<const std::byte*>(&real), GridSampleType::Float32),
              2.5);
    EXPECT_EQ(ByteSize(GridSampleType::UInt8), 1u);
    EXPECT_EQ(ByteSize(GridSampleType::Int16), 2u);
    EXPECT_EQ(ByteSize(GridSampleType::Float32), 4u);
}

TEST(Raster, OpensByExtension) {
    const auto tiff = OpenRaster(Fixture("float_raw.tif"));
    EXPECT_EQ(tiff->Type(), GridSampleType::Float32);
    EXPECT_FALSE(tiff->Grid().has_value());
    const auto geo = OpenRaster(Fixture("int16_bigtiff_bigendian.tif"));
    ASSERT_TRUE(geo->Grid().has_value());
    EXPECT_DOUBLE_EQ(geo->Grid()->west, -180.0);
    EXPECT_DOUBLE_EQ(geo->Grid()->north, 90.0);
    const auto png = OpenRaster(fs::path(ERNIE_TEST_FIXTURES) / "images" / "grey16.png");
    EXPECT_EQ(png->Type(), GridSampleType::UInt16);
    EXPECT_THROW(OpenRaster(Temp("nothing.xyz")), std::runtime_error);
}

TEST(Raster, MosaicsReadAcrossTheirTiles) {
    // Two copies of the RGB fixture side by side: a rectangle over the seam reads from both.
    const fs::path directory = Temp("mosaic");
    fs::create_directories(directory);
    fs::copy_file(Fixture("rgb_lzw_predictor2.tif"), directory / "a.tif",
                  fs::copy_options::overwrite_existing);
    fs::copy_file(Fixture("rgb_tiled_deflate.tif"), directory / "b.tif",
                  fs::copy_options::overwrite_existing);
    WriteText(directory / "moon.mosaic", "# west to east\na.tif b.tif\n");
    const auto mosaic = OpenRaster(directory / "moon.mosaic");
    EXPECT_EQ(mosaic->Width(), 2 * kWidth);
    EXPECT_EQ(mosaic->Height(), kHeight);
    std::vector<std::byte> part(4 * 3);
    mosaic->Read(kWidth - 2, 5, 4, 1, part);
    for (std::uint32_t i = 0; i < 4; ++i) {
        EXPECT_EQ(At<std::uint8_t>(part, i * 3), ((kWidth - 2 + i) % kWidth * 7) % 256) << i;
    }
    WriteText(directory / "ragged.mosaic", "a.tif b.tif\na.tif\n");
    EXPECT_THROW(OpenRaster(directory / "ragged.mosaic"), std::invalid_argument);
}

// ----------------------------------------------------------------------------------------------
// DEMs

/// The map projection of `WriteDem`'s labels by default: 4 samples a degree from 40 S and 340 E.
constexpr const char* kCylindrical = "  MAP_PROJECTION_TYPE = \"SIMPLE CYLINDRICAL\"\n"
                                     "  MAP_RESOLUTION = 4 <pix/deg>\n"
                                     "  A_AXIS_RADIUS = 1737.4 <km>\n"
                                     "  MAXIMUM_LATITUDE = -40 <deg>\n"
                                     "  WESTERNMOST_LONGITUDE = 340 <deg>\n";

constexpr double kMoonRadius = 1737400.0;

/// A LOLA LDEM-style label and 16-bit image of `heights` (metres, `lines` rows from the northern)
/// in a directory of its own, half-metre steps above a 1737.4 km sphere, on map projection
/// `projection` (the IMAGE_MAP_PROJECTION object's lines). Returns the label's path.
fs::path WriteDem(const std::string& name, const std::vector<double>& heights, int lines,
                  int samples, const char* projection = kCylindrical) {
    const fs::path directory = Temp("dem_" + name);
    fs::create_directories(directory);
    std::ofstream image(directory / "TEST_DEM.IMG", std::ios::binary);
    for (const double height : heights) {
        const auto dn = static_cast<std::int16_t>(std::lround(height / 0.5));
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
    return directory / "TEST_DEM.LBL";
}

/// Heights rising 10 m a line and 1 m a sample.
std::vector<double> Rising(int lines, int samples) {
    std::vector<double> heights;
    for (int l = 0; l < lines; ++l) {
        for (int s = 0; s < samples; ++s) {
            heights.push_back(10.0 * l + s);
        }
    }
    return heights;
}

TEST(CylindricalDem, ReadsLolasLdemLabelsAndSamplesBetweenCells) {
    const CylindricalDem dem = ReadCylindricalDem(WriteDem("rising", Rising(8, 12), 8, 12));
    EXPECT_EQ(dem.lines, 8u);
    EXPECT_EQ(dem.samples, 12u);
    EXPECT_EQ(dem.referenceRadius, kMoonRadius);
    // Line 2, sample 3 is centred at 40.625 S, 340.875 E; longitudes wrap; between cells the
    // heights are bilinear.
    EXPECT_DOUBLE_EQ(dem.HeightAt(-40.625, 340.875), 23.0);
    EXPECT_DOUBLE_EQ(dem.HeightAt(-40.625, -19.125), 23.0);
    EXPECT_DOUBLE_EQ(dem.HeightAt(-40.75, 340.875), 28.0);
    EXPECT_THROW(ReadCylindricalDem("/no/such/dem.lbl"), std::runtime_error);
}

TEST(CylindricalDem, ReadsNacDtmsTrueAtTheirParallelWithNullCells) {
    // A 2 x 4 grid of 32-bit reals, as LROC's NAC DTMs hold theirs: 4 pixels a degree along the
    // meridians, true at 60 S so 2 along the parallels, its last sample PDS's null.
    const fs::path directory = Temp("dem_dtm");
    fs::create_directories(directory);
    {
        std::ofstream image(directory / "DTM.IMG", std::ios::binary);
        const float values[8] = {1.0f, 2.0f, 3.0f, -3.4028227e38f, 5.0f, 6.0f, 7.0f, 8.0f};
        image.write(reinterpret_cast<const char*>(values), sizeof(values));
    }
    WriteText(directory / "DTM.LBL", "PDS_VERSION_ID = PDS3\n"
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
                                     "END\n");
    const CylindricalDem dtm = ReadCylindricalDem(directory / "DTM.LBL");
    EXPECT_DOUBLE_EQ(dtm.standardParallelDeg, -60.0);
    EXPECT_EQ(dtm.referenceRadius, kMoonRadius) << "the projection's A axis, in km";
    // Line 1, sample 2 is centred at 59.875 S, 11.25 E (two samples a degree of longitude).
    EXPECT_NEAR(dtm.HeightAt(-59.875, 11.25), 7.0, 1e-9);
    EXPECT_TRUE(std::isnan(dtm.HeightAt(-59.625, 11.75)));
    // Where it has no data a fallback stands in.
    CylindricalDem fallback;
    fallback.referenceRadius = kMoonRadius;
    fallback.lines = fallback.samples = 1;
    fallback.pixelsPerDegree = 1.0 / 360.0;
    fallback.maxLatitudeDeg = 90.0;
    fallback.heights = {100.0f};
    const LocalTerrain terrain = MakeLocalTerrain(dtm, fallback, -59.625, 11.75, 3, 1.0);
    EXPECT_NEAR(terrain.heights.As<float>()[4], 100.0, 1e-3);
}

TEST(PolarStereographicDem, ReadsLolasPolarLabelsAndProjectsThePole) {
    // A south polar grid of 10 m pixels, the pole at the centre of 21 x 21.
    constexpr int kSize = 21;
    const fs::path label = WriteDem("polar", Rising(kSize, kSize), kSize, kSize,
                                    "  MAP_PROJECTION_TYPE = \"POLAR STEREOGRAPHIC\"\n"
                                    "  A_AXIS_RADIUS = 1737.4 <km>\n"
                                    "  CENTER_LATITUDE = -90 <deg>\n"
                                    "  CENTER_LONGITUDE = 0 <deg>\n"
                                    "  MAP_SCALE = 10 <m/pix>\n"
                                    "  LINE_PROJECTION_OFFSET = 10.0 <pix>\n"
                                    "  SAMPLE_PROJECTION_OFFSET = 10.0 <pix>\n");
    const PolarStereographicDem dem = ReadPolarStereographicDem(label);
    EXPECT_EQ(dem.lines, 21u);
    EXPECT_EQ(dem.mapScale, 10.0);
    EXPECT_EQ(dem.referenceRadius, kMoonRadius);
    const glm::dvec2 pole = dem.ToGrid(-90.0, 123.0);
    EXPECT_NEAR(pole.x, 10.0, 1e-9);
    EXPECT_NEAR(pole.y, 10.0, 1e-9);
    // 20 m from the pole: along 90 E two samples right (x), along the prime meridian two lines up
    // (y), as LOLA's grids lie.
    const double latitude =
        glm::degrees(2.0 * std::atan(20.0 / (2.0 * kMoonRadius)) - 0.5 * std::numbers::pi);
    EXPECT_NEAR(dem.HeightAt(latitude, 90.0), 10.0 * 10 + 12, 1e-6);
    EXPECT_NEAR(dem.HeightAt(latitude, 0.0), 10.0 * 8 + 10, 1e-6);
    EXPECT_NEAR(dem.HeightAt(latitude, 180.0), 10.0 * 12 + 10, 1e-6);
    EXPECT_THROW(ReadCylindricalDem(label), std::invalid_argument);
    EXPECT_THROW(ReadPolarStereographicDem(WriteDem("notpolar", std::vector<double>(4, 0.0), 2, 2)),
                 std::invalid_argument);
    // A local terrain near the pole: flat ground about the point.
    const LocalTerrain terrain = MakeLocalTerrain(dem, -89.9999, 0.0, 3, 1.0);
    EXPECT_EQ(terrain.heights.Shape()[0], 3);
}

TEST(LocalTerrain, HoldsThePlanetsCurvatureAndFramesThePoint) {
    // A flat sphere about 41 S, 341 E: the heightfield drops as the sphere does, and the frame
    // takes the point to its origin.
    const CylindricalDem dem =
        ReadCylindricalDem(WriteDem("flat", std::vector<double>(8 * 12, 0.0), 8, 12));
    const LocalTerrain terrain = MakeLocalTerrain(dem, -41.0, 341.0, 21, 1000.0);
    ASSERT_EQ(terrain.heights.Shape()[0], 21);
    const float* heights = terrain.heights.As<float>();
    EXPECT_NEAR(heights[10 * 21 + 10], 0.0, 1e-3);
    const double drop = std::sqrt(kMoonRadius * kMoonRadius - 1e4 * 1e4) - kMoonRadius;
    EXPECT_NEAR(heights[10 * 21 + 20], drop, 1e-2);
    EXPECT_EQ(terrain.origin, glm::dvec2(-10000.0, 10000.0));
    const glm::dvec3 point = kMoonRadius * BodyDirection(-41.0, 341.0);
    const glm::dvec3 local(terrain.bodyToLocal * glm::dvec4(point * 1.001, 1.0));
    EXPECT_NEAR(local.x, 0.0, 1e-6);
    EXPECT_NEAR(local.y, 0.0, 1e-6);
    EXPECT_NEAR(local.z, 0.001 * kMoonRadius, 1e-6);
    EXPECT_THROW(MakeLocalTerrain(dem, 0.0, 0.0, 1, 1.0), std::invalid_argument);
}

}  // namespace
}  // namespace ernie
