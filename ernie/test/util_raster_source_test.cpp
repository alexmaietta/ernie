// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/pds.h"
#include "ernie/util/raster_source.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using ernie::PdsImage;
using ernie::RasterSampleType;

namespace {

std::filesystem::path Temp(const std::string& name) {
    return std::filesystem::path(::testing::TempDir()) / name;
}

void WriteText(const std::filesystem::path& path,
               const std::string& text) {
    std::ofstream(path, std::ios::binary) << text;
}

}  // namespace

TEST(PdsImage, DetachedLabelOfLolaHeights) {
    // A 6 x 4 grid as LOLA's LDEM files hold it: little-endian int16
    // 100x - 300y in a .img, scaled by 0.5 about a 1737.4 km sphere,
    // described by a .lbl in capitals beside it, the whole Moon from
    // 0 degrees east; -32768 missing.
    const std::filesystem::path data = Temp("ldem_test.img");
    {
        std::ofstream file(data, std::ios::binary);
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 6; ++x) {
                const auto v =
                    static_cast<std::int16_t>(100 * x - 300 * y);
                file.write(reinterpret_cast<const char*>(&v), 2);
            }
        }
    }
    WriteText(Temp("LDEM_TEST.LBL"),
              "PDS_VERSION_ID = \"PDS3\"\n"
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
    EXPECT_EQ(image.Type(), RasterSampleType::Int16);
    EXPECT_EQ(image.Scale(), 0.5);
    EXPECT_EQ(image.Offset(), 1737400.0);
    EXPECT_EQ(image.NoData(), -32768.0);
    EXPECT_EQ(image.Label("DESCRIPTION"), "Heights, over two lines");
    ASSERT_TRUE(image.Grid().has_value());
    EXPECT_EQ(image.Grid()->west, 0.0);
    EXPECT_EQ(image.Grid()->north, 90.0);
    EXPECT_NEAR(image.Grid()->degreesAcross, 60.0, 1e-9);
    std::vector<std::byte> part(3 * 2 * 2);
    image.Read(2, 1, 3, 2, part);
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 3; ++x) {
            std::int16_t v;
            std::memcpy(&v, part.data() + (y * 3 + x) * 2, 2);
            EXPECT_EQ(v, 100 * (x + 2) - 300 * (y + 1));
        }
    }
}

TEST(PdsImage, AttachedLabelOfBigEndianFloats) {
    // A label of 128 bytes ahead of 3 x 2 big-endian floats x + 10y,
    // the image at its second record.
    std::string label =
        "PDS_VERSION_ID = PDS3\nRECORD_BYTES = 128\n^IMAGE = 2\n"
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
                bytes.push_back(
                    static_cast<char>((u >> (8 * b)) & 0xFF));
            }
        }
    }
    WriteText(Temp("attached.img"), bytes);
    const PdsImage image(Temp("attached.img"));
    EXPECT_EQ(image.Type(), RasterSampleType::Float32);
    const std::vector<std::byte> all = image.ReadAll();
    for (int i = 0; i < 6; ++i) {
        float v;
        std::memcpy(&v, all.data() + i * 4, 4);
        EXPECT_EQ(v, static_cast<float>(i % 3 + 10 * (i / 3)));
    }
    EXPECT_FALSE(image.Grid().has_value());
}

TEST(RasterSource, OpensByExtension) {
    const auto tiff =
        ernie::OpenRaster(std::filesystem::path(ERNIE_TEST_FIXTURES) /
                          "tiff" / "float_raw.tif");
    EXPECT_EQ(tiff->Type(), RasterSampleType::Float32);
    EXPECT_FALSE(tiff->Grid().has_value());
    const auto geo =
        ernie::OpenRaster(std::filesystem::path(ERNIE_TEST_FIXTURES) /
                          "tiff" / "int16_bigtiff_bigendian.tif");
    ASSERT_TRUE(geo->Grid().has_value());
    EXPECT_DOUBLE_EQ(geo->Grid()->west, -180.0);
    EXPECT_DOUBLE_EQ(geo->Grid()->north, 90.0);
    EXPECT_THROW(ernie::OpenRaster(Temp("nothing.xyz")),
                 std::runtime_error);
}
