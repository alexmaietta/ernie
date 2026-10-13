// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/tiff.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using ernie::TiffImage;

namespace {

constexpr std::uint32_t kWidth = 37;
constexpr std::uint32_t kHeight = 23;

std::filesystem::path Fixture(const std::string& name) {
    return std::filesystem::path(ERNIE_TEST_FIXTURES) / "tiff" / name;
}

/// Sample `i` of `bytes` as T.
template <class T>
T At(const std::vector<std::byte>& bytes, std::size_t i) {
    T value;
    std::memcpy(&value, bytes.data() + i * sizeof(T), sizeof(T));
    return value;
}

void ExpectRgb(const TiffImage& image) {
    ASSERT_EQ(image.Width(), kWidth);
    ASSERT_EQ(image.Height(), kHeight);
    ASSERT_EQ(image.Channels(), 3u);
    ASSERT_EQ(image.Type(), TiffImage::SampleType::UInt8);
    const std::vector<std::byte> all = image.ReadAll();
    for (std::uint32_t y = 0; y < kHeight; ++y) {
        for (std::uint32_t x = 0; x < kWidth; ++x) {
            const std::size_t p = (std::size_t{y} * kWidth + x) * 3;
            ASSERT_EQ(At<std::uint8_t>(all, p), (x * 7) % 256);
            ASSERT_EQ(At<std::uint8_t>(all, p + 1), (y * 11) % 256);
            ASSERT_EQ(At<std::uint8_t>(all, p + 2),
                      ((x + y) * 3) % 256);
        }
    }
    // A rectangle across strips or tiles reads as the whole's.
    std::vector<std::byte> part(5 * 13 * 3);
    image.Read(11, 6, 5, 13, part);
    for (std::uint32_t y = 0; y < 13; ++y) {
        for (std::uint32_t x = 0; x < 5; ++x) {
            const std::size_t p = (std::size_t{y} * 5 + x) * 3;
            EXPECT_EQ(At<std::uint8_t>(part, p),
                      ((x + 11) * 7) % 256);
            EXPECT_EQ(At<std::uint8_t>(part, p + 1),
                      ((y + 6) * 11) % 256);
        }
    }
}

void ExpectFloat(const TiffImage& image) {
    ASSERT_EQ(image.Channels(), 1u);
    ASSERT_EQ(image.Type(), TiffImage::SampleType::Float32);
    const std::vector<std::byte> all = image.ReadAll();
    for (std::uint32_t y = 0; y < kHeight; ++y) {
        for (std::uint32_t x = 0; x < kWidth; ++x) {
            ASSERT_EQ(At<float>(all, std::size_t{y} * kWidth + x),
                      static_cast<float>(x * 0.5 - y * 1.25 + 0.125))
                << x << ", " << y;
        }
    }
}

}  // namespace

TEST(Tiff, LzwWithHorizontalPredictionInStrips) {
    // LROC's colour mosaics' encoding.
    ExpectRgb(TiffImage(Fixture("rgb_lzw_predictor2.tif")));
}

TEST(Tiff, DeflateInTiles) {
    ExpectRgb(TiffImage(Fixture("rgb_tiled_deflate.tif")));
}

TEST(Tiff, PlanesStoredApart) {
    // USGS's Viking colour mosaic of Mars keeps each channel's strips
    // apart.
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
    // int16 100x - 300y, LZW, the whole Earth in degrees, no-data
    // -32768.
    const TiffImage image(Fixture("int16_bigtiff_bigendian.tif"));
    ASSERT_EQ(image.Type(), TiffImage::SampleType::Int16);
    const std::vector<std::byte> all = image.ReadAll();
    for (std::uint32_t y = 0; y < kHeight; ++y) {
        for (std::uint32_t x = 0; x < kWidth; ++x) {
            ASSERT_EQ(
                At<std::int16_t>(all, std::size_t{y} * kWidth + x),
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
}
