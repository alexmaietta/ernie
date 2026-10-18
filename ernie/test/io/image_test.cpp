// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/io/image.h"

#include <ImfRgbaFile.h>
#include <glm/glm.hpp>
#include <gtest/gtest.h>
#include <stb_image_write.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using ernie::DecodeSlopeDeviation;
using ernie::EncodeSlopeDeviation;
using ernie::Image;
using ernie::ImageEncoding;
using ernie::LinearToSrgb;
using ernie::LoadImageFile;
using ernie::MipChain;
using ernie::MipFilter;
using ernie::SrgbToLinear;
using ernie::Tensor;
using ernie::ToRgba8;
using ernie::WriteExr;

namespace {

class ImageFiles : public testing::Test {
  protected:
    // A directory of the test's own: ctest runs the tests as
    // processes side by side, and gtest's random seed comes from the
    // clock, so tests started together would share one by the seed.
    void SetUp() override {
        const std::string test = ::testing::UnitTest::GetInstance()->current_test_info()->name();
        dir_ = std::filesystem::temp_directory_path() /
               ("ernie_image_test_" + test + "_" + std::to_string(std::random_device{}()));
        std::filesystem::create_directories(dir_);
    }

    void TearDown() override {
        std::filesystem::remove_all(dir_);
    }

    std::filesystem::path dir_;
};

}  // namespace

TEST(Srgb, MatchesTheStandardCurveAndRoundTrips) {
    EXPECT_FLOAT_EQ(SrgbToLinear(0.0f), 0.0f);
    EXPECT_FLOAT_EQ(SrgbToLinear(1.0f), 1.0f);
    EXPECT_NEAR(SrgbToLinear(0.5f), 0.21404f, 1e-5f);
    EXPECT_NEAR(SrgbToLinear(0.02f), 0.02f / 12.92f, 1e-7f);
    for (float v = 0.0f; v <= 1.0f; v += 0.05f) {
        EXPECT_NEAR(LinearToSrgb(SrgbToLinear(v)), v, 1e-5f);
    }
}

TEST(MipChain, HalvesDownToOneTexelAndAveragesBlocks) {
    Image image(5, 2);
    for (std::uint32_t y = 0; y < 2; ++y) {
        for (std::uint32_t x = 0; x < 5; ++x) {
            image.At(x, y) = glm::vec4(static_cast<float>(x + 10 * y));
        }
    }
    const std::vector<Image> mips = MipChain(image, MipFilter::Average);
    ASSERT_EQ(mips.size(), 3u);
    EXPECT_EQ(mips[1].width, 2u);
    EXPECT_EQ(mips[1].height, 1u);
    EXPECT_EQ(mips[2].width, 1u);
    EXPECT_EQ(mips[2].height, 1u);
    EXPECT_FLOAT_EQ(mips[1].At(0, 0).r, (0 + 1 + 10 + 11) / 4.0f);
    EXPECT_FLOAT_EQ(mips[1].At(1, 0).r, (2 + 3 + 12 + 13) / 4.0f);
    EXPECT_EQ(MipChain(Image(1, 1), MipFilter::Average).size(), 1u);
    EXPECT_EQ(MipChain(Image(1, 8), MipFilter::Average).size(), 4u);
}

TEST(MipChain, UnitVectorFilterRenormalisesNormals) {
    Image normals(2, 1);
    const glm::vec3 a = glm::normalize(glm::vec3(0.8f, 0.0f, 0.6f));
    const glm::vec3 b = glm::normalize(glm::vec3(-0.8f, 0.0f, 0.6f));
    normals.At(0, 0) = glm::vec4(0.5f * a + 0.5f, 1.0f);
    normals.At(1, 0) = glm::vec4(0.5f * b + 0.5f, 1.0f);
    const Image top = MipChain(normals, MipFilter::UnitVector).back();
    const glm::vec3 n = 2.0f * glm::vec3(top.At(0, 0)) - 1.0f;
    EXPECT_NEAR(glm::length(n), 1.0f, 1e-5f);
    EXPECT_NEAR(n.z, 1.0f, 1e-5f);

    Image opposite(2, 1);
    opposite.At(0, 0) = glm::vec4(1.0f, 0.5f, 0.5f, 1.0f);
    opposite.At(1, 0) = glm::vec4(0.0f, 0.5f, 0.5f, 1.0f);
    const glm::vec4 flat = MipChain(opposite, MipFilter::UnitVector).back().At(0, 0);
    EXPECT_FLOAT_EQ(flat.b, 1.0f) << "degenerate averages fall back to straight up";
}

TEST(MipChain, NormalSlopesKeepTheSlopesAveragingHides) {
    // Slopes of +0.5 and -0.5 along x: flat on average, deviating by
    // 0.5 along x and 0 along y, so sqrt(0.125) per axis.
    const auto tilted = [](float slope) {
        const glm::vec3 n = glm::normalize(glm::vec3(-slope, 0.0f, 1.0f));
        return glm::vec4(0.5f * n + 0.5f, 1.0f);
    };
    Image normals(2, 2);
    normals.At(0, 0) = normals.At(1, 1) = tilted(0.5f);
    normals.At(1, 0) = normals.At(0, 1) = tilted(-0.5f);
    const std::vector<Image> mips = MipChain(normals, MipFilter::NormalSlopes);
    EXPECT_EQ(mips[0].At(0, 0).a, 0.0f) << "level 0's texels are points";
    const glm::vec4 top = mips.back().At(0, 0);
    EXPECT_NEAR(top.b, 1.0f, 1e-6f) << "flat on average";
    EXPECT_NEAR(DecodeSlopeDeviation(top.a), std::sqrt(0.125f), 1e-5f);

    // Deviations within texels carry up a level, and add to those
    // between them.
    Image twice(4, 2);
    for (std::uint32_t x = 0; x < 4; ++x) {
        twice.At(x, 0) = tilted(x % 2 == 0 ? 0.5f : -0.5f);
        twice.At(x, 1) = tilted(x % 2 == 0 ? -0.5f : 0.5f);
    }
    EXPECT_NEAR(DecodeSlopeDeviation(MipChain(twice, MipFilter::NormalSlopes).back().At(0, 0).a),
                std::sqrt(0.125f), 1e-5f);
    EXPECT_NEAR(DecodeSlopeDeviation(EncodeSlopeDeviation(3.0f)), 3.0f, 1e-5f);
}

TEST(ImageMean, AveragesEveryPixel) {
    Image image(2, 1);
    image.At(0, 0) = glm::vec4(0.2f, 0.4f, 0.6f, 1.0f);
    image.At(1, 0) = glm::vec4(0.4f, 0.0f, 0.2f, 0.0f);
    EXPECT_NEAR(image.Mean().r, 0.3f, 1e-6f);
    EXPECT_NEAR(image.Mean().a, 0.5f, 1e-6f);
    EXPECT_EQ(Image().Mean(), glm::vec4(0.0f));
}

TEST(ToRgba8, QuantisesWithOrWithoutTheSrgbCurve) {
    Image image(2, 1);
    image.At(0, 0) = glm::vec4(SrgbToLinear(128.0f / 255.0f), 1.5f, -1.0f, 0.5f);
    image.At(1, 0) = glm::vec4(1.0f);
    const auto srgb = ToRgba8(image, ImageEncoding::Srgb);
    EXPECT_EQ(srgb[0], 128);
    EXPECT_EQ(srgb[1], 255);
    EXPECT_EQ(srgb[2], 0);
    EXPECT_EQ(srgb[3], 128) << "alpha is linear";
    const auto linear = ToRgba8(image, ImageEncoding::Linear);
    EXPECT_EQ(linear[0], 55);
}

TEST_F(ImageFiles, LoadsEightBitPngWithEitherEncoding) {
    const std::array<std::uint8_t, 8> rgba = {128, 0, 255, 255, 64, 32, 16, 128};
    const auto path = dir_ / "colour.png";
    ASSERT_NE(stbi_write_png(path.string().c_str(), 2, 1, 4, rgba.data(), 8), 0);

    const Image linear = LoadImageFile(path, ImageEncoding::Linear);
    ASSERT_EQ(linear.width, 2u);
    ASSERT_EQ(linear.height, 1u);
    EXPECT_NEAR(linear.At(0, 0).r, 128.0f / 255.0f, 1e-6f);
    EXPECT_NEAR(linear.At(1, 0).a, 128.0f / 255.0f, 1e-6f);
    const Image srgb = LoadImageFile(path, ImageEncoding::Srgb);
    EXPECT_NEAR(srgb.At(0, 0).r, SrgbToLinear(128.0f / 255.0f), 1e-6f);
    EXPECT_NEAR(srgb.At(1, 0).a, 128.0f / 255.0f, 1e-6f) << "alpha stays linear";

    const std::array<std::uint8_t, 2> grey = {0, 200};
    const auto greyPath = dir_ / "GREY.PNG";
    ASSERT_NE(stbi_write_png(greyPath.string().c_str(), 2, 1, 1, grey.data(), 2), 0);
    const Image expanded = LoadImageFile(greyPath, ImageEncoding::Linear);
    EXPECT_NEAR(expanded.At(1, 0).g, 200.0f / 255.0f, 1e-6f);
    EXPECT_FLOAT_EQ(expanded.At(1, 0).a, 1.0f);
}

TEST_F(ImageFiles, LoadsSixteenBitPng) {
    const Image image =
        LoadImageFile(std::filesystem::path(ERNIE_TEST_FIXTURES) / "images" / "grey16.png",
                      ImageEncoding::Linear);
    ASSERT_EQ(image.width, 2u);
    EXPECT_FLOAT_EQ(image.At(0, 0).r, 0.0f);
    EXPECT_FLOAT_EQ(image.At(1, 0).b, 1.0f);
    EXPECT_NEAR(image.At(0, 1).r, 32768.0f / 65535.0f, 1e-7f);
    EXPECT_NEAR(image.At(1, 1).r, 1000.0f / 65535.0f, 1e-7f) << "16-bit precision kept";
}

TEST_F(ImageFiles, LoadsJpeg) {
    const std::array<std::uint8_t, 3 * 16> grey([] {
        std::array<std::uint8_t, 3 * 16> a{};
        a.fill(100);
        return a;
    }());
    const auto path = dir_ / "flat.jpg";
    ASSERT_NE(stbi_write_jpg(path.string().c_str(), 4, 4, 3, grey.data(), 95), 0);
    const Image image = LoadImageFile(path, ImageEncoding::Linear);
    EXPECT_EQ(image.width, 4u);
    EXPECT_NEAR(image.At(2, 2).r, 100.0f / 255.0f, 3.0f / 255.0f);
}

namespace {

/// Writes `rgba` (width x height, row by row) as a half-float EXR
/// with `compression`.
void WriteExr(const std::filesystem::path& path, const std::vector<Imf::Rgba>& rgba, int width,
              int height, Imf::Compression compression) {
    Imf::RgbaOutputFile file(path.string().c_str(), width, height, Imf::WRITE_RGBA, 1.0f,
                             Imath::V2f(0.0f, 0.0f), 1.0f, Imf::INCREASING_Y, compression);
    file.setFrameBuffer(rgba.data(), 1, static_cast<std::size_t>(width));
    file.writePixels(height);
}

}  // namespace

TEST_F(ImageFiles, LoadsFloatExrAsLinear) {
    const std::vector<Imf::Rgba> rgba = {{0.25f, 2.0f, -0.5f, 1.0f}, {0.0f, 0.125f, 10.0f, 0.5f}};
    const auto path = dir_ / "data.exr";
    WriteExr(path, rgba, 2, 1, Imf::ZIP_COMPRESSION);
    const Image image = LoadImageFile(path, ImageEncoding::Srgb);
    ASSERT_EQ(image.width, 2u);
    EXPECT_FLOAT_EQ(image.At(0, 0).r, 0.25f) << "EXR ignores the sRGB encoding";
    EXPECT_FLOAT_EQ(image.At(0, 0).g, 2.0f);
    EXPECT_FLOAT_EQ(image.At(0, 0).b, -0.5f);
    EXPECT_FLOAT_EQ(image.At(1, 0).b, 10.0f);
    EXPECT_FLOAT_EQ(image.At(1, 0).a, 0.5f);
}

TEST_F(ImageFiles, LoadsExrInLossyAndLosslessCompressions) {
    // A smooth gradient, which the lossy DWA codecs keep closely;
    // Poly Haven's normal maps are DWAA.
    constexpr int kSize = 64;
    std::vector<Imf::Rgba> rgba;
    for (int y = 0; y < kSize; ++y) {
        for (int x = 0; x < kSize; ++x) {
            rgba.emplace_back(static_cast<float>(x) / kSize, static_cast<float>(y) / kSize, 0.5f,
                              1.0f);
        }
    }
    for (const auto& [compression, tolerance] :
         {std::pair(Imf::PIZ_COMPRESSION, 0.0f), std::pair(Imf::DWAA_COMPRESSION, 0.02f),
          std::pair(Imf::DWAB_COMPRESSION, 0.02f)}) {
        const auto path = dir_ / ("gradient" + std::to_string(compression) + ".exr");
        WriteExr(path, rgba, kSize, kSize, compression);
        const Image image = LoadImageFile(path, ImageEncoding::Linear);
        ASSERT_EQ(image.width, static_cast<std::uint32_t>(kSize));
        EXPECT_NEAR(image.At(40, 10).r, 40.0f / kSize, tolerance + 1e-3f) << compression;
        EXPECT_NEAR(image.At(40, 10).g, 10.0f / kSize, tolerance + 1e-3f) << compression;
        EXPECT_NEAR(image.At(40, 10).b, 0.5f, tolerance + 1e-3f) << compression;
    }
}

TEST_F(ImageFiles, ReportsMissingCorruptAndUnsupportedFiles) {
    EXPECT_THROW(LoadImageFile(dir_ / "missing.png", ImageEncoding::Linear), std::runtime_error);
    EXPECT_THROW(LoadImageFile(dir_ / "missing.exr", ImageEncoding::Linear), std::runtime_error);
    std::ofstream(dir_ / "corrupt.png") << "not a png";
    EXPECT_THROW(LoadImageFile(dir_ / "corrupt.png", ImageEncoding::Linear), std::runtime_error);
    std::ofstream(dir_ / "texture.tga") << "tga";
    EXPECT_THROW(LoadImageFile(dir_ / "texture.tga", ImageEncoding::Linear), std::runtime_error);
}

TEST_F(ImageFiles, WritesExrThatReadsBack) {
    // Two rows of three RGBA floats, beyond [0, 1] as HDR is.
    std::vector<float> values(2 * 3 * 4);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = 0.25f * static_cast<float>(i);
    }
    const Tensor image = Tensor::Copy<float>(values, {2, 3, 4});
    WriteExr(dir_ / "hdr.exr", image);
    const Image read = LoadImageFile(dir_ / "hdr.exr", ImageEncoding::Linear);
    ASSERT_EQ(read.width, 3u);
    ASSERT_EQ(read.height, 2u);
    // Row 1, column 2: elements 20 to 23.
    EXPECT_FLOAT_EQ(read.At(2, 1).r, 5.0f);
    EXPECT_FLOAT_EQ(read.At(2, 1).a, 5.75f);
}

TEST_F(ImageFiles, WritesExrOfIdsAndRejectsOtherShapes) {
    const std::vector<std::uint32_t> ids{1, 2, 3, 4};
    WriteExr(dir_ / "ids.exr", Tensor::Copy<std::uint32_t>(ids, {2, 2}));
    EXPECT_GT(std::filesystem::file_size(dir_ / "ids.exr"), 0u);
    const std::vector<float> flat{1.0f, 2.0f};
    EXPECT_THROW(WriteExr(dir_ / "flat.exr", Tensor::Copy<float>(flat, {2})),
                 std::invalid_argument);
}

TEST_F(ImageFiles, ReadsExrFixturesOfOtherWriters) {
    const std::filesystem::path exr = std::filesystem::path(ERNIE_TEST_FIXTURES) / "exr";
    const Image window = LoadImageFile(exr / "rgba_half_window.exr", ImageEncoding::Linear);
    ASSERT_EQ(window.width, 5u) << "the data window, not the display window";
    ASSERT_EQ(window.height, 3u);
    for (std::uint32_t y = 0; y < 3; ++y) {
        for (std::uint32_t x = 0; x < 5; ++x) {
            EXPECT_EQ(window.At(x, y),
                      glm::vec4(0.25f * float(x), std::exp2(float(y)), -0.5f, 0.5f))
                << x << ", " << y;
        }
    }

    const Image zip = LoadImageFile(exr / "rgb_float_zip.exr", ImageEncoding::Linear);
    ASSERT_EQ(zip.width, 16u);
    ASSERT_EQ(zip.height, 16u);
    for (const auto& [x, y] : {std::pair(0u, 0u), std::pair(7u, 3u), std::pair(15u, 15u)}) {
        EXPECT_EQ(zip.At(x, y),
                  glm::vec4(float(x) / 16.0f, float(y) / 16.0f, 100.0f + float(x), 1.0f))
            << "no alpha reads as 1: " << x << ", " << y;
    }

    const Image grey = LoadImageFile(exr / "y_half_zips.exr", ImageEncoding::Linear);
    ASSERT_EQ(grey.width, 4u);
    EXPECT_EQ(grey.At(3, 1), glm::vec4(glm::vec3(2.5f), 1.0f)) << "luminance reads as grey";
    EXPECT_EQ(grey.At(1, 0), glm::vec4(glm::vec3(0.5f), 1.0f));
}
