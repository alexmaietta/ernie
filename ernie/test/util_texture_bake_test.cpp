// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/texture_bake.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>
#include <stb_image_write.h>

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using ernie::BakeOptions;
using ernie::BakeTexture;
using ernie::EncodeTexture;
using ernie::Image;
using ernie::ImageEncoding;
using ernie::LoadTextureFile;
using ernie::MipChain;
using ernie::MipFilter;
using ernie::TextureData;
using ernie::TextureFormat;
using ernie::TextureRole;

namespace {

class TextureBake : public testing::Test {
  protected:
    void SetUp() override {
        dir_ = std::filesystem::path(::testing::TempDir()) /
               ("ernie_texture_bake_" +
                std::string(::testing::UnitTest::GetInstance()
                                ->current_test_info()
                                ->name()));
        std::filesystem::create_directories(dir_);
    }

    void TearDown() override {
        std::filesystem::remove_all(dir_);
    }

    /// A `width` x `height` RGBA PNG of `colour(x, y)`.
    template <typename Colour>
    std::filesystem::path Png(const std::string& name, int width,
                              int height, Colour colour) const {
        std::vector<std::uint8_t> texels;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const glm::u8vec4 c = colour(x, y);
                texels.insert(texels.end(), {c.r, c.g, c.b, c.a});
            }
        }
        const auto path = dir_ / name;
        EXPECT_NE(stbi_write_png(path.string().c_str(), width, height,
                                 4, texels.data(), width * 4),
                  0);
        return path;
    }

    std::filesystem::path dir_;
};

}  // namespace

TEST_F(TextureBake, ColourBakesToMippedSrgbBc7) {
    // 100 x 60: mips down to 1 x 1, the edge blocks partly past the
    // image.
    const auto input = Png("colour.png", 100, 60, [](int x, int y) {
        return glm::u8vec4(x * 2, y * 4, 128, 255);
    });
    const auto output = dir_ / "colour.dds";
    const TextureData baked = BakeTexture(input, output);
    EXPECT_EQ(baked.format, TextureFormat::Bc7Srgb);
    EXPECT_EQ(baked.width, 100u);
    EXPECT_EQ(baked.height, 60u);
    EXPECT_EQ(baked.mips, 7u);
    // 25 x 15 blocks of 16 bytes, then smaller mips.
    EXPECT_EQ(baked.Mip(0, 0).size(), 25u * 15u * 16u);

    const TextureData read = LoadTextureFile(output);
    EXPECT_EQ(read.format, baked.format);
    EXPECT_EQ(read.mips, baked.mips);
    EXPECT_EQ(read.bytes, baked.bytes) << "the DDS file reads back";

    BakeOptions linear;
    linear.srgb = false;
    linear.format = TextureFormat::Bc1;
    EXPECT_EQ(BakeTexture(input, dir_ / "linear.dds", linear).format,
              TextureFormat::Bc1);
}

TEST_F(TextureBake, NormalMapsBakeToBc5AndDataStaysLinear) {
    const auto input = Png("normal.png", 32, 32, [](int, int) {
        return glm::u8vec4(191, 128, 238, 255);
    });
    BakeOptions normal;
    normal.role = TextureRole::Normal;
    const TextureData baked =
        BakeTexture(input, dir_ / "normal.dds", normal);
    EXPECT_EQ(baked.format, TextureFormat::Bc5);
    EXPECT_EQ(baked.mips, 6u);
    EXPECT_EQ(LoadTextureFile(dir_ / "normal.dds").format,
              TextureFormat::Bc5);

    BakeOptions data;
    data.role = TextureRole::Data;
    EXPECT_EQ(BakeTexture(input, dir_ / "data.dds", data).format,
              TextureFormat::Bc7);
    data.format = TextureFormat::Bc4;
    EXPECT_EQ(BakeTexture(input, dir_ / "data4.dds", data).format,
              TextureFormat::Bc4);
}

TEST_F(TextureBake, EncodingRejectsWhatItCannotWrite) {
    const Image image(8, 8, glm::vec4(0.5f));
    const std::vector<Image> mips =
        MipChain(image, MipFilter::Average);
    EXPECT_THROW(EncodeTexture(mips, TextureFormat::Bc6hUfloat,
                               ImageEncoding::Linear),
                 std::invalid_argument);
    std::vector<Image> skipped{mips[0], mips[2]};
    EXPECT_THROW(EncodeTexture(skipped, TextureFormat::Bc7,
                               ImageEncoding::Linear),
                 std::invalid_argument);
    // RGBA8 stores the 8-bit values as they are.
    const TextureData plain = EncodeTexture(
        mips, TextureFormat::Rgba8, ImageEncoding::Linear);
    EXPECT_EQ(plain.Mip(0, 0).size(), 8u * 8u * 4u);
    EXPECT_EQ(static_cast<int>(plain.Mip(0, 0)[0]), 128);
    EXPECT_THROW(BakeTexture(dir_ / "missing.png", dir_ / "out.dds"),
                 std::runtime_error);
}
