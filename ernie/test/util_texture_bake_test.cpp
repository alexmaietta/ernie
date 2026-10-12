// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/texture_bake.h"
#include "ernie/util/texture_cache.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>
#include <stb_image_write.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using ernie::BakeMosaic;
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
    ASSERT_EQ(read.Data().size(), baked.Data().size());
    EXPECT_TRUE(std::equal(read.Data().begin(), read.Data().end(),
                           baked.Data().begin()))
        << "the DDS file reads back";
    EXPECT_NE(read.mapping, nullptr) << "read in place";

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

TEST_F(TextureBake, EightBitBakesMatchFloatOnes) {
    // An 8-bit sRGB PNG (37 x 23, so mips round down at odd edges)
    // bakes without floats; its mips match the float chain's to
    // within a step or two of 8-bit rounding.
    const auto input = Png("eight.png", 37, 23, [](int x, int y) {
        return glm::u8vec4(x * 7, y * 11, (x * y) % 256,
                           200 + x % 50);
    });
    BakeOptions options;
    options.format = TextureFormat::Rgba8;
    const TextureData eight =
        BakeTexture(input, dir_ / "eight.dds", options);
    EXPECT_EQ(eight.format, TextureFormat::Rgba8Srgb);
    const TextureData floats = EncodeTexture(
        MipChain(ernie::LoadImageFile(input, ImageEncoding::Srgb),
                 MipFilter::Average),
        TextureFormat::Rgba8Srgb, ImageEncoding::Srgb);
    ASSERT_EQ(eight.mips, floats.mips);
    for (std::uint32_t mip = 0; mip < eight.mips; ++mip) {
        const auto a = eight.Mip(0, mip);
        const auto b = floats.Mip(0, mip);
        ASSERT_EQ(a.size(), b.size());
        int worst = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            worst = std::max(worst, std::abs(static_cast<int>(a[i]) -
                                             static_cast<int>(b[i])));
        }
        EXPECT_LE(worst, 2) << "mip " << mip;
    }
}

TEST_F(TextureBake, MosaicsBakeAsTheirJoinedImage) {
    // A 2 x 2 mosaic bakes byte for byte as the image its tiles make:
    // 64 x 48 tiles keep mips 0 to 2 per tile, 30 x 30 ones none.
    const auto colour = [](int x, int y) {
        return glm::u8vec4(x * 3, y * 5, (x ^ y) & 0xff, 255);
    };
    for (const auto& [tileWidth, tileHeight] :
         {std::pair{64, 48}, std::pair{30, 30}}) {
        const std::string size = std::to_string(tileWidth) + "_" +
                                 std::to_string(tileHeight);
        const auto whole = Png("whole" + size + ".png", 2 * tileWidth,
                               2 * tileHeight, colour);
        std::vector<std::filesystem::path> tiles;
        for (int row = 0; row < 2; ++row) {
            for (int column = 0; column < 2; ++column) {
                tiles.push_back(
                    Png("tile" + size + std::to_string(row) +
                            std::to_string(column) + ".png",
                        tileWidth, tileHeight, [&](int x, int y) {
                            return colour(column * tileWidth + x,
                                          row * tileHeight + y);
                        }));
            }
        }
        for (const TextureFormat format :
             {TextureFormat::Rgba8, TextureFormat::Bc7}) {
            BakeOptions options;
            options.format = format;
            BakeTexture(whole, dir_ / "whole.dds", options);
            const TextureData layout =
                BakeMosaic(tiles, 2, dir_ / "mosaic.dds", options);
            EXPECT_EQ(layout.width,
                      static_cast<std::uint32_t>(2 * tileWidth));
            const TextureData a = LoadTextureFile(dir_ / "whole.dds");
            const TextureData b =
                LoadTextureFile(dir_ / "mosaic.dds");
            ASSERT_EQ(a.mips, b.mips);
            ASSERT_EQ(a.Data().size(), b.Data().size());
            EXPECT_TRUE(std::equal(a.Data().begin(), a.Data().end(),
                                   b.Data().begin()))
                << size << ", format " << static_cast<int>(format);
        }
    }

    const std::vector<std::filesystem::path> three(
        3, dir_ / "tile64_4800.png");
    EXPECT_THROW(BakeMosaic(three, 2, dir_ / "bad.dds"),
                 std::invalid_argument)
        << "not whole rows";
    const std::vector<std::filesystem::path> mixed{
        dir_ / "tile64_4800.png", dir_ / "tile30_3000.png"};
    EXPECT_THROW(BakeMosaic(mixed, 2, dir_ / "bad.dds"),
                 std::invalid_argument)
        << "two sizes";
    BakeOptions normal;
    normal.role = TextureRole::Normal;
    EXPECT_THROW(BakeMosaic(mixed, 2, dir_ / "bad.dds", normal),
                 std::invalid_argument);
}

TEST_F(TextureBake, DecodedBlocksMatchTheirSources) {
    // An 8 x 8 diagonal gradient's mips (its colours on one line, as
    // BC1 and BC7's one subset fit) baked to BC1, BC4, BC5 and BC7
    // decode to the same channels near it; BC2 is not decoded.
    Image image(8, 8);
    for (std::uint32_t y = 0; y < 8; ++y) {
        for (std::uint32_t x = 0; x < 8; ++x) {
            const float t = static_cast<float>(x + y) / 14.0f;
            image.At(x, y) = glm::vec4(t, 1.0f - t, 0.5f * t, 1.0f);
        }
    }
    const std::vector<Image> mips =
        MipChain(image, MipFilter::Average);
    const std::pair<TextureFormat, TextureFormat> kFormats[] = {
        {TextureFormat::Bc1Srgb, TextureFormat::Rgba8Srgb},
        {TextureFormat::Bc4, TextureFormat::R8},
        {TextureFormat::Bc5, TextureFormat::Rg8},
        {TextureFormat::Bc7, TextureFormat::Rgba8}};
    for (const auto& [format, decodedFormat] : kFormats) {
        const ImageEncoding encoding = ernie::InfoOf(format).srgb
                                           ? ImageEncoding::Srgb
                                           : ImageEncoding::Linear;
        const TextureData decoded = ernie::DecodeBlocks(
            EncodeTexture(mips, format, encoding));
        ASSERT_EQ(decoded.format, decodedFormat);
        ASSERT_EQ(decoded.mips, 4u);
        const std::uint32_t channels =
            ernie::InfoOf(decodedFormat).channels;
        const std::vector<std::uint8_t> source =
            ernie::ToRgba8(image, encoding);
        const std::span<const std::byte> texels = decoded.Mip(0, 0);
        ASSERT_EQ(texels.size(), std::size_t{64} * channels);
        // BC1's four colours a block, 5:6:5 bit endpoints, err most.
        const int tolerance =
            format == TextureFormat::Bc1Srgb ? 26 : 12;
        for (std::size_t i = 0; i < 64; ++i) {
            for (std::uint32_t c = 0; c < channels; ++c) {
                EXPECT_NEAR(
                    static_cast<int>(texels[i * channels + c]),
                    static_cast<int>(source[i * 4 + c]), tolerance)
                    << "format " << static_cast<int>(format)
                    << ", texel " << i << ", channel " << c;
            }
        }
    }
    TextureData bc2;
    bc2.format = TextureFormat::Bc2;
    bc2.width = 4;
    bc2.height = 4;
    bc2.bytes.resize(16);
    bc2.offsets = {0};
    EXPECT_THROW(ernie::DecodeBlocks(bc2), std::invalid_argument);
}

TEST_F(TextureBake, TheTextureCacheKeepsTheRecentlyUsed) {
    // Three 16 x 16 RGBA8 textures in a cache with room for two: the
    // one least recently used goes. Without a folder nothing is kept.
    TextureData texture = EncodeTexture(
        std::vector<Image>{Image(16, 16, glm::vec4(0.25f))},
        TextureFormat::Rgba8, ImageEncoding::Linear);
    const std::uintmax_t bytes = texture.bytes.size() + 200;
    const ernie::TextureCache cache(dir_ / "cache", 2 * bytes);
    EXPECT_TRUE(cache.Enabled());
    EXPECT_FALSE(cache.Read("a"));
    cache.Write("a", texture);
    texture.bytes[0] = std::byte{1};
    cache.Write("b", texture);
    const std::optional<TextureData> a = cache.Read("a");
    ASSERT_TRUE(a);
    EXPECT_EQ(a->width, 16u);
    EXPECT_EQ(a->Mip(0, 0)[0], std::byte{64});
    cache.Write("c", texture);
    EXPECT_TRUE(cache.Read("a"));
    EXPECT_FALSE(cache.Read("b"));
    const std::optional<TextureData> c = cache.Read("c");
    ASSERT_TRUE(c);
    EXPECT_EQ(c->Mip(0, 0)[0], std::byte{1});

    const ernie::TextureCache none(std::nullopt);
    EXPECT_FALSE(none.Enabled());
    none.Write("a", texture);
    EXPECT_FALSE(none.Read("a"));
}
