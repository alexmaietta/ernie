// SPDX-License-Identifier: BSD-3-Clause
/// @file texture_test.cpp
/// @brief Tests of stored textures: formats, DDS and KTX2 reading, bakes, BC decoding and the
/// cache.
#include "ernie/io/texture.h"

#include "ernie/util/bytes.h"

#include <glm/glm.hpp>
#include <gtest/gtest.h>
#include <stb_image_write.h>
#include <zstd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ernie {

// ----------------------------------------------------------------------------------------------
// Stored textures

namespace {

using Bytes = std::vector<std::byte>;

void Put32(Bytes& file, std::size_t at, std::uint32_t value) {
    if (file.size() < at + 4) {
        file.resize(at + 4);
    }
    std::memcpy(file.data() + at, &value, 4);
}

void Put64(Bytes& file, std::size_t at, std::uint64_t value) {
    Put32(file, at, static_cast<std::uint32_t>(value));
    Put32(file, at + 4, static_cast<std::uint32_t>(value >> 32));
}

/// Bytes counting up from `first`, so each mip's bytes are its own.
Bytes Ramp(std::size_t count, std::size_t first = 0) {
    Bytes bytes(count);
    for (std::size_t i = 0; i < count; ++i) {
        bytes[i] = static_cast<std::byte>((first + i) & 0xffu);
    }
    return bytes;
}

/// A DDS header (`DDS_HEADER`) of `width` x `height` with `mips`
/// mips, its pixel format's flags, FourCC, bit count and masks, then
/// `payload`. A FourCC of "DX10" takes `dx10` (format, dimension,
/// misc flags, array size) after the header.
Bytes Dds(std::uint32_t width, std::uint32_t height, std::uint32_t mips, std::uint32_t pixelFlags,
          std::uint32_t fourCc, const Bytes& payload, std::array<std::uint32_t, 4> dx10 = {},
          std::uint32_t caps2 = 0, std::uint32_t bits = 0,
          std::array<std::uint32_t, 4> masks = {}) {
    Bytes file;
    Put32(file, 0, FourCc("DDS "));
    Put32(file, 4, 124);
    Put32(file, 8, 0x1007 | (mips > 1 ? 0x20000u : 0u));
    Put32(file, 12, height);
    Put32(file, 16, width);
    Put32(file, 28, mips);
    Put32(file, 76, 32);
    Put32(file, 80, pixelFlags);
    Put32(file, 84, fourCc);
    Put32(file, 88, bits);
    for (std::size_t i = 0; i < 4; ++i) {
        Put32(file, 92 + 4 * i, masks[i]);
    }
    Put32(file, 112, caps2);
    Put32(file, 124, 0);
    if (fourCc == FourCc("DX10")) {
        for (std::size_t i = 0; i < 4; ++i) {
            Put32(file, 128 + 4 * i, dx10[i]);
        }
        Put32(file, 144, 0);
    }
    file.insert(file.end(), payload.begin(), payload.end());
    return file;
}

constexpr std::uint32_t kFourCcFlag = 0x4;

/// A KTX2 file of VkFormat `vkFormat`, `width` x `height`, with
/// `layers` layers (0 for none) and one image per level of `levels`,
/// stored smallest level first (the specification's suggestion) at
/// the offsets the level index gives.
Bytes Ktx2(std::uint32_t vkFormat, std::uint32_t width, std::uint32_t height, std::uint32_t layers,
           const std::vector<Bytes>& levels, std::uint32_t supercompression = 0) {
    const std::array<std::uint8_t, 12> identifier{0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32,
                                                  0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
    Bytes file(80 + 24 * levels.size());
    std::memcpy(file.data(), identifier.data(), 12);
    Put32(file, 12, vkFormat);
    Put32(file, 16, 1);
    Put32(file, 20, width);
    Put32(file, 24, height);
    Put32(file, 32, layers);
    Put32(file, 36, 1);
    Put32(file, 40, static_cast<std::uint32_t>(levels.size()));
    Put32(file, 44, supercompression);
    for (std::size_t level = levels.size(); level-- > 0;) {
        const std::size_t entry = 80 + 24 * level;
        Put64(file, entry, file.size());
        Put64(file, entry + 8, levels[level].size());
        Put64(file, entry + 16, levels[level].size());
        file.insert(file.end(), levels[level].begin(), levels[level].end());
    }
    return file;
}

Bytes Slice(const Bytes& bytes, std::size_t at, std::size_t count) {
    return Bytes(bytes.begin() + static_cast<std::ptrdiff_t>(at),
                 bytes.begin() + static_cast<std::ptrdiff_t>(at + count));
}

Bytes Copy(std::span<const std::byte> bytes) {
    return Bytes(bytes.begin(), bytes.end());
}

}  // namespace

TEST(TextureFile, BlockFormatsCountWholeBlocks) {
    EXPECT_EQ(ImageBytes(TextureFormat::Bc1, 8, 8), 32u);
    EXPECT_EQ(ImageBytes(TextureFormat::Bc1, 5, 5), 32u);
    EXPECT_EQ(ImageBytes(TextureFormat::Bc7, 1, 1), 16u);
    EXPECT_EQ(ImageBytes(TextureFormat::Rgba8, 3, 2), 24u);
    EXPECT_EQ(ImageBytes(TextureFormat::Rgba16F, 2, 2), 32u);
    EXPECT_EQ(WithSrgb(TextureFormat::Bc7, true), TextureFormat::Bc7Srgb);
    EXPECT_EQ(WithSrgb(TextureFormat::Rgba8Srgb, false), TextureFormat::Rgba8);
    EXPECT_EQ(WithSrgb(TextureFormat::Bc5, true), TextureFormat::Bc5);
    EXPECT_TRUE(IsTextureFile("a/b.DDS"));
    EXPECT_TRUE(IsTextureFile("b.ktx2"));
    EXPECT_FALSE(IsTextureFile("b.png"));
    EXPECT_FALSE(IsTextureFile("b.ktx"));
}

TEST(TextureFile, MipsHalveDownToOneTexel) {
    EXPECT_EQ(MipCount(1, 1), 1u);
    EXPECT_EQ(MipCount(8, 3), 4u);
    EXPECT_EQ(MipCount(9, 1), 4u);
    EXPECT_EQ(MipSide(5, 1), 2u);
    EXPECT_EQ(MipSide(5, 3), 1u);
    EXPECT_EQ(MipSide(1, 4), 1u);
}

TEST(TextureFile, EveryFormatRoundTripsThroughDds) {
    const std::filesystem::path path =
        std::filesystem::path(testing::TempDir()) / "ernie_texture_round_trip.dds";
    for (std::uint32_t value = 0; value <= static_cast<std::uint32_t>(TextureFormat::Bc7Srgb);
         ++value) {
        TextureData texture;
        texture.format = static_cast<TextureFormat>(value);
        texture.width = 8;
        texture.height = 4;
        texture.bytes = Ramp(ImageBytes(texture.format, 8, 4));
        texture.offsets = {0};
        WriteDdsFile(path, texture);
        const TextureData read = LoadTextureFile(path);
        EXPECT_EQ(read.format, texture.format) << value;
        EXPECT_EQ(Copy(read.Data()), texture.bytes) << value;
    }
    std::filesystem::remove(path);
}

TEST(TextureFile, LegacyDdsReadsDxt1WithEveryMip) {
    // 8 x 8 down to 1 x 1: 2 x 2 blocks, then one block each.
    const Bytes payload = Ramp(32 + 8 + 8 + 8);
    const TextureData texture = ReadDds(Dds(8, 8, 4, kFourCcFlag, FourCc("DXT1"), payload));
    EXPECT_EQ(texture.format, TextureFormat::Bc1);
    EXPECT_EQ(texture.width, 8u);
    EXPECT_EQ(texture.mips, 4u);
    EXPECT_EQ(texture.layers, 1u);
    EXPECT_EQ(Copy(texture.Mip(0, 0)), Slice(payload, 0, 32));
    EXPECT_EQ(Copy(texture.Mip(0, 3)), Slice(payload, 48, 8));
    EXPECT_THROW(texture.Mip(0, 4), std::out_of_range);

    EXPECT_EQ(ReadDds(Dds(4, 4, 1, kFourCcFlag, FourCc("ATI2"), Ramp(16))).format,
              TextureFormat::Bc5);
}

TEST(TextureFile, DdsDx10ReadsArraysAndCubeMaps) {
    // Two layers of one 4 x 4 BC7 block, sRGB.
    const Bytes payload = Ramp(32);
    const TextureData array =
        ReadDds(Dds(4, 4, 1, kFourCcFlag, FourCc("DX10"), payload, {99, 3, 0, 2}));
    EXPECT_EQ(array.format, TextureFormat::Bc7Srgb);
    EXPECT_EQ(array.layers, 2u);
    EXPECT_EQ(Copy(array.Mip(1, 0)), Slice(payload, 16, 16));

    // A cube map's six faces are layers; a legacy one says so in its
    // caps.
    const TextureData cube =
        ReadDds(Dds(4, 4, 1, kFourCcFlag, FourCc("DX10"), Ramp(6 * 16), {83, 3, 0x4, 1}));
    EXPECT_TRUE(cube.cube);
    EXPECT_EQ(cube.layers, 6u);
    EXPECT_EQ(cube.format, TextureFormat::Bc5);
    EXPECT_EQ(ReadDds(Dds(4, 4, 1, kFourCcFlag, FourCc("DXT5"), Ramp(6 * 16), {}, 0xFE00)).layers,
              6u);
}

TEST(TextureFile, DdsReadsRgbaAndBgraMasks) {
    constexpr std::uint32_t kRgb = 0x40 | 0x1;
    const TextureData rgba = ReadDds(
        Dds(2, 1, 1, kRgb, 0, Ramp(8), {}, 0, 32, {0xffu, 0xff00u, 0xff0000u, 0xff000000u}));
    EXPECT_EQ(rgba.format, TextureFormat::Rgba8);
    const TextureData bgra = ReadDds(
        Dds(2, 1, 1, kRgb, 0, Ramp(8), {}, 0, 32, {0xff0000u, 0xff00u, 0xffu, 0xff000000u}));
    EXPECT_EQ(bgra.format, TextureFormat::Bgra8);
    // 24-bit RGB.
    const Bytes rgb = Dds(2, 1, 1, kRgb, 0, Ramp(6), {}, 0, 24, {0xffu, 0xff00u, 0xff0000u, 0});
    EXPECT_THROW(ReadDds(rgb), std::runtime_error);
}

TEST(TextureFile, DdsRejectsWhatItCannotRead) {
    const Bytes good = Dds(8, 8, 1, kFourCcFlag, FourCc("DXT1"), Ramp(32));
    EXPECT_NO_THROW(ReadDds(good));
    // Truncated payload, then a header cut short.
    EXPECT_THROW(ReadDds(Slice(good, 0, good.size() - 1)), std::runtime_error);
    EXPECT_THROW(ReadDds(Slice(good, 0, 60)), std::runtime_error);
    Bytes magic = good;
    magic[0] = std::byte{'X'};
    EXPECT_THROW(ReadDds(magic), std::runtime_error);
    EXPECT_THROW(ReadDds(Dds(8, 8, 1, kFourCcFlag, FourCc("ABCD"), Ramp(32))), std::runtime_error);
    // A 3D texture.
    const Bytes volume = Dds(4, 4, 1, kFourCcFlag, FourCc("DX10"), Ramp(16), {71, 4, 0, 1});
    EXPECT_THROW(ReadDds(volume), std::runtime_error);
    // More mips than the size has.
    EXPECT_THROW(ReadDds(Dds(4, 4, 4, kFourCcFlag, FourCc("DXT1"), Ramp(64))), std::runtime_error);
    // A size of zero.
    EXPECT_THROW(ReadDds(Dds(0, 4, 1, kFourCcFlag, FourCc("DXT1"), Ramp(8))), std::runtime_error);
}

TEST(TextureFile, Ktx2ReadsLevelsByTheirIndex) {
    // RGBA8 (VkFormat 37) 2 x 2 and 1 x 1, stored smallest first.
    const Bytes level0 = Ramp(16, 1);
    const Bytes level1 = Ramp(4, 100);
    const TextureData texture = ReadKtx2(Ktx2(37, 2, 2, 0, {level0, level1}));
    EXPECT_EQ(texture.format, TextureFormat::Rgba8);
    EXPECT_EQ(texture.mips, 2u);
    EXPECT_EQ(texture.layers, 1u);
    EXPECT_EQ(Copy(texture.Mip(0, 0)), level0);
    EXPECT_EQ(Copy(texture.Mip(0, 1)), level1);

    // Two layers of BC5 (141) in one level, layer by layer.
    const Bytes layers = Ramp(32);
    const TextureData array = ReadKtx2(Ktx2(141, 4, 4, 2, {layers}));
    EXPECT_EQ(array.format, TextureFormat::Bc5);
    EXPECT_EQ(array.layers, 2u);
    EXPECT_EQ(Copy(array.Mip(1, 0)), Slice(layers, 16, 16));
    EXPECT_EQ(ReadKtx2(Ktx2(146, 4, 4, 0, {Ramp(16)})).format, TextureFormat::Bc7Srgb);
}

TEST(TextureFile, Ktx2DecompressesZstdLevels) {
    // RGBA8 2 x 2 and 1 x 1, each level a zstd frame, its
    // uncompressed length in the level index.
    const std::vector<Bytes> levels{Ramp(16, 1), Ramp(4, 100)};
    std::vector<Bytes> frames;
    for (const Bytes& level : levels) {
        Bytes frame(ZSTD_compressBound(level.size()));
        const std::size_t size =
            ZSTD_compress(frame.data(), frame.size(), level.data(), level.size(), 3);
        ASSERT_EQ(ZSTD_isError(size), 0u);
        frame.resize(size);
        frames.push_back(frame);
    }
    Bytes file = Ktx2(37, 2, 2, 0, frames, 2);
    for (std::size_t level = 0; level < levels.size(); ++level) {
        Put64(file, 80 + 24 * level + 16, levels[level].size());
    }
    const TextureData texture = ReadKtx2(file);
    EXPECT_EQ(texture.format, TextureFormat::Rgba8);
    EXPECT_EQ(Copy(texture.Mip(0, 0)), levels[0]);
    EXPECT_EQ(Copy(texture.Mip(0, 1)), levels[1]);

    // An uncompressed length other than the level's.
    Put64(file, 80 + 16, 12);
    EXPECT_THROW(ReadKtx2(file), std::runtime_error);
}

TEST(TextureFile, Ktx2TranscodesBasisUniversalToBc) {
    // Fixtures encoded by Basis Universal's tool (fixtures/ktx2):
    // colour to BC7 with every mip, normal maps' x and y to BC5.
    const auto load = [](const char* name) {
        return LoadTextureFile(std::filesystem::path(ERNIE_TEST_FIXTURES) / "ktx2" / name);
    };
    for (const char* name : {"quadrants_etc1s.ktx2", "quadrants_uastc_zstd.ktx2"}) {
        const TextureData colour = load(name);
        EXPECT_EQ(colour.format, TextureFormat::Bc7) << name;
        EXPECT_EQ(colour.width, 16u) << name;
        EXPECT_EQ(colour.height, 16u) << name;
        EXPECT_EQ(colour.mips, 5u) << name;
        EXPECT_EQ(colour.Mip(0, 0).size(), 16u * 16u) << name;
        EXPECT_EQ(colour.Mip(0, 4).size(), 16u) << name;
    }
    for (const char* name : {"tilted_normal_etc1s.ktx2", "tilted_normal_uastc.ktx2"}) {
        const TextureData normal = load(name);
        EXPECT_EQ(normal.format, TextureFormat::Bc5) << name;
        EXPECT_EQ(normal.mips, 1u) << name;
    }
}

TEST(TextureFile, Ktx2RejectsWhatItCannotRead) {
    // No VkFormat but no Basis Universal data, then zstd levels that
    // are not zstd, then zlib supercompression.
    try {
        ReadKtx2(Ktx2(0, 4, 4, 0, {Ramp(16)}, 1));
        FAIL() << "read Basis Universal data that is not";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("Basis Universal"), std::string::npos);
    }
    EXPECT_THROW(ReadKtx2(Ktx2(145, 4, 4, 0, {Ramp(16)}, 2)), std::runtime_error);
    EXPECT_THROW(ReadKtx2(Ktx2(145, 4, 4, 0, {Ramp(16)}, 3)), std::runtime_error);
    // A level of the wrong length, then an unknown VkFormat.
    EXPECT_THROW(ReadKtx2(Ktx2(37, 2, 2, 0, {Ramp(12)})), std::runtime_error);
    EXPECT_THROW(ReadKtx2(Ktx2(1000, 2, 2, 0, {Ramp(16)})), std::runtime_error);
    Bytes truncated = Ktx2(37, 2, 2, 0, {Ramp(16)});
    truncated.resize(truncated.size() - 1);
    EXPECT_THROW(ReadKtx2(truncated), std::runtime_error);
    EXPECT_THROW(ReadKtx2(Ramp(100)), std::runtime_error);
}

TEST(TextureFile, LoadsFilesByExtension) {
    const std::filesystem::path dir =
        std::filesystem::path(::testing::TempDir()) / "ernie_texture_file_test";
    std::filesystem::create_directories(dir);
    const auto write = [&](const std::string& name, const Bytes& bytes) {
        std::ofstream(dir / name, std::ios::binary)
            .write(reinterpret_cast<const char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
        return dir / name;
    };
    EXPECT_EQ(
        LoadTextureFile(write("a.DDS", Dds(4, 4, 1, kFourCcFlag, FourCc("DXT1"), Ramp(8)))).format,
        TextureFormat::Bc1);
    EXPECT_EQ(LoadTextureFile(write("a.ktx2", Ktx2(37, 1, 1, 0, {Ramp(4)}))).format,
              TextureFormat::Rgba8);
    // Stored texels are read in place from the mapped file; the mips
    // hold the file's bytes where they lie.
    const Bytes payload = Ramp(32 + 8 + 8 + 8, 3);
    const TextureData dds =
        LoadTextureFile(write("mips.dds", Dds(8, 8, 4, kFourCcFlag, FourCc("DXT1"), payload)));
    EXPECT_NE(dds.mapping, nullptr);
    EXPECT_TRUE(dds.bytes.empty()) << "no copy";
    EXPECT_EQ(Copy(dds.Mip(0, 3)), Slice(payload, 48, 8));
    const Bytes level0 = Ramp(16, 1);
    const Bytes level1 = Ramp(4, 100);
    const TextureData ktx2 =
        LoadTextureFile(write("levels.ktx2", Ktx2(37, 2, 2, 0, {level0, level1})));
    EXPECT_NE(ktx2.mapping, nullptr);
    EXPECT_EQ(Copy(ktx2.Mip(0, 0)), level0);
    EXPECT_EQ(Copy(ktx2.Mip(0, 1)), level1);
    // Transcoded Basis Universal texels are decoded into memory.
    const TextureData basis = LoadTextureFile(std::filesystem::path(ERNIE_TEST_FIXTURES) / "ktx2" /
                                              "quadrants_uastc_zstd.ktx2");
    EXPECT_EQ(basis.mapping, nullptr);
    EXPECT_FALSE(basis.bytes.empty());
    // Errors name the file.
    try {
        LoadTextureFile(write("bad.dds", Ramp(10)));
        FAIL() << "read a malformed file";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("bad.dds"), std::string::npos);
    }
    EXPECT_THROW(LoadTextureFile(dir / "missing.dds"), std::runtime_error);
    std::filesystem::remove_all(dir);
}

// ----------------------------------------------------------------------------------------------
// Baking and the cache

namespace {

class TextureBake : public testing::Test {
  protected:
    void SetUp() override {
        dir_ = std::filesystem::path(::testing::TempDir()) /
               ("ernie_texture_bake_" +
                std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
        std::filesystem::create_directories(dir_);
    }

    void TearDown() override {
        std::filesystem::remove_all(dir_);
    }

    /// A `width` x `height` RGBA PNG of `colour(x, y)`.
    template <typename Colour>
    std::filesystem::path Png(const std::string& name, int width, int height, Colour colour) const {
        std::vector<std::uint8_t> texels;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const glm::u8vec4 c = colour(x, y);
                texels.insert(texels.end(), {c.r, c.g, c.b, c.a});
            }
        }
        const auto path = dir_ / name;
        EXPECT_NE(stbi_write_png(path.string().c_str(), width, height, 4, texels.data(), width * 4),
                  0);
        return path;
    }

    std::filesystem::path dir_;
};

}  // namespace

TEST_F(TextureBake, ColourBakesToMippedSrgbBc7) {
    // 100 x 60: mips down to 1 x 1, the edge blocks partly past the
    // image.
    const auto input = Png("colour.png", 100, 60,
                           [](int x, int y) { return glm::u8vec4(x * 2, y * 4, 128, 255); });
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
    EXPECT_TRUE(std::equal(read.Data().begin(), read.Data().end(), baked.Data().begin()))
        << "the DDS file reads back";
    EXPECT_NE(read.mapping, nullptr) << "read in place";

    BakeOptions linear;
    linear.srgb = false;
    linear.format = TextureFormat::Bc1;
    EXPECT_EQ(BakeTexture(input, dir_ / "linear.dds", linear).format, TextureFormat::Bc1);
}

TEST_F(TextureBake, NormalMapsBakeToBc5AndDataStaysLinear) {
    const auto input =
        Png("normal.png", 32, 32, [](int, int) { return glm::u8vec4(191, 128, 238, 255); });
    BakeOptions normal;
    normal.role = TextureRole::Normal;
    const TextureData baked = BakeTexture(input, dir_ / "normal.dds", normal);
    EXPECT_EQ(baked.format, TextureFormat::Bc5);
    EXPECT_EQ(baked.mips, 6u);
    EXPECT_EQ(LoadTextureFile(dir_ / "normal.dds").format, TextureFormat::Bc5);

    BakeOptions data;
    data.role = TextureRole::Data;
    EXPECT_EQ(BakeTexture(input, dir_ / "data.dds", data).format, TextureFormat::Bc7);
    data.format = TextureFormat::Bc4;
    EXPECT_EQ(BakeTexture(input, dir_ / "data4.dds", data).format, TextureFormat::Bc4);
}

TEST_F(TextureBake, EncodingRejectsWhatItCannotWrite) {
    const Image image(8, 8, glm::vec4(0.5f));
    const std::vector<Image> mips = MipChain(image, MipFilter::Average);
    EXPECT_THROW(EncodeTexture(mips, TextureFormat::Bc6hUfloat, ImageEncoding::Linear),
                 std::invalid_argument);
    std::vector<Image> skipped{mips[0], mips[2]};
    EXPECT_THROW(EncodeTexture(skipped, TextureFormat::Bc7, ImageEncoding::Linear),
                 std::invalid_argument);
    // RGBA8 stores the 8-bit values as they are.
    const TextureData plain = EncodeTexture(mips, TextureFormat::Rgba8, ImageEncoding::Linear);
    EXPECT_EQ(plain.Mip(0, 0).size(), 8u * 8u * 4u);
    EXPECT_EQ(static_cast<int>(plain.Mip(0, 0)[0]), 128);
    EXPECT_THROW(BakeTexture(dir_ / "missing.png", dir_ / "out.dds"), std::runtime_error);
}

TEST_F(TextureBake, EightBitBakesMatchFloatOnes) {
    // An 8-bit sRGB PNG (37 x 23, so mips round down at odd edges)
    // bakes without floats; its mips match the float chain's to
    // within a step or two of 8-bit rounding.
    const auto input = Png("eight.png", 37, 23, [](int x, int y) {
        return glm::u8vec4(x * 7, y * 11, (x * y) % 256, 200 + x % 50);
    });
    BakeOptions options;
    options.format = TextureFormat::Rgba8;
    const TextureData eight = BakeTexture(input, dir_ / "eight.dds", options);
    EXPECT_EQ(eight.format, TextureFormat::Rgba8Srgb);
    const TextureData floats = EncodeTexture(
        MipChain(ernie::LoadImageFile(input, ImageEncoding::Srgb), MipFilter::Average),
        TextureFormat::Rgba8Srgb, ImageEncoding::Srgb);
    ASSERT_EQ(eight.mips, floats.mips);
    for (std::uint32_t mip = 0; mip < eight.mips; ++mip) {
        const auto a = eight.Mip(0, mip);
        const auto b = floats.Mip(0, mip);
        ASSERT_EQ(a.size(), b.size());
        int worst = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            worst = std::max(worst, std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i])));
        }
        EXPECT_LE(worst, 2) << "mip " << mip;
    }
}

TEST_F(TextureBake, MosaicsBakeAsTheirJoinedImage) {
    // A 2 x 2 mosaic bakes byte for byte as the image its tiles make:
    // 64 x 48 tiles keep mips 0 to 2 per tile, 30 x 30 ones none.
    const auto colour = [](int x, int y) { return glm::u8vec4(x * 3, y * 5, (x ^ y) & 0xff, 255); };
    for (const auto& [tileWidth, tileHeight] : {std::pair{64, 48}, std::pair{30, 30}}) {
        const std::string size = std::to_string(tileWidth) + "_" + std::to_string(tileHeight);
        const auto whole = Png("whole" + size + ".png", 2 * tileWidth, 2 * tileHeight, colour);
        std::vector<std::filesystem::path> tiles;
        for (int row = 0; row < 2; ++row) {
            for (int column = 0; column < 2; ++column) {
                tiles.push_back(
                    Png("tile" + size + std::to_string(row) + std::to_string(column) + ".png",
                        tileWidth, tileHeight, [&](int x, int y) {
                            return colour(column * tileWidth + x, row * tileHeight + y);
                        }));
            }
        }
        for (const TextureFormat format : {TextureFormat::Rgba8, TextureFormat::Bc7}) {
            BakeOptions options;
            options.format = format;
            BakeTexture(whole, dir_ / "whole.dds", options);
            const TextureData layout = BakeMosaic(tiles, 2, dir_ / "mosaic.dds", options);
            EXPECT_EQ(layout.width, static_cast<std::uint32_t>(2 * tileWidth));
            const TextureData a = LoadTextureFile(dir_ / "whole.dds");
            const TextureData b = LoadTextureFile(dir_ / "mosaic.dds");
            ASSERT_EQ(a.mips, b.mips);
            ASSERT_EQ(a.Data().size(), b.Data().size());
            EXPECT_TRUE(std::equal(a.Data().begin(), a.Data().end(), b.Data().begin()))
                << size << ", format " << static_cast<int>(format);
        }
    }

    const std::vector<std::filesystem::path> three(3, dir_ / "tile64_4800.png");
    EXPECT_THROW(BakeMosaic(three, 2, dir_ / "bad.dds"), std::invalid_argument) << "not whole rows";
    const std::vector<std::filesystem::path> mixed{dir_ / "tile64_4800.png",
                                                   dir_ / "tile30_3000.png"};
    EXPECT_THROW(BakeMosaic(mixed, 2, dir_ / "bad.dds"), std::invalid_argument) << "two sizes";
    BakeOptions normal;
    normal.role = TextureRole::Normal;
    EXPECT_THROW(BakeMosaic(mixed, 2, dir_ / "bad.dds", normal), std::invalid_argument);
}

TEST_F(TextureBake, RasterBakesMatchWholeBakesBandByBand) {
    // 600 x 2100: bands of 1024, 1024 and 52 rows, their mips 0 to 8
    // made band by band (the last band's ending at mip 5), the rest
    // joined; a bake a few rows of blocks a step is the same.
    const auto input = Png("tall.png", 600, 2100, [](int x, int y) {
        return glm::u8vec4(x * 7, y * 3, (x * y) & 0xff, 255 - y % 7);
    });
    for (const TextureFormat format : {TextureFormat::Rgba8, TextureFormat::Bc7}) {
        BakeOptions options;
        options.format = format;
        BakeTexture(input, dir_ / "whole.dds", options);
        const TextureData whole = LoadTextureFile(dir_ / "whole.dds");
        for (const std::uint64_t texels : {~std::uint64_t{0}, std::uint64_t{20000}}) {
            ernie::RasterBake bake(ernie::OpenRaster(input), dir_ / "banded.dds", options);
            int steps = 1;
            while (!bake.Step(texels, true)) {
                ++steps;
            }
            EXPECT_EQ(bake.Encoded(), bake.Total());
            // 1.68 million texels in all.
            if (texels < bake.Total()) {
                EXPECT_GE(steps, 80);
            }
            const TextureData banded = LoadTextureFile(dir_ / "banded.dds");
            ASSERT_EQ(banded.mips, whole.mips);
            ASSERT_EQ(banded.Data().size(), whole.Data().size());
            for (std::uint32_t mip = 0; mip < whole.mips; ++mip) {
                const auto a = whole.Mip(0, mip);
                const auto b = banded.Mip(0, mip);
                EXPECT_TRUE(std::equal(a.begin(), a.end(), b.begin()))
                    << "format " << static_cast<int>(format) << ", mip " << mip << ", " << texels
                    << " texels a step";
            }
        }
    }
}

TEST_F(TextureBake, AnUnfinishedRasterBakeRemovesItsFile) {
    const auto input =
        Png("big.png", 512, 3000, [](int x, int y) { return glm::u8vec4(x, y, 0, 255); });
    const auto output = dir_ / "unfinished.dds";
    {
        ernie::RasterBake bake(ernie::OpenRaster(input), output);
        EXPECT_FALSE(bake.Step(4096, true));
        EXPECT_TRUE(std::filesystem::exists(output));
        EXPECT_LT(bake.Encoded(), bake.Total());
    }
    EXPECT_FALSE(std::filesystem::exists(output));

    BakeOptions normal;
    normal.role = TextureRole::Normal;
    EXPECT_THROW(ernie::RasterBake(ernie::OpenRaster(input), dir_ / "normal.dds", normal),
                 std::invalid_argument);
}

TEST_F(TextureBake, MosaicFilesNameTheirTilesRowByRow) {
    std::vector<std::filesystem::path> tiles;
    for (int row = 0; row < 2; ++row) {
        for (int column = 0; column < 3; ++column) {
            tiles.push_back(Png(
                "tile" + std::to_string(row) + std::to_string(column) + ".png", 8, 4,
                [&](int x, int y) { return glm::u8vec4(column * 8 + x, row * 4 + y, 0, 255); }));
        }
    }
    {
        std::ofstream file(dir_ / "world.mosaic");
        file << "# Two rows of three\n"
             << "tile00.png tile01.png  tile02.png\n\n"
             << "tile10.png tile11.png tile12.png  # bottom\n";
    }
    const std::unique_ptr<ernie::GridSource> mosaic = ernie::OpenRaster(dir_ / "world.mosaic");
    ASSERT_EQ(mosaic->Width(), 24u);
    ASSERT_EQ(mosaic->Height(), 8u);
    EXPECT_EQ(mosaic->Parts().size(), 6u);
    // A rectangle across all six tiles reads as the joined image.
    std::vector<std::byte> pixels(20 * 6 * 4);
    mosaic->Read(2, 1, 20, 6, pixels);
    for (std::uint32_t y = 0; y < 6; ++y) {
        for (std::uint32_t x = 0; x < 20; ++x) {
            const std::size_t at = (y * 20 + x) * 4;
            ASSERT_EQ(static_cast<int>(pixels[at]), 2 + x);
            ASSERT_EQ(static_cast<int>(pixels[at + 1]), 1 + y);
        }
    }
    {
        std::ofstream file(dir_ / "ragged.mosaic");
        file << "tile00.png tile01.png\ntile10.png\n";
    }
    EXPECT_THROW(ernie::OpenRaster(dir_ / "ragged.mosaic"), std::invalid_argument);
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
    const std::vector<Image> mips = MipChain(image, MipFilter::Average);
    const std::pair<TextureFormat, TextureFormat> kFormats[] = {
        {TextureFormat::Bc1Srgb, TextureFormat::Rgba8Srgb},
        {TextureFormat::Bc4, TextureFormat::R8},
        {TextureFormat::Bc5, TextureFormat::Rg8},
        {TextureFormat::Bc7, TextureFormat::Rgba8}};
    for (const auto& [format, decodedFormat] : kFormats) {
        const ImageEncoding encoding =
            ernie::FormatInfo(format).srgb ? ImageEncoding::Srgb : ImageEncoding::Linear;
        const TextureData decoded = ernie::DecodeBlocks(EncodeTexture(mips, format, encoding));
        ASSERT_EQ(decoded.format, decodedFormat);
        ASSERT_EQ(decoded.mips, 4u);
        const std::uint32_t channels = ernie::FormatInfo(decodedFormat).channels;
        const std::vector<std::uint8_t> source = ernie::ToRgba8(image, encoding);
        const std::span<const std::byte> texels = decoded.Mip(0, 0);
        ASSERT_EQ(texels.size(), std::size_t{64} * channels);
        // BC1's four colours a block, 5:6:5 bit endpoints, err most.
        const int tolerance = format == TextureFormat::Bc1Srgb ? 26 : 12;
        for (std::size_t i = 0; i < 64; ++i) {
            for (std::uint32_t c = 0; c < channels; ++c) {
                EXPECT_NEAR(static_cast<int>(texels[i * channels + c]),
                            static_cast<int>(source[i * 4 + c]), tolerance)
                    << "format " << static_cast<int>(format) << ", texel " << i << ", channel "
                    << c;
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
    TextureData texture = EncodeTexture(std::vector<Image>{Image(16, 16, glm::vec4(0.25f))},
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

    // A bake written in parts beside the cache, published: kept,
    // though larger than the cache, the others going.
    const std::optional<std::filesystem::path> partial = cache.PartialPath("big");
    ASSERT_TRUE(partial);
    {
        const auto png =
            Png("big.png", 64, 64, [](int x, int y) { return glm::u8vec4(x, y, 0, 255); });
        RasterBake(OpenRaster(png), *partial).Finish();
    }
    EXPECT_FALSE(cache.Read("big"));
    EXPECT_TRUE(cache.Publish(*partial, "big"));
    EXPECT_FALSE(std::filesystem::exists(*partial));
    const std::optional<TextureData> big = cache.Read("big");
    ASSERT_TRUE(big);
    EXPECT_EQ(big->width, 64u);
    EXPECT_FALSE(cache.Read("a"));
    EXPECT_FALSE(cache.Read("c"));

    const ernie::TextureCache none(std::nullopt);
    EXPECT_FALSE(none.Enabled());
    none.Write("a", texture);
    EXPECT_FALSE(none.Read("a"));
    EXPECT_FALSE(none.PartialPath("a"));
}

TEST_F(TextureBake, Rgba8MipsHalveAnImageIn8Bits) {
    // A 6 x 4 PNG: read as RGBA8, mips 6 x 4, 3 x 2 and 1 x 1, each
    // the average of the last; its size from its header alone; and no
    // size for a file that is not an 8-bit PNG or JPEG.
    std::vector<std::uint8_t> texels;
    for (int i = 0; i < 6 * 4; ++i) {
        texels.insert(texels.end(), {static_cast<std::uint8_t>(i * 10), 128, 0, 255});
    }
    const std::filesystem::path png = dir_ / "six_by_four.png";
    ASSERT_NE(stbi_write_png(png.string().c_str(), 6, 4, 4, texels.data(), 6 * 4), 0);
    EXPECT_EQ(ernie::Rgba8ImageSize(png), glm::uvec2(6, 4));
    EXPECT_FALSE(ernie::Rgba8ImageSize(dir_ / "missing.png"));
    const std::vector<std::vector<std::uint8_t>> mips = ernie::LoadRgba8Mips(png, false);
    ASSERT_EQ(mips.size(), 3u);
    EXPECT_EQ(mips[0], texels);
    EXPECT_EQ(mips[1].size(), std::size_t{3 * 2 * 4});
    EXPECT_EQ(mips[2].size(), std::size_t{4});
    // The first 2 x 2: red 0, 10, 60 and 70.
    EXPECT_NEAR(mips[1][0], 35, 1);
    EXPECT_EQ(mips[1][1], 128);
}

}  // namespace ernie
