// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/texture_file.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using ernie::ImageBytes;
using ernie::IsTextureFile;
using ernie::LoadTextureFile;
using ernie::ReadDds;
using ernie::ReadKtx2;
using ernie::TextureData;
using ernie::TextureFormat;
using ernie::WithSrgb;

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

std::uint32_t FourCc(const char (&code)[5]) {
    std::uint32_t value = 0;
    std::memcpy(&value, code, 4);
    return value;
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
Bytes Dds(std::uint32_t width, std::uint32_t height,
          std::uint32_t mips, std::uint32_t pixelFlags,
          std::uint32_t fourCc, const Bytes& payload,
          std::array<std::uint32_t, 4> dx10 = {},
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
Bytes Ktx2(std::uint32_t vkFormat, std::uint32_t width,
           std::uint32_t height, std::uint32_t layers,
           const std::vector<Bytes>& levels,
           std::uint32_t supercompression = 0) {
    const std::array<std::uint8_t, 12> identifier{
        0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32,
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
        file.insert(file.end(), levels[level].begin(),
                    levels[level].end());
    }
    return file;
}

Bytes Slice(const Bytes& bytes, std::size_t at, std::size_t count) {
    return Bytes(bytes.begin() + static_cast<std::ptrdiff_t>(at),
                 bytes.begin() +
                     static_cast<std::ptrdiff_t>(at + count));
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
    EXPECT_EQ(WithSrgb(TextureFormat::Bc7, true),
              TextureFormat::Bc7Srgb);
    EXPECT_EQ(WithSrgb(TextureFormat::Rgba8Srgb, false),
              TextureFormat::Rgba8);
    EXPECT_EQ(WithSrgb(TextureFormat::Bc5, true), TextureFormat::Bc5);
    EXPECT_TRUE(IsTextureFile("a/b.DDS"));
    EXPECT_TRUE(IsTextureFile("b.ktx2"));
    EXPECT_FALSE(IsTextureFile("b.png"));
    EXPECT_FALSE(IsTextureFile("b.ktx"));
}

TEST(TextureFile, LegacyDdsReadsDxt1WithEveryMip) {
    // 8 x 8 down to 1 x 1: 2 x 2 blocks, then one block each.
    const Bytes payload = Ramp(32 + 8 + 8 + 8);
    const TextureData texture =
        ReadDds(Dds(8, 8, 4, kFourCcFlag, FourCc("DXT1"), payload));
    EXPECT_EQ(texture.format, TextureFormat::Bc1);
    EXPECT_EQ(texture.width, 8u);
    EXPECT_EQ(texture.mips, 4u);
    EXPECT_EQ(texture.layers, 1u);
    EXPECT_EQ(Copy(texture.Mip(0, 0)), Slice(payload, 0, 32));
    EXPECT_EQ(Copy(texture.Mip(0, 3)), Slice(payload, 48, 8));
    EXPECT_THROW(texture.Mip(0, 4), std::out_of_range);

    EXPECT_EQ(
        ReadDds(Dds(4, 4, 1, kFourCcFlag, FourCc("ATI2"), Ramp(16)))
            .format,
        TextureFormat::Bc5);
}

TEST(TextureFile, DdsDx10ReadsArraysAndCubeMaps) {
    // Two layers of one 4 x 4 BC7 block, sRGB.
    const Bytes payload = Ramp(32);
    const TextureData array =
        ReadDds(Dds(4, 4, 1, kFourCcFlag, FourCc("DX10"), payload,
                    {99, 3, 0, 2}));
    EXPECT_EQ(array.format, TextureFormat::Bc7Srgb);
    EXPECT_EQ(array.layers, 2u);
    EXPECT_EQ(Copy(array.Mip(1, 0)), Slice(payload, 16, 16));

    // A cube map's six faces are layers; a legacy one says so in its
    // caps.
    const TextureData cube =
        ReadDds(Dds(4, 4, 1, kFourCcFlag, FourCc("DX10"),
                    Ramp(6 * 16), {83, 3, 0x4, 1}));
    EXPECT_TRUE(cube.cube);
    EXPECT_EQ(cube.layers, 6u);
    EXPECT_EQ(cube.format, TextureFormat::Bc5);
    EXPECT_EQ(ReadDds(Dds(4, 4, 1, kFourCcFlag, FourCc("DXT5"),
                          Ramp(6 * 16), {}, 0xFE00))
                  .layers,
              6u);
}

TEST(TextureFile, DdsReadsRgbaAndBgraMasks) {
    constexpr std::uint32_t kRgb = 0x40 | 0x1;
    const TextureData rgba =
        ReadDds(Dds(2, 1, 1, kRgb, 0, Ramp(8), {}, 0, 32,
                    {0xffu, 0xff00u, 0xff0000u, 0xff000000u}));
    EXPECT_EQ(rgba.format, TextureFormat::Rgba8);
    const TextureData bgra =
        ReadDds(Dds(2, 1, 1, kRgb, 0, Ramp(8), {}, 0, 32,
                    {0xff0000u, 0xff00u, 0xffu, 0xff000000u}));
    EXPECT_EQ(bgra.format, TextureFormat::Bgra8);
    // 24-bit RGB.
    const Bytes rgb = Dds(2, 1, 1, kRgb, 0, Ramp(6), {}, 0, 24,
                          {0xffu, 0xff00u, 0xff0000u, 0});
    EXPECT_THROW(ReadDds(rgb), std::runtime_error);
}

TEST(TextureFile, DdsRejectsWhatItCannotRead) {
    const Bytes good =
        Dds(8, 8, 1, kFourCcFlag, FourCc("DXT1"), Ramp(32));
    EXPECT_NO_THROW(ReadDds(good));
    // Truncated payload, then a header cut short.
    EXPECT_THROW(ReadDds(Slice(good, 0, good.size() - 1)),
                 std::runtime_error);
    EXPECT_THROW(ReadDds(Slice(good, 0, 60)), std::runtime_error);
    Bytes magic = good;
    magic[0] = std::byte{'X'};
    EXPECT_THROW(ReadDds(magic), std::runtime_error);
    EXPECT_THROW(
        ReadDds(Dds(8, 8, 1, kFourCcFlag, FourCc("ABCD"), Ramp(32))),
        std::runtime_error);
    // A 3D texture.
    const Bytes volume = Dds(4, 4, 1, kFourCcFlag, FourCc("DX10"),
                             Ramp(16), {71, 4, 0, 1});
    EXPECT_THROW(ReadDds(volume), std::runtime_error);
    // More mips than the size has.
    EXPECT_THROW(
        ReadDds(Dds(4, 4, 4, kFourCcFlag, FourCc("DXT1"), Ramp(64))),
        std::runtime_error);
    // A size of zero.
    EXPECT_THROW(
        ReadDds(Dds(0, 4, 1, kFourCcFlag, FourCc("DXT1"), Ramp(8))),
        std::runtime_error);
}

TEST(TextureFile, Ktx2ReadsLevelsByTheirIndex) {
    // RGBA8 (VkFormat 37) 2 x 2 and 1 x 1, stored smallest first.
    const Bytes level0 = Ramp(16, 1);
    const Bytes level1 = Ramp(4, 100);
    const TextureData texture =
        ReadKtx2(Ktx2(37, 2, 2, 0, {level0, level1}));
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
    EXPECT_EQ(ReadKtx2(Ktx2(146, 4, 4, 0, {Ramp(16)})).format,
              TextureFormat::Bc7Srgb);
}

TEST(TextureFile, Ktx2RejectsWhatItCannotRead) {
    // Basis Universal (no VkFormat) and zstd need libktx.
    try {
        ReadKtx2(Ktx2(0, 4, 4, 0, {Ramp(16)}, 1));
        FAIL() << "read a supercompressed file";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("libktx"),
                  std::string::npos);
    }
    EXPECT_THROW(ReadKtx2(Ktx2(145, 4, 4, 0, {Ramp(16)}, 2)),
                 std::runtime_error);
    // A level of the wrong length, then an unknown VkFormat.
    EXPECT_THROW(ReadKtx2(Ktx2(37, 2, 2, 0, {Ramp(12)})),
                 std::runtime_error);
    EXPECT_THROW(ReadKtx2(Ktx2(1000, 2, 2, 0, {Ramp(16)})),
                 std::runtime_error);
    Bytes truncated = Ktx2(37, 2, 2, 0, {Ramp(16)});
    truncated.resize(truncated.size() - 1);
    EXPECT_THROW(ReadKtx2(truncated), std::runtime_error);
    EXPECT_THROW(ReadKtx2(Ramp(100)), std::runtime_error);
}

TEST(TextureFile, LoadsFilesByExtension) {
    const std::filesystem::path dir =
        std::filesystem::path(::testing::TempDir()) /
        "ernie_texture_file_test";
    std::filesystem::create_directories(dir);
    const auto write = [&](const std::string& name,
                           const Bytes& bytes) {
        std::ofstream(dir / name, std::ios::binary)
            .write(reinterpret_cast<const char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
        return dir / name;
    };
    EXPECT_EQ(
        LoadTextureFile(write("a.DDS", Dds(4, 4, 1, kFourCcFlag,
                                           FourCc("DXT1"), Ramp(8))))
            .format,
        TextureFormat::Bc1);
    EXPECT_EQ(
        LoadTextureFile(write("a.ktx2", Ktx2(37, 1, 1, 0, {Ramp(4)})))
            .format,
        TextureFormat::Rgba8);
    // Errors name the file.
    try {
        LoadTextureFile(write("bad.dds", Ramp(10)));
        FAIL() << "read a malformed file";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("bad.dds"),
                  std::string::npos);
    }
    EXPECT_THROW(LoadTextureFile(dir / "missing.dds"),
                 std::runtime_error);
    std::filesystem::remove_all(dir);
}
