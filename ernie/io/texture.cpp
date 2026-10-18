// SPDX-License-Identifier: BSD-3-Clause
/// @file texture.cpp
/// @brief Texture formats, DDS and KTX2 reading and writing, Basis transcoding, BC encoding and
/// decoding, whole, banded and mosaic bakes, and the texture cache.
#include "ernie/io/texture.h"

#include "ernie/util/bytes.h"
#include "ernie/util/disk_cache.h"
#include "ernie/util/parallel.h"
#include "ernie/util/profile.h"
#include "ernie/util/text.h"

#include <basisu_transcoder.h>
#include <bc7decomp.h>
#include <bc7enc.h>
#include <rgbcx.h>
#include <stb_image.h>
#include <zstd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace ernie {

namespace fs = std::filesystem;

namespace {

using F = TextureFormat;

// ----------------------------------------------------------------------------------------------
// Formats

/// A format's layout and the codes DDS (DXGI) and KTX2 (VkFormat) name it by.
struct FormatRow {
    TextureFormat format;
    TextureFormatInfo info;
    std::uint32_t dxgi;
    std::uint32_t vk;
};

/// Every format, in `TextureFormat`'s order.
constexpr std::array<FormatRow, 22> kFormats{{
    {F::R8, {1, 1, 1, false}, 61, 9},
    {F::Rg8, {1, 2, 2, false}, 49, 16},
    {F::Rgba8, {1, 4, 4, false}, 28, 37},
    {F::Rgba8Srgb, {1, 4, 4, true}, 29, 43},
    {F::Bgra8, {1, 4, 4, false}, 87, 44},
    {F::Bgra8Srgb, {1, 4, 4, true}, 91, 50},
    {F::Rgba16F, {1, 8, 4, false}, 10, 97},
    {F::Rgba32F, {1, 16, 4, false}, 2, 109},
    {F::Bc1, {4, 8, 4, false}, 71, 133},
    {F::Bc1Srgb, {4, 8, 4, true}, 72, 134},
    {F::Bc2, {4, 16, 4, false}, 74, 135},
    {F::Bc2Srgb, {4, 16, 4, true}, 75, 136},
    {F::Bc3, {4, 16, 4, false}, 77, 137},
    {F::Bc3Srgb, {4, 16, 4, true}, 78, 138},
    {F::Bc4, {4, 8, 1, false}, 80, 139},
    {F::Bc4Snorm, {4, 8, 1, false}, 81, 140},
    {F::Bc5, {4, 16, 2, false}, 83, 141},
    {F::Bc5Snorm, {4, 16, 2, false}, 84, 142},
    {F::Bc6hUfloat, {4, 16, 3, false}, 95, 143},
    {F::Bc6hSfloat, {4, 16, 3, false}, 96, 144},
    {F::Bc7, {4, 16, 4, false}, 98, 145},
    {F::Bc7Srgb, {4, 16, 4, true}, 99, 146},
}};

constexpr bool InFormatOrder() {
    for (std::size_t i = 0; i < kFormats.size(); ++i) {
        if (static_cast<std::size_t>(kFormats[i].format) != i) {
            return false;
        }
    }
    return true;
}
static_assert(InFormatOrder(), "kFormats is indexed by TextureFormat");

const FormatRow& Row(TextureFormat format) {
    const auto index = static_cast<std::size_t>(format);
    if (index >= kFormats.size()) {
        throw std::invalid_argument("Unknown texture format");
    }
    return kFormats[index];
}

template <typename Matches> std::optional<TextureFormat> FindFormat(const Matches& matches) {
    const auto row = std::find_if(kFormats.begin(), kFormats.end(), matches);
    return row == kFormats.end() ? std::nullopt : std::optional(row->format);
}

std::optional<TextureFormat> FromDxgi(std::uint32_t dxgi) {
    return FindFormat([&](const FormatRow& row) { return row.dxgi == dxgi; });
}

std::optional<TextureFormat> FromVkFormat(std::uint32_t vk) {
    // BC1_RGB (131, 132) reads as BC1_RGBA: they differ only in the alpha of blocks in
    // three-colour mode.
    if (vk == 131 || vk == 132) {
        vk += 2;
    }
    return FindFormat([&](const FormatRow& row) { return row.vk == vk; });
}

// ----------------------------------------------------------------------------------------------
// Layout

// Bounds that keep every size below in range, far past what devices sample whole (virtual
// textures page larger ones: global mosaics run to 86400 texels across).
constexpr std::uint32_t kMaxSide = 1u << 20;
constexpr std::uint32_t kMaxLayers = 2048;

/// A texture of `format`, `width` x `height` with `layers` layers and `mips` mips, its bytes still
/// to fill (with `allocate`, else left to a mapped file): checks the shape and lays out `offsets`
/// layer by layer.
TextureData Shape(TextureFormat format, std::uint32_t width, std::uint32_t height,
                  std::uint32_t layers, std::uint32_t mips, bool allocate = true) {
    if (width == 0 || height == 0 || width > kMaxSide || height > kMaxSide) {
        throw std::runtime_error("Texture has a size of " + std::to_string(width) + " x " +
                                 std::to_string(height));
    }
    if (layers == 0 || layers > kMaxLayers) {
        throw std::runtime_error("Texture has " + std::to_string(layers) + " layers");
    }
    if (mips == 0 || mips > MipCount(width, height)) {
        throw std::runtime_error("Texture has " + std::to_string(mips) + " mips");
    }
    TextureData data;
    data.format = format;
    data.width = width;
    data.height = height;
    data.layers = layers;
    data.mips = mips;
    std::size_t total = 0;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        for (std::uint32_t mip = 0; mip < mips; ++mip) {
            data.offsets.push_back(total);
            total += ImageBytes(format, MipSide(width, mip), MipSide(height, mip));
        }
    }
    if (allocate) {
        data.bytes.resize(total);
    }
    return data;
}

/// Bytes of every layer's mips of `texture`, as `Shape` laid them out.
std::size_t ShapeBytes(const TextureData& texture) {
    const std::uint32_t last = texture.mips - 1;
    return texture.offsets.back() +
           ImageBytes(texture.format, MipSide(texture.width, last), MipSide(texture.height, last));
}

/// A single-layer texture of a format the bakes write, laid out as `Shape`.
/// @throws std::invalid_argument for a format it does not write.
TextureData Allocate(TextureFormat format, std::uint32_t width, std::uint32_t height,
                     std::uint32_t mips, bool allocate = true) {
    constexpr std::array kWritten{F::Rgba8,   F::Rgba8Srgb, F::Bc1, F::Bc1Srgb, F::Bc3,
                                  F::Bc3Srgb, F::Bc4,       F::Bc5, F::Bc7,     F::Bc7Srgb};
    if (std::find(kWritten.begin(), kWritten.end(), format) == kWritten.end()) {
        throw std::invalid_argument("Textures are not encoded to this format");
    }
    return Shape(format, width, height, 1, mips, allocate);
}

// ----------------------------------------------------------------------------------------------
// DDS and KTX2 reading

/// The header fields of both formats: little-endian 32-bit words.
std::uint32_t U32(std::span<const std::byte> file, std::size_t at) {
    return ReadLittleEndian<std::uint32_t>(file, at);
}

/// The format of a DDS file without a DX10 header, from its pixel format (`DDS_PIXELFORMAT` at
/// `at`).
TextureFormat LegacyDdsFormat(std::span<const std::byte> file, std::size_t at) {
    constexpr std::uint32_t kFourCcFlag = 0x4;
    constexpr std::uint32_t kRgbFlag = 0x40;
    const std::uint32_t flags = U32(file, at + 4);
    if ((flags & kFourCcFlag) != 0) {
        const std::uint32_t code = U32(file, at + 8);
        switch (code) {
        case FourCc("DXT1"):
            return F::Bc1;
        case FourCc("DXT2"):
        case FourCc("DXT3"):
            return F::Bc2;
        case FourCc("DXT4"):
        case FourCc("DXT5"):
            return F::Bc3;
        case FourCc("ATI1"):
        case FourCc("BC4U"):
            return F::Bc4;
        case FourCc("BC4S"):
            return F::Bc4Snorm;
        case FourCc("ATI2"):
        case FourCc("BC5U"):
            return F::Bc5;
        case FourCc("BC5S"):
            return F::Bc5Snorm;
        case 113:
            return F::Rgba16F;  // A16B16G16R16F
        case 116:
            return F::Rgba32F;  // A32B32G32R32F
        default:
            throw std::runtime_error("DDS file has an unsupported FourCC code " +
                                     std::to_string(code));
        }
    }
    const std::uint32_t bits = U32(file, at + 12);
    const std::uint32_t red = U32(file, at + 16);
    const std::uint32_t green = U32(file, at + 20);
    const std::uint32_t blue = U32(file, at + 24);
    if ((flags & kRgbFlag) != 0 && bits == 32 && green == 0xff00u) {
        if (red == 0xffu && blue == 0xff0000u) {
            return F::Rgba8;
        }
        if (red == 0xff0000u && blue == 0xffu) {
            return F::Bgra8;
        }
    }
    throw std::runtime_error("DDS file has an unsupported uncompressed layout");
}

/// A DDS file's texture: copied into `bytes` with `copy`, else read from `file` in place
/// (`mapped`, the offsets into it).
TextureData ParseDds(std::span<const std::byte> file, bool copy) {
    // "DDS ", then the 124-byte DDS_HEADER, its pixel format at 76.
    constexpr std::size_t kHeader = 4;
    constexpr std::size_t kPixelFormat = kHeader + 72;
    if (U32(file, 0) != FourCc("DDS ") || U32(file, kHeader) != 124) {
        throw std::runtime_error("Not a DDS file");
    }
    constexpr std::uint32_t kMipCountFlag = 0x20000;
    constexpr std::uint32_t kDepthFlag = 0x800000;
    constexpr std::uint32_t kCubeMap = 0x200;
    const std::uint32_t flags = U32(file, kHeader + 4);
    const std::uint32_t height = U32(file, kHeader + 8);
    const std::uint32_t width = U32(file, kHeader + 12);
    const std::uint32_t depth = U32(file, kHeader + 20);
    const std::uint32_t mipCount = U32(file, kHeader + 24);
    const std::uint32_t caps2 = U32(file, kHeader + 108);
    if ((flags & kDepthFlag) != 0 && depth > 1) {
        throw std::runtime_error("DDS volume textures are not read");
    }
    const std::uint32_t mips = (flags & kMipCountFlag) != 0 && mipCount > 0 ? mipCount : 1;

    std::size_t data = kHeader + 124;
    TextureFormat format;
    std::uint32_t layers = 1;
    bool cube = (caps2 & kCubeMap) != 0;
    if (U32(file, kPixelFormat + 8) == FourCc("DX10")) {
        // DDS_HEADER_DXT10: format, dimension, misc flags, array size.
        const std::uint32_t dxgi = U32(file, data);
        const std::uint32_t dimension = U32(file, data + 4);
        constexpr std::uint32_t kTexture2d = 3;
        constexpr std::uint32_t kCubeFlag = 0x4;
        if (dimension != kTexture2d) {
            throw std::runtime_error("DDS file is not a 2D texture (dimension " +
                                     std::to_string(dimension) + ")");
        }
        const std::optional<TextureFormat> known = FromDxgi(dxgi);
        if (!known) {
            throw std::runtime_error("DDS file has an unsupported DXGI format " +
                                     std::to_string(dxgi));
        }
        format = *known;
        cube = (U32(file, data + 8) & kCubeFlag) != 0;
        layers = std::max(1u, U32(file, data + 12));
        data += 20;
    } else {
        format = LegacyDdsFormat(file, kPixelFormat);
    }
    if (cube) {
        if (layers > kMaxLayers / 6) {
            throw std::runtime_error("DDS file has too many cube maps");
        }
        layers *= 6;
    }
    TextureData texture = Shape(format, width, height, layers, mips, copy);
    texture.cube = cube;
    // Each layer's mips follow the one before, as `TextureData` lays them out.
    const std::size_t total = ShapeBytes(texture);
    if (file.size() < data || file.size() - data < total) {
        throw std::runtime_error("DDS file is truncated");
    }
    if (copy) {
        std::memcpy(texture.bytes.data(), file.data() + data, total);
    } else {
        texture.mapped = file.subspan(data, total);
    }
    return texture;
}

/// A KTX2 file of Basis Universal's ETC1S or UASTC formats transcoded to BC blocks: two-channel
/// data (normal maps' x and y) to BC5, HDR to BC6H, the rest to BC7, sRGB as the file's transfer
/// function says.
TextureData TranscodeBasis(std::span<const std::byte> file) {
    ERNIE_ZONE("Transcode Basis");
    static std::once_flag initialised;
    std::call_once(initialised, basist::basisu_transcoder_init);
    basist::ktx2_transcoder ktx2;
    if (file.size() > UINT32_MAX ||
        !ktx2.init(file.data(), static_cast<std::uint32_t>(file.size())) ||
        !ktx2.start_transcoding()) {
        throw std::runtime_error("KTX2 file holds no Basis Universal data it can transcode");
    }
    TextureFormat format = F::Bc7;
    basist::transcoder_texture_format target = basist::transcoder_texture_format::cTFBC7_RGBA;
    // BC5 takes x from red and y from alpha unless told otherwise.
    int first = -1;
    int second = -1;
    const auto channel0 = ktx2.get_dfd_channel_id0();
    const bool twoChannels =
        ktx2.is_etc1s() ? channel0 == basist::KTX2_DF_CHANNEL_ETC1S_RRR &&
                              ktx2.get_dfd_channel_id1() == basist::KTX2_DF_CHANNEL_ETC1S_GGG
                        : channel0 == basist::KTX2_DF_CHANNEL_UASTC_RRRG ||
                              channel0 == basist::KTX2_DF_CHANNEL_UASTC_RG;
    if (ktx2.is_hdr()) {
        format = F::Bc6hUfloat;
        target = basist::transcoder_texture_format::cTFBC6H;
    } else if (twoChannels) {
        format = F::Bc5;
        target = basist::transcoder_texture_format::cTFBC5_RG;
        if (!ktx2.is_etc1s() && channel0 == basist::KTX2_DF_CHANNEL_UASTC_RG) {
            first = 0;
            second = 1;
        }
    } else if (ktx2.get_dfd_transfer_func() == basist::KTX2_KHR_DF_TRANSFER_SRGB) {
        format = F::Bc7Srgb;
    }
    const std::uint32_t faces = ktx2.get_faces();
    const std::uint32_t layers = std::max(1u, ktx2.get_layers());
    const std::uint32_t levels = std::max(1u, ktx2.get_levels());
    if (layers > kMaxLayers / std::max(1u, faces)) {
        throw std::runtime_error("KTX2 file has too many layers");
    }
    TextureData texture =
        Shape(format, ktx2.get_width(), ktx2.get_height(), layers * faces, levels);
    texture.cube = faces == 6;
    for (std::uint32_t level = 0; level < levels; ++level) {
        const std::uint32_t width = MipSide(texture.width, level);
        const std::uint32_t height = MipSide(texture.height, level);
        const auto blocks = static_cast<std::uint32_t>(((width + 3) / 4) * ((height + 3) / 4));
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            for (std::uint32_t face = 0; face < faces; ++face) {
                const std::size_t slot = (std::size_t{layer} * faces + face) * levels + level;
                if (!ktx2.transcode_image_level(level, layer, face,
                                                texture.bytes.data() + texture.offsets[slot],
                                                blocks, target, 0, 0, 0, first, second)) {
                    throw std::runtime_error("KTX2 file's level " + std::to_string(level) +
                                             " does not transcode");
                }
            }
        }
    }
    return texture;
}

/// A KTX2 file's texture: its levels as `ParseDds` reads, in place without `copy` unless they are
/// supercompressed (decoded into `bytes`).
TextureData ParseKtx2(std::span<const std::byte> file, bool copy) {
    constexpr std::array<std::uint8_t, 12> kIdentifier{0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32,
                                                       0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
    if (file.size() < 80 || std::memcmp(file.data(), kIdentifier.data(), 12) != 0) {
        throw std::runtime_error("Not a KTX2 file");
    }
    const std::uint32_t vkFormat = U32(file, 12);
    const std::uint32_t width = U32(file, 20);
    const std::uint32_t height = U32(file, 24);
    const std::uint32_t depth = U32(file, 28);
    const std::uint32_t layerCount = U32(file, 32);
    const std::uint32_t faces = U32(file, 36);
    const std::uint32_t levels = std::max(1u, U32(file, 40));
    const std::uint32_t supercompression = U32(file, 44);
    if (depth > 1) {
        throw std::runtime_error("KTX2 volume textures are not read");
    }
    if (faces != 1 && faces != 6) {
        throw std::runtime_error("KTX2 file has " + std::to_string(faces) + " faces");
    }
    // Basis Universal's formats (ETC1S, often BasisLZ-supercompressed, and UASTC) have no
    // VkFormat.
    if (vkFormat == 0) {
        return TranscodeBasis(file);
    }
    constexpr std::uint32_t kZstd = 2;
    if (supercompression != 0 && supercompression != kZstd) {
        throw std::runtime_error("KTX2 file uses supercompression scheme " +
                                 std::to_string(supercompression) +
                                 ", which is not read (only zstd)");
    }
    const std::optional<TextureFormat> format = FromVkFormat(vkFormat);
    if (!format) {
        throw std::runtime_error("KTX2 file has an unsupported VkFormat " +
                                 std::to_string(vkFormat));
    }
    const std::uint32_t arrays = std::max(1u, layerCount);
    if (arrays > kMaxLayers / faces) {
        throw std::runtime_error("KTX2 file has too many layers");
    }
    // Stored levels are read in place: each image is contiguous in the file, the offsets pointing
    // at it.
    const bool inPlace = !copy && supercompression == 0;
    TextureData texture = Shape(*format, width, height, arrays * faces, levels, !inPlace);
    texture.cube = faces == 6;
    if (inPlace) {
        texture.mapped = file;
    }
    // The level index follows the 80-byte header: each level's offset, stored length and
    // uncompressed length, level 0 the largest. A level holds every layer's faces in turn.
    std::vector<std::byte> decompressed;
    for (std::uint32_t level = 0; level < levels; ++level) {
        const std::size_t entry = 80 + std::size_t{level} * 24;
        const auto offset = ReadLittleEndian<std::uint64_t>(file, entry);
        const auto length = ReadLittleEndian<std::uint64_t>(file, entry + 8);
        const auto uncompressed = ReadLittleEndian<std::uint64_t>(file, entry + 16);
        const std::size_t image =
            ImageBytes(*format, MipSide(width, level), MipSide(height, level));
        const std::size_t expected = image * texture.layers;
        if (offset > file.size() || file.size() - offset < length ||
            (supercompression == 0 ? length : uncompressed) != expected) {
            throw std::runtime_error("KTX2 file has a malformed level " + std::to_string(level));
        }
        if (inPlace) {
            for (std::uint32_t layer = 0; layer < texture.layers; ++layer) {
                texture.offsets[std::size_t{layer} * levels + level] =
                    static_cast<std::size_t>(offset) + std::size_t{layer} * image;
            }
            continue;
        }
        const std::byte* source = file.data() + offset;
        if (supercompression == kZstd) {
            decompressed.resize(expected);
            const std::size_t size = ZSTD_decompress(decompressed.data(), expected, source,
                                                     static_cast<std::size_t>(length));
            if (ZSTD_isError(size) != 0 || size != expected) {
                throw std::runtime_error("KTX2 file's level " + std::to_string(level) +
                                         " does not decompress");
            }
            source = decompressed.data();
        }
        for (std::uint32_t layer = 0; layer < texture.layers; ++layer) {
            std::memcpy(texture.bytes.data() + texture.offsets[std::size_t{layer} * levels + level],
                        source + std::size_t{layer} * image, image);
        }
    }
    return texture;
}

// ----------------------------------------------------------------------------------------------
// DDS writing

/// Bytes before a DDS file's texels as `WriteDdsHeader` writes them.
constexpr std::size_t kDdsHeaderBytes = (32 + 5) * 4;

/// Writes the header of a DDS file of `texture`, with the DX10 header.
void WriteDdsHeader(std::ostream& file, const TextureData& texture) {
    static_assert(std::endian::native == std::endian::little,
                  "DDS headers are written as host words");
    // "DDS ", DDS_HEADER (31 words after the magic), DDS_HEADER_DXT10.
    std::array<std::uint32_t, 32 + 5> header{};
    header[0] = FourCc("DDS ");
    header[1] = 124;
    // Caps, height, width, pixel format, linear size and mip count.
    header[2] = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000 | (texture.mips > 1 ? 0x20000u : 0u);
    header[3] = texture.height;
    header[4] = texture.width;
    header[5] =
        static_cast<std::uint32_t>(ImageBytes(texture.format, texture.width, texture.height));
    header[7] = texture.mips;
    header[19] = 32;
    header[20] = 0x4;  // FourCC
    header[21] = FourCc("DX10");
    // DDSCAPS_TEXTURE, and COMPLEX and MIPMAP with mips.
    header[27] = 0x1000 | (texture.mips > 1 ? 0x400008u : 0u);
    header[28] = texture.cube ? 0xfe00u : 0u;
    header[32] = Row(texture.format).dxgi;
    header[33] = 3;  // A 2D texture.
    header[34] = texture.cube ? 0x4u : 0u;
    header[35] = texture.cube ? texture.layers / 6 : texture.layers;
    static_assert(sizeof(header) == kDdsHeaderBytes);
    file.write(reinterpret_cast<const char*>(header.data()), sizeof(header));
}

// ----------------------------------------------------------------------------------------------
// Encoding

/// Encodes one 4 x 4 block of RGBA8 `pixels` into `block`.
void EncodeBlock(TextureFormat format, const std::uint8_t* pixels, std::byte* block,
                 const bc7enc_compress_block_params& bc7) {
    // rgbcx's effort for BC1 and BC3, of 0 to 18.
    constexpr std::uint32_t kLevel = 10;
    switch (format) {
    case F::Bc7:
    case F::Bc7Srgb:
        bc7enc_compress_block(block, pixels, &bc7);
        return;
    case F::Bc1:
    case F::Bc1Srgb:
        rgbcx::encode_bc1(kLevel, block, pixels, true, false);
        return;
    case F::Bc3:
    case F::Bc3Srgb:
        rgbcx::encode_bc3(kLevel, block, pixels);
        return;
    case F::Bc4:
        rgbcx::encode_bc4(block, pixels, 4);
        return;
    case F::Bc5:
        rgbcx::encode_bc5(block, pixels, 0, 1, 4);
        return;
    default:
        throw std::invalid_argument("Textures are not encoded to this format");
    }
}

/// `rgba` (`width` x `height` RGBA8) encoded into BC `format` blocks at `out` on the CPU, rows of
/// blocks on every thread.
void EncodeBlocks(TextureFormat format, std::span<const std::uint8_t> rgba, std::uint32_t width,
                  std::uint32_t height, std::byte* out, const bc7enc_compress_block_params& bc7) {
    ERNIE_ZONE("Encode blocks on the CPU");
    const std::uint32_t blockBytes = FormatInfo(format).blockBytes;
    const std::uint32_t across = (width + 3) / 4;
    ParallelFor((height + 3) / 4, [&](std::size_t blockRow) {
        const auto row = static_cast<std::uint32_t>(blockRow);
        std::array<std::uint8_t, 64> pixels{};
        for (std::uint32_t column = 0; column < across; ++column) {
            // Texels past the edge repeat the last row and column.
            for (std::uint32_t y = 0; y < 4; ++y) {
                const std::uint32_t sy = std::min(row * 4 + y, height - 1);
                for (std::uint32_t x = 0; x < 4; ++x) {
                    const std::uint32_t sx = std::min(column * 4 + x, width - 1);
                    std::memcpy(pixels.data() + 4 * (y * 4 + x),
                                rgba.data() + 4 * (std::size_t{sy} * width + sx), 4);
                }
            }
            EncodeBlock(format, pixels.data(),
                        out + (std::size_t{row} * across + column) * blockBytes, bc7);
        }
    });
}

/// BC7's settings for `format` at effort `quality`, the encoders initialised.
bc7enc_compress_block_params Bc7Params(TextureFormat format, int quality) {
    static std::once_flag initialised;
    std::call_once(initialised, [] {
        bc7enc_compress_block_init();
        rgbcx::init();
    });
    bc7enc_compress_block_params bc7{};
    bc7enc_compress_block_params_init(&bc7);
    if (format == F::Bc7) {
        bc7enc_compress_block_params_init_linear_weights(&bc7);
    }
    bc7.m_uber_level = static_cast<std::uint32_t>(std::clamp(quality, 0, BC7ENC_MAX_UBER_LEVEL));
    return bc7;
}

/// `rgba` (`width` x `height` RGBA8) stored as `format` at `out`: copied when uncompressed, else
/// by `encoder` where it handles the format, else on the CPU.
void EncodeRgba8(TextureFormat format, std::span<const std::uint8_t> rgba, std::uint32_t width,
                 std::uint32_t height, std::byte* out, const bc7enc_compress_block_params& bc7,
                 BlockEncoder* encoder) {
    if (!FormatInfo(format).compressed()) {
        std::memcpy(out, rgba.data(), rgba.size());
    } else if (encoder != nullptr && encoder->Encodes(format)) {
        encoder->Encode(rgba, width, height, format, out);
    } else {
        EncodeBlocks(format, rgba, width, height, out, bc7);
    }
}

/// The next mip of RGBA8 `rgba` (`width` x `height`): each texel the average of the 2 x 2 above
/// it, clamped at odd edges, as `MipChain`'s `MipFilter::Average`; with `srgb` its rgb averaged as
/// linear values.
std::vector<std::uint8_t> HalveRgba8(std::span<const std::uint8_t> rgba, std::uint32_t width,
                                     std::uint32_t height, bool srgb) {
    ERNIE_ZONE("Halve RGBA8");
    static const std::array<float, 256> kLinear = [] {
        std::array<float, 256> table{};
        for (std::size_t i = 0; i < table.size(); ++i) {
            table[i] = SrgbToLinear(static_cast<float>(i) / 255.0f);
        }
        return table;
    }();
    // Linear back to sRGB bytes by table rather than a power a texel, the most of the halving's
    // time: 2^14 steps of linear, finer than half a byte of sRGB even near black, where its slope
    // is 12.92.
    constexpr std::size_t kSteps = 1u << 14;
    static const std::array<std::uint8_t, kSteps + 1> kSrgb = [] {
        std::array<std::uint8_t, kSteps + 1> table{};
        for (std::size_t i = 0; i < table.size(); ++i) {
            table[i] = static_cast<std::uint8_t>(
                std::lround(LinearToSrgb(static_cast<float>(i) / kSteps) * 255.0f));
        }
        return table;
    }();
    const std::uint32_t w = MipSide(width, 1);
    const std::uint32_t h = MipSide(height, 1);
    std::vector<std::uint8_t> half(std::size_t{w} * h * 4);
    ParallelFor(h, [&](std::size_t line) {
        const auto y = static_cast<std::uint32_t>(line);
        for (std::uint32_t x = 0; x < w; ++x) {
            for (std::uint32_t c = 0; c < 4; ++c) {
                float sum = 0.0f;
                for (std::uint32_t dy = 0; dy < 2; ++dy) {
                    for (std::uint32_t dx = 0; dx < 2; ++dx) {
                        const std::uint32_t sx = std::min(2 * x + dx, width - 1);
                        const std::uint32_t sy = std::min(2 * y + dy, height - 1);
                        const std::uint8_t value = rgba[(std::size_t{sy} * width + sx) * 4 + c];
                        sum += srgb && c < 3 ? kLinear[value] : static_cast<float>(value) / 255.0f;
                    }
                }
                const float mean = std::clamp(sum / 4.0f, 0.0f, 1.0f);
                half[(std::size_t{y} * w + x) * 4 + c] =
                    srgb && c < 3 ? kSrgb[static_cast<std::size_t>(std::lround(mean * kSteps))]
                                  : static_cast<std::uint8_t>(std::lround(mean * 255.0f));
            }
        }
    });
    return half;
}

/// An 8-bit PNG or JPEG decoded as RGBA8, in stb's buffer.
struct Rgba8Image {
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> texels{nullptr, &stbi_image_free};
    std::uint32_t width = 0;
    std::uint32_t height = 0;

    std::span<const std::uint8_t> Texels() const {
        return {texels.get(), std::size_t{width} * height * 4};
    }
};

Rgba8Image DecodeRgba8(const fs::path& file) {
    ERNIE_ZONE("Decode image");
    const std::string name = file.string();
    int width = 0;
    int height = 0;
    int channels = 0;
    Rgba8Image image;
    image.texels.reset(stbi_load(name.c_str(), &width, &height, &channels, 4));
    if (image.texels == nullptr) {
        throw std::runtime_error("Cannot read " + name + ": " + stbi_failure_reason());
    }
    image.width = static_cast<std::uint32_t>(width);
    image.height = static_cast<std::uint32_t>(height);
    return image;
}

/// The format and sRGB encoding a bake stores in.
TextureFormat BakeFormat(const BakeOptions& options, bool srgb) {
    const TextureFormat role = options.role == TextureRole::Normal ? F::Bc5 : F::Bc7;
    return WithSrgb(options.format.value_or(role), srgb);
}

bool BakesSrgb(const BakeOptions& options) {
    return options.role == TextureRole::Colour && options.srgb;
}

}  // namespace

// ----------------------------------------------------------------------------------------------
// Formats

TextureFormatInfo FormatInfo(TextureFormat format) {
    return Row(format).info;
}

TextureFormat WithSrgb(TextureFormat format, bool srgb) {
    constexpr std::array<std::array<F, 2>, 6> kPairs{{
        {F::Rgba8, F::Rgba8Srgb},
        {F::Bgra8, F::Bgra8Srgb},
        {F::Bc1, F::Bc1Srgb},
        {F::Bc2, F::Bc2Srgb},
        {F::Bc3, F::Bc3Srgb},
        {F::Bc7, F::Bc7Srgb},
    }};
    for (const auto& pair : kPairs) {
        if (format == pair[0] || format == pair[1]) {
            return pair[srgb ? 1 : 0];
        }
    }
    return format;
}

std::size_t ImageBytes(TextureFormat format, std::uint32_t width, std::uint32_t height) {
    const TextureFormatInfo info = FormatInfo(format);
    const std::size_t across = (width + info.blockSize - 1) / info.blockSize;
    const std::size_t down = (height + info.blockSize - 1) / info.blockSize;
    return across * down * info.blockBytes;
}

std::uint32_t MipCount(std::uint32_t width, std::uint32_t height) {
    return std::max(1u, static_cast<std::uint32_t>(std::bit_width(std::max(width, height))));
}

// ----------------------------------------------------------------------------------------------
// Stored textures

std::span<const std::byte> TextureData::Mip(std::uint32_t layer, std::uint32_t mip) const {
    if (layer >= layers || mip >= mips) {
        throw std::out_of_range("Texture has no layer " + std::to_string(layer) + " mip " +
                                std::to_string(mip));
    }
    const std::size_t at = offsets[std::size_t{layer} * mips + mip];
    return Data().subspan(at, ImageBytes(format, MipSide(width, mip), MipSide(height, mip)));
}

bool IsTextureFile(const fs::path& path) {
    const std::string extension = ToLower(path.extension().string());
    return extension == ".dds" || extension == ".ktx2";
}

TextureData LoadTextureFile(const fs::path& path) {
    ERNIE_ZONE("Load texture file");
    if (!IsTextureFile(path)) {
        throw std::runtime_error(path.string() + " is neither a .dds nor a .ktx2 file");
    }
    // Mapped, so texels are read from disk only as they are used: a virtual texture's tiles in
    // view, not the whole file.
    auto mapping = std::make_shared<const MappedFile>(path);
    try {
        TextureData texture = ToLower(path.extension().string()) == ".dds"
                                  ? ParseDds(mapping->Bytes(), false)
                                  : ParseKtx2(mapping->Bytes(), false);
        if (!texture.mapped.empty()) {
            texture.mapping = std::move(mapping);
        }
        return texture;
    } catch (const std::runtime_error& error) {
        throw std::runtime_error(path.string() + ": " + error.what());
    }
}

TextureData ReadDds(std::span<const std::byte> file) {
    return ParseDds(file, true);
}

TextureData ReadKtx2(std::span<const std::byte> file) {
    return ParseKtx2(file, true);
}

void WriteDdsFile(const fs::path& path, const TextureData& texture) {
    ERNIE_ZONE("Write DDS");
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot write " + path.string());
    }
    WriteDdsHeader(file, texture);
    // Layer after layer, each layer's mips in turn: `TextureData`'s order.
    const std::span<const std::byte> data = texture.Data();
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size()));
    if (!file) {
        throw std::runtime_error("Cannot write " + path.string());
    }
}

TextureData DecodeBlocks(const TextureData& texture) {
    const TextureFormatInfo info = FormatInfo(texture.format);
    if (!info.compressed()) {
        return texture;
    }
    ERNIE_ZONE("Decode BC blocks");
    TextureFormat decoded = F::Rgba8;
    switch (texture.format) {
    case F::Bc1:
    case F::Bc3:
    case F::Bc7:
        break;
    case F::Bc1Srgb:
    case F::Bc3Srgb:
    case F::Bc7Srgb:
        decoded = F::Rgba8Srgb;
        break;
    case F::Bc4:
        decoded = F::R8;
        break;
    case F::Bc5:
        decoded = F::Rg8;
        break;
    default:
        throw std::invalid_argument("BC2, BC6H and signed BC4 and BC5 are not decoded");
    }
    const std::uint32_t channels = FormatInfo(decoded).channels;

    TextureData out = Shape(decoded, texture.width, texture.height, texture.layers, texture.mips);
    out.cube = texture.cube;
    for (std::uint32_t layer = 0; layer < texture.layers; ++layer) {
        for (std::uint32_t mip = 0; mip < texture.mips; ++mip) {
            const std::uint32_t w = MipSide(texture.width, mip);
            const std::uint32_t h = MipSide(texture.height, mip);
            const std::span<const std::byte> blocks = texture.Mip(layer, mip);
            std::byte* texels = out.bytes.data() + out.offsets[std::size_t{layer} * out.mips + mip];
            const std::uint32_t across = (w + 3) / 4;
            for (std::uint32_t by = 0; by < (h + 3) / 4; ++by) {
                for (std::uint32_t bx = 0; bx < across; ++bx) {
                    const std::byte* block =
                        blocks.data() + (std::size_t{by} * across + bx) * info.blockBytes;
                    // The block's 4 x 4 texels as RGBA8.
                    std::array<std::uint8_t, 64> rgba{};
                    switch (texture.format) {
                    case F::Bc1:
                    case F::Bc1Srgb:
                        rgbcx::unpack_bc1(block, rgba.data());
                        break;
                    case F::Bc3:
                    case F::Bc3Srgb:
                        rgbcx::unpack_bc3(block, rgba.data());
                        break;
                    case F::Bc4:
                        rgbcx::unpack_bc4(block, rgba.data());
                        break;
                    case F::Bc5:
                        rgbcx::unpack_bc5(block, rgba.data());
                        break;
                    default:
                        bc7decomp::unpack_bc7(
                            block, reinterpret_cast<bc7decomp::color_rgba*>(rgba.data()));
                        break;
                    }
                    for (std::uint32_t i = 0; i < 16; ++i) {
                        const std::uint32_t x = bx * 4 + i % 4;
                        const std::uint32_t y = by * 4 + i / 4;
                        if (x < w && y < h) {
                            std::memcpy(texels + (std::size_t{y} * w + x) * channels,
                                        rgba.data() + std::size_t{i} * 4, channels);
                        }
                    }
                }
            }
        }
    }
    return out;
}

// ----------------------------------------------------------------------------------------------
// Baking

std::optional<glm::uvec2> Rgba8ImageSize(const fs::path& file) {
    if (!IsPngOrJpeg(file)) {
        return std::nullopt;
    }
    const std::string name = file.string();
    int width = 0;
    int height = 0;
    int channels = 0;
    if (stbi_info(name.c_str(), &width, &height, &channels) == 0 ||
        stbi_is_16_bit(name.c_str()) != 0) {
        return std::nullopt;
    }
    return glm::uvec2(static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height));
}

std::vector<std::vector<std::uint8_t>> LoadRgba8Mips(const fs::path& file, bool srgb) {
    ERNIE_ZONE("Load RGBA8 mips");
    if (!Rgba8ImageSize(file)) {
        throw std::runtime_error(file.string() + " is not an 8-bit PNG or JPEG");
    }
    Rgba8Image image = DecodeRgba8(file);
    std::uint32_t w = image.width;
    std::uint32_t h = image.height;
    std::vector<std::vector<std::uint8_t>> mips;
    mips.emplace_back(image.Texels().begin(), image.Texels().end());
    image.texels.reset();
    while (w > 1 || h > 1) {
        mips.push_back(HalveRgba8(mips.back(), w, h, srgb));
        w = MipSide(w, 1);
        h = MipSide(h, 1);
    }
    return mips;
}

TextureData EncodeTexture(std::span<const Image> mips, TextureFormat format, ImageEncoding encoding,
                          int quality, BlockEncoder* encoder) {
    if (mips.empty()) {
        throw std::invalid_argument("A texture needs a mip");
    }
    TextureData texture =
        Allocate(format, mips[0].width, mips[0].height, static_cast<std::uint32_t>(mips.size()));
    for (std::uint32_t mip = 0; mip < texture.mips; ++mip) {
        if (mips[mip].width != MipSide(texture.width, mip) ||
            mips[mip].height != MipSide(texture.height, mip)) {
            throw std::invalid_argument("Mip " + std::to_string(mip) +
                                        " does not halve the one before");
        }
    }
    const bc7enc_compress_block_params bc7 = Bc7Params(format, quality);
    for (std::uint32_t mip = 0; mip < texture.mips; ++mip) {
        EncodeRgba8(format, ToRgba8(mips[mip], encoding), mips[mip].width, mips[mip].height,
                    texture.bytes.data() + texture.offsets[mip], bc7, encoder);
    }
    return texture;
}

TextureData BakeTexture(const fs::path& input, const fs::path& output, const BakeOptions& options) {
    ERNIE_ZONE("Bake texture");
    const bool srgb = BakesSrgb(options);
    const TextureFormat format = BakeFormat(options, srgb);
    // Rasters read in parts bake a band at a time.
    const std::string extension = ToLower(input.extension().string());
    if (extension == ".tif" || extension == ".tiff" || extension == ".img" || extension == ".lbl" ||
        extension == ".mosaic") {
        RasterBake(OpenRaster(input), output, options).Finish();
        return LoadTextureFile(output);
    }
    // 8-bit PNG and JPEG colour and data stay 8-bit, each mip made from the last and dropped once
    // encoded: 4 bytes a texel of the full size rather than float's 16 for the whole chain, which
    // a global mosaic needs. Normal maps (their slopes) and EXR and 16-bit images go through
    // floats.
    if (options.role != TextureRole::Normal && Rgba8ImageSize(input)) {
        Rgba8Image image = DecodeRgba8(input);
        std::uint32_t w = image.width;
        std::uint32_t h = image.height;
        TextureData texture = Allocate(format, w, h, MipCount(w, h));
        const bc7enc_compress_block_params bc7 = Bc7Params(format, options.quality);
        std::vector<std::uint8_t> halved;
        std::span<const std::uint8_t> rgba = image.Texels();
        for (std::uint32_t mip = 0; mip < texture.mips; ++mip) {
            EncodeRgba8(format, rgba, w, h, texture.bytes.data() + texture.offsets[mip], bc7,
                        options.encoder);
            if (mip + 1 < texture.mips) {
                halved = HalveRgba8(rgba, w, h, srgb);
                rgba = halved;
                // Mip 0 is read from the decoder's buffer, freed once halved.
                image.texels.reset();
                w = MipSide(w, 1);
                h = MipSide(h, 1);
            }
        }
        WriteDdsFile(output, texture);
        return texture;
    }
    const ImageEncoding encoding = srgb ? ImageEncoding::Srgb : ImageEncoding::Linear;
    const std::vector<Image> mips = MipChain(
        LoadImageFile(input, encoding),
        options.role == TextureRole::Normal ? MipFilter::NormalSlopes : MipFilter::Average);
    // Normals' mips hold n / 2 + 1/2 in rgb, as the stored-texture decode reads x and y; BC5 keeps
    // those two.
    TextureData texture = EncodeTexture(mips, format, encoding, options.quality, options.encoder);
    WriteDdsFile(output, texture);
    return texture;
}

namespace {

/// Rows of a band read at once.
constexpr std::uint32_t kBandRows = 1024;
/// Band mips waiting to be encoded, past which reading waits.
constexpr std::size_t kQueuedBytes = std::size_t{256} << 20;

/// The texels of mip `mip` from a span [`start`, `start` + `count`) of a side of `side` texels:
/// the span's start there and how many. Spans start at multiples of 2^mip, so a mip's spans tile
/// it.
std::pair<std::uint32_t, std::uint32_t> MipSpan(std::uint32_t side, std::uint32_t start,
                                                std::uint32_t count, std::uint32_t mip) {
    const std::uint32_t first = start >> mip;
    const std::uint32_t end = start + count == side ? MipSide(side, mip) : (start + count) >> mip;
    return {first, end > first ? end - first : 0};
}

/// `pixels` samples of `source`'s at `samples` as RGBA8 at `out`.
void SamplesToRgba8(const GridSource& source, const std::byte* samples, std::size_t pixels,
                    std::uint8_t* out) {
    const std::uint32_t channels = source.Channels();
    const bool wide = source.Type() == GridSampleType::UInt16;
    for (std::size_t i = 0; i < pixels; ++i) {
        std::array<std::uint8_t, 4> c{0, 0, 0, 255};
        for (std::uint32_t k = 0; k < std::min(channels, 4u); ++k) {
            const std::size_t sample = i * channels + k;
            c[k] = wide
                       ? static_cast<std::uint8_t>(LoadAs<std::uint16_t>(samples + sample * 2) >> 8)
                       : static_cast<std::uint8_t>(samples[sample]);
        }
        if (channels <= 2) {
            // Grey, and alpha.
            c = {c[0], c[0], c[0], channels == 2 ? c[1] : c[3]};
        }
        std::memcpy(out + i * 4, c.data(), 4);
    }
}

/// The rows and columns of `rgba` (`width` wide) up to `w` x `h`.
std::vector<std::uint8_t> Cropped(std::vector<std::uint8_t> rgba, std::uint32_t width,
                                  std::uint32_t w, std::uint32_t h) {
    if (w == width) {
        rgba.resize(std::size_t{w} * h * 4);
        return rgba;
    }
    std::vector<std::uint8_t> cropped(std::size_t{w} * h * 4);
    for (std::uint32_t y = 0; y < h; ++y) {
        std::memcpy(cropped.data() + std::size_t{y} * w * 4,
                    rgba.data() + std::size_t{y} * width * 4, std::size_t{w} * 4);
    }
    return cropped;
}

}  // namespace

struct RasterBake::State {
    /// A rectangle of a mip to encode: `w` x `h` texels at (`x`, `y`) of mip `mip`, RGBA8.
    struct Piece {
        std::uint32_t mip = 0;
        std::uint32_t x = 0;
        std::uint32_t y = 0;
        std::uint32_t w = 0;
        std::uint32_t h = 0;
        std::vector<std::uint8_t> rgba;
    };

    std::unique_ptr<GridSource> source;
    fs::path output;
    BakeOptions options;
    bool srgb = true;
    TextureData layout;
    TextureFormatInfo info;
    bc7enc_compress_block_params bc7{};
    /// Mips made band by band; the rest are joined.
    std::uint32_t banded = 0;
    std::fstream file;
    std::uint64_t total = 0;
    std::atomic<std::uint64_t> encoded = 0;

    std::mutex mutex;
    std::condition_variable changed;
    std::deque<Piece> pieces;
    std::size_t queued = 0;  ///< Bytes of `pieces`.
    bool read = false;       ///< Every piece queued.
    bool stop = false;
    std::exception_ptr failure;

    /// The piece being encoded, and its rows done.
    std::optional<Piece> current;
    std::uint32_t row = 0;
    bool done = false;
    std::thread reader;  ///< Joined by ~RasterBake.

    /// Queues `piece`, once there is room.
    void Queue(Piece piece) {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return stop || pieces.empty() || queued < kQueuedBytes; });
        if (stop) {
            return;
        }
        queued += piece.rgba.size();
        pieces.push_back(std::move(piece));
        changed.notify_all();
    }

    bool Stopping() {
        const std::scoped_lock lock(mutex);
        return stop;
    }

    /// The reading thread: bands' mips, then the joined ones.
    void Read() {
        try {
            ReadBands();
        } catch (...) {
            const std::scoped_lock lock(mutex);
            failure = std::current_exception();
        }
        const std::scoped_lock lock(mutex);
        read = true;
        changed.notify_all();
    }

    void ReadBands() {
        ERNIE_ZONE("Read raster bands");
        const std::uint32_t width = layout.width;
        const std::uint32_t height = layout.height;
        const std::size_t pixel = std::size_t{source->Channels()} * source->SampleBytes();
        const std::uint32_t joinedWidth = MipSpan(width, 0, width, banded).second;
        const std::uint32_t joinedHeight = MipSpan(height, 0, height, banded).second;
        std::vector<std::uint8_t> joined;
        if (banded < layout.mips) {
            joined.resize(std::size_t{joinedWidth} * joinedHeight * 4);
        }
        std::vector<std::byte> samples;
        for (const GridRect& part : source->Parts()) {
            for (std::uint32_t y = part.y; y < part.y + part.height; y += kBandRows) {
                if (Stopping()) {
                    return;
                }
                const std::uint32_t rows = std::min(kBandRows, part.y + part.height - y);
                std::uint32_t w = part.width;
                std::uint32_t h = rows;
                samples.resize(std::size_t{w} * h * pixel);
                std::vector<std::uint8_t> rgba(std::size_t{w} * h * 4);
                {
                    ERNIE_ZONE("Read band");
                    source->Read(part.x, y, w, h, samples);
                    SamplesToRgba8(*source, samples.data(), std::size_t{w} * h, rgba.data());
                }
                for (std::uint32_t mip = 0;; ++mip) {
                    const auto [x0, across] = MipSpan(width, part.x, part.width, mip);
                    const auto [y0, down] = MipSpan(height, y, rows, mip);
                    if (across == 0 || down == 0) {
                        break;
                    }
                    // Halving keeps a texel of a side of one, which the whole image's mip may not
                    // have here.
                    if (across < w || down < h) {
                        rgba = Cropped(std::move(rgba), w, across, down);
                        w = across;
                        h = down;
                    }
                    if (mip == banded) {
                        for (std::uint32_t r = 0; r < h; ++r) {
                            std::memcpy(joined.data() +
                                            (std::size_t{y0 + r} * joinedWidth + x0) * 4,
                                        rgba.data() + std::size_t{r} * w * 4, std::size_t{w} * 4);
                        }
                        break;
                    }
                    const bool last = mip + 1 == layout.mips;
                    std::vector<std::uint8_t> half =
                        last ? std::vector<std::uint8_t>{} : HalveRgba8(rgba, w, h, srgb);
                    Queue(Piece{mip, x0, y0, w, h, std::move(rgba)});
                    if (last || Stopping()) {
                        break;
                    }
                    rgba = std::move(half);
                    w = MipSide(w, 1);
                    h = MipSide(h, 1);
                }
            }
        }
        source->Release();
        std::uint32_t w = joinedWidth;
        std::uint32_t h = joinedHeight;
        for (std::uint32_t mip = banded; mip < layout.mips; ++mip) {
            if (Stopping()) {
                return;
            }
            std::vector<std::uint8_t> half = mip + 1 < layout.mips ? HalveRgba8(joined, w, h, srgb)
                                                                   : std::vector<std::uint8_t>{};
            Queue(Piece{mip, 0, 0, w, h, std::move(joined)});
            joined = std::move(half);
            w = MipSide(w, 1);
            h = MipSide(h, 1);
        }
    }
};

RasterBake::RasterBake(std::unique_ptr<GridSource> source, const fs::path& output,
                       const BakeOptions& options)
    : state_(std::make_unique<State>()) {
    State& s = *state_;
    if (options.role == TextureRole::Normal) {
        throw std::invalid_argument("Normal maps are not baked by bands");
    }
    if (source->Type() != GridSampleType::UInt8 && source->Type() != GridSampleType::UInt16) {
        throw std::invalid_argument("Textures bake from 8- or 16-bit samples");
    }
    s.source = std::move(source);
    s.output = output;
    s.options = options;
    s.srgb = BakesSrgb(options);
    const TextureFormat format = BakeFormat(options, s.srgb);
    const std::uint32_t width = s.source->Width();
    const std::uint32_t height = s.source->Height();
    const std::uint32_t mips = MipCount(width, height);
    s.layout = Allocate(format, width, height, mips, false);
    s.info = FormatInfo(format);
    s.bc7 = Bc7Params(format, options.quality);
    for (std::uint32_t mip = 0; mip < mips; ++mip) {
        s.total += std::uint64_t{MipSide(width, mip)} * MipSide(height, mip);
    }
    // Bands start at multiples of `alignment` texels: their mips are made band by band while their
    // starts there are of whole blocks and halve exactly into the next.
    const std::vector<GridRect> parts = s.source->Parts();
    std::uint32_t alignment = kBandRows;
    for (const GridRect& part : parts) {
        alignment = std::gcd(alignment, std::gcd(part.x, part.y));
        if (part.x + part.width != width) {
            alignment = std::gcd(alignment, part.width);
        }
        if (part.y + part.height != height) {
            alignment = std::gcd(alignment, part.height);
        }
    }
    const bool whole = parts.size() == 1 && height <= kBandRows;
    s.banded = whole ? mips : 0;
    while (!whole && s.banded < mips &&
           alignment % (std::max(s.info.blockSize, 2u) << s.banded) == 0) {
        ++s.banded;
    }
    s.file.open(output, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!s.file) {
        throw std::runtime_error("Cannot write " + output.string());
    }
    WriteDdsHeader(s.file, s.layout);
    s.reader = std::thread([&s] { s.Read(); });
}

RasterBake::~RasterBake() {
    State& s = *state_;
    {
        const std::scoped_lock lock(s.mutex);
        s.stop = true;
        s.changed.notify_all();
    }
    if (s.reader.joinable()) {
        s.reader.join();
    }
    if (!s.done) {
        s.file.close();
        std::error_code error;
        fs::remove(s.output, error);
    }
}

const TextureData& RasterBake::Layout() const {
    return state_->layout;
}

std::uint64_t RasterBake::Encoded() const {
    return state_->encoded;
}

std::uint64_t RasterBake::Total() const {
    return state_->total;
}

const TextureData& RasterBake::Finish() {
    while (!Step(~std::uint64_t{0}, true)) {
    }
    return Layout();
}

bool RasterBake::Step(std::uint64_t texels, bool wait) {
    ERNIE_ZONE("Raster bake step");
    State& s = *state_;
    std::uint64_t spent = 0;
    const std::uint32_t side = s.info.blockSize;
    std::vector<std::byte> blocks;
    while (!s.done) {
        if (!s.current) {
            std::unique_lock lock(s.mutex);
            if (wait) {
                s.changed.wait(lock, [&] { return !s.pieces.empty() || s.read; });
            }
            if (s.failure) {
                std::rethrow_exception(s.failure);
            }
            if (s.pieces.empty()) {
                if (!s.read) {
                    return false;
                }
                lock.unlock();
                s.file.close();
                if (!s.file) {
                    throw std::runtime_error("Cannot write " + s.output.string());
                }
                s.done = true;
                break;
            }
            s.current = std::move(s.pieces.front());
            s.pieces.pop_front();
            s.queued -= s.current->rgba.size();
            s.row = 0;
            s.changed.notify_all();
        }
        if (spent >= texels) {
            return false;
        }
        // Whole rows of blocks, about the texels left.
        const State::Piece& piece = *s.current;
        const std::uint64_t left = texels - spent;
        const auto fit = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(piece.h, left / piece.w / side * side));
        const std::uint32_t rows = std::min(piece.h - s.row, std::max(fit, side));
        const std::span<const std::uint8_t> rgba(
            piece.rgba.data() + std::size_t{s.row} * piece.w * 4, std::size_t{rows} * piece.w * 4);
        blocks.resize(ImageBytes(s.layout.format, piece.w, rows));
        {
            ERNIE_ZONE("Encode band");
            EncodeRgba8(s.layout.format, rgba, piece.w, rows, blocks.data(), s.bc7,
                        s.options.encoder);
        }
        // Each row of blocks in its place in the mip.
        const std::size_t mipAcross = (MipSide(s.layout.width, piece.mip) + side - 1) / side;
        const std::size_t rowBytes = (piece.w + side - 1) / side * s.info.blockBytes;
        for (std::uint32_t y = 0; y < (rows + side - 1) / side; ++y) {
            const std::size_t at =
                kDdsHeaderBytes + s.layout.offsets[piece.mip] +
                ((std::size_t{piece.y + s.row} / side + y) * mipAcross + piece.x / side) *
                    s.info.blockBytes;
            s.file.seekp(static_cast<std::streamoff>(at));
            s.file.write(reinterpret_cast<const char*>(blocks.data() + y * rowBytes),
                         static_cast<std::streamsize>(rowBytes));
        }
        if (!s.file) {
            throw std::runtime_error("Cannot write " + s.output.string());
        }
        spent += std::uint64_t{piece.w} * rows;
        s.encoded += std::uint64_t{piece.w} * rows;
        s.row += rows;
        if (s.row >= piece.h) {
            s.current.reset();
        }
    }
    return true;
}

TextureData BakeMosaic(std::span<const fs::path> tiles, std::uint32_t columns,
                       const fs::path& output, const BakeOptions& options) {
    ERNIE_ZONE("Bake mosaic");
    return RasterBake(std::make_unique<MosaicRaster>(
                          std::vector<fs::path>(tiles.begin(), tiles.end()), columns),
                      output, options)
        .Finish();
}

// ----------------------------------------------------------------------------------------------
// Cache

namespace {

/// Bumped when what the cache stores, or how it keys it, changes.
constexpr std::string_view kCacheFormat = "ernie-texture-1";

}  // namespace

std::optional<fs::path> TextureCache::DefaultDirectory() {
    return CacheDirectory("ERNIE_TEXTURE_CACHE", "textures");
}

TextureCache::TextureCache(std::optional<fs::path> directory, std::uintmax_t capacity)
    : directory_(std::move(directory)), capacity_(capacity) {}

fs::path TextureCache::EntryPath(std::string_view key) const {
    CacheKey hash;
    hash.Add(kCacheFormat);
    hash.Add(key);
    return *directory_ / (hash.Hex() + ".dds");
}

std::optional<TextureData> TextureCache::Read(std::string_view key) const {
    if (!directory_) {
        return std::nullopt;
    }
    ERNIE_ZONE("Texture cache read");
    const fs::path path = EntryPath(key);
    std::error_code error;
    if (!fs::exists(path, error)) {
        return std::nullopt;
    }
    try {
        TextureData texture = LoadTextureFile(path);
        // Its modification time is its last use, for `Prune`.
        fs::last_write_time(path, fs::file_time_type::clock::now(), error);
        return texture;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

void TextureCache::Write(std::string_view key, const TextureData& texture) const {
    const std::optional<fs::path> partial = PartialPath(key);
    if (!partial) {
        return;
    }
    ERNIE_ZONE("Texture cache write");
    try {
        WriteDdsFile(*partial, texture);
    } catch (const std::exception&) {
        std::error_code error;
        fs::remove(*partial, error);
        return;
    }
    Publish(*partial, key);
}

std::optional<fs::path> TextureCache::PartialPath(std::string_view key) const {
    if (!directory_) {
        return std::nullopt;
    }
    std::error_code error;
    fs::create_directories(*directory_, error);
    // Written beside its entry, then renamed over it, so that no reader (in this process or
    // another) sees a part of it.
    fs::path partial = EntryPath(key);
    partial += ".part" + std::to_string(std::random_device{}());
    return partial;
}

bool TextureCache::Publish(const fs::path& partial, std::string_view key) const {
    std::error_code error;
    if (!directory_) {
        fs::remove(partial, error);
        return false;
    }
    const fs::path path = EntryPath(key);
    fs::rename(partial, path, error);
    if (error) {
        fs::remove(partial, error);
        return false;
    }
    Prune(path);
    return true;
}

void TextureCache::Remove(std::string_view key) const {
    if (directory_) {
        std::error_code error;
        fs::remove(EntryPath(key), error);
    }
}

void TextureCache::Prune(const fs::path& kept) const {
    ERNIE_ZONE("Texture cache prune");
    struct Entry {
        fs::path path;
        fs::file_time_type used;
        std::uintmax_t bytes;
    };
    std::vector<Entry> entries;
    std::uintmax_t total = 0;
    std::error_code error;
    const fs::file_time_type now = fs::file_time_type::clock::now();
    for (const fs::directory_entry& file : fs::directory_iterator(*directory_, error)) {
        const fs::file_time_type used = file.last_write_time(error);
        if (error) {
            continue;
        }
        // Parts a day old are of writes that never finished.
        if (file.path().extension().string().starts_with(".part")) {
            if (now - used > std::chrono::hours(24)) {
                fs::remove(file.path(), error);
            }
            continue;
        }
        if (file.path().extension() != ".dds") {
            continue;
        }
        Entry entry{file.path(), used, file.file_size(error)};
        if (error) {
            continue;
        }
        total += entry.bytes;
        entries.push_back(std::move(entry));
    }
    if (total <= capacity_) {
        return;
    }
    std::sort(entries.begin(), entries.end(),
              [](const Entry& a, const Entry& b) { return a.used < b.used; });
    for (const Entry& entry : entries) {
        if (total <= capacity_) {
            break;
        }
        if (entry.path != kept && fs::remove(entry.path, error)) {
            total -= entry.bytes;
        }
    }
}

}  // namespace ernie
