// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/texture_file.h"

#include "ernie/util/profile.h"

#include <basisu_transcoder.h>
#include <bc7decomp.h>
#include <rgbcx.h>
#include <zstd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

namespace ernie {

namespace {

// Bounds that keep every size below in range, far past what devices
// sample whole (virtual textures page larger ones: global mosaics run
// to 86400 texels across).
constexpr std::uint32_t kMaxSide = 1u << 20;
constexpr std::uint32_t kMaxLayers = 2048;

std::uint32_t U32(std::span<const std::byte> file, std::size_t at) {
    if (at + 4 > file.size()) {
        throw std::runtime_error("Texture file is truncated");
    }
    std::uint32_t value = 0;
    std::memcpy(&value, file.data() + at, 4);
    return value;  // Little-endian, as both formats and hosts are.
}

std::uint64_t U64(std::span<const std::byte> file, std::size_t at) {
    return U32(file, at) |
           (static_cast<std::uint64_t>(U32(file, at + 4)) << 32);
}

constexpr std::uint32_t FourCc(const char (&code)[5]) {
    return static_cast<std::uint32_t>(code[0]) |
           (static_cast<std::uint32_t>(code[1]) << 8) |
           (static_cast<std::uint32_t>(code[2]) << 16) |
           (static_cast<std::uint32_t>(code[3]) << 24);
}

std::string LowerExtension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(),
                   extension.begin(), [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return extension;
}

std::uint32_t MipCount(std::uint32_t width, std::uint32_t height) {
    std::uint32_t count = 1;
    for (std::uint32_t side = std::max(width, height); side > 1;
         side >>= 1) {
        ++count;
    }
    return count;
}

/// A texture of `format`, `width` x `height` with `layers` layers and
/// `mips` mips, its bytes still to fill (with `allocate`, else left
/// to a mapped file): checks the shape and lays out `offsets` layer
/// by layer.
TextureData Shape(TextureFormat format, std::uint32_t width,
                  std::uint32_t height, std::uint32_t layers,
                  std::uint32_t mips, bool allocate = true) {
    if (width == 0 || height == 0 || width > kMaxSide ||
        height > kMaxSide) {
        throw std::runtime_error("Texture file has a size of " +
                                 std::to_string(width) + " x " +
                                 std::to_string(height));
    }
    if (layers == 0 || layers > kMaxLayers) {
        throw std::runtime_error("Texture file has " +
                                 std::to_string(layers) + " layers");
    }
    if (mips == 0 || mips > MipCount(width, height)) {
        throw std::runtime_error("Texture file has " +
                                 std::to_string(mips) + " mips");
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
            total += ImageBytes(format, std::max(1u, width >> mip),
                                std::max(1u, height >> mip));
        }
    }
    if (allocate) {
        data.bytes.resize(total);
    }
    return data;
}

/// Bytes of every layer's mips of `texture`, as `Shape` laid them
/// out.
std::size_t ShapeBytes(const TextureData& texture) {
    const std::uint32_t last = texture.mips - 1;
    return texture.offsets.back() +
           ImageBytes(texture.format,
                      std::max(1u, texture.width >> last),
                      std::max(1u, texture.height >> last));
}

std::optional<TextureFormat> FromDxgi(std::uint32_t dxgi) {
    switch (dxgi) {
    case 2:
        return TextureFormat::Rgba32F;
    case 10:
        return TextureFormat::Rgba16F;
    case 28:
        return TextureFormat::Rgba8;
    case 29:
        return TextureFormat::Rgba8Srgb;
    case 49:
        return TextureFormat::Rg8;
    case 61:
        return TextureFormat::R8;
    case 71:
        return TextureFormat::Bc1;
    case 72:
        return TextureFormat::Bc1Srgb;
    case 74:
        return TextureFormat::Bc2;
    case 75:
        return TextureFormat::Bc2Srgb;
    case 77:
        return TextureFormat::Bc3;
    case 78:
        return TextureFormat::Bc3Srgb;
    case 80:
        return TextureFormat::Bc4;
    case 81:
        return TextureFormat::Bc4Snorm;
    case 83:
        return TextureFormat::Bc5;
    case 84:
        return TextureFormat::Bc5Snorm;
    case 87:
        return TextureFormat::Bgra8;
    case 91:
        return TextureFormat::Bgra8Srgb;
    case 95:
        return TextureFormat::Bc6hUfloat;
    case 96:
        return TextureFormat::Bc6hSfloat;
    case 98:
        return TextureFormat::Bc7;
    case 99:
        return TextureFormat::Bc7Srgb;
    default:
        return std::nullopt;
    }
}

std::optional<TextureFormat> FromVkFormat(std::uint32_t vk) {
    switch (vk) {
    case 9:
        return TextureFormat::R8;  // R8_UNORM
    case 16:
        return TextureFormat::Rg8;  // R8G8_UNORM
    case 37:
        return TextureFormat::Rgba8;  // R8G8B8A8_UNORM
    case 43:
        return TextureFormat::Rgba8Srgb;  // R8G8B8A8_SRGB
    case 44:
        return TextureFormat::Bgra8;  // B8G8R8A8_UNORM
    case 50:
        return TextureFormat::Bgra8Srgb;  // B8G8R8A8_SRGB
    case 97:
        return TextureFormat::Rgba16F;  // R16G16B16A16_SFLOAT
    case 109:
        return TextureFormat::Rgba32F;  // R32G32B32A32_SFLOAT
    // BC1_RGB read as BC1_RGBA: they differ only in the alpha of
    // blocks in three-colour mode.
    case 131:
    case 133:
        return TextureFormat::Bc1;
    case 132:
    case 134:
        return TextureFormat::Bc1Srgb;
    case 135:
        return TextureFormat::Bc2;
    case 136:
        return TextureFormat::Bc2Srgb;
    case 137:
        return TextureFormat::Bc3;
    case 138:
        return TextureFormat::Bc3Srgb;
    case 139:
        return TextureFormat::Bc4;
    case 140:
        return TextureFormat::Bc4Snorm;
    case 141:
        return TextureFormat::Bc5;
    case 142:
        return TextureFormat::Bc5Snorm;
    case 143:
        return TextureFormat::Bc6hUfloat;
    case 144:
        return TextureFormat::Bc6hSfloat;
    case 145:
        return TextureFormat::Bc7;
    case 146:
        return TextureFormat::Bc7Srgb;
    default:
        return std::nullopt;
    }
}

/// The format of a DDS file without a DX10 header, from its pixel
/// format (`DDS_PIXELFORMAT` at `at`).
TextureFormat LegacyDdsFormat(std::span<const std::byte> file,
                              std::size_t at) {
    constexpr std::uint32_t kFourCcFlag = 0x4;
    constexpr std::uint32_t kRgbFlag = 0x40;
    const std::uint32_t flags = U32(file, at + 4);
    if ((flags & kFourCcFlag) != 0) {
        const std::uint32_t code = U32(file, at + 8);
        switch (code) {
        case FourCc("DXT1"):
            return TextureFormat::Bc1;
        case FourCc("DXT2"):
        case FourCc("DXT3"):
            return TextureFormat::Bc2;
        case FourCc("DXT4"):
        case FourCc("DXT5"):
            return TextureFormat::Bc3;
        case FourCc("ATI1"):
        case FourCc("BC4U"):
            return TextureFormat::Bc4;
        case FourCc("BC4S"):
            return TextureFormat::Bc4Snorm;
        case FourCc("ATI2"):
        case FourCc("BC5U"):
            return TextureFormat::Bc5;
        case FourCc("BC5S"):
            return TextureFormat::Bc5Snorm;
        case 113:
            return TextureFormat::Rgba16F;  // A16B16G16R16F
        case 116:
            return TextureFormat::Rgba32F;  // A32B32G32R32F
        default:
            throw std::runtime_error(
                "DDS file has an unsupported FourCC code " +
                std::to_string(code));
        }
    }
    const std::uint32_t bits = U32(file, at + 12);
    const std::uint32_t red = U32(file, at + 16);
    const std::uint32_t green = U32(file, at + 20);
    const std::uint32_t blue = U32(file, at + 24);
    if ((flags & kRgbFlag) != 0 && bits == 32 && green == 0xff00u) {
        if (red == 0xffu && blue == 0xff0000u) {
            return TextureFormat::Rgba8;
        }
        if (red == 0xff0000u && blue == 0xffu) {
            return TextureFormat::Bgra8;
        }
    }
    throw std::runtime_error(
        "DDS file has an unsupported uncompressed layout");
}

/// A KTX2 file of Basis Universal's ETC1S or UASTC formats transcoded
/// to BC blocks: two-channel data (normal maps' x and y) to BC5, HDR
/// to BC6H, the rest to BC7, sRGB as the file's transfer function
/// says.
TextureData TranscodeBasis(std::span<const std::byte> file) {
    ERNIE_ZONE("Transcode Basis");
    static std::once_flag initialised;
    std::call_once(initialised, basist::basisu_transcoder_init);
    basist::ktx2_transcoder ktx2;
    if (file.size() > UINT32_MAX ||
        !ktx2.init(file.data(),
                   static_cast<std::uint32_t>(file.size())) ||
        !ktx2.start_transcoding()) {
        throw std::runtime_error("KTX2 file holds no Basis Universal "
                                 "data it can transcode");
    }
    TextureFormat format = TextureFormat::Bc7;
    basist::transcoder_texture_format target =
        basist::transcoder_texture_format::cTFBC7_RGBA;
    // BC5 takes x from red and y from alpha unless told otherwise.
    int first = -1;
    int second = -1;
    const auto channel0 = ktx2.get_dfd_channel_id0();
    const bool twoChannels =
        ktx2.is_etc1s()
            ? channel0 == basist::KTX2_DF_CHANNEL_ETC1S_RRR &&
                  ktx2.get_dfd_channel_id1() ==
                      basist::KTX2_DF_CHANNEL_ETC1S_GGG
            : channel0 == basist::KTX2_DF_CHANNEL_UASTC_RRRG ||
                  channel0 == basist::KTX2_DF_CHANNEL_UASTC_RG;
    if (ktx2.is_hdr()) {
        format = TextureFormat::Bc6hUfloat;
        target = basist::transcoder_texture_format::cTFBC6H;
    } else if (twoChannels) {
        format = TextureFormat::Bc5;
        target = basist::transcoder_texture_format::cTFBC5_RG;
        if (!ktx2.is_etc1s() &&
            channel0 == basist::KTX2_DF_CHANNEL_UASTC_RG) {
            first = 0;
            second = 1;
        }
    } else if (ktx2.get_dfd_transfer_func() ==
               basist::KTX2_KHR_DF_TRANSFER_SRGB) {
        format = TextureFormat::Bc7Srgb;
    }
    const std::uint32_t faces = ktx2.get_faces();
    const std::uint32_t layers = std::max(1u, ktx2.get_layers());
    const std::uint32_t levels = std::max(1u, ktx2.get_levels());
    if (layers > kMaxLayers / std::max(1u, faces)) {
        throw std::runtime_error("KTX2 file has too many layers");
    }
    TextureData texture =
        Shape(format, ktx2.get_width(), ktx2.get_height(),
              layers * faces, levels);
    texture.cube = faces == 6;
    for (std::uint32_t level = 0; level < levels; ++level) {
        const std::uint32_t width =
            std::max(1u, texture.width >> level);
        const std::uint32_t height =
            std::max(1u, texture.height >> level);
        const auto blocks = static_cast<std::uint32_t>(
            ((width + 3) / 4) * ((height + 3) / 4));
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            for (std::uint32_t face = 0; face < faces; ++face) {
                const std::size_t slot =
                    (std::size_t{layer} * faces + face) * levels +
                    level;
                if (!ktx2.transcode_image_level(
                        level, layer, face,
                        texture.bytes.data() + texture.offsets[slot],
                        blocks, target, 0, 0, 0, first, second)) {
                    throw std::runtime_error("KTX2 file's level " +
                                             std::to_string(level) +
                                             " does not transcode");
                }
            }
        }
    }
    return texture;
}

}  // namespace

TextureFormatInfo InfoOf(TextureFormat format) {
    using F = TextureFormat;
    switch (format) {
    case F::R8:
        return {1, 1, 1, false};
    case F::Rg8:
        return {1, 2, 2, false};
    case F::Rgba8:
    case F::Bgra8:
        return {1, 4, 4, false};
    case F::Rgba8Srgb:
    case F::Bgra8Srgb:
        return {1, 4, 4, true};
    case F::Rgba16F:
        return {1, 8, 4, false};
    case F::Rgba32F:
        return {1, 16, 4, false};
    case F::Bc1:
        return {4, 8, 4, false};
    case F::Bc1Srgb:
        return {4, 8, 4, true};
    case F::Bc2:
    case F::Bc3:
    case F::Bc7:
        return {4, 16, 4, false};
    case F::Bc2Srgb:
    case F::Bc3Srgb:
    case F::Bc7Srgb:
        return {4, 16, 4, true};
    case F::Bc4:
    case F::Bc4Snorm:
        return {4, 8, 1, false};
    case F::Bc5:
    case F::Bc5Snorm:
        return {4, 16, 2, false};
    case F::Bc6hUfloat:
    case F::Bc6hSfloat:
        return {4, 16, 3, false};
    }
    throw std::invalid_argument("Unknown texture format");
}

TextureFormat WithSrgb(TextureFormat format, bool srgb) {
    using F = TextureFormat;
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

std::size_t ImageBytes(TextureFormat format, std::uint32_t width,
                       std::uint32_t height) {
    const TextureFormatInfo info = InfoOf(format);
    const std::size_t across =
        (width + info.blockSize - 1) / info.blockSize;
    const std::size_t down =
        (height + info.blockSize - 1) / info.blockSize;
    return across * down * info.blockBytes;
}

std::span<const std::byte> TextureData::Mip(std::uint32_t layer,
                                            std::uint32_t mip) const {
    if (layer >= layers || mip >= mips) {
        throw std::out_of_range("Texture has no layer " +
                                std::to_string(layer) + " mip " +
                                std::to_string(mip));
    }
    const std::size_t at = offsets[std::size_t{layer} * mips + mip];
    return Data().subspan(
        at, ImageBytes(format, std::max(1u, width >> mip),
                       std::max(1u, height >> mip)));
}

bool IsTextureFile(const std::filesystem::path& path) {
    const std::string extension = LowerExtension(path);
    return extension == ".dds" || extension == ".ktx2";
}

namespace {

/// A DDS file's texture: copied into `bytes` with `copy`, else read
/// from `file` in place (`mapped`, the offsets into it).
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
    const std::uint32_t mips =
        (flags & kMipCountFlag) != 0 && mipCount > 0 ? mipCount : 1;

    std::size_t data = kHeader + 124;
    TextureFormat format;
    std::uint32_t layers = 1;
    bool cube = (caps2 & kCubeMap) != 0;
    if (U32(file, kPixelFormat + 8) == FourCc("DX10")) {
        // DDS_HEADER_DXT10: format, dimension, misc flags, array
        // size.
        const std::uint32_t dxgi = U32(file, data);
        const std::uint32_t dimension = U32(file, data + 4);
        constexpr std::uint32_t kTexture2d = 3;
        constexpr std::uint32_t kCubeFlag = 0x4;
        if (dimension != kTexture2d) {
            throw std::runtime_error(
                "DDS file is not a 2D texture (dimension " +
                std::to_string(dimension) + ")");
        }
        const std::optional<TextureFormat> known = FromDxgi(dxgi);
        if (!known) {
            throw std::runtime_error(
                "DDS file has an unsupported DXGI format " +
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
            throw std::runtime_error(
                "DDS file has too many cube maps");
        }
        layers *= 6;
    }
    TextureData texture =
        Shape(format, width, height, layers, mips, copy);
    texture.cube = cube;
    // Each layer's mips follow the one before, as `TextureData` lays
    // them out.
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

/// A KTX2 file's texture: its levels as `ParseDds` reads, in place
/// without `copy` unless they are supercompressed (decoded into
/// `bytes`).
TextureData ParseKtx2(std::span<const std::byte> file, bool copy) {
    constexpr std::array<std::uint8_t, 12> kIdentifier{
        0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32,
        0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
    if (file.size() < 80 ||
        std::memcmp(file.data(), kIdentifier.data(), 12) != 0) {
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
        throw std::runtime_error("KTX2 file has " +
                                 std::to_string(faces) + " faces");
    }
    // Basis Universal's formats (ETC1S, often
    // BasisLZ-supercompressed, and UASTC) have no VkFormat.
    if (vkFormat == 0) {
        return TranscodeBasis(file);
    }
    constexpr std::uint32_t kZstd = 2;
    if (supercompression != 0 && supercompression != kZstd) {
        throw std::runtime_error(
            "KTX2 file uses supercompression scheme " +
            std::to_string(supercompression) +
            ", which is not read (only zstd)");
    }
    const std::optional<TextureFormat> format =
        FromVkFormat(vkFormat);
    if (!format) {
        throw std::runtime_error(
            "KTX2 file has an unsupported VkFormat " +
            std::to_string(vkFormat));
    }
    const std::uint32_t arrays = std::max(1u, layerCount);
    if (arrays > kMaxLayers / faces) {
        throw std::runtime_error("KTX2 file has too many layers");
    }
    // Stored levels are read in place: each image is contiguous in
    // the file, the offsets pointing at it.
    const bool inPlace = !copy && supercompression == 0;
    TextureData texture = Shape(*format, width, height,
                                arrays * faces, levels, !inPlace);
    texture.cube = faces == 6;
    if (inPlace) {
        texture.mapped = file;
    }
    // The level index follows the 80-byte header: each level's
    // offset, stored length and uncompressed length, level 0 the
    // largest. A level holds every layer's faces in turn.
    std::vector<std::byte> decompressed;
    for (std::uint32_t level = 0; level < levels; ++level) {
        const std::size_t entry = 80 + std::size_t{level} * 24;
        const std::uint64_t offset = U64(file, entry);
        const std::uint64_t length = U64(file, entry + 8);
        const std::uint64_t uncompressed = U64(file, entry + 16);
        const std::size_t image =
            ImageBytes(*format, std::max(1u, width >> level),
                       std::max(1u, height >> level));
        const std::size_t expected = image * texture.layers;
        if (offset > file.size() || file.size() - offset < length ||
            (supercompression == 0 ? length : uncompressed) !=
                expected) {
            throw std::runtime_error(
                "KTX2 file has a malformed level " +
                std::to_string(level));
        }
        if (inPlace) {
            for (std::uint32_t layer = 0; layer < texture.layers;
                 ++layer) {
                texture.offsets[std::size_t{layer} * levels + level] =
                    static_cast<std::size_t>(offset) +
                    std::size_t{layer} * image;
            }
            continue;
        }
        const std::byte* source = file.data() + offset;
        if (supercompression == kZstd) {
            decompressed.resize(expected);
            const std::size_t size =
                ZSTD_decompress(decompressed.data(), expected, source,
                                static_cast<std::size_t>(length));
            if (ZSTD_isError(size) != 0 || size != expected) {
                throw std::runtime_error("KTX2 file's level " +
                                         std::to_string(level) +
                                         " does not decompress");
            }
            source = decompressed.data();
        }
        for (std::uint32_t layer = 0; layer < texture.layers;
             ++layer) {
            std::memcpy(
                texture.bytes.data() +
                    texture
                        .offsets[std::size_t{layer} * levels + level],
                source + std::size_t{layer} * image, image);
        }
    }
    return texture;
}

}  // namespace

TextureData ReadDds(std::span<const std::byte> file) {
    return ParseDds(file, true);
}

TextureData ReadKtx2(std::span<const std::byte> file) {
    return ParseKtx2(file, true);
}

TextureData LoadTextureFile(const std::filesystem::path& path) {
    ERNIE_ZONE("Load texture file");
    const std::string extension = LowerExtension(path);
    if (extension != ".dds" && extension != ".ktx2") {
        throw std::runtime_error(
            path.string() + " is neither a .dds nor a .ktx2 file");
    }
    // Mapped, so texels are read from disk only as they are used: a
    // virtual texture's tiles in view, not the whole file.
    auto mapping = std::make_shared<const MappedFile>(path);
    try {
        TextureData texture =
            extension == ".dds" ? ParseDds(mapping->Bytes(), false)
                                : ParseKtx2(mapping->Bytes(), false);
        if (!texture.mapped.empty()) {
            texture.mapping = std::move(mapping);
        }
        return texture;
    } catch (const std::runtime_error& error) {
        throw std::runtime_error(path.string() + ": " + error.what());
    }
}

TextureData DecodeBlocks(const TextureData& texture) {
    const TextureFormatInfo info = InfoOf(texture.format);
    if (!info.compressed()) {
        return texture;
    }
    ERNIE_ZONE("Decode BC blocks");
    TextureFormat decoded = TextureFormat::Rgba8;
    switch (texture.format) {
    case TextureFormat::Bc1:
    case TextureFormat::Bc3:
    case TextureFormat::Bc7:
        break;
    case TextureFormat::Bc1Srgb:
    case TextureFormat::Bc3Srgb:
    case TextureFormat::Bc7Srgb:
        decoded = TextureFormat::Rgba8Srgb;
        break;
    case TextureFormat::Bc4:
        decoded = TextureFormat::R8;
        break;
    case TextureFormat::Bc5:
        decoded = TextureFormat::Rg8;
        break;
    default:
        throw std::invalid_argument(
            "BC2, BC6H and signed BC4 and BC5 are not decoded");
    }
    const std::uint32_t channels = InfoOf(decoded).channels;

    TextureData out;
    out.format = decoded;
    out.width = texture.width;
    out.height = texture.height;
    out.layers = texture.layers;
    out.mips = texture.mips;
    out.cube = texture.cube;
    for (std::uint32_t layer = 0; layer < texture.layers; ++layer) {
        for (std::uint32_t mip = 0; mip < texture.mips; ++mip) {
            const std::uint32_t w =
                std::max(1u, texture.width >> mip);
            const std::uint32_t h =
                std::max(1u, texture.height >> mip);
            const std::span<const std::byte> blocks =
                texture.Mip(layer, mip);
            out.offsets.push_back(out.bytes.size());
            const std::size_t start = out.bytes.size();
            out.bytes.resize(start + ImageBytes(decoded, w, h));
            std::byte* texels = out.bytes.data() + start;
            const std::uint32_t across = (w + 3) / 4;
            for (std::uint32_t by = 0; by < (h + 3) / 4; ++by) {
                for (std::uint32_t bx = 0; bx < across; ++bx) {
                    const std::byte* block =
                        blocks.data() +
                        (std::size_t{by} * across + bx) *
                            info.blockBytes;
                    // The block's 4 x 4 texels as RGBA8.
                    std::array<std::uint8_t, 64> rgba{};
                    switch (texture.format) {
                    case TextureFormat::Bc1:
                    case TextureFormat::Bc1Srgb:
                        rgbcx::unpack_bc1(block, rgba.data());
                        break;
                    case TextureFormat::Bc3:
                    case TextureFormat::Bc3Srgb:
                        rgbcx::unpack_bc3(block, rgba.data());
                        break;
                    case TextureFormat::Bc4:
                        rgbcx::unpack_bc4(block, rgba.data());
                        break;
                    case TextureFormat::Bc5:
                        rgbcx::unpack_bc5(block, rgba.data());
                        break;
                    default:
                        bc7decomp::unpack_bc7(
                            block,
                            reinterpret_cast<bc7decomp::color_rgba*>(
                                rgba.data()));
                        break;
                    }
                    for (std::uint32_t i = 0; i < 16; ++i) {
                        const std::uint32_t x = bx * 4 + i % 4;
                        const std::uint32_t y = by * 4 + i / 4;
                        if (x >= w || y >= h) {
                            continue;
                        }
                        std::memcpy(texels +
                                        (std::size_t{y} * w + x) *
                                            channels,
                                    rgba.data() + std::size_t{i} * 4,
                                    channels);
                    }
                }
            }
        }
    }
    return out;
}

}  // namespace ernie
