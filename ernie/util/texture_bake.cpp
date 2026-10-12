// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/texture_bake.h"

#include "ernie/util/profile.h"

#include <bc7enc.h>
#include <rgbcx.h>
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ernie {

namespace {

/// The DXGI format DDS files name `format` by.
std::uint32_t DxgiOf(TextureFormat format) {
    using F = TextureFormat;
    switch (format) {
    case F::Rgba32F:
        return 2;
    case F::Rgba16F:
        return 10;
    case F::Rgba8:
        return 28;
    case F::Rgba8Srgb:
        return 29;
    case F::Rg8:
        return 49;
    case F::R8:
        return 61;
    case F::Bc1:
        return 71;
    case F::Bc1Srgb:
        return 72;
    case F::Bc2:
        return 74;
    case F::Bc2Srgb:
        return 75;
    case F::Bc3:
        return 77;
    case F::Bc3Srgb:
        return 78;
    case F::Bc4:
        return 80;
    case F::Bc4Snorm:
        return 81;
    case F::Bc5:
        return 83;
    case F::Bc5Snorm:
        return 84;
    case F::Bgra8:
        return 87;
    case F::Bgra8Srgb:
        return 91;
    case F::Bc6hUfloat:
        return 95;
    case F::Bc6hSfloat:
        return 96;
    case F::Bc7:
        return 98;
    case F::Bc7Srgb:
        return 99;
    }
    throw std::runtime_error(
        "No DXGI format for this texture format");
}

/// Encodes one 4 x 4 block of RGBA8 `pixels` into `block`.
void EncodeBlock(TextureFormat format, const std::uint8_t* pixels,
                 std::byte* block,
                 const bc7enc_compress_block_params& bc7) {
    // rgbcx's effort for BC1 and BC3, of 0 to 18.
    constexpr std::uint32_t kLevel = 10;
    using F = TextureFormat;
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
        throw std::invalid_argument(
            "Textures are not encoded to this format");
    }
}

/// `mip`, stored as 8-bit RGBA in `bytes` (`width` x `height`),
/// encoded into BC `format` blocks at `out`, rows of blocks shared
/// among threads.
/// Runs `body(row)` for every row below `rows`, the rows shared among
/// the hardware's threads.
template <typename Body>
void ForEachRow(std::uint32_t rows, const Body& body) {
    std::atomic<std::uint32_t> next{0};
    const auto work = [&] {
        for (std::uint32_t row = next++; row < rows; row = next++) {
            body(row);
        }
    };
    const unsigned threads = std::max(
        1u, std::min(std::thread::hardware_concurrency(), rows));
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < threads; ++t) {
        pool.emplace_back(work);
    }
    work();
    for (std::thread& thread : pool) {
        thread.join();
    }
}

/// `rgba` (`width` x `height` RGBA8) encoded into BC `format` blocks
/// at `out`, rows of blocks on every thread.
void EncodeBlocks(TextureFormat format,
                  std::span<const std::uint8_t> rgba,
                  std::uint32_t width, std::uint32_t height,
                  std::byte* out,
                  const bc7enc_compress_block_params& bc7) {
    ERNIE_ZONE("Encode blocks on the CPU");
    const TextureFormatInfo info = InfoOf(format);
    const std::uint32_t across = (width + 3) / 4;
    const std::uint32_t down = (height + 3) / 4;
    ForEachRow(down, [&](std::uint32_t row) {
        std::array<std::uint8_t, 64> pixels{};
        for (std::uint32_t column = 0; column < across; ++column) {
            // Texels past the edge repeat the last row and column.
            for (std::uint32_t y = 0; y < 4; ++y) {
                const std::uint32_t sy =
                    std::min(row * 4 + y, height - 1);
                for (std::uint32_t x = 0; x < 4; ++x) {
                    const std::uint32_t sx =
                        std::min(column * 4 + x, width - 1);
                    std::memcpy(
                        pixels.data() + 4 * (y * 4 + x),
                        rgba.data() +
                            4 * (std::size_t{sy} * width + sx),
                        4);
                }
            }
            EncodeBlock(format, pixels.data(),
                        out + (std::size_t{row} * across + column) *
                                  info.blockBytes,
                        bc7);
        }
    });
}

/// A texture of `format`, `width` x `height` with `mips` mips, its
/// blocks still to encode (with `allocate`; else only laid out).
/// @throws std::invalid_argument for a format it does not write.
TextureData Allocate(TextureFormat format, std::uint32_t width,
                     std::uint32_t height, std::uint32_t mips,
                     bool allocate = true) {
    using F = TextureFormat;
    constexpr std::array<F, 10> kWritten{
        F::Rgba8,   F::Rgba8Srgb, F::Bc1, F::Bc1Srgb, F::Bc3,
        F::Bc3Srgb, F::Bc4,       F::Bc5, F::Bc7,     F::Bc7Srgb};
    if (std::find(kWritten.begin(), kWritten.end(), format) ==
        kWritten.end()) {
        throw std::invalid_argument(
            "Textures are not encoded to this format");
    }
    TextureData texture;
    texture.format = format;
    texture.width = width;
    texture.height = height;
    texture.mips = mips;
    std::size_t total = 0;
    for (std::uint32_t mip = 0; mip < mips; ++mip) {
        texture.offsets.push_back(total);
        total += ImageBytes(format, std::max(1u, width >> mip),
                            std::max(1u, height >> mip));
    }
    if (allocate) {
        texture.bytes.resize(total);
    }
    return texture;
}

/// BC7's settings for `format` at effort `quality`, the encoders
/// initialised.
bc7enc_compress_block_params Bc7Params(TextureFormat format,
                                       int quality) {
    static std::once_flag initialised;
    std::call_once(initialised, [] {
        bc7enc_compress_block_init();
        rgbcx::init();
    });
    bc7enc_compress_block_params bc7{};
    bc7enc_compress_block_params_init(&bc7);
    if (format == TextureFormat::Bc7) {
        bc7enc_compress_block_params_init_linear_weights(&bc7);
    }
    bc7.m_uber_level = static_cast<std::uint32_t>(
        std::clamp(quality, 0, BC7ENC_MAX_UBER_LEVEL));
    return bc7;
}

/// Encodes mip `mip` of `texture` from its RGBA8 texels `rgba`.
void EncodeMip(TextureData& texture, std::uint32_t mip,
               std::span<const std::uint8_t> rgba,
               const bc7enc_compress_block_params& bc7,
               BlockEncoder* encoder) {
    const std::uint32_t width = std::max(1u, texture.width >> mip);
    const std::uint32_t height = std::max(1u, texture.height >> mip);
    std::byte* out = texture.bytes.data() + texture.offsets[mip];
    if (!InfoOf(texture.format).compressed()) {
        std::memcpy(out, rgba.data(), rgba.size());
        return;
    }
    if (encoder != nullptr && encoder->Encodes(texture.format)) {
        encoder->Encode(rgba, width, height, texture.format, out);
        return;
    }
    EncodeBlocks(texture.format, rgba, width, height, out, bc7);
}

/// The next mip of RGBA8 `rgba` (`width` x `height`): each texel the
/// average of the 2 x 2 above it, clamped at odd edges, as
/// `MipChain`'s `MipFilter::Average`; with `srgb` its rgb averaged as
/// linear values.
std::vector<std::uint8_t>
HalveRgba8(std::span<const std::uint8_t> rgba, std::uint32_t width,
           std::uint32_t height, bool srgb) {
    ERNIE_ZONE("Halve RGBA8");
    static const std::array<float, 256> kLinear = [] {
        std::array<float, 256> table{};
        for (int i = 0; i < 256; ++i) {
            table[static_cast<std::size_t>(i)] =
                SrgbToLinear(static_cast<float>(i) / 255.0f);
        }
        return table;
    }();
    const std::uint32_t w = std::max(1u, width / 2);
    const std::uint32_t h = std::max(1u, height / 2);
    std::vector<std::uint8_t> half(std::size_t{w} * h * 4);
    ForEachRow(h, [&](std::uint32_t y) {
        for (std::uint32_t x = 0; x < w; ++x) {
            for (std::uint32_t c = 0; c < 4; ++c) {
                float sum = 0.0f;
                for (std::uint32_t dy = 0; dy < 2; ++dy) {
                    for (std::uint32_t dx = 0; dx < 2; ++dx) {
                        const std::uint32_t sx =
                            std::min(2 * x + dx, width - 1);
                        const std::uint32_t sy =
                            std::min(2 * y + dy, height - 1);
                        const std::uint8_t value =
                            rgba[(std::size_t{sy} * width + sx) * 4 +
                                 c];
                        sum +=
                            srgb && c < 3
                                ? kLinear[value]
                                : static_cast<float>(value) / 255.0f;
                    }
                }
                float mean = sum / 4.0f;
                if (srgb && c < 3) {
                    mean = LinearToSrgb(mean);
                }
                half[(std::size_t{y} * w + x) * 4 + c] =
                    static_cast<std::uint8_t>(std::lround(
                        std::clamp(mean, 0.0f, 1.0f) * 255.0f));
            }
        }
    });
    return half;
}

}  // namespace

std::optional<glm::uvec2>
Rgba8ImageSize(const std::filesystem::path& file) {
    std::string extension = file.extension().string();
    std::transform(extension.begin(), extension.end(),
                   extension.begin(), [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    if (extension != ".png" && extension != ".jpg" &&
        extension != ".jpeg") {
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
    return glm::uvec2(static_cast<std::uint32_t>(width),
                      static_cast<std::uint32_t>(height));
}

std::vector<std::vector<std::uint8_t>>
LoadRgba8Mips(const std::filesystem::path& file, bool srgb) {
    ERNIE_ZONE("Load RGBA8 mips");
    const std::string name = file.string();
    if (!Rgba8ImageSize(file)) {
        throw std::runtime_error(name +
                                 " is not an 8-bit PNG or JPEG");
    }
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* data = [&] {
        ERNIE_ZONE("Decode image");
        return stbi_load(name.c_str(), &width, &height, &channels, 4);
    }();
    if (data == nullptr) {
        throw std::runtime_error("Cannot read " + name + ": " +
                                 stbi_failure_reason());
    }
    const std::unique_ptr<stbi_uc, decltype(&stbi_image_free)>
        decoded(data, &stbi_image_free);
    auto w = static_cast<std::uint32_t>(width);
    auto h = static_cast<std::uint32_t>(height);
    std::vector<std::vector<std::uint8_t>> mips;
    mips.emplace_back(decoded.get(),
                      decoded.get() + std::size_t{w} * h * 4);
    while (w > 1 || h > 1) {
        std::vector<std::uint8_t> half =
            HalveRgba8(mips.back(), w, h, srgb);
        mips.push_back(std::move(half));
        w = std::max(1u, w / 2);
        h = std::max(1u, h / 2);
    }
    return mips;
}

TextureData EncodeTexture(std::span<const Image> mips,
                          TextureFormat format,
                          ImageEncoding encoding, int quality,
                          BlockEncoder* encoder) {
    if (mips.empty()) {
        throw std::invalid_argument("A texture needs a mip");
    }
    TextureData texture =
        Allocate(format, mips[0].width, mips[0].height,
                 static_cast<std::uint32_t>(mips.size()));
    for (std::uint32_t mip = 0; mip < texture.mips; ++mip) {
        if (mips[mip].width != std::max(1u, texture.width >> mip) ||
            mips[mip].height != std::max(1u, texture.height >> mip)) {
            throw std::invalid_argument(
                "Mip " + std::to_string(mip) +
                " does not halve the one before");
        }
    }
    const bc7enc_compress_block_params bc7 =
        Bc7Params(format, quality);
    for (std::uint32_t mip = 0; mip < texture.mips; ++mip) {
        EncodeMip(texture, mip, ToRgba8(mips[mip], encoding), bc7,
                  encoder);
    }
    return texture;
}

namespace {

/// Bytes before a DDS file's texels as `WriteDdsHeader` writes them.
constexpr std::size_t kDdsHeaderBytes = (32 + 5) * 4;

/// Writes the header of a DDS file of `texture`, with the DX10
/// header.
void WriteDdsHeader(std::ostream& file, const TextureData& texture) {
    // "DDS ", DDS_HEADER (31 words after the magic),
    // DDS_HEADER_DXT10.
    std::array<std::uint32_t, 32 + 5> header{};
    std::memcpy(header.data(), "DDS ", 4);
    header[1] = 124;
    // Caps, height, width, pixel format, linear size and mip count.
    header[2] = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000 |
                (texture.mips > 1 ? 0x20000u : 0u);
    header[3] = texture.height;
    header[4] = texture.width;
    header[5] = static_cast<std::uint32_t>(
        ImageBytes(texture.format, texture.width, texture.height));
    header[7] = texture.mips;
    header[19] = 32;
    header[20] = 0x4;  // FourCC
    std::memcpy(&header[21], "DX10", 4);
    // DDSCAPS_TEXTURE, and COMPLEX and MIPMAP with mips.
    header[27] = 0x1000 | (texture.mips > 1 ? 0x400008u : 0u);
    header[28] = texture.cube ? 0xfe00u : 0u;
    header[32] = DxgiOf(texture.format);
    header[33] = 3;  // A 2D texture.
    header[34] = texture.cube ? 0x4u : 0u;
    header[35] = texture.cube ? texture.layers / 6 : texture.layers;
    static_assert(sizeof(header) == kDdsHeaderBytes);
    file.write(reinterpret_cast<const char*>(header.data()),
               sizeof(header));
}

}  // namespace

void WriteDdsFile(const std::filesystem::path& path,
                  const TextureData& texture) {
    ERNIE_ZONE("Write DDS");
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot write " + path.string());
    }
    WriteDdsHeader(file, texture);
    // Layer after layer, each layer's mips in turn: `TextureData`'s
    // order.
    const std::span<const std::byte> data = texture.Data();
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size()));
    if (!file) {
        throw std::runtime_error("Cannot write " + path.string());
    }
}

TextureData BakeTexture(const std::filesystem::path& input,
                        const std::filesystem::path& output,
                        const BakeOptions& options) {
    ERNIE_ZONE("Bake texture");
    const bool srgb =
        options.role == TextureRole::Colour && options.srgb;
    TextureFormat format = options.role == TextureRole::Normal
                               ? TextureFormat::Bc5
                               : TextureFormat::Bc7;
    if (options.format) {
        format = *options.format;
    }
    format = WithSrgb(format, srgb);
    const ImageEncoding encoding =
        srgb ? ImageEncoding::Srgb : ImageEncoding::Linear;
    // 8-bit PNG and JPEG colour and data stay 8-bit, each mip made
    // from the last and dropped once encoded: 4 bytes a texel of the
    // full size rather than float's 16 for the whole chain, which a
    // global mosaic needs. Normal maps (their slopes) and EXR and
    // 16-bit images go through floats.
    std::string extension = input.extension().string();
    std::transform(extension.begin(), extension.end(),
                   extension.begin(), [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    const std::string name = input.string();
    if (options.role != TextureRole::Normal &&
        (extension == ".png" || extension == ".jpg" ||
         extension == ".jpeg") &&
        stbi_is_16_bit(name.c_str()) == 0) {
        int width = 0;
        int height = 0;
        int channels = 0;
        stbi_uc* data =
            stbi_load(name.c_str(), &width, &height, &channels, 4);
        if (data == nullptr) {
            throw std::runtime_error("Cannot read " + name + ": " +
                                     stbi_failure_reason());
        }
        // Mip 0 is read from the decoder's buffer, freed once halved.
        std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> decoded(
            data, &stbi_image_free);
        auto w = static_cast<std::uint32_t>(width);
        auto h = static_cast<std::uint32_t>(height);
        std::vector<std::uint8_t> halved;
        std::span<const std::uint8_t> rgba(decoded.get(),
                                           std::size_t{w} * h * 4);
        std::uint32_t mips = 1;
        for (std::uint32_t side = std::max(w, h); side > 1;
             side >>= 1) {
            ++mips;
        }
        TextureData texture = Allocate(format, w, h, mips);
        const bc7enc_compress_block_params bc7 =
            Bc7Params(format, options.quality);
        for (std::uint32_t mip = 0; mip < mips; ++mip) {
            EncodeMip(texture, mip, rgba, bc7, options.encoder);
            if (mip + 1 < mips) {
                halved = HalveRgba8(rgba, w, h, srgb);
                rgba = halved;
                decoded.reset();
                w = std::max(1u, w / 2);
                h = std::max(1u, h / 2);
            }
        }
        WriteDdsFile(output, texture);
        return texture;
    }
    const Image image = LoadImageFile(input, encoding);
    const std::vector<Image> mips =
        MipChain(image, options.role == TextureRole::Normal
                            ? MipFilter::NormalSlopes
                            : MipFilter::Average);
    // Normals' mips hold n / 2 + 1/2 in rgb, as the stored-texture
    // decode reads x and y; BC5 keeps those two.
    TextureData texture = EncodeTexture(
        mips, format, encoding, options.quality, options.encoder);
    WriteDdsFile(output, texture);
    return texture;
}

TextureData BakeMosaic(std::span<const std::filesystem::path> tiles,
                       std::uint32_t columns,
                       const std::filesystem::path& output,
                       const BakeOptions& options) {
    ERNIE_ZONE("Bake mosaic");
    if (columns == 0 || tiles.empty() ||
        tiles.size() % columns != 0) {
        throw std::invalid_argument("A mosaic needs whole rows of " +
                                    std::to_string(columns) +
                                    " tiles");
    }
    if (options.role == TextureRole::Normal) {
        throw std::invalid_argument(
            "Normal maps are not baked as mosaics");
    }
    const auto rows =
        static_cast<std::uint32_t>(tiles.size() / columns);
    // Every tile an 8-bit PNG or JPEG of one size.
    std::uint32_t tileWidth = 0;
    std::uint32_t tileHeight = 0;
    for (const std::filesystem::path& tile : tiles) {
        const std::string name = tile.string();
        int width = 0;
        int height = 0;
        int channels = 0;
        if (stbi_info(name.c_str(), &width, &height, &channels) ==
                0 ||
            stbi_is_16_bit(name.c_str()) != 0) {
            throw std::runtime_error("Cannot read " + name +
                                     " as an 8-bit PNG or JPEG");
        }
        if (tileWidth == 0) {
            tileWidth = static_cast<std::uint32_t>(width);
            tileHeight = static_cast<std::uint32_t>(height);
        } else if (static_cast<std::uint32_t>(width) != tileWidth ||
                   static_cast<std::uint32_t>(height) != tileHeight) {
            throw std::invalid_argument(
                "A mosaic's tiles are all one size: " + name +
                " is not " + std::to_string(tileWidth) + " x " +
                std::to_string(tileHeight));
        }
    }
    const bool srgb =
        options.role == TextureRole::Colour && options.srgb;
    const TextureFormat format =
        WithSrgb(options.format.value_or(TextureFormat::Bc7), srgb);
    const std::uint32_t width = columns * tileWidth;
    const std::uint32_t height = rows * tileHeight;
    std::uint32_t mips = 1;
    for (std::uint32_t side = std::max(width, height); side > 1;
         side >>= 1) {
        ++mips;
    }
    const TextureData layout =
        Allocate(format, width, height, mips, false);
    const TextureFormatInfo info = InfoOf(format);
    // Mips stay per tile while they halve exactly and split into
    // whole blocks at the tiles' edges (21600 texels: mips 0 to 3);
    // the rest are made from those tiles' last mips joined.
    std::uint32_t perTile = 0;
    while (perTile + 1 < mips && tileWidth % (4u << perTile) == 0 &&
           tileHeight % (4u << perTile) == 0) {
        ++perTile;
    }
    std::ofstream file(output, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("Cannot write " + output.string());
    }
    WriteDdsHeader(file, layout);
    // Writes `blocks` (`across` x `down` blocks of mip `mip`) at
    // block
    // (`column`, `row`) of the mip.
    const auto write = [&](std::uint32_t mip, std::uint32_t column,
                           std::uint32_t row, std::uint32_t across,
                           std::uint32_t down,
                           const std::vector<std::byte>& blocks) {
        const std::uint32_t mipWidth = std::max(1u, width >> mip);
        const std::size_t mipAcross =
            (mipWidth + info.blockSize - 1) / info.blockSize;
        for (std::uint32_t y = 0; y < down; ++y) {
            file.seekp(static_cast<std::streamoff>(
                kDdsHeaderBytes + layout.offsets[mip] +
                ((row + y) * mipAcross + column) * info.blockBytes));
            file.write(reinterpret_cast<const char*>(
                           blocks.data() +
                           std::size_t{y} * across * info.blockBytes),
                       static_cast<std::streamsize>(across *
                                                    info.blockBytes));
        }
    };
    const bc7enc_compress_block_params bc7 =
        Bc7Params(format, options.quality);
    const auto encode = [&](std::span<const std::uint8_t> rgba,
                            std::uint32_t w, std::uint32_t h) {
        std::vector<std::byte> blocks(ImageBytes(format, w, h));
        if (info.compressed()) {
            if (options.encoder != nullptr &&
                options.encoder->Encodes(format)) {
                options.encoder->Encode(rgba, w, h, format,
                                        blocks.data());
            } else {
                EncodeBlocks(format, rgba, w, h, blocks.data(), bc7);
            }
        } else {
            std::memcpy(blocks.data(), rgba.data(), rgba.size());
        }
        return blocks;
    };

    // The tiles' last per-tile mips, joined.
    const std::uint32_t joinedWidth = width >> perTile;
    const std::uint32_t joinedHeight = height >> perTile;
    std::vector<std::uint8_t> joined(std::size_t{joinedWidth} *
                                     joinedHeight * 4);
    // The next tile decodes on its own thread while this one encodes.
    using Decoded =
        std::unique_ptr<stbi_uc, decltype(&stbi_image_free)>;
    const auto decode = [&](std::size_t index) {
        const std::string name = tiles[index].string();
        int w = 0;
        int h = 0;
        int channels = 0;
        stbi_uc* data = stbi_load(name.c_str(), &w, &h, &channels, 4);
        if (data == nullptr) {
            throw std::runtime_error("Cannot read " + name + ": " +
                                     stbi_failure_reason());
        }
        return Decoded(data, &stbi_image_free);
    };
    std::future<Decoded> next =
        std::async(std::launch::async, decode, 0);
    for (std::size_t index = 0; index < tiles.size(); ++index) {
        Decoded decoded = next.get();
        if (index + 1 < tiles.size()) {
            next = std::async(std::launch::async, decode, index + 1);
        }
        const auto column =
            static_cast<std::uint32_t>(index % columns);
        const auto row = static_cast<std::uint32_t>(index / columns);
        std::uint32_t w = tileWidth;
        std::uint32_t h = tileHeight;
        std::vector<std::uint8_t> halved;
        std::span<const std::uint8_t> rgba(decoded.get(),
                                           std::size_t{w} * h * 4);
        for (std::uint32_t mip = 0; mip < perTile; ++mip) {
            const std::uint32_t across =
                (w + info.blockSize - 1) / info.blockSize;
            const std::uint32_t down =
                (h + info.blockSize - 1) / info.blockSize;
            write(mip, column * across, row * down, across, down,
                  encode(rgba, w, h));
            halved = HalveRgba8(rgba, w, h, srgb);
            rgba = halved;
            decoded.reset();
            w /= 2;
            h /= 2;
        }
        for (std::uint32_t y = 0; y < h; ++y) {
            std::memcpy(joined.data() + ((std::size_t{row} * h + y) *
                                             joinedWidth +
                                         std::size_t{column} * w) *
                                            4,
                        rgba.data() + std::size_t{y} * w * 4,
                        std::size_t{w} * 4);
        }
    }
    // The joined mips, down to 1 x 1.
    std::uint32_t w = joinedWidth;
    std::uint32_t h = joinedHeight;
    for (std::uint32_t mip = perTile; mip < mips; ++mip) {
        write(mip, 0, 0, (w + info.blockSize - 1) / info.blockSize,
              (h + info.blockSize - 1) / info.blockSize,
              encode(joined, w, h));
        if (mip + 1 < mips) {
            joined = HalveRgba8(joined, w, h, srgb);
            w = std::max(1u, w / 2);
            h = std::max(1u, h / 2);
        }
    }
    if (!file) {
        throw std::runtime_error("Cannot write " + output.string());
    }
    return layout;
}

}  // namespace ernie
