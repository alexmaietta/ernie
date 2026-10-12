// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/texture_bake.h"

#include <bc7enc.h>
#include <rgbcx.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <fstream>
#include <mutex>
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
void EncodeBlocks(TextureFormat format,
                  const std::vector<std::uint8_t>& rgba,
                  std::uint32_t width, std::uint32_t height,
                  std::byte* out,
                  const bc7enc_compress_block_params& bc7) {
    const TextureFormatInfo info = InfoOf(format);
    const std::uint32_t across = (width + 3) / 4;
    const std::uint32_t down = (height + 3) / 4;
    std::atomic<std::uint32_t> nextRow{0};
    const auto work = [&] {
        std::array<std::uint8_t, 64> pixels{};
        for (std::uint32_t row = nextRow++; row < down;
             row = nextRow++) {
            for (std::uint32_t column = 0; column < across;
                 ++column) {
                // Texels past the edge repeat the last row and
                // column.
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
                            out +
                                (std::size_t{row} * across + column) *
                                    info.blockBytes,
                            bc7);
            }
        }
    };
    const unsigned threads = std::max(
        1u, std::min(std::thread::hardware_concurrency(), down));
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < threads; ++t) {
        pool.emplace_back(work);
    }
    work();
    for (std::thread& thread : pool) {
        thread.join();
    }
}

}  // namespace

TextureData EncodeTexture(std::span<const Image> mips,
                          TextureFormat format,
                          ImageEncoding encoding, int quality) {
    using F = TextureFormat;
    constexpr std::array<F, 10> kWritten{
        F::Rgba8,   F::Rgba8Srgb, F::Bc1, F::Bc1Srgb, F::Bc3,
        F::Bc3Srgb, F::Bc4,       F::Bc5, F::Bc7,     F::Bc7Srgb};
    if (std::find(kWritten.begin(), kWritten.end(), format) ==
        kWritten.end()) {
        throw std::invalid_argument(
            "Textures are not encoded to this format");
    }
    if (mips.empty()) {
        throw std::invalid_argument("A texture needs a mip");
    }
    TextureData texture;
    texture.format = format;
    texture.width = mips[0].width;
    texture.height = mips[0].height;
    texture.mips = static_cast<std::uint32_t>(mips.size());
    for (std::uint32_t mip = 0; mip < texture.mips; ++mip) {
        const std::uint32_t width =
            std::max(1u, texture.width >> mip);
        const std::uint32_t height =
            std::max(1u, texture.height >> mip);
        if (mips[mip].width != width || mips[mip].height != height) {
            throw std::invalid_argument(
                "Mip " + std::to_string(mip) +
                " does not halve the one before");
        }
        texture.offsets.push_back(texture.bytes.size());
        texture.bytes.resize(texture.bytes.size() +
                             ImageBytes(format, width, height));
    }

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
    bc7.m_uber_level = static_cast<std::uint32_t>(
        std::clamp(quality, 0, BC7ENC_MAX_UBER_LEVEL));
    for (std::uint32_t mip = 0; mip < texture.mips; ++mip) {
        const Image& image = mips[mip];
        const std::vector<std::uint8_t> rgba =
            ToRgba8(image, encoding);
        std::byte* out = texture.bytes.data() + texture.offsets[mip];
        if (!InfoOf(format).compressed()) {
            std::memcpy(out, rgba.data(), rgba.size());
            continue;
        }
        EncodeBlocks(format, rgba, image.width, image.height, out,
                     bc7);
    }
    return texture;
}

void WriteDdsFile(const std::filesystem::path& path,
                  const TextureData& texture) {
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
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot write " + path.string());
    }
    file.write(reinterpret_cast<const char*>(header.data()),
               sizeof(header));
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
    const Image image = LoadImageFile(input, encoding);
    const std::vector<Image> mips =
        MipChain(image, options.role == TextureRole::Normal
                            ? MipFilter::NormalSlopes
                            : MipFilter::Average);
    // Normals' mips hold n / 2 + 1/2 in rgb, as the stored-texture
    // decode reads x and y; BC5 keeps those two.
    TextureData texture =
        EncodeTexture(mips, format, encoding, options.quality);
    WriteDdsFile(output, texture);
    return texture;
}

}  // namespace ernie
