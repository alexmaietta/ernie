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
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
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
    // Linear back to sRGB bytes by table rather than a power a texel,
    // the most of the halving's time: 2^14 steps of linear, finer
    // than half a byte of sRGB even near black, where its slope
    // is 12.92.
    constexpr std::size_t kSteps = 1u << 14;
    static const std::array<std::uint8_t, kSteps + 1> kSrgb = [] {
        std::array<std::uint8_t, kSteps + 1> table{};
        for (std::size_t i = 0; i < table.size(); ++i) {
            table[i] = static_cast<std::uint8_t>(std::lround(
                LinearToSrgb(static_cast<float>(i) / kSteps) *
                255.0f));
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
                const float mean = std::clamp(sum / 4.0f, 0.0f, 1.0f);
                half[(std::size_t{y} * w + x) * 4 + c] =
                    srgb && c < 3 ? kSrgb[static_cast<std::size_t>(
                                        std::lround(mean * kSteps))]
                                  : static_cast<std::uint8_t>(
                                        std::lround(mean * 255.0f));
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

namespace {

/// Rows of a band read at once.
constexpr std::uint32_t kBandRows = 1024;
/// Band mips waiting to be encoded, past which reading waits.
constexpr std::size_t kQueuedBytes = std::size_t{256} << 20;

/// The texels of mip `mip` from a span [`start`, `start` + `count`)
/// of a side of `side` texels: the span's start there and how many.
/// Spans start at multiples of 2^mip, so a mip's spans tile it.
std::pair<std::uint32_t, std::uint32_t> MipSpan(std::uint32_t side,
                                                std::uint32_t start,
                                                std::uint32_t count,
                                                std::uint32_t mip) {
    const std::uint32_t first = start >> mip;
    const std::uint32_t end = start + count == side
                                  ? std::max(1u, side >> mip)
                                  : (start + count) >> mip;
    return {first, end > first ? end - first : 0};
}

/// `pixels` samples of `source`'s at `samples` as RGBA8 at `out`.
void ToRgba8(const RasterSource& source, const std::byte* samples,
             std::size_t pixels, std::uint8_t* out) {
    const std::uint32_t channels = source.Channels();
    const bool wide = source.Type() == RasterSampleType::UInt16;
    for (std::size_t i = 0; i < pixels; ++i) {
        std::array<std::uint8_t, 4> c{0, 0, 0, 255};
        for (std::uint32_t k = 0; k < std::min(channels, 4u); ++k) {
            if (wide) {
                std::uint16_t value = 0;
                std::memcpy(&value, samples + (i * channels + k) * 2,
                            2);
                c[k] = static_cast<std::uint8_t>(value >> 8);
            } else {
                c[k] = static_cast<std::uint8_t>(
                    samples[i * channels + k]);
            }
        }
        if (channels <= 2) {
            // Grey, and alpha.
            c = {c[0], c[0], c[0], channels == 2 ? c[1] : c[3]};
        }
        std::memcpy(out + i * 4, c.data(), 4);
    }
}

/// The rows and columns of `rgba` (`width` wide) up to `w` x `h`.
std::vector<std::uint8_t> Cropped(std::vector<std::uint8_t> rgba,
                                  std::uint32_t width,
                                  std::uint32_t w, std::uint32_t h) {
    if (w == width) {
        rgba.resize(std::size_t{w} * h * 4);
        return rgba;
    }
    std::vector<std::uint8_t> cropped(std::size_t{w} * h * 4);
    for (std::uint32_t y = 0; y < h; ++y) {
        std::memcpy(cropped.data() + std::size_t{y} * w * 4,
                    rgba.data() + std::size_t{y} * width * 4,
                    std::size_t{w} * 4);
    }
    return cropped;
}

}  // namespace

struct RasterBake::State {
    /// A rectangle of a mip to encode: `w` x `h` texels at (`x`, `y`)
    /// of mip `mip`, RGBA8.
    struct Piece {
        std::uint32_t mip = 0;
        std::uint32_t x = 0;
        std::uint32_t y = 0;
        std::uint32_t w = 0;
        std::uint32_t h = 0;
        std::vector<std::uint8_t> rgba;
    };

    std::unique_ptr<RasterSource> source;
    std::filesystem::path output;
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
    std::jthread reader;

    /// Queues `piece`, once there is room.
    void Queue(Piece piece) {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] {
            return stop || pieces.empty() || queued < kQueuedBytes;
        });
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
        const std::size_t pixel =
            std::size_t{source->Channels()} * source->SampleBytes();
        const std::uint32_t joinedWidth =
            MipSpan(width, 0, width, banded).second;
        const std::uint32_t joinedHeight =
            MipSpan(height, 0, height, banded).second;
        std::vector<std::uint8_t> joined;
        if (banded < layout.mips) {
            joined.resize(std::size_t{joinedWidth} * joinedHeight *
                          4);
        }
        std::vector<std::byte> samples;
        for (const RasterRect& part : source->Parts()) {
            for (std::uint32_t y = part.y; y < part.y + part.height;
                 y += kBandRows) {
                if (Stopping()) {
                    return;
                }
                const std::uint32_t rows =
                    std::min(kBandRows, part.y + part.height - y);
                std::uint32_t w = part.width;
                std::uint32_t h = rows;
                samples.resize(std::size_t{w} * h * pixel);
                std::vector<std::uint8_t> rgba(std::size_t{w} * h *
                                               4);
                {
                    ERNIE_ZONE("Read band");
                    source->Read(part.x, y, w, h, samples);
                    ToRgba8(*source, samples.data(),
                            std::size_t{w} * h, rgba.data());
                }
                for (std::uint32_t mip = 0;; ++mip) {
                    const auto [x0, across] =
                        MipSpan(width, part.x, part.width, mip);
                    const auto [y0, down] =
                        MipSpan(height, y, rows, mip);
                    if (across == 0 || down == 0) {
                        break;
                    }
                    // Halving keeps a texel of a side of one, which
                    // the whole image's mip may not have here.
                    if (across < w || down < h) {
                        rgba =
                            Cropped(std::move(rgba), w, across, down);
                        w = across;
                        h = down;
                    }
                    if (mip == banded) {
                        for (std::uint32_t r = 0; r < h; ++r) {
                            std::memcpy(
                                joined.data() + (std::size_t{y0 + r} *
                                                     joinedWidth +
                                                 x0) *
                                                    4,
                                rgba.data() + std::size_t{r} * w * 4,
                                std::size_t{w} * 4);
                        }
                        break;
                    }
                    const bool last = mip + 1 == layout.mips;
                    std::vector<std::uint8_t> half =
                        last ? std::vector<std::uint8_t>{}
                             : HalveRgba8(rgba, w, h, srgb);
                    Queue(Piece{mip, x0, y0, w, h, std::move(rgba)});
                    if (last || Stopping()) {
                        break;
                    }
                    rgba = std::move(half);
                    w = std::max(1u, w / 2);
                    h = std::max(1u, h / 2);
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
            std::vector<std::uint8_t> half =
                mip + 1 < layout.mips ? HalveRgba8(joined, w, h, srgb)
                                      : std::vector<std::uint8_t>{};
            Queue(Piece{mip, 0, 0, w, h, std::move(joined)});
            joined = std::move(half);
            w = std::max(1u, w / 2);
            h = std::max(1u, h / 2);
        }
    }
};

RasterBake::RasterBake(std::unique_ptr<RasterSource> source,
                       const std::filesystem::path& output,
                       const BakeOptions& options)
    : state_(std::make_unique<State>()) {
    State& s = *state_;
    if (options.role == TextureRole::Normal) {
        throw std::invalid_argument(
            "Normal maps are not baked by bands");
    }
    if (source->Type() != RasterSampleType::UInt8 &&
        source->Type() != RasterSampleType::UInt16) {
        throw std::invalid_argument(
            "Textures bake from 8- or 16-bit samples");
    }
    s.source = std::move(source);
    s.output = output;
    s.options = options;
    s.srgb = options.role == TextureRole::Colour && options.srgb;
    const TextureFormat format =
        WithSrgb(options.format.value_or(TextureFormat::Bc7), s.srgb);
    const std::uint32_t width = s.source->Width();
    const std::uint32_t height = s.source->Height();
    std::uint32_t mips = 1;
    for (std::uint32_t side = std::max(width, height); side > 1;
         side >>= 1) {
        ++mips;
    }
    s.layout = Allocate(format, width, height, mips, false);
    s.info = InfoOf(format);
    s.bc7 = Bc7Params(format, options.quality);
    for (std::uint32_t mip = 0; mip < mips; ++mip) {
        s.total += std::uint64_t{std::max(1u, width >> mip)} *
                   std::max(1u, height >> mip);
    }
    // Bands start at multiples of `alignment` texels: their mips are
    // made band by band while their starts there are of whole blocks
    // and halve exactly into the next.
    const std::vector<RasterRect> parts = s.source->Parts();
    std::uint32_t alignment = kBandRows;
    for (const RasterRect& part : parts) {
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
           alignment % (std::max(s.info.blockSize, 2u) << s.banded) ==
               0) {
        ++s.banded;
    }
    s.file.open(output,
                std::ios::binary | std::ios::out | std::ios::trunc);
    if (!s.file) {
        throw std::runtime_error("Cannot write " + output.string());
    }
    WriteDdsHeader(s.file, s.layout);
    s.reader = std::jthread([&s] { s.Read(); });
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
        std::filesystem::remove(s.output, error);
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
                s.changed.wait(lock, [&] {
                    return !s.pieces.empty() || s.read;
                });
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
                    throw std::runtime_error("Cannot write " +
                                             s.output.string());
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
        const auto fit =
            static_cast<std::uint32_t>(std::min<std::uint64_t>(
                piece.h, left / piece.w / side * side));
        const std::uint32_t rows =
            std::min(piece.h - s.row, std::max(fit, side));
        const std::span<const std::uint8_t> rgba(
            piece.rgba.data() + std::size_t{s.row} * piece.w * 4,
            std::size_t{rows} * piece.w * 4);
        const TextureFormat format = s.layout.format;
        blocks.resize(ImageBytes(format, piece.w, rows));
        {
            ERNIE_ZONE("Encode band");
            if (!s.info.compressed()) {
                std::memcpy(blocks.data(), rgba.data(), rgba.size());
            } else if (s.options.encoder != nullptr &&
                       s.options.encoder->Encodes(format)) {
                s.options.encoder->Encode(rgba, piece.w, rows, format,
                                          blocks.data());
            } else {
                EncodeBlocks(format, rgba, piece.w, rows,
                             blocks.data(), s.bc7);
            }
        }
        // Each row of blocks in its place in the mip.
        const std::size_t mipAcross =
            (std::max(1u, s.layout.width >> piece.mip) + side - 1) /
            side;
        const std::size_t across = (piece.w + side - 1) / side;
        const std::size_t rowBytes = across * s.info.blockBytes;
        for (std::uint32_t y = 0; y < (rows + side - 1) / side; ++y) {
            const std::size_t at =
                kDdsHeaderBytes + s.layout.offsets[piece.mip] +
                ((std::size_t{piece.y + s.row} / side + y) *
                     mipAcross +
                 piece.x / side) *
                    s.info.blockBytes;
            s.file.seekp(static_cast<std::streamoff>(at));
            s.file.write(reinterpret_cast<const char*>(blocks.data() +
                                                       y * rowBytes),
                         static_cast<std::streamsize>(rowBytes));
        }
        if (!s.file) {
            throw std::runtime_error("Cannot write " +
                                     s.output.string());
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

TextureData BakeMosaic(std::span<const std::filesystem::path> tiles,
                       std::uint32_t columns,
                       const std::filesystem::path& output,
                       const BakeOptions& options) {
    ERNIE_ZONE("Bake mosaic");
    RasterBake bake(std::make_unique<MosaicRaster>(
                        std::vector<std::filesystem::path>(
                            tiles.begin(), tiles.end()),
                        columns),
                    output, options);
    while (!bake.Step(~std::uint64_t{0}, true)) {
    }
    return bake.Layout();
}

}  // namespace ernie
