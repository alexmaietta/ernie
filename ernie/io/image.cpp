// SPDX-License-Identifier: BSD-3-Clause
/// @file image.cpp
/// @brief Image loading, EXR writing, sRGB conversion and mip chains.
#include "ernie/io/image.h"

#include "ernie/util/profiler.h"
#include "ernie/util/text.h"

#include <ImfChannelList.h>
#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfOutputFile.h>
#include <ImfRgbaFile.h>
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace ernie {

namespace {

Image LoadExr(const std::filesystem::path& path) {
    try {
        // RGBA as half floats, whatever the file holds: missing alpha
        // reads as 1, luminance images as grey, and every compression
        // (RLE, ZIP, PIZ, PXR24, B44, DWAA, DWAB, HTJ2K) decodes,
        // scanline or tiled.
        Imf::RgbaInputFile file(path.string().c_str());
        const Imath::Box2i window = file.dataWindow();
        const int width = window.max.x - window.min.x + 1;
        const int height = window.max.y - window.min.y + 1;
        std::vector<Imf::Rgba> texels(static_cast<std::size_t>(width) *
                                      static_cast<std::size_t>(height));
        // Strides that put the window's first pixel at the buffer's
        // start.
        file.setFrameBuffer(texels.data() - window.min.x -
                                static_cast<std::ptrdiff_t>(window.min.y) * width,
                            1, static_cast<std::size_t>(width));
        file.readPixels(window.min.y, window.max.y);
        Image image(static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height));
        for (std::size_t i = 0; i < image.pixels.size(); ++i) {
            const Imf::Rgba& t = texels[i];
            image.pixels[i] = glm::vec4(static_cast<float>(t.r), static_cast<float>(t.g),
                                        static_cast<float>(t.b), static_cast<float>(t.a));
        }
        return image;
    } catch (const std::exception& error) {
        throw std::runtime_error("Cannot read " + path.string() + ": " + error.what());
    }
}

template <typename T>
Image Decode(T* data, int width, int height, float scale, ImageEncoding encoding) {
    Image image(static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height));
    for (std::size_t i = 0; i < image.pixels.size(); ++i) {
        glm::vec4 value;
        for (int c = 0; c < 4; ++c) {
            value[c] = static_cast<float>(data[4 * i + static_cast<std::size_t>(c)]) * scale;
        }
        if (encoding == ImageEncoding::Srgb) {
            value = glm::vec4(SrgbToLinear(value.r), SrgbToLinear(value.g), SrgbToLinear(value.b),
                              value.a);
        }
        image.pixels[i] = value;
    }
    stbi_image_free(data);
    return image;
}

Image LoadStb(const std::filesystem::path& path, ImageEncoding encoding) {
    const std::string name = path.string();
    int width = 0;
    int height = 0;
    int channels = 0;
    if (stbi_is_16_bit(name.c_str()) != 0) {
        if (stbi_us* data = stbi_load_16(name.c_str(), &width, &height, &channels, 4)) {
            return Decode(data, width, height, 1.0f / 65535.0f, encoding);
        }
    } else if (stbi_uc* data = stbi_load(name.c_str(), &width, &height, &channels, 4)) {
        return Decode(data, width, height, 1.0f / 255.0f, encoding);
    }
    throw std::runtime_error("Cannot read " + name + ": " + stbi_failure_reason());
}

}  // namespace

glm::vec4 Image::Mean() const {
    glm::dvec4 sum(0.0);
    for (const glm::vec4& p : pixels) {
        sum += glm::dvec4(p);
    }
    return pixels.empty() ? glm::vec4(0.0f) : glm::vec4(sum / static_cast<double>(pixels.size()));
}

void WriteExr(const std::filesystem::path& path, const Tensor& image) {
    if ((image.Rank() != 2 && image.Rank() != 3) || image.Location().kind != DeviceKind::Cpu) {
        throw std::invalid_argument("EXR images are (rows, cols[, channels]) on the CPU");
    }
    const auto rows = static_cast<int>(image.Shape()[0]);
    const auto cols = static_cast<int>(image.Shape()[1]);
    const int channels = image.Rank() == 3 ? static_cast<int>(image.Shape()[2]) : 1;
    const DType type = image.Type();
    if (type != DType::Float32 && type != DType::UInt32 && type != DType::UInt16 &&
        type != DType::UInt8) {
        throw std::invalid_argument("EXR images are float32 or unsigned integers");
    }
    const Tensor contiguous = image.IsContiguous() ? image : image.Contiguous();
    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) *
                              static_cast<std::size_t>(channels);
    // Floats and uint32 as they are; narrower integers as floats.
    const bool uints = type == DType::UInt32;
    std::vector<float> floats;
    if (type == DType::UInt16 || type == DType::UInt8) {
        floats.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            floats[i] =
                static_cast<float>(type == DType::UInt16 ? contiguous.As<std::uint16_t>()[i]
                                                         : contiguous.As<std::uint8_t>()[i]);
        }
    }
    char* base = floats.empty() ? static_cast<char*>(const_cast<void*>(contiguous.Data()))
                                : reinterpret_cast<char*>(floats.data());
    static constexpr std::array<const char*, 4> kRgba{"R", "G", "B", "A"};
    const auto name = [&](int c) -> std::string {
        if (channels == 1) {
            return "Y";
        }
        return channels <= 4 ? kRgba[static_cast<std::size_t>(c)] : std::to_string(c);
    };
    try {
        Imf::Header header(cols, rows);
        header.compression() = Imf::ZIP_COMPRESSION;
        const Imf::PixelType pixel = uints ? Imf::UINT : Imf::FLOAT;
        Imf::FrameBuffer frame;
        const std::size_t texel = static_cast<std::size_t>(channels) * sizeof(float);
        for (int c = 0; c < channels; ++c) {
            header.channels().insert(name(c), Imf::Channel(pixel));
            frame.insert(name(c),
                         Imf::Slice(pixel, base + static_cast<std::size_t>(c) * sizeof(float),
                                    texel, texel * static_cast<std::size_t>(cols)));
        }
        Imf::OutputFile file(path.string().c_str(), header);
        file.setFrameBuffer(frame);
        file.writePixels(rows);
    } catch (const std::exception& error) {
        throw std::runtime_error("Cannot write " + path.string() + ": " + error.what());
    }
}

Image LoadImageFile(const std::filesystem::path& path, ImageEncoding encoding) {
    ERNIE_ZONE("Decode image");
    const std::string extension = ToLower(path.extension().string());
    if (extension == ".exr") {
        return LoadExr(path);
    }
    if (IsPngOrJpeg(path)) {
        return LoadStb(path, encoding);
    }
    throw std::runtime_error("Unsupported image format: " + path.string());
}

bool IsPngOrJpeg(const std::filesystem::path& path) {
    const std::string extension = ToLower(path.extension().string());
    return extension == ".png" || extension == ".jpg" || extension == ".jpeg";
}

float SrgbToLinear(float value) {
    return value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
}

float LinearToSrgb(float value) {
    return value <= 0.0031308f ? value * 12.92f : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
}

float EncodeSlopeDeviation(float sigma) {
    const float s = std::max(sigma, 0.0f);
    return s / (1.0f + s);
}

float DecodeSlopeDeviation(float encoded) {
    const float e = std::clamp(encoded, 0.0f, 0.999f);
    return e / (1.0f - e);
}

namespace {

/// The slope (dz/dx, dz/dy) of a normal stored as rgb in [0, 1].
glm::vec2 NormalSlope(const glm::vec4& texel) {
    const glm::vec3 n = 2.0f * glm::vec3(texel) - 1.0f;
    return -glm::vec2(n) / std::max(n.z, kMinNormalZ);
}

/// The 2 x 2 slope-space filter of `NormalSlopes`: the mean slope's
/// normal, and the slopes' deviation from it, within and between the
/// four texels, per axis.
glm::vec4 FilterSlopes(const std::array<glm::vec4, 4>& texels) {
    glm::vec2 mean(0.0f);
    float square = 0.0f;  // Mean of sx^2 + sy^2 over everything the
                          // texels cover.
    for (const glm::vec4& texel : texels) {
        const glm::vec2 slope = NormalSlope(texel);
        const float sigma = DecodeSlopeDeviation(texel.a);
        mean += 0.25f * slope;
        square += 0.25f * (glm::dot(slope, slope) + 2.0f * sigma * sigma);
    }
    const float variance = std::max(0.5f * (square - glm::dot(mean, mean)), 0.0f);
    const glm::vec3 n = glm::normalize(glm::vec3(-mean, 1.0f));
    return glm::vec4(0.5f * n + 0.5f, EncodeSlopeDeviation(std::sqrt(variance)));
}

}  // namespace

std::vector<Image> MipChain(const Image& image, MipFilter filter) {
    ERNIE_ZONE("Mip chain");
    std::vector<Image> levels{image};
    if (filter == MipFilter::NormalSlopes) {
        for (glm::vec4& texel : levels.back().pixels) {
            texel.a = 0.0f;
        }
    }
    while (levels.back().width > 1 || levels.back().height > 1) {
        const Image& above = levels.back();
        Image level(std::max(1u, above.width / 2), std::max(1u, above.height / 2));
        for (std::uint32_t y = 0; y < level.height; ++y) {
            const std::uint32_t y0 = std::min(2 * y, above.height - 1);
            const std::uint32_t y1 = std::min(2 * y + 1, above.height - 1);
            for (std::uint32_t x = 0; x < level.width; ++x) {
                const std::uint32_t x0 = std::min(2 * x, above.width - 1);
                const std::uint32_t x1 = std::min(2 * x + 1, above.width - 1);
                if (filter == MipFilter::NormalSlopes) {
                    level.At(x, y) = FilterSlopes(
                        {above.At(x0, y0), above.At(x1, y0), above.At(x0, y1), above.At(x1, y1)});
                    continue;
                }
                glm::vec4 texel = 0.25f * (above.At(x0, y0) + above.At(x1, y0) + above.At(x0, y1) +
                                           above.At(x1, y1));
                if (filter == MipFilter::UnitVector) {
                    glm::vec3 n = 2.0f * glm::vec3(texel) - 1.0f;
                    const float length = glm::length(n);
                    n = length > 0.0f ? n / length : glm::vec3(0.0f, 0.0f, 1.0f);
                    texel = glm::vec4(0.5f * n + 0.5f, texel.a);
                }
                level.At(x, y) = texel;
            }
        }
        levels.push_back(std::move(level));
    }
    return levels;
}

std::vector<std::uint8_t> ToRgba8(const Image& image, ImageEncoding encoding) {
    ERNIE_ZONE("Quantise to RGBA8");
    std::vector<std::uint8_t> bytes(image.pixels.size() * 4);
    auto quantise = [](float value) {
        return static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
    };
    for (std::size_t i = 0; i < image.pixels.size(); ++i) {
        const glm::vec4& p = image.pixels[i];
        for (int c = 0; c < 3; ++c) {
            const float value = encoding == ImageEncoding::Srgb ? LinearToSrgb(p[c]) : p[c];
            bytes[4 * i + static_cast<std::size_t>(c)] = quantise(value);
        }
        bytes[4 * i + 3] = quantise(p.a);
    }
    return bytes;
}

Image ImageFromTensor(const Tensor& tensor, ImageEncoding encoding) {
    const bool grey = tensor.Rank() == 2;
    const std::int64_t channels = grey ? 1 : (tensor.Rank() == 3 ? tensor.Shape()[2] : 0);
    const bool typed = tensor.Type() == DType::Float32 || tensor.Type() == DType::UInt8;
    if (!typed || (channels != 1 && channels != 3 && channels != 4) ||
        tensor.Location().kind != DeviceKind::Cpu || tensor.Shape()[0] < 1 ||
        tensor.Shape()[1] < 1) {
        throw std::invalid_argument("expected a float32 or uint8 (rows, cols[, 1, 3 or 4]) "
                                    "CPU tensor");
    }
    const Tensor packed = tensor.Contiguous();
    Image image(static_cast<std::uint32_t>(tensor.Shape()[1]),
                static_cast<std::uint32_t>(tensor.Shape()[0]));
    const bool bytes = tensor.Type() == DType::UInt8;
    const auto* floats = bytes ? nullptr : packed.As<float>();
    const auto* uints = bytes ? packed.As<std::uint8_t>() : nullptr;
    const auto value = [&](std::size_t index, bool colour) {
        if (!bytes) {
            return floats[index];
        }
        const float stored = static_cast<float>(uints[index]) / 255.0f;
        return colour && encoding == ImageEncoding::Srgb ? SrgbToLinear(stored) : stored;
    };
    const auto stride = static_cast<std::size_t>(channels);
    for (std::size_t i = 0; i < image.pixels.size(); ++i) {
        glm::vec4& pixel = image.pixels[i];
        const std::size_t base = i * stride;
        if (channels == 1) {
            pixel = glm::vec4(glm::vec3(value(base, true)), 1.0f);
        } else {
            pixel = glm::vec4(value(base, true), value(base + 1, true), value(base + 2, true),
                              channels == 4 ? value(base + 3, false) : 1.0f);
        }
    }
    return image;
}

}  // namespace ernie
