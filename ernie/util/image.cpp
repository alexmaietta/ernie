// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/image.h"

#include <ImfRgbaFile.h>
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace ernie {

namespace {

std::string Lowercase(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return text;
}

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
        std::vector<Imf::Rgba> texels(
            static_cast<std::size_t>(width) *
            static_cast<std::size_t>(height));
        // Strides that put the window's first pixel at the buffer's
        // start.
        file.setFrameBuffer(
            texels.data() - window.min.x -
                static_cast<std::ptrdiff_t>(window.min.y) * width,
            1, static_cast<std::size_t>(width));
        file.readPixels(window.min.y, window.max.y);
        Image image(static_cast<std::uint32_t>(width),
                    static_cast<std::uint32_t>(height));
        for (std::size_t i = 0; i < image.pixels.size(); ++i) {
            const Imf::Rgba& t = texels[i];
            image.pixels[i] = glm::vec4(
                static_cast<float>(t.r), static_cast<float>(t.g),
                static_cast<float>(t.b), static_cast<float>(t.a));
        }
        return image;
    } catch (const std::exception& error) {
        throw std::runtime_error("Cannot read " + path.string() +
                                 ": " + error.what());
    }
}

template <typename T>
Image Decode(T* data, int width, int height, float scale,
             ImageEncoding encoding) {
    Image image(static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(height));
    for (std::size_t i = 0; i < image.pixels.size(); ++i) {
        glm::vec4 value;
        for (int c = 0; c < 4; ++c) {
            value[c] =
                static_cast<float>(
                    data[4 * i + static_cast<std::size_t>(c)]) *
                scale;
        }
        if (encoding == ImageEncoding::Srgb) {
            value = glm::vec4(SrgbToLinear(value.r),
                              SrgbToLinear(value.g),
                              SrgbToLinear(value.b), value.a);
        }
        image.pixels[i] = value;
    }
    stbi_image_free(data);
    return image;
}

Image LoadStb(const std::filesystem::path& path,
              ImageEncoding encoding) {
    const std::string name = path.string();
    int width = 0;
    int height = 0;
    int channels = 0;
    if (stbi_is_16_bit(name.c_str()) != 0) {
        if (stbi_us* data = stbi_load_16(name.c_str(), &width,
                                         &height, &channels, 4)) {
            return Decode(data, width, height, 1.0f / 65535.0f,
                          encoding);
        }
    } else if (stbi_uc* data = stbi_load(name.c_str(), &width,
                                         &height, &channels, 4)) {
        return Decode(data, width, height, 1.0f / 255.0f, encoding);
    }
    throw std::runtime_error("Cannot read " + name + ": " +
                             stbi_failure_reason());
}

}  // namespace

glm::vec4 Image::Mean() const {
    glm::dvec4 sum(0.0);
    for (const glm::vec4& p : pixels) {
        sum += glm::dvec4(p);
    }
    return pixels.empty()
               ? glm::vec4(0.0f)
               : glm::vec4(sum / static_cast<double>(pixels.size()));
}

Image LoadImageFile(const std::filesystem::path& path,
                    ImageEncoding encoding) {
    const std::string extension =
        Lowercase(path.extension().string());
    if (extension == ".exr") {
        return LoadExr(path);
    }
    if (extension == ".png" || extension == ".jpg" ||
        extension == ".jpeg") {
        return LoadStb(path, encoding);
    }
    throw std::runtime_error("Unsupported image format: " +
                             path.string());
}

float SrgbToLinear(float value) {
    return value <= 0.04045f
               ? value / 12.92f
               : std::pow((value + 0.055f) / 1.055f, 2.4f);
}

float LinearToSrgb(float value) {
    return value <= 0.0031308f
               ? value * 12.92f
               : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
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
glm::vec2 SlopeOf(const glm::vec4& texel) {
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
        const glm::vec2 slope = SlopeOf(texel);
        const float sigma = DecodeSlopeDeviation(texel.a);
        mean += 0.25f * slope;
        square +=
            0.25f * (glm::dot(slope, slope) + 2.0f * sigma * sigma);
    }
    const float variance =
        std::max(0.5f * (square - glm::dot(mean, mean)), 0.0f);
    const glm::vec3 n = glm::normalize(glm::vec3(-mean, 1.0f));
    return glm::vec4(0.5f * n + 0.5f,
                     EncodeSlopeDeviation(std::sqrt(variance)));
}

}  // namespace

std::vector<Image> MipChain(const Image& image, MipFilter filter) {
    std::vector<Image> levels{image};
    if (filter == MipFilter::NormalSlopes) {
        for (glm::vec4& texel : levels.back().pixels) {
            texel.a = 0.0f;
        }
    }
    while (levels.back().width > 1 || levels.back().height > 1) {
        const Image& above = levels.back();
        Image level(std::max(1u, above.width / 2),
                    std::max(1u, above.height / 2));
        for (std::uint32_t y = 0; y < level.height; ++y) {
            const std::uint32_t y0 =
                std::min(2 * y, above.height - 1);
            const std::uint32_t y1 =
                std::min(2 * y + 1, above.height - 1);
            for (std::uint32_t x = 0; x < level.width; ++x) {
                const std::uint32_t x0 =
                    std::min(2 * x, above.width - 1);
                const std::uint32_t x1 =
                    std::min(2 * x + 1, above.width - 1);
                if (filter == MipFilter::NormalSlopes) {
                    level.At(x, y) = FilterSlopes(
                        {above.At(x0, y0), above.At(x1, y0),
                         above.At(x0, y1), above.At(x1, y1)});
                    continue;
                }
                glm::vec4 texel =
                    0.25f * (above.At(x0, y0) + above.At(x1, y0) +
                             above.At(x0, y1) + above.At(x1, y1));
                if (filter == MipFilter::UnitVector) {
                    glm::vec3 n = 2.0f * glm::vec3(texel) - 1.0f;
                    const float length = glm::length(n);
                    n = length > 0.0f ? n / length
                                      : glm::vec3(0.0f, 0.0f, 1.0f);
                    texel = glm::vec4(0.5f * n + 0.5f, texel.a);
                }
                level.At(x, y) = texel;
            }
        }
        levels.push_back(std::move(level));
    }
    return levels;
}

std::vector<std::uint8_t> ToRgba8(const Image& image,
                                  ImageEncoding encoding) {
    std::vector<std::uint8_t> bytes(image.pixels.size() * 4);
    auto quantise = [](float value) {
        return static_cast<std::uint8_t>(
            std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
    };
    for (std::size_t i = 0; i < image.pixels.size(); ++i) {
        const glm::vec4& p = image.pixels[i];
        for (int c = 0; c < 3; ++c) {
            const float value = encoding == ImageEncoding::Srgb
                                    ? LinearToSrgb(p[c])
                                    : p[c];
            bytes[4 * i + static_cast<std::size_t>(c)] =
                quantise(value);
        }
        bytes[4 * i + 3] = quantise(p.a);
    }
    return bytes;
}

}  // namespace ernie
