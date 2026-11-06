// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <vector>

#include "ernie/device/tensor.h"
#include "ernie/export.h"
#include "ernie/util/span.h"

namespace ernie {

enum class ImageEncoding : uint8_t { Srgb, Linear };

struct Rgba {
    float r = 0;
    float g = 0;
    float b = 0;
    float a = 0;

    bool operator==(const Rgba& other) const {
        return r == other.r && g == other.g && b == other.b && a == other.a;
    }

    bool operator!=(const Rgba& other) const {
        return !(*this == other);
    }
};

struct ERNIE_API Image {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<Rgba> pixels;

    Image() = default;

    Image(uint32_t w, uint32_t h, Rgba fill = {})
        : width(w), height(h), pixels(size_t(w) * h, fill) {}

    Rgba& At(uint32_t x, uint32_t y) {
        return pixels[size_t(y) * width + x];
    }

    const Rgba& At(uint32_t x, uint32_t y) const {
        return pixels[size_t(y) * width + x];
    }

    Rgba Mean() const;
};

struct Rgba8Image {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint32_t> texels;
};

ERNIE_API Image LoadImageFile(const std::filesystem::path& path, ImageEncoding encoding);
ERNIE_API Image LoadImageBytes(Span<const std::byte> bytes, std::string_view name,
                               ImageEncoding encoding);
ERNIE_API bool IsPngOrJpeg(const std::filesystem::path& path);

ERNIE_API Rgba8Image LoadRgba8File(const std::filesystem::path& path);
ERNIE_API Rgba8Image LoadRgba8Bytes(Span<const std::byte> bytes, std::string_view name);

struct ERNIE_API Pixels {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t channels = 1;
    device::DType type = device::DType::UInt8;
    std::vector<std::byte> bytes;

    Pixels() = default;
    Pixels(const device::Tensor& tensor);
    Pixels(Span<const std::byte> data, device::DType elementType, int64_t rowCount,
           int64_t colCount, int64_t channelCount = 1);

    template <class T>
    Pixels(Span<const T> values, int64_t rowCount, int64_t colCount, int64_t channelCount = 1)
        : Pixels(Span<const std::byte>(reinterpret_cast<const std::byte*>(values.data()),
                                       values.size_bytes()),
                 device::ElementType<T>(), rowCount, colCount, channelCount) {}

    template <class T>
    Pixels(const std::vector<T>& values, int64_t rowCount, int64_t colCount,
           int64_t channelCount = 1)
        : Pixels(Span<const T>(values), rowCount, colCount, channelCount) {}

    template <class T> Span<const T> Values() const {
        if (device::ElementType<T>() != type) {
            throw std::invalid_argument("Values of a type unlike the pixels' elements");
        }
        return {reinterpret_cast<const T*>(bytes.data()), bytes.size() / sizeof(T)};
    }
};

ERNIE_API Image ImageFromTensor(const device::Tensor& tensor, ImageEncoding encoding);
ERNIE_API std::vector<uint8_t> ToRgba8(const Image& image, ImageEncoding encoding);

ERNIE_API void WritePng(const std::filesystem::path& path, const Image& image);
ERNIE_API void WritePng(const std::filesystem::path& path, const Pixels& image);
ERNIE_API void WriteJpeg(const std::filesystem::path& path, const Image& image, int quality = 92);
ERNIE_API void WriteJpeg(const std::filesystem::path& path, const Pixels& image, int quality = 92);
ERNIE_API void WriteExr(const std::filesystem::path& path, const Image& image);
ERNIE_API void WriteExr(const std::filesystem::path& path, const Pixels& image);
ERNIE_API std::filesystem::path WriteImage(std::filesystem::path path, const Pixels& image);

enum class Colormap : uint8_t { Grayscale, Turbo, Viridis };

struct ColorizeOptions {
    Colormap colormap = Colormap::Grayscale;
    std::optional<float> minimum;
    std::optional<float> maximum;
    bool logarithmic = false;
    bool reversed = false;
    bool centered = false;
    uint32_t channel = 0;

    bool operator==(const ColorizeOptions& other) const {
        return std::tie(colormap, minimum, maximum, logarithmic, reversed, centered, channel) ==
               std::tie(other.colormap, other.minimum, other.maximum, other.logarithmic,
                        other.reversed, other.centered, other.channel);
    }
};

inline constexpr ColorizeOptions kDepthColors{Colormap::Grayscale, std::nullopt, std::nullopt, true,
                                              true};

ERNIE_API std::array<uint8_t, 4> ColormapColor(Colormap colormap, float position);

ERNIE_API Pixels ColorizeScalars(const Pixels& values, const ColorizeOptions& options = {});
ERNIE_API Pixels ColorizeHdr(const Pixels& color, float exposure = 0);
ERNIE_API Pixels ColorizeNormals(const Pixels& normals);
ERNIE_API Pixels ColorizeIds(const Pixels& ids);
ERNIE_API std::array<uint8_t, 4> IdColor(Span<const uint32_t> words);
ERNIE_API device::Tensor ColorizeScalars(const device::Tensor& values,
                                         const ColorizeOptions& options = {});
ERNIE_API device::Tensor ColorizeScalars(const device::Tensor& values,
                                         const ColorizeOptions& options, device::Tensor& bounds);
ERNIE_API device::Tensor ColorizeHdr(const device::Tensor& color, float exposure = 0);
ERNIE_API device::Tensor ColorizeNormals(const device::Tensor& normals);
ERNIE_API device::Tensor ColorizeIds(const device::Tensor& ids);

}
