// SPDX-License-Identifier: BSD-3-Clause
#include "util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ernie::demos {

namespace {

void CheckImage(const Tensor& image) {
    const std::vector<std::int64_t>& shape = image.Shape();
    if (shape.size() != 3 || image.Type() != DType::UInt8 ||
        image.Location().kind != DeviceKind::Cpu) {
        throw std::invalid_argument(
            "Expected a uint8 (rows, cols, channels) CPU image");
    }
}

double Decode(double c) {
    return c <= 0.04045 ? c / 12.92
                        : std::pow((c + 0.055) / 1.055, 2.4);
}

}  // namespace

Tensor SideBySide(std::span<const Tensor> images, int gap) {
    if (images.empty()) {
        throw std::invalid_argument("SideBySide needs images");
    }
    for (const Tensor& image : images) {
        CheckImage(image);
    }
    const std::int64_t rows = images[0].Shape()[0];
    const std::int64_t channels = images[0].Shape()[2];
    std::int64_t cols =
        gap * static_cast<std::int64_t>(images.size() - 1);
    for (const Tensor& image : images) {
        if (image.Shape()[0] != rows ||
            image.Shape()[2] != channels) {
            throw std::invalid_argument(
                "SideBySide needs images of one height and channel "
                "count");
        }
        cols += image.Shape()[1];
    }
    Tensor result =
        Tensor::Zeros(DType::UInt8, {rows, cols, channels});
    auto* to = static_cast<std::uint8_t*>(result.Data());
    std::int64_t left = 0;
    for (const Tensor& image : images) {
        const std::int64_t width = image.Shape()[1];
        const auto* from =
            static_cast<const std::uint8_t*>(image.Data());
        for (std::int64_t row = 0; row < rows; ++row) {
            std::memcpy(to + (row * cols + left) * channels,
                        from + row * width * channels,
                        static_cast<std::size_t>(width * channels));
        }
        left += width + gap;
    }
    return result;
}

glm::dvec3 SrgbToLinear(glm::dvec3 srgb) {
    return {Decode(srgb.x), Decode(srgb.y), Decode(srgb.z)};
}

glm::dvec3 MeanLinearRgb(const Tensor& image) {
    CheckImage(image);
    const std::int64_t pixels = image.Shape()[0] * image.Shape()[1];
    const std::int64_t channels = image.Shape()[2];
    const auto* texels =
        static_cast<const std::uint8_t*>(image.Data());
    glm::dvec3 sum(0.0);
    for (std::int64_t i = 0; i < pixels; ++i) {
        for (int c = 0; c < 3; ++c) {
            sum[c] += texels[i * channels + c];
        }
    }
    return SrgbToLinear(sum / (255.0 * static_cast<double>(pixels)));
}

double ColourDeviation(const Tensor& image) {
    CheckImage(image);
    const std::int64_t pixels = image.Shape()[0] * image.Shape()[1];
    const std::int64_t channels = image.Shape()[2];
    const auto* texels =
        static_cast<const std::uint8_t*>(image.Data());
    double sum = 0.0;
    double squares = 0.0;
    for (std::int64_t i = 0; i < pixels; ++i) {
        for (int c = 0; c < 3; ++c) {
            const double value = texels[i * channels + c];
            sum += value;
            squares += value * value;
        }
    }
    const double count = 3.0 * static_cast<double>(pixels);
    const double mean = sum / count;
    return std::sqrt(std::max(squares / count - mean * mean, 0.0));
}

}  // namespace ernie::demos
