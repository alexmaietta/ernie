// SPDX-License-Identifier: BSD-3-Clause
#include "images.h"

#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ernie::bench {

namespace {

/// The RGBA8 colour of texel `i` of a var's CPU tensor: colour as is,
/// distances darkening with range, normals and HDR mapped to [0, 1],
/// ids hashed.
std::array<std::uint8_t, 4> Visualise(const Tensor& tensor,
                                      const std::string& var,
                                      std::size_t i, int channels) {
    const auto byte = [](double value) {
        return static_cast<std::uint8_t>(
            std::clamp(value, 0.0, 1.0) * 255.0 + 0.5);
    };
    if (tensor.Type() == DType::UInt8) {
        const std::uint8_t* t = tensor.As<std::uint8_t>() + i * 4;
        return {t[0], t[1], t[2], 255};
    }
    if (tensor.Type() == DType::UInt32) {
        std::uint32_t h = tensor.As<std::uint32_t>()[i] * 2654435761u;
        h ^= h >> 15;
        return {static_cast<std::uint8_t>(h),
                static_cast<std::uint8_t>(h >> 8),
                static_cast<std::uint8_t>(h >> 16), 255};
    }
    if (tensor.Type() != DType::Float32) {
        return {0, 0, 0, 255};
    }
    const float* f =
        tensor.As<float>() + i * static_cast<std::size_t>(channels);
    if (channels == 1) {
        const double d = f[0];
        const std::uint8_t grey = std::isfinite(d) && d > 0.0
                                      ? byte(1.0 / (1.0 + d / 5.0))
                                      : 0;
        return {grey, grey, grey, 255};
    }
    std::array<std::uint8_t, 4> rgba{0, 0, 0, 255};
    for (int c = 0; c < std::min(channels, 3); ++c) {
        const double x = f[c];
        rgba[static_cast<std::size_t>(c)] = var == "Normal"
                                                ? byte(0.5 * x + 0.5)
                                                : byte(x / (1.0 + x));
    }
    return rgba;
}

/// Writes up to 16 views of `tensor`, spread through the batch, as a
/// 4-wide grid.
void WriteSheet(const std::filesystem::path& path,
                const Tensor& tensor, const std::string& var) {
    std::vector<std::int64_t> shape = tensor.Shape();
    if (tensor.Type() == DType::Float32 &&
        var == "SpectralRadiance") {
        return;  // bands first: no colour image
    }
    // One view is (rows, cols) for a channel, else (rows, cols,
    // channels); a batch adds the view first.
    const bool oneChannel =
        tensor.Type() == DType::UInt32 || var.starts_with("Distance");
    const std::size_t viewDims = oneChannel ? 2 : 3;
    if (shape.size() == viewDims) {
        shape.insert(shape.begin(), 1);
    }
    const int views = static_cast<int>(shape[0]);
    const int rows = static_cast<int>(shape[1]);
    const int cols = static_cast<int>(shape[2]);
    const int channels = oneChannel ? 1 : static_cast<int>(shape[3]);
    const int shown = std::min(views, 16);
    const int across = std::min(shown, 4);
    const int down = (shown + across - 1) / across;
    std::vector<std::uint8_t> sheet(
        static_cast<std::size_t>(across * cols) *
            static_cast<std::size_t>(down * rows) * 4,
        0);
    for (int s = 0; s < shown; ++s) {
        const int view = s * views / shown;
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                const std::size_t from =
                    (static_cast<std::size_t>(view) *
                         static_cast<std::size_t>(rows) +
                     static_cast<std::size_t>(y)) *
                        static_cast<std::size_t>(cols) +
                    static_cast<std::size_t>(x);
                const std::size_t to =
                    (static_cast<std::size_t>((s / across) * rows +
                                              y) *
                         static_cast<std::size_t>(across * cols) +
                     static_cast<std::size_t>((s % across) * cols +
                                              x)) *
                    4;
                const auto rgba =
                    Visualise(tensor, var, from, channels);
                std::copy(rgba.begin(), rgba.end(),
                          sheet.begin() +
                              static_cast<std::ptrdiff_t>(to));
            }
        }
    }
    if (stbi_write_png(path.string().c_str(), across * cols,
                       down * rows, 4, sheet.data(),
                       across * cols * 4) == 0) {
        throw std::runtime_error("Cannot write " + path.string());
    }
}

}  // namespace

void WriteSheets(const std::filesystem::path& directory,
                 const std::string& stem,
                 const RenderProductSetOutputs& outputs) {
    std::filesystem::create_directories(directory);
    for (const auto& [path, product] : outputs) {
        for (const auto& [name, var] : product.vars) {
            RenderVarOutput output = var;
            const Tensor tensor = output.Map(MapDevice::Cpu).tensor;
            WriteSheet(directory / (stem + "_" + name + ".png"),
                       tensor, name);
        }
    }
}

}  // namespace ernie::bench
