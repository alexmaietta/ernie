// SPDX-License-Identifier: BSD-3-Clause
/// @file image.h
/// @brief Float RGBA images: loading, EXR output, sRGB and mips.
#pragma once

#include "ernie/export.h"

#include "ernie/util/tensor.h"

#include "ernie/util/glm.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace ernie {

/// @brief How 8- and 16-bit stored values map to linear values.
enum class ImageEncoding {
    /// IEC 61966-2-1 sRGB transfer function, for colour textures.
    Srgb,
    /// Stored values are linear, for normal and data maps.
    Linear,
};

/// @brief How `MipChain` filters each level.
enum class MipFilter {
    /// 2 x 2 box filter of every channel.
    Average,
    /// Box filter of tangent-space normals (rgb in [0, 1]),
    /// renormalised.
    UnitVector,
    /// Box filter of tangent-space normals in slope space. rgb holds
    /// the normal of the mean slope (z at least `kMinNormalZ`);
    /// alpha the per-axis slope deviation as `EncodeSlopeDeviation`
    /// (0 at level 0). Shaders recover the variance filtering hides.
    NormalSlopes,
};

/// Least normal z slopes are taken from, keeping slopes below 20.
inline constexpr float kMinNormalZ = 0.05f;

/// @brief Packs a slope deviation into [0, 1) for 8-bit storage.
/// @param sigma The deviation, 0 or more.
/// @return sigma / (1 + sigma), finest near 0.
ERNIE_API float EncodeSlopeDeviation(float sigma);
/// @brief Inverts `EncodeSlopeDeviation`.
/// @param encoded The stored value.
/// @return The slope deviation.
ERNIE_API float DecodeSlopeDeviation(float encoded);

/// @brief An RGBA image of linear floats, rows top to bottom.
struct ERNIE_API Image {
    /// Width, in pixels.
    std::uint32_t width = 0;
    /// Height, in pixels.
    std::uint32_t height = 0;
    /// Pixels, row by row.
    std::vector<glm::vec4> pixels;

    Image() = default;
    /// @brief Allocates a filled image.
    /// @param w    Width, in pixels.
    /// @param h    Height, in pixels.
    /// @param fill The initial value of every pixel.
    Image(std::uint32_t w, std::uint32_t h, const glm::vec4& fill = glm::vec4(0.0f))
        : width(w), height(h), pixels(static_cast<std::size_t>(w) * h, fill) {}

    /// @return The pixel at (`x`, `y`).
    glm::vec4& At(std::uint32_t x, std::uint32_t y) {
        return pixels[static_cast<std::size_t>(y) * width + x];
    }

    /// @return The pixel at (`x`, `y`).
    const glm::vec4& At(std::uint32_t x, std::uint32_t y) const {
        return pixels[static_cast<std::size_t>(y) * width + x];
    }

    /// @return The mean of every pixel.
    glm::vec4 Mean() const;
};

/// @brief Loads a PNG (8/16-bit), JPEG or OpenEXR file.
/// Grey replicates to rgb; missing alpha reads as 1. EXR reads
/// scanline and tiled images in any compression, DWAA/DWAB included.
/// @param path     The file.
/// @param encoding How PNG and JPEG values decode; EXR is linear.
/// @return The image.
/// @throws std::runtime_error If the file cannot be read or decoded,
///         or is not .png, .jpg, .jpeg or .exr.
ERNIE_API Image LoadImageFile(const std::filesystem::path& path, ImageEncoding encoding);

/// @brief Tests for a .png, .jpg or .jpeg extension, in any case.
/// @param path The file.
/// @return Whether it is a PNG or JPEG, read by stb.
ERNIE_API bool IsPngOrJpeg(const std::filesystem::path& path);

/// @brief Writes a CPU tensor as a ZIP-compressed OpenEXR file.
/// Channels are float32, or uint32 for uint32 data such as ids;
/// narrower integers become exact floats. One channel is Y, two RG,
/// three RGB, four RGBA, more are numbered.
/// @param path  The file.
/// @param image (rows, cols) or (rows, cols, channels), rows top to
///              bottom.
/// @throws std::invalid_argument For another rank or type.
/// @throws std::runtime_error If the file cannot be written.
ERNIE_API void WriteExr(const std::filesystem::path& path, const Tensor& image);

/// @brief Decodes one sRGB channel to linear (IEC 61966-2-1).
/// @param value The encoded value in [0, 1].
/// @return The linear value.
ERNIE_API float SrgbToLinear(float value);
/// @brief Encodes one linear channel to sRGB (IEC 61966-2-1).
/// @param value The linear value in [0, 1].
/// @return The encoded value.
ERNIE_API float LinearToSrgb(float value);

/// @brief Builds the mip chain down to 1 x 1.
/// Level n is max(1, floor(size / 2^n)) per side, as WebGPU expects;
/// each texel filters the 2 x 2 above it, clamped at odd edges.
/// @param image  Level 0.
/// @param filter How levels are filtered.
/// @return Every level, level 0 first.
///
/// References:
/// - M. Toksvig, "Mipmapping Normal Maps", Journal of Graphics Tools
///   10(3), 2005.
/// - M. Olano and D. Baker, "LEAN Mapping", I3D 2010.
ERNIE_API std::vector<Image> MipChain(const Image& image, MipFilter filter);

/// @brief Quantises to 8-bit RGBA.
/// @param image    The image.
/// @param encoding `Srgb` encodes rgb as sRGB; alpha stays linear.
/// @return Four bytes per pixel.
ERNIE_API std::vector<std::uint8_t> ToRgba8(const Image& image, ImageEncoding encoding);

/// @brief A texture tensor as an image.
/// @details Grey replicates into rgb and a missing alpha reads as 1.
/// @param tensor   A float32 (linear) or uint8 (rows, cols) or (rows,
///                 cols, 1, 3 or 4) CPU tensor.
/// @param encoding How uint8 colour is encoded: sRGB or linear.
/// @return The image.
/// @throws std::invalid_argument For another kind of tensor.
ERNIE_API Image ImageFromTensor(const Tensor& tensor, ImageEncoding encoding);

}  // namespace ernie
