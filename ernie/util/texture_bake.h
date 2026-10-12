// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/util/image.h"
#include "ernie/util/texture_file.h"

#include <filesystem>
#include <optional>
#include <span>

namespace ernie {

/// What a texture holds, which decides how it is mipped and stored.
enum class TextureRole {
    Colour,  ///< Colour, sRGB unless said otherwise; BC7.
    /// Tangent-space normals as n / 2 + 1/2, mipped as normals
    /// (`MipFilter::NormalSlopes`) and stored as x and y (BC5), the
    /// shader rebuilding z.
    Normal,
    Data,  ///< Linear values such as roughness or metallic; BC7.
};

/// Encodes RGBA8 images into BC blocks away from the CPU, as
/// `GpuBlockEncoder` does on the GPU: bakes use it for the formats it
/// encodes, the CPU's encoders for the rest.
class BlockEncoder {
  public:
    virtual ~BlockEncoder() = default;

    virtual bool Encodes(TextureFormat format) const = 0;

    /// `rgba` (`width` x `height` RGBA8, rows top to bottom) as
    /// `format`'s blocks at `out` (`ImageBytes(format, width,
    /// height)`), past the edges repeating the last row and column.
    virtual void Encode(std::span<const std::uint8_t> rgba,
                        std::uint32_t width, std::uint32_t height,
                        TextureFormat format, std::byte* out) = 0;
};

/// How `BakeTexture` stores a texture.
struct BakeOptions {
    TextureRole role = TextureRole::Colour;
    /// The stored format; by default the role's. BC1, BC3, BC4 (red),
    /// BC5 (red and green), BC7 and RGBA8 are written; BC6H is not.
    std::optional<TextureFormat> format;
    /// Colour stored as sRGB (the default) or linear.
    bool srgb = true;
    /// BC7's effort on the CPU, 0 (fastest) to 4.
    int quality = 1;
    /// Encodes the formats it can in place of the CPU (on the GPU);
    /// null for the CPU's encoders alone.
    BlockEncoder* encoder = nullptr;
};

/// `mips` (full size first, as `MipChain` makes them) stored in
/// `format`, 8-bit values encoded with `encoding`; BC blocks over a
/// mip's edge repeat its last row and column; `quality` is BC7's
/// effort, 0 to 4. Encodes with `encoder` where it encodes `format`,
/// else on every hardware thread.
/// @throws std::invalid_argument for a format it does not write, or
/// mips that do not halve.
TextureData EncodeTexture(std::span<const Image> mips,
                          TextureFormat format,
                          ImageEncoding encoding, int quality = 1,
                          BlockEncoder* encoder = nullptr);

/// Writes `texture` as a DDS file with the DX10 header, every layer's
/// mips (`ReadDds` reads it back).
/// @throws std::runtime_error if the file cannot be written, or for a
/// format DDS has no code for here.
void WriteDdsFile(const std::filesystem::path& path,
                  const TextureData& texture);

/// The width and height of an 8-bit PNG or JPEG, read from its
/// header; none for any other file (16-bit, another format,
/// unreadable).
std::optional<glm::uvec2>
Rgba8ImageSize(const std::filesystem::path& file);

/// An 8-bit PNG or JPEG decoded as RGBA8 (rows top to bottom) and its
/// mips down to 1 x 1, each the 2 x 2 average of the last, with
/// `srgb` its rgb averaged as linear values: in 8 bits throughout,
/// for images too large for floats (a 21600 x 10800 Earth is 3.7 GB
/// of them, its RGBA8 mips 1.2 GB). Full size first.
/// @throws std::runtime_error if it cannot be read, or is not an
/// 8-bit PNG or JPEG.
std::vector<std::vector<std::uint8_t>>
LoadRgba8Mips(const std::filesystem::path& file, bool srgb);

/// Bakes the PNG, JPEG or EXR image `input` into a DDS file `output`
/// for Ernie's textures: every mip down to 1 x 1, filtered as its
/// role says before compression, stored as `options` says; what
/// virtual textures need (`ernie:virtualTexture`), and a quarter to
/// an eighth of the memory of uncompressed RGBA8. Returns the stored
/// texture.
///
/// 8-bit PNG and JPEG colour and data bake in 8 bits, each mip made
/// from the last and dropped once encoded (about 4 bytes a texel of
/// the image beyond the output); normal maps, EXR and 16-bit images
/// through floats (16 bytes a texel for every mip).
///
/// References:
/// - R. Geldreich, "bc7enc_rdo" (bc7enc and rgbcx).
///   https://github.com/richgel999/bc7enc_rdo
/// @throws std::runtime_error if the input cannot be read or the
/// output written; std::invalid_argument as `EncodeTexture`.
TextureData BakeTexture(const std::filesystem::path& input,
                        const std::filesystem::path& output,
                        const BakeOptions& options = {});

/// As `BakeTexture`, for an image too large to hold whole: a mosaic
/// of `tiles`, 8-bit PNG or JPEG images of one size, `columns` to a
/// row, rows top to bottom (a global mosaic's tiles, such as Blue
/// Marble's eight 21600 x 21600 ones). Each tile is read and its top
/// mips encoded into the file in turn, as far as they halve exactly
/// and split into whole blocks at the tiles' edges; the smaller mips
/// come from those tiles' last mips joined. So at most two tiles'
/// texels (one decoding as the other encodes) and the joined mips are
/// held. Normal maps are not baked as mosaics. Returns the stored
/// texture's layout, its texels in the file only.
/// @throws std::invalid_argument for tiles that are not whole rows or
/// not of one size, or normal maps; std::runtime_error if a tile
/// cannot be read or the output written.
TextureData BakeMosaic(std::span<const std::filesystem::path> tiles,
                       std::uint32_t columns,
                       const std::filesystem::path& output,
                       const BakeOptions& options = {});

}  // namespace ernie
