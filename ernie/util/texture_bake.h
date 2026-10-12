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

/// How `BakeTexture` stores a texture.
struct BakeOptions {
    TextureRole role = TextureRole::Colour;
    /// The stored format; by default the role's. BC1, BC3, BC4 (red),
    /// BC5 (red and green), BC7 and RGBA8 are written; BC6H is not.
    std::optional<TextureFormat> format;
    /// Colour stored as sRGB (the default) or linear.
    bool srgb = true;
    /// BC7's effort, 0 (fastest) to 4.
    int quality = 1;
};

/// `mips` (full size first, as `MipChain` makes them) stored in
/// `format`, 8-bit values encoded with `encoding`; BC blocks over a
/// mip's edge repeat its last row and column; `quality` is BC7's
/// effort, 0 to 4. Encodes on every hardware thread.
/// @throws std::invalid_argument for a format it does not write, or
/// mips that do not halve.
TextureData EncodeTexture(std::span<const Image> mips,
                          TextureFormat format,
                          ImageEncoding encoding, int quality = 1);

/// Writes `texture` as a DDS file with the DX10 header, every layer's
/// mips (`ReadDds` reads it back).
/// @throws std::runtime_error if the file cannot be written, or for a
/// format DDS has no code for here.
void WriteDdsFile(const std::filesystem::path& path,
                  const TextureData& texture);

/// Bakes the PNG, JPEG or EXR image `input` into a DDS file `output`
/// for Ernie's textures: every mip down to 1 x 1, filtered as its
/// role says before compression, stored as `options` says; what
/// virtual textures need (`ernie:virtualTexture`), and a quarter to
/// an eighth of the memory of uncompressed RGBA8. Returns the stored
/// texture.
///
/// References:
/// - R. Geldreich, "bc7enc_rdo" (bc7enc and rgbcx).
///   https://github.com/richgel999/bc7enc_rdo
/// @throws std::runtime_error if the input cannot be read or the
/// output written; std::invalid_argument as `EncodeTexture`.
TextureData BakeTexture(const std::filesystem::path& input,
                        const std::filesystem::path& output,
                        const BakeOptions& options = {});

}  // namespace ernie
