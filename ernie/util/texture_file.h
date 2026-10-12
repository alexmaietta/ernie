// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace ernie {

/// How a stored texture's texels are laid out: plain texels, or BC1
/// to BC7 blocks of 4 x 4 texels, uploaded as they are.
enum class TextureFormat : std::uint32_t {
    R8,
    Rg8,
    Rgba8,
    Rgba8Srgb,
    Bgra8,
    Bgra8Srgb,
    Rgba16F,
    Rgba32F,
    Bc1,  ///< With one-bit alpha (DXT1).
    Bc1Srgb,
    Bc2,
    Bc2Srgb,
    Bc3,
    Bc3Srgb,
    Bc4,
    Bc4Snorm,
    Bc5,
    Bc5Snorm,
    Bc6hUfloat,
    Bc6hSfloat,
    Bc7,
    Bc7Srgb,
};

/// What a `TextureFormat` stores.
struct TextureFormatInfo {
    std::uint32_t blockSize = 1;   ///< Texels along a block's side.
    std::uint32_t blockBytes = 4;  ///< Bytes of a block (or texel).
    std::uint32_t channels = 4;    ///< Channels it decodes to.
    bool srgb = false;             ///< Colour decoded from sRGB.
    bool compressed() const {
        return blockSize > 1;
    }
};

TextureFormatInfo InfoOf(TextureFormat format);

/// `format` with or without sRGB decoding, where it has both
/// variants (8-bit RGBA and BGRA, BC1, BC2, BC3 and BC7); others come
/// back as they are.
TextureFormat WithSrgb(TextureFormat format, bool srgb);

/// Bytes of a `width` x `height` image of `format`: whole blocks.
std::size_t ImageBytes(TextureFormat format, std::uint32_t width,
                       std::uint32_t height);

/// A texture as a DDS or KTX2 file stores it, not decoded: each
/// layer's mips of `format`, mip 0 full size and each next halved
/// (at least 1), with a cube map's six faces as layers.
struct TextureData {
    TextureFormat format = TextureFormat::Rgba8;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t layers = 1;
    std::uint32_t mips = 1;
    bool cube = false;
    std::vector<std::byte> bytes;
    /// Where each mip starts in `bytes`, layer by layer
    /// (`layer * mips + mip`).
    std::vector<std::size_t> offsets;

    /// The bytes of mip `mip` of layer `layer`.
    /// @throws std::out_of_range for a layer or mip it lacks.
    std::span<const std::byte> Mip(std::uint32_t layer,
                                   std::uint32_t mip) const;
};

/// Whether `path` names a file `LoadTextureFile` reads: .dds or
/// .ktx2, in any case.
bool IsTextureFile(const std::filesystem::path& path);

/// Reads a DDS or KTX2 file as stored (`TextureData`), by extension.
///
/// DDS: the DX10 header's DXGI formats, the legacy DXT1 to DXT5,
/// ATI1, ATI2, BC4U/S and BC5U/S codes, and 32-bit RGBA or BGRA
/// masks; 2D textures, arrays and cube maps, not volumes. KTX2: the
/// formats above by their VkFormat, stored or zstd-supercompressed;
/// and Basis Universal's ETC1S (BasisLZ) and UASTC, transcoded to BC
/// blocks: BC5 for two channels (a normal map's x and y, labelled
/// ETC1S RRR and GGG or UASTC RRRG or RG, as `toktx --normal_mode`
/// writes them), BC6H for HDR and BC7 otherwise.
///
/// References:
/// - Microsoft, "DDS programming guide" (DDS_HEADER,
/// DDS_HEADER_DXT10).
///   https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dx-graphics-dds-pguide
/// - Khronos, "KTX File Format Specification", version 2.0.
///   https://registry.khronos.org/KTX/specs/2.0/ktxspec.v2.html
/// - Binomial, "Basis Universal" (the transcoder).
///   https://github.com/BinomialLLC/basis_universal
/// @throws std::runtime_error if the file cannot be read, is
/// malformed or truncated, or holds a format or layout it does not
/// read.
TextureData LoadTextureFile(const std::filesystem::path& path);

/// `LoadTextureFile` of a file's contents.
TextureData ReadDds(std::span<const std::byte> file);
TextureData ReadKtx2(std::span<const std::byte> file);

}  // namespace ernie
