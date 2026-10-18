// SPDX-License-Identifier: BSD-3-Clause
/// @file texture.h
/// @brief Stored textures: formats, DDS and KTX2 reading, BC decoding, baking images into mipped,
/// block-compressed DDS files, and a disk cache of baked textures.
#pragma once

#include "ernie/export.h"

#include "ernie/io/image.h"
#include "ernie/io/raster.h"
#include "ernie/util/file.h"

#include "ernie/util/glm.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace ernie {

// ----------------------------------------------------------------------------------------------
// Formats

/// @brief A stored texture's texel layout.
/// Plain texels, or BC1-BC7 blocks of 4 x 4 uploaded as they are.
enum class TextureFormat : std::uint32_t {
    R8,
    Rg8,
    Rgba8,
    Rgba8Srgb,
    Bgra8,
    Bgra8Srgb,
    Rgba16F,
    Rgba32F,
    /// With one-bit alpha (DXT1).
    Bc1,
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

/// @brief What a `TextureFormat` stores.
struct TextureFormatInfo {
    /// Texels along a block's side; 1 when uncompressed.
    std::uint32_t blockSize = 1;
    /// Bytes per block (or texel).
    std::uint32_t blockBytes = 4;
    /// Channels it decodes to.
    std::uint32_t channels = 4;
    /// Whether colour decodes from sRGB.
    bool srgb = false;
    /// @return Whether the format is block-compressed.
    bool compressed() const {
        return blockSize > 1;
    }
};

/// @brief Describes a texture format.
/// @param format The format.
/// @return Its block size, bytes, channels and sRGB flag.
ERNIE_API TextureFormatInfo FormatInfo(TextureFormat format);

/// @brief Switches sRGB decoding on or off where a format has both variants (RGBA8, BGRA8, BC1,
/// BC2, BC3, BC7).
/// @param format The format.
/// @param srgb   Whether to decode from sRGB.
/// @return The variant, or `format` unchanged.
ERNIE_API TextureFormat WithSrgb(TextureFormat format, bool srgb);

/// @brief Bytes of an image in whole blocks.
/// @param format The format.
/// @param width  Width, in texels.
/// @param height Height, in texels.
/// @return The size in bytes.
ERNIE_API std::size_t ImageBytes(TextureFormat format, std::uint32_t width, std::uint32_t height);

/// @brief Mips of a full chain, down to 1 x 1.
/// @param width  Width of mip 0, in texels.
/// @param height Height of mip 0, in texels.
/// @return The number of mips.
ERNIE_API std::uint32_t MipCount(std::uint32_t width, std::uint32_t height);

/// @brief A side of a mip.
/// @param side The side at mip 0, in texels.
/// @param mip  The mip.
/// @return The side halved `mip` times, at least 1.
constexpr std::uint32_t MipSide(std::uint32_t side, std::uint32_t mip) {
    return side >> mip > 0 ? side >> mip : 1u;
}

// ----------------------------------------------------------------------------------------------
// Stored textures

/// @brief A texture as a DDS or KTX2 file stores it, not decoded.
/// Each layer holds `mips` levels, halving down from full size; a cube map's six faces are layers.
struct ERNIE_API TextureData {
    /// The texel format.
    TextureFormat format = TextureFormat::Rgba8;
    /// Width of mip 0, in texels.
    std::uint32_t width = 0;
    /// Height of mip 0, in texels.
    std::uint32_t height = 0;
    /// Array layers.
    std::uint32_t layers = 1;
    /// Mips per layer.
    std::uint32_t mips = 1;
    /// Whether the layers are cube faces.
    bool cube = false;
    /// The owned texels; empty when read in place from `mapping`.
    std::vector<std::byte> bytes;
    /// The mapped file holding the texels; null when in `bytes`.
    std::shared_ptr<const MappedFile> mapping;
    /// The texels within `mapping`.
    std::span<const std::byte> mapped;
    /// Where each mip starts in `Data()`, at `layer * mips + mip`.
    std::vector<std::size_t> offsets;

    /// @return The texels: the mapped file's, else `bytes`.
    std::span<const std::byte> Data() const {
        return mapping != nullptr ? mapped : std::span<const std::byte>(bytes);
    }

    /// @brief One mip of one layer.
    /// @param layer The layer.
    /// @param mip   The mip.
    /// @return Its bytes.
    /// @throws std::out_of_range If the layer or mip does not exist.
    std::span<const std::byte> Mip(std::uint32_t layer, std::uint32_t mip) const;
};

/// @brief Tests for a .dds or .ktx2 extension, in any case.
/// @param path The file.
/// @return Whether `LoadTextureFile` reads it.
ERNIE_API bool IsTextureFile(const std::filesystem::path& path);

/// @brief Reads a DDS or KTX2 file as stored, by extension.
/// The file is mapped and read in place, so texels load from disk only as used; supercompressed
/// KTX2 levels decode into memory. DDS: DXGI formats, legacy DXTn/ATIn/BC4/BC5 codes and 32-bit
/// RGBA or BGRA; 2D, arrays and cubes, not volumes. KTX2: the same formats, stored or zstd; Basis
/// ETC1S and UASTC transcode to BC5 for two channels (`toktx --normal_mode`), BC6H for HDR, else
/// BC7.
/// @param path The file.
/// @return The texture.
/// @throws std::runtime_error If the file cannot be read, is malformed or truncated, or has an
///         unsupported format.
///
/// References:
/// - Microsoft, "DDS programming guide" (DDS_HEADER, DDS_HEADER_DXT10).
///   https://learn.microsoft.com/en-us/windows/win32/direct3ddds/dx-graphics-dds-pguide
/// - Khronos, "KTX File Format Specification", version 2.0.
///   https://registry.khronos.org/KTX/specs/2.0/ktxspec.v2.html
/// - Binomial, "Basis Universal" (the transcoder). https://github.com/BinomialLLC/basis_universal
ERNIE_API TextureData LoadTextureFile(const std::filesystem::path& path);

/// @brief Parses a DDS file's contents (`LoadTextureFile`).
/// @param file The bytes.
/// @return The texture, owning its texels.
ERNIE_API TextureData ReadDds(std::span<const std::byte> file);

/// @brief Parses a KTX2 file's contents (`LoadTextureFile`).
/// @param file The bytes.
/// @return The texture, owning its texels.
ERNIE_API TextureData ReadKtx2(std::span<const std::byte> file);

/// @brief Writes a DDS file with the DX10 header (`ReadDds` reads it back).
/// @param path    The file.
/// @param texture Every layer's mips.
/// @throws std::runtime_error If the file cannot be written.
ERNIE_API void WriteDdsFile(const std::filesystem::path& path, const TextureData& texture);

/// @brief Decodes BC blocks for devices that cannot sample them.
/// BC1/BC3/BC7 become RGBA8 (sRGB kept), BC4 R8 and BC5 RG8, every layer and mip.
/// @param texture The texture.
/// @return The decoded texture; uncompressed ones unchanged.
/// @throws std::invalid_argument For BC2, BC6H and signed BC4/BC5.
///
/// References:
/// - R. Geldreich, "bc7enc_rdo" (rgbcx's BC1 to BC5 decoders, bc7decomp's BC7), MIT or public
///   domain. https://github.com/richgel999/bc7enc_rdo
ERNIE_API TextureData DecodeBlocks(const TextureData& texture);

// ----------------------------------------------------------------------------------------------
// Baking

/// @brief What a texture holds, deciding how it is mipped and stored.
enum class TextureRole {
    /// Colour, sRGB unless said otherwise; BC7.
    Colour,
    /// Tangent-space normals as n / 2 + 1/2, mipped as `MipFilter::NormalSlopes`, stored as x and
    /// y (BC5); shaders rebuild z.
    Normal,
    /// Linear values such as roughness or metallic; BC7.
    Data,
};

/// @brief Encodes RGBA8 images into BC blocks off the CPU.
/// Bakes use it for the formats it handles (`GpuBlockEncoder`), the CPU encoders for the rest.
class ERNIE_API BlockEncoder {
  public:
    virtual ~BlockEncoder() = default;

    /// @param format A block format.
    /// @return Whether this encoder handles it.
    virtual bool Encodes(TextureFormat format) const = 0;

    /// @brief Encodes an image; edges repeat the last row and column.
    /// @param rgba   RGBA8 texels, rows top to bottom.
    /// @param width  Width, in texels.
    /// @param height Height, in texels.
    /// @param format The block format.
    /// @param out    `ImageBytes(format, width, height)` bytes.
    virtual void Encode(std::span<const std::uint8_t> rgba, std::uint32_t width,
                        std::uint32_t height, TextureFormat format, std::byte* out) = 0;
};

/// @brief How `BakeTexture` stores a texture.
struct BakeOptions {
    /// What the texture holds.
    TextureRole role = TextureRole::Colour;
    /// The stored format; the role's by default. BC1, BC3, BC4, BC5, BC7 and RGBA8 are written;
    /// BC6H is not.
    std::optional<TextureFormat> format;
    /// Whether colour is stored as sRGB rather than linear.
    bool srgb = true;
    /// BC7 effort on the CPU, 0 (fastest) to 4.
    int quality = 1;
    /// An encoder used where it can, such as on the GPU; null for the CPU's alone.
    BlockEncoder* encoder = nullptr;
};

/// @brief Encodes a mip chain into a stored format.
/// BC blocks past a mip's edge repeat its last row and column.
/// @param mips     Levels as `MipChain` makes them, full size first.
/// @param format   The stored format.
/// @param encoding How 8-bit values are encoded.
/// @param quality  BC7 effort, 0 to 4.
/// @param encoder  Used where it handles `format`; else all CPU threads.
/// @return The texture.
/// @throws std::invalid_argument For an unsupported format, or mips that do not halve.
ERNIE_API TextureData EncodeTexture(std::span<const Image> mips, TextureFormat format,
                                    ImageEncoding encoding, int quality = 1,
                                    BlockEncoder* encoder = nullptr);

/// @brief Reads an 8-bit PNG or JPEG's size from its header.
/// @param file The image.
/// @return Width and height; none for other or unreadable files.
ERNIE_API std::optional<glm::uvec2> Rgba8ImageSize(const std::filesystem::path& file);

/// @brief Decodes an 8-bit PNG or JPEG and mips it in 8 bits.
/// For images too large for floats (a 21600 x 10800 Earth is 3.7 GB as floats, 1.2 GB as RGBA8
/// mips). Each mip is the 2 x 2 average of the last.
/// @param file The image.
/// @param srgb Whether rgb is averaged as linear light.
/// @return RGBA8 levels down to 1 x 1, full size first.
/// @throws std::runtime_error If it cannot be read or is not an 8-bit PNG or JPEG.
ERNIE_API std::vector<std::vector<std::uint8_t>> LoadRgba8Mips(const std::filesystem::path& file,
                                                               bool srgb);

/// @brief Bakes an image into a mipped DDS texture.
/// Mips are filtered by role, then compressed: what virtual textures need, at a quarter to an
/// eighth of RGBA8's memory. 8-bit PNG and JPEG bake in 8 bits; normals, EXR and 16-bit go
/// through floats. TIFF, PDS3 and `.mosaic` bake in bands (`RasterBake`) and return mapped from
/// `output`.
/// @param input   The source image.
/// @param output  The DDS file to write.
/// @param options How to store it.
/// @return The stored texture.
/// @throws std::runtime_error If the input cannot be read or the output written.
/// @throws std::invalid_argument As `EncodeTexture`.
///
/// References:
/// - R. Geldreich, "bc7enc_rdo" (bc7enc and rgbcx). https://github.com/richgel999/bc7enc_rdo
ERNIE_API TextureData BakeTexture(const std::filesystem::path& input,
                                  const std::filesystem::path& output,
                                  const BakeOptions& options = {});

/// @brief Bakes a colour or data raster into DDS a band at a time.
/// For images larger than memory, JPEG's limit or one file, and to spread a bake over frames. A
/// worker reads 1024-row bands and mips them exactly; `Step` encodes ready work on the caller's
/// thread. At most about 256 MB of band mips wait. Samples are 8- or 16-bit (high byte kept); 1
/// channel is grey, 2 grey and alpha, 3 RGB.
class ERNIE_API RasterBake {
  public:
    /// @brief Starts a bake and writes the file's header.
    /// @param source  The raster.
    /// @param output  The DDS file.
    /// @param options How to store it.
    /// @throws std::invalid_argument For normal maps or samples not 8- or 16-bit.
    /// @throws std::runtime_error If `output` cannot be written.
    RasterBake(std::unique_ptr<GridSource> source, const std::filesystem::path& output,
               const BakeOptions& options = {});
    /// @brief Stops the bake; an unfinished file is removed.
    ~RasterBake();
    RasterBake(const RasterBake&) = delete;
    RasterBake& operator=(const RasterBake&) = delete;

    /// @return The stored texture's layout; texels are in the file.
    const TextureData& Layout() const;

    /// @brief Encodes and writes what the reader has ready.
    /// @param texels About how many texels to encode; at least a band's row of blocks.
    /// @param wait   Whether to wait for the reader.
    /// @return True once the file is complete.
    /// @throws std::exception Whatever reading or writing threw.
    bool Step(std::uint64_t texels, bool wait = false);

    /// @brief Encodes everything left, waiting for the reader.
    /// @return The stored texture's layout.
    const TextureData& Finish();

    /// @return Texels encoded so far, over every mip.
    std::uint64_t Encoded() const;
    /// @return Texels to encode in all.
    std::uint64_t Total() const;

  private:
    struct State;
    std::unique_ptr<State> state_;
};

/// @brief Bakes a mosaic of 8-bit PNG or JPEG tiles, as `BakeTexture`.
/// Each tile's top mips are encoded in turn; smaller mips come from the tiles' last mips joined.
/// At most two tiles are held at once. Normal maps are not supported.
/// @param tiles   Tiles of one size, row by row, top to bottom.
/// @param columns Tiles per row.
/// @param output  The DDS file.
/// @param options How to store it.
/// @return The stored layout; texels are in the file.
/// @throws std::invalid_argument For tiles not filling whole rows or of different sizes, or normal
///         maps.
/// @throws std::runtime_error If a tile cannot be read or the output written.
ERNIE_API TextureData BakeMosaic(std::span<const std::filesystem::path> tiles,
                                 std::uint32_t columns, const std::filesystem::path& output,
                                 const BakeOptions& options = {});

// ----------------------------------------------------------------------------------------------
// Cache

/// @brief Keeps prepared textures between runs as DDS files.
/// A texture is decoded, mipped and encoded once until its file or read settings change. Least
/// recently used entries go past the capacity. Any thread may read; I/O failures count as misses.
class ERNIE_API TextureCache {
  public:
    /// @brief The default folder, from `ERNIE_TEXTURE_CACHE` (empty or "0" for none) or
    /// `<user cache>/ernie/textures`.
    /// @return The folder, or nothing when disabled.
    static std::optional<std::filesystem::path> DefaultDirectory();

    /// @brief Opens a cache.
    /// @param directory The folder; none disables the cache.
    /// @param capacity  The size limit in bytes; 16 GB by default (a global Blue Marble mosaic is
    ///                  5 GB of BC7).
    explicit TextureCache(std::optional<std::filesystem::path> directory = DefaultDirectory(),
                          std::uintmax_t capacity = std::uintmax_t{16} << 30);

    /// @return Whether the cache has a folder.
    bool Enabled() const {
        return directory_.has_value();
    }

    /// @brief Reads a cached texture and marks it as just used.
    /// @param key The entry's key.
    /// @return The texture, mapped as by `LoadTextureFile`; none if absent or unreadable.
    std::optional<TextureData> Read(std::string_view key) const;

    /// @brief Caches a texture atomically, then prunes.
    /// @param key     The entry's key.
    /// @param texture The texture.
    void Write(std::string_view key, const TextureData& texture) const;

    /// @brief A new file for a DDS written in parts (`RasterBake`), then cached by `Publish`.
    /// @param key The entry's key.
    /// @return The path, or none without a cache.
    std::optional<std::filesystem::path> PartialPath(std::string_view key) const;

    /// @brief Moves a complete DDS file into the cache, then prunes others.
    /// @param partial The file from `PartialPath`.
    /// @param key     The entry's key.
    /// @return False if it could not be moved; it is then removed.
    bool Publish(const std::filesystem::path& partial, std::string_view key) const;

    /// @brief Forgets a cached texture, if any.
    /// @param key The entry's key.
    void Remove(std::string_view key) const;

  private:
    std::filesystem::path EntryPath(std::string_view key) const;
    /// Removes the least recently used past the capacity, but not `kept`.
    void Prune(const std::filesystem::path& kept) const;

    /// The folder, if enabled.
    std::optional<std::filesystem::path> directory_;
    /// The size limit, in bytes.
    std::uintmax_t capacity_;
};

}  // namespace ernie
