// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/util/mapped_file.h"
#include "ernie/util/raster_source.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ernie {

/// A TIFF or GeoTIFF image read in place from its mapped file: its
/// layout, and rectangles of its samples decoded on demand, so that
/// images larger than memory (planetary mosaics and elevation models,
/// gigapixel photographs) read a part at a time (`RasterSource`).
///
/// Classic TIFF and BigTIFF, either byte order; strips or tiles of
/// interleaved samples (PlanarConfiguration 1); 8-, 16- and 32-bit
/// unsigned or signed integers and 32-bit floats; uncompressed, LZW
/// or Deflate, with horizontal (2) or floating-point (3) prediction.
/// Of GeoTIFF, the pixel scale, tie point and GDAL's no-data value.
///
/// References:
/// - Adobe, "TIFF Revision 6.0", 1992 (the layout, LZW, predictor 2).
/// - Adobe, "Adobe Photoshop TIFF Technical Note 3", 2005 (Deflate,
///   the floating-point predictor).
/// - Aware Systems, "BigTIFF" (64-bit offsets).
///   https://www.awaresystems.be/imaging/tiff/bigtiff.html
/// - OGC, "GeoTIFF Standard" 1.1, 2019 (ModelPixelScale,
///   ModelTiepoint); GDAL's GDAL_NODATA tag (42113).
class TiffImage : public RasterSource {
  public:
    using SampleType = RasterSampleType;

    /// @throws std::runtime_error if the file cannot be read, is not
    /// a TIFF, or holds a layout or encoding it does not read.
    explicit TiffImage(const std::filesystem::path& path);

    std::uint32_t Width() const override {
        return width_;
    }
    std::uint32_t Height() const override {
        return height_;
    }
    std::uint32_t Channels() const override {
        return channels_;
    }
    SampleType Type() const override {
        return type_;
    }

    /// GeoTIFF's ModelPixelScale (x, y, z) and ModelTiepoint (raster
    /// i, j, k to model x, y, z), when present.
    std::optional<std::array<double, 3>> PixelScale() const {
        return pixelScale_;
    }
    std::optional<std::array<double, 6>> Tiepoint() const {
        return tiepoint_;
    }
    /// GDAL's no-data value, when present.
    std::optional<double> NoData() const override {
        return noData_;
    }
    /// From the tie point and pixel scale, when both are present and
    /// the model space is in degrees (a geographic grid).
    std::optional<LatLongGrid> Grid() const override;

    /// @throws std::out_of_range for a rectangle past the image;
    /// std::runtime_error for a damaged strip or tile.
    void Read(std::uint32_t x, std::uint32_t y, std::uint32_t w,
              std::uint32_t h,
              std::span<std::byte> out) const override;

  private:
    /// Strip or tile `chunk` decoded (predictor undone, host byte
    /// order): `chunkWidth_` by its rows; the last one decoded is
    /// kept for the next `Read`.
    const std::vector<std::byte>& Chunk(std::size_t chunk) const;

    std::shared_ptr<const MappedFile> file_;
    bool bigEndian_ = false;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    std::uint32_t channels_ = 1;
    std::uint32_t bits_ = 8;
    SampleType type_ = SampleType::UInt8;
    std::uint32_t compression_ = 1;
    std::uint32_t predictor_ = 1;
    /// Strips are `width_` wide and `chunkHeight_` rows; tiles
    /// `chunkWidth_` by `chunkHeight_`, `chunksAcross_` to a row.
    bool tiled_ = false;
    std::uint32_t chunkWidth_ = 0;
    std::uint32_t chunkHeight_ = 0;
    std::uint32_t chunksAcross_ = 1;
    std::vector<std::uint64_t> offsets_;
    std::vector<std::uint64_t> byteCounts_;
    std::optional<std::array<double, 3>> pixelScale_;
    std::optional<std::array<double, 6>> tiepoint_;
    std::optional<double> noData_;
    mutable std::size_t cachedChunk_ = ~std::size_t{0};
    mutable std::vector<std::byte> cached_;
};

}  // namespace ernie
