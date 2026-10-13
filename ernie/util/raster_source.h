// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ernie {

/// How a raster's samples are stored.
enum class RasterSampleType {
    UInt8,
    UInt16,
    Int16,
    UInt32,
    Int32,
    Float32,
};

/// Bytes of one sample of `type`.
std::uint32_t SampleBytesOf(RasterSampleType type);

/// Where a raster in simple cylindrical (equirectangular) projection
/// lies: degrees east of its west edge and north of its north edge,
/// and degrees per pixel across and down (positive).
struct LatLongGrid {
    double west = -180.0;
    double north = 90.0;
    double degreesAcross = 0.0;
    double degreesDown = 0.0;
};

/// A rectangle of a raster's pixels.
struct RasterRect {
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

/// A raster image read a rectangle at a time, whatever its file: so
/// that planetary mosaics, elevation models and gigapixel photographs
/// larger than memory are baked, resampled and streamed in parts.
/// Rasters are read by one thread at a time.
class RasterSource {
  public:
    virtual ~RasterSource() = default;

    virtual std::uint32_t Width() const = 0;
    virtual std::uint32_t Height() const = 0;
    /// Samples per pixel (1 grey or elevation, 3 RGB, 4 RGBA).
    virtual std::uint32_t Channels() const = 0;
    virtual RasterSampleType Type() const = 0;

    std::uint32_t SampleBytes() const {
        return SampleBytesOf(Type());
    }

    /// The samples of columns [x, x + w) and rows [y, y + h), rows
    /// top to bottom, interleaved, in the host's byte order: w * h *
    /// Channels() * SampleBytes() bytes at `out`.
    /// @throws std::out_of_range for a rectangle past the image;
    /// std::runtime_error for damaged data.
    virtual void Read(std::uint32_t x, std::uint32_t y,
                      std::uint32_t w, std::uint32_t h,
                      std::span<std::byte> out) const = 0;

    /// A sample's physical value is sample * Scale() + Offset()
    /// (PDS's SCALING_FACTOR and OFFSET; 1 and 0 for most images).
    virtual double Scale() const {
        return 1.0;
    }
    virtual double Offset() const {
        return 0.0;
    }
    /// The sample marking no data, when there is one.
    virtual std::optional<double> NoData() const {
        return std::nullopt;
    }
    /// Its place in latitude and longitude, when its file says.
    virtual std::optional<LatLongGrid> Grid() const {
        return std::nullopt;
    }

    /// Rectangles covering the image best read one after another,
    /// each whole before the next (a mosaic's tiles, each decoded
    /// whole); the whole image by default.
    virtual std::vector<RasterRect> Parts() const {
        return {RasterRect{0, 0, Width(), Height()}};
    }

    /// Lets go of what reading holds in memory (a decoded image), to
    /// be read again if asked.
    virtual void Release() const {}

    /// The whole image (`Read` of every pixel).
    std::vector<std::byte> ReadAll() const;
};

/// One raster of a grid of tiles of one size, type and channels,
/// `columns` to a row, rows top to bottom: a global mosaic too large
/// for one file (Blue Marble's eight 21600 x 21600 tiles). Its parts
/// are its tiles, read (decoded) one at a time; its grid, its first
/// tile's spread over the rest.
class MosaicRaster : public RasterSource {
  public:
    /// @throws std::invalid_argument for tiles that are not whole
    /// rows or not alike; what `OpenRaster` throws.
    MosaicRaster(const std::vector<std::filesystem::path>& tiles,
                 std::uint32_t columns);

    std::uint32_t Width() const override {
        return columns_ * tileWidth_;
    }
    std::uint32_t Height() const override {
        return rows_ * tileHeight_;
    }
    std::uint32_t Channels() const override {
        return tiles_.front()->Channels();
    }
    RasterSampleType Type() const override {
        return tiles_.front()->Type();
    }
    double Scale() const override {
        return tiles_.front()->Scale();
    }
    double Offset() const override {
        return tiles_.front()->Offset();
    }
    std::optional<double> NoData() const override {
        return tiles_.front()->NoData();
    }
    std::optional<LatLongGrid> Grid() const override {
        return tiles_.front()->Grid();
    }
    std::vector<RasterRect> Parts() const override;
    void Release() const override;

    void Read(std::uint32_t x, std::uint32_t y, std::uint32_t w,
              std::uint32_t h,
              std::span<std::byte> out) const override;

  private:
    std::vector<std::unique_ptr<RasterSource>> tiles_;
    std::uint32_t columns_ = 0;
    std::uint32_t rows_ = 0;
    std::uint32_t tileWidth_ = 0;
    std::uint32_t tileHeight_ = 0;
    /// The tile read last, released as another is read.
    mutable std::size_t current_ = 0;
};

/// The mosaic a `.mosaic` file describes: a row of tiles' file names
/// (beside it, or paths relative to it) to a line, top to bottom,
/// separated by spaces; `#` begins a comment.
/// @throws std::runtime_error if it cannot be read;
/// std::invalid_argument for rows of different lengths, or as
/// `MosaicRaster`.
std::unique_ptr<MosaicRaster>
OpenMosaic(const std::filesystem::path& path);

/// The raster at `path`, by extension: TIFF and GeoTIFF (.tif,
/// .tiff, read in place), PDS3 images (.img with a detached .lbl, or
/// the .lbl itself; read in place), 8- and 16-bit PNG and JPEG
/// (decoded whole into memory at the first read) and mosaics of them
/// (`.mosaic`, `OpenMosaic`).
/// @throws std::runtime_error if it cannot be read.
std::unique_ptr<RasterSource>
OpenRaster(const std::filesystem::path& path);

}  // namespace ernie
