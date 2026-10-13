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

    /// The whole image (`Read` of every pixel).
    std::vector<std::byte> ReadAll() const;
};

/// The raster at `path`, by extension: TIFF and GeoTIFF (.tif,
/// .tiff, read in place), PDS3 images (.img with a detached .lbl, or
/// the .lbl itself; read in place), and 8-bit PNG and JPEG (decoded
/// whole into memory).
/// @throws std::runtime_error if it cannot be read.
std::unique_ptr<RasterSource>
OpenRaster(const std::filesystem::path& path);

}  // namespace ernie
