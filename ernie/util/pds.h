// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/util/mapped_file.h"
#include "ernie/util/raster_source.h"

#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>

namespace ernie {

/// A PDS3 image read in place from its mapped file: NASA's Planetary
/// Data System's elevation models and mosaics (LOLA's and MOLA's
/// gridded heights among them), their label attached at the file's
/// start or in a .lbl file beside it.
///
/// One band of 8-, 16- or 32-bit integers (signed or not) or IEEE
/// floats, in either byte order; SCALING_FACTOR and OFFSET to
/// physical values; MISSING_CONSTANT as no data; and a simple
/// cylindrical (equirectangular) map projection's grid.
///
/// References:
/// - NASA PDS, "Planetary Data System Standards Reference" 3.8, 2009
///   (the label's syntax, chapter 12; IMAGE objects, appendix A.21;
///   IMAGE_MAP_PROJECTION, appendix A.22).
/// - LOLA team, "LRO LOLA Gridded Data Record" (LDEM_*.LBL).
class PdsImage : public RasterSource {
  public:
    /// The image of label or data file `path` (the label's ^IMAGE
    /// names the data beside it).
    /// @throws std::runtime_error if the label cannot be found or
    /// read, or describes samples it does not read.
    explicit PdsImage(const std::filesystem::path& path);

    std::uint32_t Width() const override {
        return width_;
    }
    std::uint32_t Height() const override {
        return height_;
    }
    std::uint32_t Channels() const override {
        return 1;
    }
    RasterSampleType Type() const override {
        return type_;
    }
    double Scale() const override {
        return scale_;
    }
    double Offset() const override {
        return offset_;
    }
    std::optional<double> NoData() const override {
        return noData_;
    }
    std::optional<LatLongGrid> Grid() const override {
        return grid_;
    }

    void Read(std::uint32_t x, std::uint32_t y, std::uint32_t w,
              std::uint32_t h,
              std::span<std::byte> out) const override;

    /// The label's value for `key` (the first if several objects
    /// name it), quotes and units removed; none if absent.
    std::optional<std::string> Label(const std::string& key) const;

  private:
    std::map<std::string, std::string> label_;
    std::shared_ptr<const MappedFile> file_;
    std::uint64_t start_ = 0;  ///< The first sample's byte.
    bool bigEndian_ = false;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    RasterSampleType type_ = RasterSampleType::UInt8;
    double scale_ = 1.0;
    double offset_ = 0.0;
    std::optional<double> noData_;
    std::optional<LatLongGrid> grid_;
};

}  // namespace ernie
