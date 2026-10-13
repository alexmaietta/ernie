// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/raster_source.h"

#include "ernie/util/pds.h"
#include "ernie/util/tiff.h"

#include <stb_image.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ernie {

namespace {

/// An 8- or 16-bit PNG or JPEG, its header read at once and its
/// pixels decoded whole into memory at the first read.
class DecodedRaster : public RasterSource {
  public:
    explicit DecodedRaster(const std::filesystem::path& path)
        : name_(path.string()) {
        int width = 0;
        int height = 0;
        int channels = 0;
        if (stbi_info(name_.c_str(), &width, &height, &channels) ==
            0) {
            throw std::runtime_error("Cannot read " + name_ + ": " +
                                     stbi_failure_reason());
        }
        if (stbi_is_16_bit(name_.c_str()) != 0) {
            type_ = RasterSampleType::UInt16;
        }
        width_ = static_cast<std::uint32_t>(width);
        height_ = static_cast<std::uint32_t>(height);
        channels_ = static_cast<std::uint32_t>(channels);
    }

    std::uint32_t Width() const override {
        return width_;
    }
    std::uint32_t Height() const override {
        return height_;
    }
    std::uint32_t Channels() const override {
        return channels_;
    }
    RasterSampleType Type() const override {
        return type_;
    }

    void Read(std::uint32_t x, std::uint32_t y, std::uint32_t w,
              std::uint32_t h,
              std::span<std::byte> out) const override {
        if (x > width_ || w > width_ - x || y > height_ ||
            h > height_ - y) {
            throw std::out_of_range("A rectangle past the image");
        }
        const std::size_t pixel =
            std::size_t{channels_} * SampleBytes();
        if (out.size() < std::size_t{w} * h * pixel) {
            throw std::out_of_range(
                "Too little room for the rectangle");
        }
        if (pixels_ == nullptr) {
            Decode();
        }
        const auto* bytes =
            static_cast<const std::byte*>(pixels_.get());
        for (std::uint32_t row = 0; row < h; ++row) {
            std::memcpy(out.data() + std::size_t{row} * w * pixel,
                        bytes + (std::size_t{y + row} * width_ + x) *
                                    pixel,
                        std::size_t{w} * pixel);
        }
    }

    void Release() const override {
        pixels_.reset();
    }

  private:
    void Decode() const {
        int width = 0;
        int height = 0;
        int channels = 0;
        void* data =
            type_ == RasterSampleType::UInt16
                ? static_cast<void*>(stbi_load_16(
                      name_.c_str(), &width, &height, &channels, 0))
                : static_cast<void*>(stbi_load(
                      name_.c_str(), &width, &height, &channels, 0));
        if (data == nullptr) {
            throw std::runtime_error("Cannot read " + name_ + ": " +
                                     stbi_failure_reason());
        }
        pixels_.reset(data);
        if (static_cast<std::uint32_t>(width) != width_ ||
            static_cast<std::uint32_t>(height) != height_ ||
            static_cast<std::uint32_t>(channels) != channels_) {
            pixels_.reset();
            throw std::runtime_error(name_ +
                                     " changed as it was read");
        }
    }

    struct Free {
        void operator()(void* data) const {
            stbi_image_free(data);
        }
    };

    std::string name_;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    std::uint32_t channels_ = 0;
    RasterSampleType type_ = RasterSampleType::UInt8;
    mutable std::unique_ptr<void, Free> pixels_;
};

}  // namespace

std::uint32_t SampleBytesOf(RasterSampleType type) {
    switch (type) {
    case RasterSampleType::UInt8:
        return 1;
    case RasterSampleType::UInt16:
    case RasterSampleType::Int16:
        return 2;
    case RasterSampleType::UInt32:
    case RasterSampleType::Int32:
    case RasterSampleType::Float32:
        break;
    }
    return 4;
}

std::vector<std::byte> RasterSource::ReadAll() const {
    std::vector<std::byte> out(std::size_t{Width()} * Height() *
                               Channels() * SampleBytes());
    Read(0, 0, Width(), Height(), out);
    return out;
}

MosaicRaster::MosaicRaster(
    const std::vector<std::filesystem::path>& tiles,
    std::uint32_t columns)
    : columns_(columns) {
    if (columns == 0 || tiles.empty() ||
        tiles.size() % columns != 0) {
        throw std::invalid_argument("A mosaic needs whole rows of " +
                                    std::to_string(columns) +
                                    " tiles");
    }
    rows_ = static_cast<std::uint32_t>(tiles.size() / columns);
    for (const std::filesystem::path& tile : tiles) {
        tiles_.push_back(OpenRaster(tile));
        const RasterSource& first = *tiles_.front();
        const RasterSource& last = *tiles_.back();
        if (last.Width() != first.Width() ||
            last.Height() != first.Height() ||
            last.Channels() != first.Channels() ||
            last.Type() != first.Type()) {
            throw std::invalid_argument(
                "A mosaic's tiles are alike: " + tile.string() +
                " is not as " + tiles.front().string());
        }
    }
    tileWidth_ = tiles_.front()->Width();
    tileHeight_ = tiles_.front()->Height();
}

std::vector<RasterRect> MosaicRaster::Parts() const {
    std::vector<RasterRect> parts;
    for (std::uint32_t row = 0; row < rows_; ++row) {
        for (std::uint32_t column = 0; column < columns_; ++column) {
            parts.push_back({column * tileWidth_, row * tileHeight_,
                             tileWidth_, tileHeight_});
        }
    }
    return parts;
}

void MosaicRaster::Release() const {
    for (const std::unique_ptr<RasterSource>& tile : tiles_) {
        tile->Release();
    }
}

void MosaicRaster::Read(std::uint32_t x, std::uint32_t y,
                        std::uint32_t w, std::uint32_t h,
                        std::span<std::byte> out) const {
    if (x > Width() || w > Width() - x || y > Height() ||
        h > Height() - y) {
        throw std::out_of_range("A rectangle past the mosaic");
    }
    const std::size_t pixel = std::size_t{Channels()} * SampleBytes();
    if (out.size() < std::size_t{w} * h * pixel) {
        throw std::out_of_range("Too little room for the rectangle");
    }
    std::vector<std::byte> part;
    for (std::uint32_t row = y / tileHeight_;
         row < rows_ && row * tileHeight_ < y + h; ++row) {
        for (std::uint32_t column = x / tileWidth_;
             column < columns_ && column * tileWidth_ < x + w;
             ++column) {
            const std::size_t index =
                std::size_t{row} * columns_ + column;
            if (index != current_) {
                tiles_[current_]->Release();
                current_ = index;
            }
            // The rectangle's overlap with the tile, in the tile.
            const std::uint32_t left =
                std::max(x, column * tileWidth_);
            const std::uint32_t top = std::max(y, row * tileHeight_);
            const std::uint32_t right =
                std::min(x + w, (column + 1) * tileWidth_);
            const std::uint32_t bottom =
                std::min(y + h, (row + 1) * tileHeight_);
            const std::uint32_t across = right - left;
            part.resize(std::size_t{across} * (bottom - top) * pixel);
            tiles_[index]->Read(left - column * tileWidth_,
                                top - row * tileHeight_, across,
                                bottom - top, part);
            for (std::uint32_t r = top; r < bottom; ++r) {
                std::memcpy(
                    out.data() +
                        (std::size_t{r - y} * w + (left - x)) * pixel,
                    part.data() +
                        std::size_t{r - top} * across * pixel,
                    std::size_t{across} * pixel);
            }
        }
    }
}

std::unique_ptr<MosaicRaster>
OpenMosaic(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot read " + path.string());
    }
    std::vector<std::filesystem::path> tiles;
    std::size_t columns = 0;
    std::string line;
    while (std::getline(file, line)) {
        if (const std::size_t comment = line.find('#');
            comment != std::string::npos) {
            line.resize(comment);
        }
        std::istringstream names(line);
        std::size_t count = 0;
        for (std::string name; names >> name; ++count) {
            tiles.push_back(path.parent_path() / name);
        }
        if (count == 0) {
            continue;
        }
        if (columns != 0 && count != columns) {
            throw std::invalid_argument(
                path.string() + ": a row of " +
                std::to_string(count) + " tiles, not " +
                std::to_string(columns));
        }
        columns = count;
    }
    return std::make_unique<MosaicRaster>(
        tiles, static_cast<std::uint32_t>(columns));
}

std::unique_ptr<RasterSource>
OpenRaster(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(),
                   extension.begin(), [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    if (extension == ".tif" || extension == ".tiff") {
        return std::make_unique<TiffImage>(path);
    }
    if (extension == ".img" || extension == ".lbl") {
        return std::make_unique<PdsImage>(path);
    }
    if (extension == ".mosaic") {
        return OpenMosaic(path);
    }
    if (extension == ".png" || extension == ".jpg" ||
        extension == ".jpeg") {
        return std::make_unique<DecodedRaster>(path);
    }
    throw std::runtime_error(
        path.string() + " is not a raster read here (TIFF, PDS, PNG, "
                        "JPEG or a mosaic of them)");
}

}  // namespace ernie
