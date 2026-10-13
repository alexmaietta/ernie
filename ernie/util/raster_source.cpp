// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/raster_source.h"

#include "ernie/util/pds.h"
#include "ernie/util/tiff.h"

#include <stb_image.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ernie {

namespace {

/// An 8- or 16-bit PNG or JPEG, decoded whole into memory.
class DecodedRaster : public RasterSource {
  public:
    explicit DecodedRaster(const std::filesystem::path& path) {
        const std::string name = path.string();
        int width = 0;
        int height = 0;
        int channels = 0;
        void* data = nullptr;
        if (stbi_is_16_bit(name.c_str()) != 0) {
            data = stbi_load_16(name.c_str(), &width, &height,
                                &channels, 0);
            type_ = RasterSampleType::UInt16;
        } else {
            data = stbi_load(name.c_str(), &width, &height, &channels,
                             0);
        }
        if (data == nullptr) {
            throw std::runtime_error("Cannot read " + name + ": " +
                                     stbi_failure_reason());
        }
        width_ = static_cast<std::uint32_t>(width);
        height_ = static_cast<std::uint32_t>(height);
        channels_ = static_cast<std::uint32_t>(channels);
        const auto* bytes = static_cast<const std::byte*>(data);
        pixels_.assign(bytes, bytes + std::size_t{width_} * height_ *
                                          channels_ * SampleBytes());
        stbi_image_free(data);
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
        for (std::uint32_t row = 0; row < h; ++row) {
            std::memcpy(out.data() + std::size_t{row} * w * pixel,
                        pixels_.data() +
                            (std::size_t{y + row} * width_ + x) *
                                pixel,
                        std::size_t{w} * pixel);
        }
    }

  private:
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    std::uint32_t channels_ = 0;
    RasterSampleType type_ = RasterSampleType::UInt8;
    std::vector<std::byte> pixels_;
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
    if (extension == ".png" || extension == ".jpg" ||
        extension == ".jpeg") {
        return std::make_unique<DecodedRaster>(path);
    }
    throw std::runtime_error(
        path.string() + " is not a raster read here (TIFF, PDS, PNG "
                        "or JPEG)");
}

}  // namespace ernie
