// SPDX-License-Identifier: BSD-3-Clause
/// @file raster.cpp
/// @brief Raster reading: decoded images, mosaics, PDS3 labels and images, TIFF strips and tiles,
/// and planetary DEMs.
#include "ernie/io/raster.h"

#include "ernie/io/image.h"

#include "ernie/util/bytes.h"
#include "ernie/util/file.h"
#include "ernie/util/profile.h"
#include "ernie/util/text.h"

#include <stb_image.h>

#include <algorithm>
#include <bit>
#include <climits>
#include <cstring>
#include <fstream>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace ernie {

namespace fs = std::filesystem;

namespace {

// ----------------------------------------------------------------------------------------------
// Decoded images

/// An 8- or 16-bit PNG or JPEG, its header read at once and its pixels decoded whole into memory
/// at the first read.
class DecodedRaster : public GridSource {
  public:
    explicit DecodedRaster(const fs::path& path) : name_(path.string()) {
        int width = 0;
        int height = 0;
        int channels = 0;
        if (stbi_info(name_.c_str(), &width, &height, &channels) == 0) {
            throw std::runtime_error("Cannot read " + name_ + ": " + stbi_failure_reason());
        }
        if (stbi_is_16_bit(name_.c_str()) != 0) {
            type_ = GridSampleType::UInt16;
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
    GridSampleType Type() const override {
        return type_;
    }

    void Read(std::uint32_t x, std::uint32_t y, std::uint32_t w, std::uint32_t h,
              std::span<std::byte> out) const override {
        CheckRead(x, y, w, h, out);
        if (pixels_ == nullptr) {
            Decode();
        }
        const std::size_t pixel = std::size_t{channels_} * SampleBytes();
        const auto* bytes = static_cast<const std::byte*>(pixels_.get());
        for (std::uint32_t row = 0; row < h; ++row) {
            std::memcpy(out.data() + std::size_t{row} * w * pixel,
                        bytes + (std::size_t{y + row} * width_ + x) * pixel,
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
            type_ == GridSampleType::UInt16
                ? static_cast<void*>(stbi_load_16(name_.c_str(), &width, &height, &channels, 0))
                : static_cast<void*>(stbi_load(name_.c_str(), &width, &height, &channels, 0));
        if (data == nullptr) {
            throw std::runtime_error("Cannot read " + name_ + ": " + stbi_failure_reason());
        }
        pixels_.reset(data);
        if (static_cast<std::uint32_t>(width) != width_ ||
            static_cast<std::uint32_t>(height) != height_ ||
            static_cast<std::uint32_t>(channels) != channels_) {
            pixels_.reset();
            throw std::runtime_error(name_ + " changed as it was read");
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
    GridSampleType type_ = GridSampleType::UInt8;
    mutable std::unique_ptr<void, Free> pixels_;
};

// ----------------------------------------------------------------------------------------------
// PDS3 label syntax

/// `line` without its comments; `inComment` says whether one is open at its start and at its end.
std::string WithoutComments(const std::string& line, bool& inComment) {
    std::string text;
    for (std::size_t at = 0; at < line.size();) {
        if (inComment) {
            const std::size_t end = line.find("*/", at);
            if (end == std::string::npos) {
                break;
            }
            inComment = false;
            at = end + 2;
        } else if (line.compare(at, 2, "/*") == 0) {
            inComment = true;
            at += 2;
        } else {
            text.push_back(line[at++]);
        }
    }
    return text;
}

/// Whether `value` opens a quote, parenthesis or brace it does not close, so it goes on over the
/// next line.
bool Unclosed(const std::string& value) {
    int depth = 0;
    bool quoted = false;
    for (const char c : value) {
        if (c == '"') {
            quoted = !quoted;
        } else if (!quoted && (c == '(' || c == '{')) {
            ++depth;
        } else if (!quoted && (c == ')' || c == '}')) {
            --depth;
        }
    }
    return quoted || depth > 0;
}

// ----------------------------------------------------------------------------------------------
// TIFF layout

// TIFF tags read.
constexpr std::uint16_t kImageWidth = 256;
constexpr std::uint16_t kImageLength = 257;
constexpr std::uint16_t kBitsPerSample = 258;
constexpr std::uint16_t kCompression = 259;
constexpr std::uint16_t kStripOffsets = 273;
constexpr std::uint16_t kSamplesPerPixel = 277;
constexpr std::uint16_t kRowsPerStrip = 278;
constexpr std::uint16_t kStripByteCounts = 279;
constexpr std::uint16_t kPlanarConfiguration = 284;
constexpr std::uint16_t kPredictor = 317;
constexpr std::uint16_t kTileWidth = 322;
constexpr std::uint16_t kTileLength = 323;
constexpr std::uint16_t kTileOffsets = 324;
constexpr std::uint16_t kTileByteCounts = 325;
constexpr std::uint16_t kSampleFormat = 339;
constexpr std::uint16_t kModelPixelScale = 33550;
constexpr std::uint16_t kModelTiepoint = 33922;
constexpr std::uint16_t kGdalNoData = 42113;

/// The size of a TIFF field type: BYTE, ASCII, SHORT, LONG, ..., DOUBLE, LONG8; 0 if unknown.
std::size_t TypeBytes(std::uint16_t type) {
    switch (type) {
    case 1:  // BYTE
    case 2:  // ASCII
    case 6:  // SBYTE
    case 7:  // UNDEFINED
        return 1;
    case 3:  // SHORT
    case 8:  // SSHORT
        return 2;
    case 4:   // LONG
    case 9:   // SLONG
    case 11:  // FLOAT
        return 4;
    case 5:   // RATIONAL
    case 10:  // SRATIONAL
    case 12:  // DOUBLE
    case 16:  // LONG8
    case 17:  // SLONG8
        return 8;
    default:
        return 0;
    }
}

/// Reads a TIFF's integers in its byte order, checking every access against the file's end.
class TiffReader {
  public:
    TiffReader(std::span<const std::byte> bytes, bool bigEndian)
        : bytes_(bytes), bigEndian_(bigEndian) {}

    std::uint64_t Unsigned(std::uint64_t at, std::size_t size) const {
        if (at > bytes_.size() || size > bytes_.size() - at) {
            throw std::runtime_error("TIFF: read past the file's end");
        }
        std::uint64_t value = 0;
        for (std::size_t i = 0; i < size; ++i) {
            const auto byte =
                std::to_integer<std::uint64_t>(bytes_[at + (bigEndian_ ? i : size - 1 - i)]);
            value = (value << 8) | byte;
        }
        return value;
    }

    double Double(std::uint64_t at) const {
        return std::bit_cast<double>(Unsigned(at, 8));
    }

  private:
    std::span<const std::byte> bytes_;
    bool bigEndian_;
};

/// One IFD entry: its type, count and where its values are (inline in the entry or at an offset).
struct TiffField {
    std::uint16_t type = 0;
    std::uint64_t count = 0;
    std::uint64_t at = 0;
};

/// TIFF's LZW: 9- to 12-bit codes, most significant bit first, 256 clearing the table and 257
/// ending; widths grow a code early.
std::vector<std::byte> DecodeLzw(std::span<const std::byte> input, std::size_t expected) {
    constexpr std::uint32_t kClear = 256;
    constexpr std::uint32_t kEnd = 257;
    std::vector<std::byte> out;
    out.reserve(expected);
    std::array<std::uint16_t, 4096> prefix{};
    std::array<std::byte, 4096> suffix{};
    std::array<std::byte, 4096> first{};
    std::array<std::uint16_t, 4096> length{};
    for (std::uint32_t i = 0; i < 256; ++i) {
        suffix[i] = first[i] = static_cast<std::byte>(i);
        length[i] = 1;
    }
    std::uint64_t bit = 0;
    const std::uint64_t bits = std::uint64_t{input.size()} * 8;
    std::uint32_t width = 9;
    std::uint32_t next = 258;
    std::int32_t old = -1;
    const auto read = [&]() -> std::uint32_t {
        if (bit + width > bits) {
            return kEnd;
        }
        std::uint32_t code = 0;
        for (std::uint32_t i = 0; i < width; ++i, ++bit) {
            const auto byte = std::to_integer<std::uint32_t>(input[bit >> 3]);
            code = (code << 1) | ((byte >> (7 - (bit & 7))) & 1);
        }
        return code;
    };
    const auto emit = [&](std::uint32_t code) {
        const std::size_t at = out.size();
        out.resize(at + length[code]);
        for (std::uint32_t c = code, i = length[code]; i-- > 0;) {
            out[at + i] = suffix[c];
            c = prefix[c];
        }
    };
    const auto add = [&](std::uint32_t from, std::byte byte) {
        if (next < 4096) {
            prefix[next] = static_cast<std::uint16_t>(from);
            suffix[next] = byte;
            first[next] = first[from];
            length[next] = static_cast<std::uint16_t>(length[from] + 1);
            ++next;
        }
        if (next >= (1u << width) - 1 && width < 12) {
            ++width;
        }
    };
    for (std::uint32_t code = read(); code != kEnd; code = read()) {
        if (code == kClear) {
            width = 9;
            next = 258;
            old = -1;
            continue;
        }
        if (old < 0) {
            if (code > 255) {
                throw std::runtime_error("TIFF: damaged LZW data");
            }
            emit(code);
        } else if (code < next) {
            emit(code);
            add(static_cast<std::uint32_t>(old), first[code]);
        } else if (code == next) {
            add(static_cast<std::uint32_t>(old), first[static_cast<std::uint32_t>(old)]);
            emit(code);
        } else {
            throw std::runtime_error("TIFF: damaged LZW data");
        }
        old = static_cast<std::int32_t>(code);
        if (out.size() >= expected) {
            break;
        }
    }
    return out;
}

/// zlib-wrapped Deflate, by stb_image's decoder.
std::vector<std::byte> DecodeDeflate(std::span<const std::byte> input, std::size_t expected) {
    if (expected > static_cast<std::size_t>(INT_MAX) ||
        input.size() > static_cast<std::size_t>(INT_MAX)) {
        throw std::runtime_error("TIFF: a Deflate strip over 2 GB");
    }
    std::vector<std::byte> out(expected);
    const int size = stbi_zlib_decode_buffer(
        reinterpret_cast<char*>(out.data()), static_cast<int>(expected),
        reinterpret_cast<const char*>(input.data()), static_cast<int>(input.size()));
    if (size < 0) {
        throw std::runtime_error("TIFF: damaged Deflate data");
    }
    out.resize(static_cast<std::size_t>(size));
    return out;
}

/// Undoes predictor 3 in place: the bytes of floats, most significant first, differenced across
/// each row.
void UndoFloatingPointPredictor(std::vector<std::byte>& data, std::size_t rowBytes,
                                std::uint32_t rows, std::uint32_t channels,
                                std::uint32_t sampleBytes) {
    static_assert(std::endian::native == std::endian::little,
                  "TIFF floats are reordered for a little-endian host");
    std::vector<std::byte> row(rowBytes);
    const std::size_t samples = rowBytes / sampleBytes;
    for (std::uint32_t y = 0; y < rows; ++y) {
        std::byte* line = data.data() + y * rowBytes;
        for (std::size_t i = channels; i < rowBytes; ++i) {
            line[i] = static_cast<std::byte>(std::to_integer<unsigned>(line[i]) +
                                             std::to_integer<unsigned>(line[i - channels]));
        }
        for (std::size_t i = 0; i < samples; ++i) {
            for (std::uint32_t b = 0; b < sampleBytes; ++b) {
                row[i * sampleBytes + b] = line[(sampleBytes - 1 - b) * samples + i];
            }
        }
        std::memcpy(line, row.data(), rowBytes);
    }
}

/// Undoes predictor 2 in place: each sample the difference from its left neighbour's.
void UndoHorizontalPredictor(std::vector<std::byte>& data, std::size_t rowBytes, std::uint32_t rows,
                             std::uint32_t channels, std::uint32_t sampleBytes) {
    const std::size_t samples = rowBytes / sampleBytes;
    for (std::uint32_t y = 0; y < rows; ++y) {
        std::byte* line = data.data() + y * rowBytes;
        for (std::size_t i = channels; i < samples; ++i) {
            switch (sampleBytes) {
            case 1:
                line[i] = static_cast<std::byte>(std::to_integer<unsigned>(line[i]) +
                                                 std::to_integer<unsigned>(line[i - channels]));
                break;
            case 2: {
                std::uint16_t a, b;
                std::memcpy(&a, line + 2 * i, 2);
                std::memcpy(&b, line + 2 * (i - channels), 2);
                a = static_cast<std::uint16_t>(a + b);
                std::memcpy(line + 2 * i, &a, 2);
                break;
            }
            default: {
                std::uint32_t a, b;
                std::memcpy(&a, line + 4 * i, 4);
                std::memcpy(&b, line + 4 * (i - channels), 4);
                a += b;
                std::memcpy(line + 4 * i, &a, 4);
                break;
            }
            }
        }
    }
}

// ----------------------------------------------------------------------------------------------
// DEMs

/// A DEM's grid: its size, its heights (metres above the reference sphere) and that sphere's
/// radius.
struct DemGrid {
    std::uint32_t lines = 0;
    std::uint32_t samples = 0;
    double referenceRadius = 0.0;
    std::vector<float> heights;
};

/// Reads a DEM's heights from its PDS image: samples times the scale, plus the offset unless the
/// offset is the reference sphere's radius (as LOLA's is; the projection's A axis is otherwise
/// the sphere, in km). PDS's real null, the missing constant and non-finite samples are NaN.
DemGrid ReadDemGrid(const PdsImage& image) {
    const double offset = image.Offset();
    const bool offsetIsRadius = offset > 1e5;
    DemGrid dem;
    dem.lines = image.Height();
    dem.samples = image.Width();
    dem.referenceRadius = offsetIsRadius
                              ? offset
                              : 1e3 * image.Keywords().Number("IMAGE_MAP_PROJECTION.A_AXIS_RADIUS");
    const double addend = offsetIsRadius ? 0.0 : offset;
    const double scale = image.Scale();
    const std::optional<double> noData = image.NoData();
    const GridSampleType type = image.Type();
    const std::uint32_t bytes = image.SampleBytes();
    const std::vector<std::byte> raw = image.ReadAll();
    dem.heights.resize(std::size_t{dem.lines} * dem.samples);
    for (std::size_t i = 0; i < dem.heights.size(); ++i) {
        const double value = DecodeSample(raw.data() + i * bytes, type);
        const bool missing =
            (noData && value == *noData) ||
            (type == GridSampleType::Float32 && !(value > -1e30 && std::isfinite(value)));
        dem.heights[i] = missing ? std::numeric_limits<float>::quiet_NaN()
                                 : static_cast<float>(value * scale + addend);
    }
    return dem;
}

/// The map projection type a DEM's label gives, checked against those accepted.
std::string DemProjection(const PdsImage& image, std::initializer_list<std::string_view> accepted) {
    const std::string projection =
        image.Keywords().Text("IMAGE_MAP_PROJECTION.MAP_PROJECTION_TYPE");
    if (std::find(accepted.begin(), accepted.end(), projection) == accepted.end()) {
        throw std::invalid_argument("The DEM's projection '" + projection + "' is not " +
                                    std::string(*accepted.begin()));
    }
    return projection;
}

/// `heights` (`lines` rows of `samples`) at fractional line `line` and sample `sample` (cell
/// centres at whole numbers), bilinear, clamped to the grid's edges; 0 for an empty grid. Corners
/// of no weight are left out, so a cell without data (NaN) reaches only the points beside it.
double SampleDem(const std::vector<float>& heights, std::uint32_t lines, std::uint32_t samples,
                 double line, double sample) {
    if (lines == 0 || samples == 0) {
        return 0.0;
    }
    const double y = std::clamp(line, 0.0, double(lines - 1));
    const double x = std::clamp(sample, 0.0, double(samples - 1));
    const auto l0 =
        std::min<std::uint32_t>(static_cast<std::uint32_t>(y), lines > 1 ? lines - 2 : 0);
    const auto s0 =
        std::min<std::uint32_t>(static_cast<std::uint32_t>(x), samples > 1 ? samples - 2 : 0);
    const std::uint32_t l1 = std::min(l0 + 1, lines - 1);
    const std::uint32_t s1 = std::min(s0 + 1, samples - 1);
    const double ty = y - l0;
    const double tx = x - s0;
    const auto at = [&](std::uint32_t l, std::uint32_t s) {
        return static_cast<double>(heights[std::size_t(l) * samples + s]);
    };
    const double weights[4] = {(1.0 - tx) * (1.0 - ty), tx * (1.0 - ty), (1.0 - tx) * ty, tx * ty};
    const double values[4] = {at(l0, s0), at(l0, s1), at(l1, s0), at(l1, s1)};
    double sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        if (weights[i] > 0.0) {
            sum += weights[i] * values[i];
        }
    }
    return sum;
}

/// The local terrain about a point of a sphere of `radius`, its surface `heightAt(latitude,
/// longitude)` above it (`MakeLocalTerrain`).
template <typename HeightAt>
LocalTerrain LocalTerrainAround(double radius, const HeightAt& heightAt, double latitudeDeg,
                                double longitudeDeg, std::uint32_t cells, double cellSize) {
    if (cells < 2 || !(cellSize > 0.0)) {
        throw std::invalid_argument("A local terrain needs at least 2 cells a positive size apart");
    }
    // The local frame: up along the point's direction, east and north along its parallel and
    // meridian.
    const glm::dvec3 up = BodyDirection(latitudeDeg, longitudeDeg);
    const double lon = glm::radians(longitudeDeg);
    const glm::dvec3 east(-std::sin(lon), std::cos(lon), 0.0);
    const glm::dvec3 north = glm::cross(up, east);
    const glm::dvec3 centre = radius * up;
    LocalTerrain terrain;
    glm::dmat4 localToBody(1.0);
    localToBody[0] = glm::dvec4(east, 0.0);
    localToBody[1] = glm::dvec4(north, 0.0);
    localToBody[2] = glm::dvec4(up, 0.0);
    localToBody[3] = glm::dvec4(centre, 1.0);
    terrain.bodyToLocal = glm::inverse(localToBody);
    terrain.cellSize = cellSize;
    const double half = 0.5 * (cells - 1) * cellSize;
    terrain.origin = glm::dvec2(-half, half);

    std::vector<float> heights(std::size_t(cells) * cells);
    for (std::uint32_t row = 0; row < cells; ++row) {
        for (std::uint32_t col = 0; col < cells; ++col) {
            const double x = -half + col * cellSize;
            const double y = half - row * cellSize;
            // The surface point whose local x and y are these: on the sphere of radius R + h
            // through them, h the DEM's height where it lands, settled in a few rounds.
            double h = 0.0;
            double z = 0.0;
            for (int i = 0; i < 4; ++i) {
                const double r = radius + h;
                z = std::sqrt(std::max(r * r - x * x - y * y, 0.0)) - radius;
                const glm::dvec3 d = glm::normalize(centre + x * east + y * north + z * up);
                const double lat = glm::degrees(std::asin(std::clamp(d.z, -1.0, 1.0)));
                const double along = glm::degrees(std::atan2(d.y, d.x));
                h = heightAt(lat, along);
                if (std::isnan(h)) {
                    h = 0.0;
                }
            }
            const double r = radius + h;
            z = std::sqrt(std::max(r * r - x * x - y * y, 0.0)) - radius;
            heights[std::size_t(row) * cells + col] = static_cast<float>(z);
        }
    }
    terrain.heights = Tensor::Copy<float>(heights, {std::int64_t(cells), std::int64_t(cells)});
    return terrain;
}

}  // namespace

// ----------------------------------------------------------------------------------------------
// Rasters

std::uint32_t ByteSize(GridSampleType type) {
    switch (type) {
    case GridSampleType::UInt8:
        return 1;
    case GridSampleType::UInt16:
    case GridSampleType::Int16:
        return 2;
    case GridSampleType::UInt32:
    case GridSampleType::Int32:
    case GridSampleType::Float32:
        break;
    }
    return 4;
}

double DecodeSample(const std::byte* sample, GridSampleType type) {
    switch (type) {
    case GridSampleType::UInt8:
        return LoadAs<std::uint8_t>(sample);
    case GridSampleType::UInt16:
        return LoadAs<std::uint16_t>(sample);
    case GridSampleType::Int16:
        return LoadAs<std::int16_t>(sample);
    case GridSampleType::UInt32:
        return LoadAs<std::uint32_t>(sample);
    case GridSampleType::Int32:
        return LoadAs<std::int32_t>(sample);
    case GridSampleType::Float32:
        break;
    }
    return LoadAs<float>(sample);
}

std::vector<std::byte> GridSource::ReadAll() const {
    std::vector<std::byte> out(std::size_t{Width()} * Height() * Channels() * SampleBytes());
    Read(0, 0, Width(), Height(), out);
    return out;
}

void GridSource::CheckRead(std::uint32_t x, std::uint32_t y, std::uint32_t w, std::uint32_t h,
                           std::span<const std::byte> out) const {
    if (x > Width() || w > Width() - x || y > Height() || h > Height() - y) {
        throw std::out_of_range("A rectangle past the raster");
    }
    if (out.size() < std::size_t{w} * h * Channels() * SampleBytes()) {
        throw std::out_of_range("Too little room for the rectangle");
    }
}

MosaicRaster::MosaicRaster(const std::vector<fs::path>& tiles, std::uint32_t columns)
    : columns_(columns) {
    if (columns == 0 || tiles.empty() || tiles.size() % columns != 0) {
        throw std::invalid_argument("A mosaic needs whole rows of " + std::to_string(columns) +
                                    " tiles");
    }
    rows_ = static_cast<std::uint32_t>(tiles.size() / columns);
    for (const fs::path& tile : tiles) {
        tiles_.push_back(OpenRaster(tile));
        const GridSource& first = *tiles_.front();
        const GridSource& last = *tiles_.back();
        if (last.Width() != first.Width() || last.Height() != first.Height() ||
            last.Channels() != first.Channels() || last.Type() != first.Type()) {
            throw std::invalid_argument("A mosaic's tiles are alike: " + tile.string() +
                                        " is not as " + tiles.front().string());
        }
    }
    tileWidth_ = tiles_.front()->Width();
    tileHeight_ = tiles_.front()->Height();
}

std::vector<GridRect> MosaicRaster::Parts() const {
    std::vector<GridRect> parts;
    for (std::uint32_t row = 0; row < rows_; ++row) {
        for (std::uint32_t column = 0; column < columns_; ++column) {
            parts.push_back({column * tileWidth_, row * tileHeight_, tileWidth_, tileHeight_});
        }
    }
    return parts;
}

void MosaicRaster::Release() const {
    for (const std::unique_ptr<GridSource>& tile : tiles_) {
        tile->Release();
    }
}

void MosaicRaster::Read(std::uint32_t x, std::uint32_t y, std::uint32_t w, std::uint32_t h,
                        std::span<std::byte> out) const {
    CheckRead(x, y, w, h, out);
    const std::size_t pixel = std::size_t{Channels()} * SampleBytes();
    std::vector<std::byte> part;
    for (std::uint32_t row = y / tileHeight_; row < rows_ && row * tileHeight_ < y + h; ++row) {
        for (std::uint32_t column = x / tileWidth_;
             column < columns_ && column * tileWidth_ < x + w; ++column) {
            const std::size_t index = std::size_t{row} * columns_ + column;
            if (index != current_) {
                tiles_[current_]->Release();
                current_ = index;
            }
            // The rectangle's overlap with the tile, in the tile.
            const std::uint32_t left = std::max(x, column * tileWidth_);
            const std::uint32_t top = std::max(y, row * tileHeight_);
            const std::uint32_t right = std::min(x + w, (column + 1) * tileWidth_);
            const std::uint32_t bottom = std::min(y + h, (row + 1) * tileHeight_);
            const std::uint32_t across = right - left;
            part.resize(std::size_t{across} * (bottom - top) * pixel);
            tiles_[index]->Read(left - column * tileWidth_, top - row * tileHeight_, across,
                                bottom - top, part);
            for (std::uint32_t r = top; r < bottom; ++r) {
                std::memcpy(out.data() + (std::size_t{r - y} * w + (left - x)) * pixel,
                            part.data() + std::size_t{r - top} * across * pixel,
                            std::size_t{across} * pixel);
            }
        }
    }
}

std::unique_ptr<MosaicRaster> OpenMosaic(const fs::path& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot read " + path.string());
    }
    std::vector<fs::path> tiles;
    std::size_t columns = 0;
    std::string line;
    while (std::getline(file, line)) {
        if (const std::size_t comment = line.find('#'); comment != std::string::npos) {
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
            throw std::invalid_argument(path.string() + ": a row of " + std::to_string(count) +
                                        " tiles, not " + std::to_string(columns));
        }
        columns = count;
    }
    return std::make_unique<MosaicRaster>(tiles, static_cast<std::uint32_t>(columns));
}

std::unique_ptr<GridSource> OpenRaster(const fs::path& path) {
    const std::string extension = ToLower(path.extension().string());
    if (extension == ".tif" || extension == ".tiff") {
        return std::make_unique<TiffImage>(path);
    }
    if (extension == ".img" || extension == ".lbl") {
        return std::make_unique<PdsImage>(path);
    }
    if (extension == ".mosaic") {
        return OpenMosaic(path);
    }
    if (IsPngOrJpeg(path)) {
        return std::make_unique<DecodedRaster>(path);
    }
    throw std::runtime_error(
        path.string() + " is not a raster read here (TIFF, PDS, PNG, JPEG or a mosaic of them)");
}

// ----------------------------------------------------------------------------------------------
// Grid views

void CheckGridView(const GridView& view, const std::string& what, std::size_t minSide) {
    if (view.width < minSide || view.height < minSide) {
        throw std::invalid_argument(what + " must be at least " + std::to_string(minSide) + " x " +
                                    std::to_string(minSide) + " cells");
    }
    if (view.data == nullptr) {
        throw std::invalid_argument(what + " has no data");
    }
    if (view.Stride() < view.width) {
        throw std::invalid_argument(what + " row stride is shorter than its width");
    }
    if (!std::isfinite(view.originX) || !std::isfinite(view.originY) ||
        !std::isfinite(view.cellSize) || !(view.cellSize > 0.0)) {
        throw std::invalid_argument(what + " needs a finite origin and a positive cell size");
    }
}

// ----------------------------------------------------------------------------------------------
// PDS3

std::string CleanPds3Value(std::string_view value) {
    std::string text(value);
    if (const std::size_t unit = text.find('<');
        unit != std::string::npos && text.front() != '"' && text.front() != '(') {
        text = Trim(text.substr(0, unit));
    }
    if (text.size() >= 2 && (text.front() == '"' || text.front() == '\'') &&
        text.back() == text.front()) {
        text = text.substr(1, text.size() - 2);
    }
    return Trim(text);
}

std::optional<std::string> Pds3Label::Get(std::string_view name) const {
    for (auto it = keywords.rbegin(); it != keywords.rend(); ++it) {
        if (name.size() == it->object.size() + it->key.size() && name.starts_with(it->object) &&
            name.ends_with(it->key)) {
            return CleanPds3Value(it->value);
        }
    }
    return std::nullopt;
}

double Pds3Label::Number(std::string_view name) const {
    const std::string text = Text(name);
    try {
        return std::stod(text);
    } catch (const std::exception&) {
        throw std::invalid_argument("PDS3 keyword " + std::string(name) + " = '" + text +
                                    "' is not a number");
    }
}

std::string Pds3Label::Text(std::string_view name) const {
    if (std::optional<std::string> value = Get(name)) {
        return *std::move(value);
    }
    throw std::invalid_argument("The PDS3 label has no " + std::string(name));
}

std::optional<std::string> Pds3Label::Find(std::string_view key) const {
    const std::string wanted = ToUpper(key);
    for (const Keyword& keyword : keywords) {
        if (ToUpper(keyword.key) == wanted) {
            return keyword.value;
        }
    }
    return std::nullopt;
}

Pds3Label ParsePds3Label(std::string_view text) {
    Pds3Label label;
    std::vector<std::string> objects;
    std::istringstream lines{std::string(text)};
    std::string line;
    std::string key;
    std::string value;
    bool inComment = false;
    bool continued = false;
    const auto finish = [&] {
        if (key == "OBJECT") {
            objects.push_back(CleanPds3Value(value));
        } else if (key == "END_OBJECT") {
            if (!objects.empty()) {
                objects.pop_back();
            }
        } else if (!key.empty()) {
            std::string object;
            for (const std::string& name : objects) {
                object += name + ".";
            }
            label.keywords.push_back({std::move(object), std::move(key), Trim(value)});
        }
        key.clear();
        value.clear();
        continued = false;
    };
    while (std::getline(lines, line)) {
        line = WithoutComments(line, inComment);
        if (continued) {
            value += ' ' + Trim(line);
        } else {
            const std::string trimmed = Trim(line);
            if (trimmed == "END") {
                break;
            }
            const std::size_t equals = trimmed.find('=');
            if (equals == std::string::npos) {
                continue;
            }
            key = Trim(trimmed.substr(0, equals));
            value = Trim(trimmed.substr(equals + 1));
        }
        continued = Unclosed(value);
        if (!continued) {
            finish();
        }
    }
    finish();
    return label;
}

PdsImage::PdsImage(const fs::path& path) {
    const std::string name = path.string();
    // The label: this file if it is one, else one beside the data, else attached at the data's
    // start.
    fs::path labelPath;
    if (ToUpper(path.extension().string()) == ".LBL") {
        labelPath = path;
    } else if (const fs::path beside =
                   FindFileAnyCase(path.parent_path(), path.stem().string() + ".lbl");
               fs::exists(beside)) {
        labelPath = beside;
    }
    std::string text;
    const bool attached = labelPath.empty();
    if (!attached) {
        text = ReadTextFile(labelPath);
    } else {
        std::ifstream file(path, std::ios::binary);
        text.resize(1 << 16);
        file.read(text.data(), static_cast<std::streamsize>(text.size()));
        text.resize(static_cast<std::size_t>(file.gcount()));
    }
    if (text.find("PDS_VERSION_ID") == std::string::npos) {
        throw std::runtime_error(name + " has no PDS3 label");
    }
    label_ = ParsePds3Label(text);
    const auto number = [&](std::string_view key,
                            std::optional<double> fallback = std::nullopt) -> double {
        const std::optional<std::string> value = Label(key);
        if (!value) {
            if (fallback) {
                return *fallback;
            }
            throw std::runtime_error(name + ": the PDS label has no " + std::string(key));
        }
        try {
            return std::stod(*value);
        } catch (const std::exception&) {
            throw std::runtime_error(name + ": the PDS label's " + std::string(key) +
                                     " is not a number");
        }
    };
    width_ = static_cast<std::uint32_t>(number("LINE_SAMPLES"));
    height_ = static_cast<std::uint32_t>(number("LINES"));
    if (number("BANDS", 1.0) != 1.0) {
        throw std::runtime_error(name + ": PDS images of several bands are not read");
    }
    const auto bits = static_cast<std::uint32_t>(number("SAMPLE_BITS"));
    const std::string sampleType = ToUpper(Label("SAMPLE_TYPE").value_or(""));
    const bool real = sampleType.find("REAL") != std::string::npos;
    const bool unsignedType = sampleType.find("UNSIGNED") != std::string::npos;
    bigEndian_ = sampleType.rfind("LSB", 0) != 0 && sampleType.rfind("PC_", 0) != 0 &&
                 sampleType.rfind("VAX", 0) != 0;
    if (real && bits == 32) {
        type_ = GridSampleType::Float32;
    } else if (!real && bits == 8) {
        type_ = GridSampleType::UInt8;
    } else if (!real && bits == 16) {
        type_ = unsignedType ? GridSampleType::UInt16 : GridSampleType::Int16;
    } else if (!real && bits == 32) {
        type_ = unsignedType ? GridSampleType::UInt32 : GridSampleType::Int32;
    } else {
        throw std::runtime_error(name + ": PDS samples " + sampleType + " of " +
                                 std::to_string(bits) + " bits are not read");
    }
    scale_ = number("SCALING_FACTOR", 1.0);
    offset_ = number("OFFSET", 0.0);
    if (Label("MISSING_CONSTANT")) {
        noData_ = number("MISSING_CONSTANT");
    }

    // The data: ^IMAGE names a file beside the label ("NAME.IMG", or ("NAME.IMG", record)), or a
    // record or byte of this one.
    const std::string pointer = label_.Find("^IMAGE").value_or(std::string());
    fs::path dataPath = path;
    std::uint64_t record = 1;
    bool bytes = false;
    if (const std::size_t quote = pointer.find('"'); quote != std::string::npos) {
        const std::size_t end = pointer.find('"', quote + 1);
        dataPath = FindFileAnyCase(path.parent_path(), pointer.substr(quote + 1, end - quote - 1));
        if (const std::size_t comma = pointer.find(',', end); comma != std::string::npos) {
            const std::string rest = pointer.substr(comma + 1);
            record = std::stoull(rest);
            bytes = ToUpper(rest).find("<BYTES>") != std::string::npos;
        }
    } else if (!pointer.empty()) {
        record = std::stoull(pointer);
        bytes = ToUpper(pointer).find("<BYTES>") != std::string::npos;
    } else if (!attached) {
        throw std::runtime_error(name + ": the PDS label has no ^IMAGE");
    }
    const auto recordBytes = static_cast<std::uint64_t>(number("RECORD_BYTES", 0.0));
    start_ = bytes ? record - 1 : (record - 1) * recordBytes;
    file_ = std::make_shared<const MappedFile>(dataPath);
    if (file_->Bytes().size() < start_ + std::uint64_t{width_} * height_ * SampleBytes()) {
        throw std::runtime_error(dataPath.string() + " is shorter than its PDS label says");
    }

    // Simple cylindrical grids.
    const std::string projection = ToUpper(Label("MAP_PROJECTION_TYPE").value_or(""));
    if ((projection == "SIMPLE CYLINDRICAL" || projection == "EQUIRECTANGULAR") &&
        number("CENTER_LATITUDE", 0.0) == 0.0 && Label("WESTERNMOST_LONGITUDE") &&
        Label("MAXIMUM_LATITUDE")) {
        LatLongGrid grid;
        grid.west = number("WESTERNMOST_LONGITUDE");
        grid.north = number("MAXIMUM_LATITUDE");
        if (Label("MAP_RESOLUTION")) {
            grid.degreesAcross = grid.degreesDown = 1.0 / number("MAP_RESOLUTION");
        } else {
            grid.degreesAcross = (number("EASTERNMOST_LONGITUDE") - grid.west) / width_;
            grid.degreesDown = (grid.north - number("MINIMUM_LATITUDE")) / height_;
        }
        grid_ = grid;
    }
}

std::optional<std::string> PdsImage::Label(std::string_view key) const {
    const std::optional<std::string> value = label_.Find(key);
    if (!value) {
        return std::nullopt;
    }
    return CleanPds3Value(*value);
}

void PdsImage::Read(std::uint32_t x, std::uint32_t y, std::uint32_t w, std::uint32_t h,
                    std::span<std::byte> out) const {
    CheckRead(x, y, w, h, out);
    const std::uint32_t bytes = SampleBytes();
    const std::span<const std::byte> data = file_->Bytes();
    for (std::uint32_t row = 0; row < h; ++row) {
        const std::span<std::byte> line =
            out.subspan(std::size_t{row} * w * bytes, std::size_t{w} * bytes);
        std::memcpy(line.data(),
                    data.data() + start_ + (std::uint64_t{y + row} * width_ + x) * bytes,
                    line.size());
        if (NeedsByteSwap(bigEndian_, bytes)) {
            SwapByteOrder(line, bytes);
        }
    }
}

// ----------------------------------------------------------------------------------------------
// TIFF

TiffImage::TiffImage(const fs::path& path) : file_(std::make_shared<const MappedFile>(path)) {
    const std::span<const std::byte> bytes = file_->Bytes();
    const std::string name = path.string();
    if (bytes.size() < 16) {
        throw std::runtime_error(name + " is not a TIFF file");
    }
    const auto b0 = std::to_integer<char>(bytes[0]);
    const auto b1 = std::to_integer<char>(bytes[1]);
    if (b0 != b1 || (b0 != 'I' && b0 != 'M')) {
        throw std::runtime_error(name + " is not a TIFF file");
    }
    bigEndian_ = b0 == 'M';
    const TiffReader r(bytes, bigEndian_);
    const std::uint64_t magic = r.Unsigned(2, 2);
    const bool big = magic == 43;
    if (magic != 42 && !big) {
        throw std::runtime_error(name + " is not a TIFF file");
    }
    // The first image's directory.
    const std::uint64_t ifd = big ? r.Unsigned(8, 8) : r.Unsigned(4, 4);
    const std::uint64_t entries = big ? r.Unsigned(ifd, 8) : r.Unsigned(ifd, 2);
    const std::size_t entryBytes = big ? 20 : 12;
    const std::size_t inlineBytes = big ? 8 : 4;
    std::vector<std::pair<std::uint16_t, TiffField>> fields;
    for (std::uint64_t i = 0; i < entries; ++i) {
        const std::uint64_t at = ifd + (big ? 8 : 2) + i * entryBytes;
        const auto tag = static_cast<std::uint16_t>(r.Unsigned(at, 2));
        TiffField field;
        field.type = static_cast<std::uint16_t>(r.Unsigned(at + 2, 2));
        field.count = big ? r.Unsigned(at + 4, 8) : r.Unsigned(at + 4, 4);
        const std::uint64_t valueAt = at + (big ? 12 : 8);
        field.at = TypeBytes(field.type) * field.count <= inlineBytes
                       ? valueAt
                       : r.Unsigned(valueAt, inlineBytes);
        fields.emplace_back(tag, field);
    }
    const auto find = [&](std::uint16_t tag) -> const TiffField* {
        for (const auto& [t, field] : fields) {
            if (t == tag) {
                return &field;
            }
        }
        return nullptr;
    };
    const auto values = [&](std::uint16_t tag) {
        std::vector<std::uint64_t> out;
        if (const TiffField* field = find(tag)) {
            const std::size_t size = TypeBytes(field->type);
            if (size == 0) {
                throw std::runtime_error(name + ": a TIFF field of an unknown type");
            }
            for (std::uint64_t i = 0; i < field->count; ++i) {
                out.push_back(r.Unsigned(field->at + i * size, size));
            }
        }
        return out;
    };
    const auto one = [&](std::uint16_t tag, std::uint64_t fallback) {
        const std::vector<std::uint64_t> v = values(tag);
        return v.empty() ? fallback : v.front();
    };
    width_ = static_cast<std::uint32_t>(one(kImageWidth, 0));
    height_ = static_cast<std::uint32_t>(one(kImageLength, 0));
    channels_ = static_cast<std::uint32_t>(one(kSamplesPerPixel, 1));
    const std::vector<std::uint64_t> bits = values(kBitsPerSample);
    bits_ = bits.empty() ? 1u : static_cast<std::uint32_t>(bits.front());
    if (width_ == 0 || height_ == 0 || channels_ == 0) {
        throw std::runtime_error(name + ": an empty TIFF image");
    }
    if (std::any_of(bits.begin(), bits.end(), [&](std::uint64_t b) { return b != bits_; })) {
        throw std::runtime_error(name + ": TIFF samples of mixed sizes");
    }
    // Planes stored apart: each channel's strips or tiles in turn.
    planes_ = one(kPlanarConfiguration, 1) == 2 ? channels_ : 1u;
    const std::uint64_t format = one(kSampleFormat, 1);
    if (format == 3 && bits_ == 32) {
        type_ = GridSampleType::Float32;
    } else if (format == 1 && bits_ == 8) {
        type_ = GridSampleType::UInt8;
    } else if (format == 1 && bits_ == 16) {
        type_ = GridSampleType::UInt16;
    } else if (format == 2 && bits_ == 16) {
        type_ = GridSampleType::Int16;
    } else if (format == 1 && bits_ == 32) {
        type_ = GridSampleType::UInt32;
    } else if (format == 2 && bits_ == 32) {
        type_ = GridSampleType::Int32;
    } else {
        throw std::runtime_error(name + ": TIFF samples of " + std::to_string(bits_) +
                                 " bits, format " + std::to_string(format) + ", are not read");
    }
    compression_ = static_cast<std::uint32_t>(one(kCompression, 1));
    if (compression_ != 1 && compression_ != 5 && compression_ != 8 && compression_ != 32946) {
        throw std::runtime_error(name + ": TIFF compression " + std::to_string(compression_) +
                                 " is not read");
    }
    predictor_ = static_cast<std::uint32_t>(one(kPredictor, 1));
    if (predictor_ > 3 || (predictor_ == 3 && type_ != GridSampleType::Float32)) {
        throw std::runtime_error(name + ": TIFF predictor " + std::to_string(predictor_) +
                                 " is not read");
    }
    tiled_ = find(kTileOffsets) != nullptr;
    if (tiled_) {
        chunkWidth_ = static_cast<std::uint32_t>(one(kTileWidth, 0));
        chunkHeight_ = static_cast<std::uint32_t>(one(kTileLength, 0));
        offsets_ = values(kTileOffsets);
        byteCounts_ = values(kTileByteCounts);
    } else {
        chunkWidth_ = width_;
        chunkHeight_ = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(one(kRowsPerStrip, height_), height_));
        offsets_ = values(kStripOffsets);
        byteCounts_ = values(kStripByteCounts);
    }
    if (chunkWidth_ == 0 || chunkHeight_ == 0) {
        throw std::runtime_error(name + ": TIFF strips or tiles of no size");
    }
    chunksAcross_ = (width_ + chunkWidth_ - 1) / chunkWidth_;
    chunksPerPlane_ = std::size_t{chunksAcross_} * ((height_ + chunkHeight_ - 1) / chunkHeight_);
    const std::size_t chunks = chunksPerPlane_ * planes_;
    cachedChunks_.assign(planes_, ~std::size_t{0});
    cached_.resize(planes_);
    if (offsets_.size() < chunks || byteCounts_.size() < chunks) {
        throw std::runtime_error(name + ": TIFF strips or tiles missing");
    }
    // GeoTIFF's grid, and GDAL's no-data value (ASCII).
    if (const TiffField* field = find(kModelPixelScale);
        field != nullptr && field->count >= 3 && field->type == 12) {
        pixelScale_ = std::array<double, 3>{r.Double(field->at), r.Double(field->at + 8),
                                            r.Double(field->at + 16)};
    }
    if (const TiffField* field = find(kModelTiepoint);
        field != nullptr && field->count >= 6 && field->type == 12) {
        std::array<double, 6> tie{};
        for (std::size_t i = 0; i < 6; ++i) {
            tie[i] = r.Double(field->at + 8 * i);
        }
        tiepoint_ = tie;
    }
    if (const TiffField* field = find(kGdalNoData);
        field != nullptr && field->type == 2 && field->count > 0) {
        std::string text;
        for (std::uint64_t i = 0; i < field->count; ++i) {
            const auto c = static_cast<char>(r.Unsigned(field->at + i, 1));
            if (c == '\0') {
                break;
            }
            text.push_back(c);
        }
        try {
            noData_ = std::stod(text);
        } catch (const std::exception&) {
            // Not a number ("nan" parses; anything else is ignored).
        }
    }
}

const std::vector<std::byte>& TiffImage::Chunk(std::size_t chunk) const {
    const std::size_t plane = chunk / chunksPerPlane_;
    if (chunk == cachedChunks_[plane]) {
        return cached_[plane];
    }
    ERNIE_ZONE("TIFF chunk");
    const std::uint32_t channels = ChunkChannels();
    const std::uint32_t row0 =
        static_cast<std::uint32_t>(chunk % chunksPerPlane_ / chunksAcross_) * chunkHeight_;
    // A strip at the image's foot holds only the rows left; tiles are always whole.
    const std::uint32_t rows = tiled_ ? chunkHeight_ : std::min(chunkHeight_, height_ - row0);
    const std::uint32_t sampleBytes = SampleBytes();
    const std::size_t rowBytes = std::size_t{chunkWidth_} * channels * sampleBytes;
    const std::size_t expected = rowBytes * rows;
    const std::span<const std::byte> file = file_->Bytes();
    const std::uint64_t offset = offsets_[chunk];
    const std::uint64_t count = byteCounts_[chunk];
    if (offset > file.size() || count > file.size() - offset) {
        throw std::runtime_error("TIFF: a strip or tile past the file's end");
    }
    const std::span<const std::byte> input = file.subspan(offset, count);
    std::vector<std::byte> data;
    switch (compression_) {
    case 5:
        data = DecodeLzw(input, expected);
        break;
    case 8:
    case 32946:
        data = DecodeDeflate(input, expected);
        break;
    default:
        data.assign(input.begin(), input.end());
        break;
    }
    if (data.size() < expected) {
        throw std::runtime_error("TIFF: a strip or tile cut short");
    }
    data.resize(expected);
    // Predictor 3 reorders the bytes itself; otherwise samples are put in host order before
    // predictor 2 is undone.
    if (predictor_ == 3) {
        UndoFloatingPointPredictor(data, rowBytes, rows, channels, sampleBytes);
    } else if (NeedsByteSwap(bigEndian_, sampleBytes)) {
        SwapByteOrder(data, sampleBytes);
    }
    if (predictor_ == 2) {
        UndoHorizontalPredictor(data, rowBytes, rows, channels, sampleBytes);
    }
    cached_[plane] = std::move(data);
    cachedChunks_[plane] = chunk;
    return cached_[plane];
}

void TiffImage::Read(std::uint32_t x, std::uint32_t y, std::uint32_t w, std::uint32_t h,
                     std::span<std::byte> out) const {
    CheckRead(x, y, w, h, out);
    const std::size_t pixelBytes = std::size_t{channels_} * SampleBytes();
    // Each plane's samples go to their channel of the pixels.
    const std::size_t sampleBytes = SampleBytes();
    const std::size_t chunkPixel = ChunkChannels() * sampleBytes;
    for (std::uint32_t plane = 0; plane < planes_; ++plane) {
        for (std::uint32_t row = y; row < y + h; ++row) {
            const std::uint32_t chunkRow = row / chunkHeight_;
            for (std::uint32_t column = x; column < x + w;) {
                const std::uint32_t chunkColumn = column / chunkWidth_;
                const std::vector<std::byte>& chunk = Chunk(
                    plane * chunksPerPlane_ + std::size_t{chunkRow} * chunksAcross_ + chunkColumn);
                const std::uint32_t left = chunkColumn * chunkWidth_;
                const std::uint32_t span = std::min(x + w, left + chunkWidth_) - column;
                std::byte* to = out.data() + (std::size_t{row - y} * w + (column - x)) * pixelBytes;
                const std::byte* from =
                    chunk.data() +
                    (std::size_t{row - chunkRow * chunkHeight_} * chunkWidth_ + (column - left)) *
                        chunkPixel;
                if (planes_ == 1) {
                    std::memcpy(to, from, span * pixelBytes);
                } else {
                    for (std::uint32_t i = 0; i < span; ++i) {
                        std::memcpy(to + i * pixelBytes + plane * sampleBytes,
                                    from + i * sampleBytes, sampleBytes);
                    }
                }
                column += span;
            }
        }
    }
}

std::optional<LatLongGrid> TiffImage::Grid() const {
    if (!pixelScale_ || !tiepoint_) {
        return std::nullopt;
    }
    const std::array<double, 3>& scale = *pixelScale_;
    const std::array<double, 6>& tie = *tiepoint_;
    const LatLongGrid grid{tie[3] - tie[0] * scale[0], tie[4] + tie[1] * scale[1], scale[0],
                           scale[1]};
    // Degrees: within a turn of longitude and between the poles (metres of a projected grid are far
    // larger).
    constexpr double kSlack = 1e-6;
    if (!(grid.degreesAcross > 0.0) || !(grid.degreesDown > 0.0) ||
        grid.degreesAcross * width_ > 360.0 + kSlack ||
        grid.degreesDown * height_ > 180.0 + kSlack || std::abs(grid.west) > 360.0 ||
        std::abs(grid.north) > 90.0 + kSlack) {
        return std::nullopt;
    }
    return grid;
}

// ----------------------------------------------------------------------------------------------
// Planetary DEMs

double CylindricalDem::HeightAt(double latitudeDeg, double longitudeDeg) const {
    const double east = std::fmod(std::fmod(longitudeDeg - westLongitudeDeg, 360.0) + 360.0, 360.0);
    return SampleDem(heights, lines, samples,
                     (maxLatitudeDeg - latitudeDeg) * pixelsPerDegree - 0.5,
                     east * pixelsPerDegree * std::cos(glm::radians(standardParallelDeg)) - 0.5);
}

CylindricalDem ReadCylindricalDem(const fs::path& labelPath) {
    const PdsImage image(labelPath);
    const std::string projection = DemProjection(image, {"SIMPLE CYLINDRICAL", "EQUIRECTANGULAR"});
    const Pds3Label& label = image.Keywords();
    DemGrid grid = ReadDemGrid(image);
    CylindricalDem dem;
    dem.lines = grid.lines;
    dem.samples = grid.samples;
    dem.referenceRadius = grid.referenceRadius;
    dem.heights = std::move(grid.heights);
    dem.pixelsPerDegree = label.Number("IMAGE_MAP_PROJECTION.MAP_RESOLUTION");
    dem.maxLatitudeDeg = label.Number("IMAGE_MAP_PROJECTION.MAXIMUM_LATITUDE");
    dem.westLongitudeDeg = label.Number("IMAGE_MAP_PROJECTION.WESTERNMOST_LONGITUDE");
    dem.standardParallelDeg =
        projection == "EQUIRECTANGULAR" && label.Get("IMAGE_MAP_PROJECTION.CENTER_LATITUDE")
            ? label.Number("IMAGE_MAP_PROJECTION.CENTER_LATITUDE")
            : 0.0;
    return dem;
}

glm::dvec2 PolarStereographicDem::ToGrid(double latitudeDeg, double longitudeDeg) const {
    const double lat = glm::radians(latitudeDeg);
    const double lon = glm::radians(longitudeDeg - centerLongitudeDeg);
    const bool south = centerLatitudeDeg < 0.0;
    // Snyder 1987, eq. 21-33 and 21-34 for the sphere (k0 = 1, true at the pole):
    // rho = 2 R tan(pi/4 -+ phi/2).
    const double rho =
        2.0 * referenceRadius * std::tan(0.25 * std::numbers::pi + (south ? 0.5 : -0.5) * lat);
    const double x = rho * std::sin(lon);
    const double y = (south ? 1.0 : -1.0) * rho * std::cos(lon);
    return {lineProjectionOffset - y / mapScale, sampleProjectionOffset + x / mapScale};
}

double PolarStereographicDem::HeightAt(double latitudeDeg, double longitudeDeg) const {
    const glm::dvec2 at = ToGrid(latitudeDeg, longitudeDeg);
    return SampleDem(heights, lines, samples, at.x, at.y);
}

PolarStereographicDem ReadPolarStereographicDem(const fs::path& labelPath) {
    const PdsImage image(labelPath);
    DemProjection(image, {"POLAR STEREOGRAPHIC"});
    const Pds3Label& label = image.Keywords();
    DemGrid grid = ReadDemGrid(image);
    PolarStereographicDem dem;
    dem.lines = grid.lines;
    dem.samples = grid.samples;
    dem.referenceRadius = grid.referenceRadius;
    dem.heights = std::move(grid.heights);
    dem.centerLatitudeDeg = label.Number("IMAGE_MAP_PROJECTION.CENTER_LATITUDE");
    if (std::abs(std::abs(dem.centerLatitudeDeg) - 90.0) > 1e-9) {
        throw std::invalid_argument(
            "The DEM's polar stereographic projection is not centred on a pole");
    }
    dem.centerLongitudeDeg = label.Get("IMAGE_MAP_PROJECTION.CENTER_LONGITUDE")
                                 ? label.Number("IMAGE_MAP_PROJECTION.CENTER_LONGITUDE")
                                 : 0.0;
    // The scale's units are read from the value as written, since `Number` drops them.
    dem.mapScale = label.Number("IMAGE_MAP_PROJECTION.MAP_SCALE");
    if (ToUpper(label.Find("MAP_SCALE").value_or("")).find("<KM") != std::string::npos) {
        dem.mapScale *= 1e3;
    }
    dem.lineProjectionOffset = label.Number("IMAGE_MAP_PROJECTION.LINE_PROJECTION_OFFSET");
    dem.sampleProjectionOffset = label.Number("IMAGE_MAP_PROJECTION.SAMPLE_PROJECTION_OFFSET");
    return dem;
}

glm::dvec3 BodyDirection(double latitudeDeg, double longitudeDeg) {
    const double lat = glm::radians(latitudeDeg);
    const double lon = glm::radians(longitudeDeg);
    return {std::cos(lat) * std::cos(lon), std::cos(lat) * std::sin(lon), std::sin(lat)};
}

LocalTerrain MakeLocalTerrain(const CylindricalDem& dem, double latitudeDeg, double longitudeDeg,
                              std::uint32_t cells, double cellSize) {
    return LocalTerrainAround(
        dem.referenceRadius, [&dem](double lat, double lon) { return dem.HeightAt(lat, lon); },
        latitudeDeg, longitudeDeg, cells, cellSize);
}

LocalTerrain MakeLocalTerrain(const PolarStereographicDem& dem, double latitudeDeg,
                              double longitudeDeg, std::uint32_t cells, double cellSize) {
    return LocalTerrainAround(
        dem.referenceRadius, [&dem](double lat, double lon) { return dem.HeightAt(lat, lon); },
        latitudeDeg, longitudeDeg, cells, cellSize);
}

LocalTerrain MakeLocalTerrain(const CylindricalDem& dem, const CylindricalDem& fallback,
                              double latitudeDeg, double longitudeDeg, std::uint32_t cells,
                              double cellSize) {
    return LocalTerrainAround(
        dem.referenceRadius,
        [&](double lat, double lon) {
            const double h = dem.HeightAt(lat, lon);
            return std::isnan(h) ? fallback.HeightAt(lat, lon) : h;
        },
        latitudeDeg, longitudeDeg, cells, cellSize);
}

}  // namespace ernie
