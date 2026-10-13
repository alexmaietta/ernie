// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/tiff.h"

#include "ernie/util/profile.h"

#include <stb_image.h>

#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ernie {

namespace {

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

// Field types: BYTE, ASCII, SHORT, LONG, ..., DOUBLE, LONG8.
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

/// Reads a TIFF's integers in its byte order, checking every access
/// against the file's end.
class Reader {
  public:
    Reader(std::span<const std::byte> bytes, bool bigEndian)
        : bytes_(bytes), bigEndian_(bigEndian) {}

    std::uint64_t Unsigned(std::uint64_t at, std::size_t size) const {
        if (at > bytes_.size() || size > bytes_.size() - at) {
            throw std::runtime_error(
                "TIFF: read past the file's end");
        }
        std::uint64_t value = 0;
        for (std::size_t i = 0; i < size; ++i) {
            const auto byte = std::to_integer<std::uint64_t>(
                bytes_[at + (bigEndian_ ? i : size - 1 - i)]);
            value = (value << 8) | byte;
        }
        return value;
    }

    double Double(std::uint64_t at) const {
        return std::bit_cast<double>(Unsigned(at, 8));
    }

    std::span<const std::byte> Bytes() const {
        return bytes_;
    }

  private:
    std::span<const std::byte> bytes_;
    bool bigEndian_;
};

/// One IFD entry: its type, count and where its values are (inline in
/// the entry or at an offset).
struct Field {
    std::uint16_t type = 0;
    std::uint64_t count = 0;
    std::uint64_t at = 0;
};

/// TIFF's LZW: 9- to 12-bit codes, most significant bit first, 256
/// clearing the table and 257 ending; widths grow a code early.
std::vector<std::byte> DecodeLzw(std::span<const std::byte> input,
                                 std::size_t expected) {
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
            const auto byte =
                std::to_integer<std::uint32_t>(input[bit >> 3]);
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
            length[next] =
                static_cast<std::uint16_t>(length[from] + 1);
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
            add(static_cast<std::uint32_t>(old),
                first[static_cast<std::uint32_t>(old)]);
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
std::vector<std::byte> DecodeDeflate(std::span<const std::byte> input,
                                     std::size_t expected) {
    if (expected > static_cast<std::size_t>(INT_MAX) ||
        input.size() > static_cast<std::size_t>(INT_MAX)) {
        throw std::runtime_error("TIFF: a Deflate strip over 2 GB");
    }
    std::vector<std::byte> out(expected);
    const int size = stbi_zlib_decode_buffer(
        reinterpret_cast<char*>(out.data()),
        static_cast<int>(expected),
        reinterpret_cast<const char*>(input.data()),
        static_cast<int>(input.size()));
    if (size < 0) {
        throw std::runtime_error("TIFF: damaged Deflate data");
    }
    out.resize(static_cast<std::size_t>(size));
    return out;
}

}  // namespace

TiffImage::TiffImage(const std::filesystem::path& path)
    : file_(std::make_shared<const MappedFile>(path)) {
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
    const Reader r(bytes, bigEndian_);
    const std::uint64_t magic = r.Unsigned(2, 2);
    const bool big = magic == 43;
    if (magic != 42 && !big) {
        throw std::runtime_error(name + " is not a TIFF file");
    }
    // The first image's directory.
    const std::uint64_t ifd =
        big ? r.Unsigned(8, 8) : r.Unsigned(4, 4);
    const std::uint64_t entries =
        big ? r.Unsigned(ifd, 8) : r.Unsigned(ifd, 2);
    const std::size_t entryBytes = big ? 20 : 12;
    const std::size_t inlineBytes = big ? 8 : 4;
    std::vector<std::pair<std::uint16_t, Field>> fields;
    for (std::uint64_t i = 0; i < entries; ++i) {
        const std::uint64_t at = ifd + (big ? 8 : 2) + i * entryBytes;
        const auto tag =
            static_cast<std::uint16_t>(r.Unsigned(at, 2));
        Field field;
        field.type =
            static_cast<std::uint16_t>(r.Unsigned(at + 2, 2));
        field.count =
            big ? r.Unsigned(at + 4, 8) : r.Unsigned(at + 4, 4);
        const std::uint64_t valueAt = at + (big ? 12 : 8);
        const std::size_t size = TypeBytes(field.type);
        field.at = size * field.count <= inlineBytes
                       ? valueAt
                       : r.Unsigned(valueAt, inlineBytes);
        fields.emplace_back(tag, field);
    }
    const auto find = [&](std::uint16_t tag) -> const Field* {
        for (const auto& [t, field] : fields) {
            if (t == tag) {
                return &field;
            }
        }
        return nullptr;
    };
    const auto values = [&](std::uint16_t tag) {
        std::vector<std::uint64_t> out;
        if (const Field* field = find(tag)) {
            const std::size_t size = TypeBytes(field->type);
            if (size == 0) {
                throw std::runtime_error(name +
                                         ": a TIFF field of an "
                                         "unknown type");
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
    bits_ =
        bits.empty() ? 1u : static_cast<std::uint32_t>(bits.front());
    if (width_ == 0 || height_ == 0 || channels_ == 0) {
        throw std::runtime_error(name + ": an empty TIFF image");
    }
    if (std::any_of(bits.begin(), bits.end(),
                    [&](std::uint64_t b) { return b != bits_; })) {
        throw std::runtime_error(name +
                                 ": TIFF samples of mixed sizes");
    }
    if (one(kPlanarConfiguration, 1) != 1) {
        throw std::runtime_error(
            name + ": TIFF planes stored apart are not read");
    }
    const std::uint64_t format = one(kSampleFormat, 1);
    if (format == 3 && bits_ == 32) {
        type_ = SampleType::Float32;
    } else if (format == 1 && bits_ == 8) {
        type_ = SampleType::UInt8;
    } else if (format == 1 && bits_ == 16) {
        type_ = SampleType::UInt16;
    } else if (format == 2 && bits_ == 16) {
        type_ = SampleType::Int16;
    } else if (format == 1 && bits_ == 32) {
        type_ = SampleType::UInt32;
    } else if (format == 2 && bits_ == 32) {
        type_ = SampleType::Int32;
    } else {
        throw std::runtime_error(
            name + ": TIFF samples of " + std::to_string(bits_) +
            " bits, format " + std::to_string(format) +
            ", are not read");
    }
    compression_ = static_cast<std::uint32_t>(one(kCompression, 1));
    if (compression_ != 1 && compression_ != 5 && compression_ != 8 &&
        compression_ != 32946) {
        throw std::runtime_error(name + ": TIFF compression " +
                                 std::to_string(compression_) +
                                 " is not read");
    }
    predictor_ = static_cast<std::uint32_t>(one(kPredictor, 1));
    if (predictor_ > 3 ||
        (predictor_ == 3 && type_ != SampleType::Float32)) {
        throw std::runtime_error(name + ": TIFF predictor " +
                                 std::to_string(predictor_) +
                                 " is not read");
    }
    tiled_ = find(kTileOffsets) != nullptr;
    if (tiled_) {
        chunkWidth_ = static_cast<std::uint32_t>(one(kTileWidth, 0));
        chunkHeight_ =
            static_cast<std::uint32_t>(one(kTileLength, 0));
        offsets_ = values(kTileOffsets);
        byteCounts_ = values(kTileByteCounts);
    } else {
        chunkWidth_ = width_;
        chunkHeight_ =
            static_cast<std::uint32_t>(std::min<std::uint64_t>(
                one(kRowsPerStrip, height_), height_));
        offsets_ = values(kStripOffsets);
        byteCounts_ = values(kStripByteCounts);
    }
    if (chunkWidth_ == 0 || chunkHeight_ == 0) {
        throw std::runtime_error(name +
                                 ": TIFF strips or tiles of no size");
    }
    chunksAcross_ = (width_ + chunkWidth_ - 1) / chunkWidth_;
    const std::size_t chunks =
        std::size_t{chunksAcross_} *
        ((height_ + chunkHeight_ - 1) / chunkHeight_);
    if (offsets_.size() < chunks || byteCounts_.size() < chunks) {
        throw std::runtime_error(name +
                                 ": TIFF strips or tiles missing");
    }
    // GeoTIFF's grid, and GDAL's no-data value (ASCII).
    if (const Field* field = find(kModelPixelScale);
        field != nullptr && field->count >= 3 && field->type == 12) {
        pixelScale_ = std::array<double, 3>{r.Double(field->at),
                                            r.Double(field->at + 8),
                                            r.Double(field->at + 16)};
    }
    if (const Field* field = find(kModelTiepoint);
        field != nullptr && field->count >= 6 && field->type == 12) {
        std::array<double, 6> tie{};
        for (std::size_t i = 0; i < 6; ++i) {
            tie[i] = r.Double(field->at + 8 * i);
        }
        tiepoint_ = tie;
    }
    if (const Field* field = find(kGdalNoData);
        field != nullptr && field->type == 2 && field->count > 0) {
        std::string text;
        for (std::uint64_t i = 0; i < field->count; ++i) {
            const auto c =
                static_cast<char>(r.Unsigned(field->at + i, 1));
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

const std::vector<std::byte>&
TiffImage::Chunk(std::size_t chunk) const {
    if (chunk == cachedChunk_) {
        return cached_;
    }
    ERNIE_ZONE("TIFF chunk");
    const std::uint32_t row0 =
        static_cast<std::uint32_t>(chunk / chunksAcross_) *
        chunkHeight_;
    // A strip at the image's foot holds only the rows left; tiles are
    // always whole.
    const std::uint32_t rows =
        tiled_ ? chunkHeight_
               : std::min(chunkHeight_, height_ - row0);
    const std::size_t pixelBytes =
        std::size_t{channels_} * SampleBytes();
    const std::size_t rowBytes =
        std::size_t{chunkWidth_} * pixelBytes;
    const std::size_t expected = rowBytes * rows;
    const std::span<const std::byte> file = file_->Bytes();
    const std::uint64_t offset = offsets_[chunk];
    const std::uint64_t count = byteCounts_[chunk];
    if (offset > file.size() || count > file.size() - offset) {
        throw std::runtime_error(
            "TIFF: a strip or tile past the file's end");
    }
    const std::span<const std::byte> input =
        file.subspan(offset, count);
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
    const std::uint32_t bytes = SampleBytes();
    // Predictor 3 differences the bytes of floats, most significant
    // first, across the row; undone before the byte order is.
    if (predictor_ == 3) {
        std::vector<std::byte> row(rowBytes);
        const std::size_t count =
            std::size_t{chunkWidth_} * channels_;
        for (std::uint32_t y = 0; y < rows; ++y) {
            std::byte* line = data.data() + y * rowBytes;
            for (std::size_t i = channels_; i < rowBytes; ++i) {
                line[i] = static_cast<std::byte>(
                    std::to_integer<unsigned>(line[i]) +
                    std::to_integer<unsigned>(line[i - channels_]));
            }
            for (std::size_t i = 0; i < count; ++i) {
                for (std::uint32_t b = 0; b < bytes; ++b) {
                    // Most significant first in the file; the host is
                    // little-endian (std::endian checked below).
                    row[i * bytes + b] =
                        line[(bytes - 1 - b) * count + i];
                }
            }
            std::memcpy(line, row.data(), rowBytes);
        }
    } else if (bigEndian_ !=
                   (std::endian::native == std::endian::big) &&
               bytes > 1) {
        for (std::size_t i = 0; i < data.size(); i += bytes) {
            std::reverse(data.begin() + i, data.begin() + i + bytes);
        }
    }
    // Predictor 2: each sample the difference from its left
    // neighbour's.
    if (predictor_ == 2) {
        const std::size_t samples =
            std::size_t{chunkWidth_} * channels_;
        for (std::uint32_t y = 0; y < rows; ++y) {
            std::byte* line = data.data() + y * rowBytes;
            for (std::size_t i = channels_; i < samples; ++i) {
                switch (bytes) {
                case 1:
                    line[i] = static_cast<std::byte>(
                        std::to_integer<unsigned>(line[i]) +
                        std::to_integer<unsigned>(
                            line[i - channels_]));
                    break;
                case 2: {
                    std::uint16_t a, b;
                    std::memcpy(&a, line + 2 * i, 2);
                    std::memcpy(&b, line + 2 * (i - channels_), 2);
                    a = static_cast<std::uint16_t>(a + b);
                    std::memcpy(line + 2 * i, &a, 2);
                    break;
                }
                default: {
                    std::uint32_t a, b;
                    std::memcpy(&a, line + 4 * i, 4);
                    std::memcpy(&b, line + 4 * (i - channels_), 4);
                    a += b;
                    std::memcpy(line + 4 * i, &a, 4);
                    break;
                }
                }
            }
        }
    }
    static_assert(
        std::endian::native == std::endian::little,
        "TIFF floats are reordered for a little-endian host");
    cached_ = std::move(data);
    cachedChunk_ = chunk;
    return cached_;
}

void TiffImage::Read(std::uint32_t x, std::uint32_t y,
                     std::uint32_t w, std::uint32_t h,
                     std::span<std::byte> out) const {
    if (x > width_ || w > width_ - x || y > height_ ||
        h > height_ - y) {
        throw std::out_of_range("TIFF: a rectangle past the image");
    }
    const std::size_t pixelBytes =
        std::size_t{channels_} * SampleBytes();
    if (out.size() < std::size_t{w} * h * pixelBytes) {
        throw std::out_of_range(
            "TIFF: too little room for the rectangle");
    }
    for (std::uint32_t row = y; row < y + h; ++row) {
        const std::uint32_t chunkRow = row / chunkHeight_;
        for (std::uint32_t column = x; column < x + w;) {
            const std::uint32_t chunkColumn = column / chunkWidth_;
            const std::vector<std::byte>& chunk = Chunk(
                std::size_t{chunkRow} * chunksAcross_ + chunkColumn);
            const std::uint32_t left = chunkColumn * chunkWidth_;
            const std::uint32_t span =
                std::min(x + w, left + chunkWidth_) - column;
            std::memcpy(
                out.data() +
                    (std::size_t{row - y} * w + (column - x)) *
                        pixelBytes,
                chunk.data() +
                    (std::size_t{row - chunkRow * chunkHeight_} *
                         chunkWidth_ +
                     (column - left)) *
                        pixelBytes,
                span * pixelBytes);
            column += span;
        }
    }
}

std::optional<LatLongGrid> TiffImage::Grid() const {
    if (!pixelScale_ || !tiepoint_) {
        return std::nullopt;
    }
    const std::array<double, 3>& scale = *pixelScale_;
    const std::array<double, 6>& tie = *tiepoint_;
    const LatLongGrid grid{tie[3] - tie[0] * scale[0],
                           tie[4] + tie[1] * scale[1], scale[0],
                           scale[1]};
    // Degrees: within a turn of longitude and between the poles
    // (metres of a projected grid are far larger).
    constexpr double kSlack = 1e-6;
    if (!(grid.degreesAcross > 0.0) || !(grid.degreesDown > 0.0) ||
        grid.degreesAcross * width_ > 360.0 + kSlack ||
        grid.degreesDown * height_ > 180.0 + kSlack ||
        std::abs(grid.west) > 360.0 ||
        std::abs(grid.north) > 90.0 + kSlack) {
        return std::nullopt;
    }
    return grid;
}

}  // namespace ernie
