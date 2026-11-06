// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/io/image.h"

#include <ImfChannelList.h>
#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfIO.h>
#include <ImfOutputFile.h>
#include <ImfRgbaFile.h>
#include <stb_image.h>
#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "ernie/device/buffer.h"
#include "ernie/device/commands.h"
#include "ernie/device/device.h"
#include "ernie/device/program.h"
#include "ernie/device/queue.h"
#include "ernie/device/slang/dispatch.h"
#include "ernie/io/kernels.h"
#include "ernie/io/kernels/colorize.h"
#include "ernie/io/kernels/encoding.h"
#include "ernie/util/error.h"
#include "ernie/util/profiler.h"
#include "ernie/util/span.h"
#include "ernie/util/text.h"

namespace ernie {

namespace {

class MemoryStream : public Imf::IStream {
  public:
    MemoryStream(std::span<const std::byte> bytes, const std::string& name)
        : Imf::IStream(name.c_str()), m_bytes(bytes) {}

    bool read(char c[], int n) override {
        if (m_at + uint64_t(n) > m_bytes.size()) {
            throw Error("{} ends early", fileName());
        }
        std::memcpy(c, m_bytes.data() + m_at, size_t(n));
        m_at += uint64_t(n);
        return m_at < m_bytes.size();
    }

    uint64_t tellg() override {
        return m_at;
    }

    void seekg(uint64_t at) override {
        m_at = at;
    }

    int64_t size() override {
        return int64_t(m_bytes.size());
    }

    bool isStatelessRead() const override {
        return true;
    }

    int64_t read(void* buffer, uint64_t count, uint64_t offset) override {
        if (offset >= m_bytes.size()) {
            return 0;
        }
        const uint64_t available = std::min<uint64_t>(count, m_bytes.size() - offset);
        std::memcpy(buffer, m_bytes.data() + offset, size_t(available));
        return int64_t(available);
    }

  private:
    std::span<const std::byte> m_bytes;
    uint64_t m_at = 0;
};

Image LoadExr(std::span<const std::byte> bytes, const std::string& name) {
    try {
        MemoryStream stream(bytes, name);
        Imf::RgbaInputFile file(stream);
        const Imath::Box2i window = file.dataWindow();
        const int width = window.max.x - window.min.x + 1;
        const int height = window.max.y - window.min.y + 1;
        std::vector<Imf::Rgba> texels(size_t(width) * size_t(height));
        file.setFrameBuffer(texels.data() - window.min.x - std::ptrdiff_t(window.min.y) * width, 1,
                            size_t(width));
        file.readPixels(window.min.y, window.max.y);

        Image image{uint32_t(width), uint32_t(height)};
        for (size_t i = 0; i < image.pixels.size(); ++i) {
            const Imf::Rgba& t = texels[i];
            image.pixels[i] = {float(t.r), float(t.g), float(t.b), float(t.a)};
        }
        return image;
    } catch (const std::exception& error) {
        throw Error("Cannot read {}: {}", name, error.what());
    }
}

template <class T>
Image Decode(T* data, int width, int height, float scale, ImageEncoding encoding) {
    Image image{uint32_t(width), uint32_t(height)};
    for (size_t i = 0; i < image.pixels.size(); ++i) {
        Rgba value{float(data[4 * i]) * scale, float(data[4 * i + 1]) * scale,
                   float(data[4 * i + 2]) * scale, float(data[4 * i + 3]) * scale};
        if (encoding == ImageEncoding::Srgb) {
            value = {DecodeSrgb(value.r), DecodeSrgb(value.g), DecodeSrgb(value.b), value.a};
        }
        image.pixels[i] = value;
    }
    stbi_image_free(data);
    return image;
}

Image LoadStb(std::span<const std::byte> bytes, const std::string& name, ImageEncoding encoding) {
    const auto* data = reinterpret_cast<const stbi_uc*>(bytes.data());
    const int size = int(bytes.size());
    int width = 0;
    int height = 0;
    int channels = 0;
    if (stbi_is_16_bit_from_memory(data, size) != 0) {
        if (stbi_us* pixels = stbi_load_16_from_memory(data, size, &width, &height, &channels, 4)) {
            return Decode(pixels, width, height, 1.0f / 65535.0f, encoding);
        }
    } else if (stbi_uc* pixels = stbi_load_from_memory(data, size, &width, &height, &channels, 4)) {
        return Decode(pixels, width, height, 1.0f / 255.0f, encoding);
    }
    throw Error("Cannot read {}: {}", name, stbi_failure_reason());
}

Image ToImage(const Pixels& pixels, ImageEncoding encoding) {
    const bool bytes = pixels.type == device::DType::UInt8;
    if ((!bytes && pixels.type != device::DType::Float32) ||
        (pixels.channels != 1 && pixels.channels != 3 && pixels.channels != 4)) {
        throw Error("Images are made of float32 or uint8 (rows, cols[, 1, 3 or 4]) pixels");
    }

    Image image{uint32_t(pixels.cols), uint32_t(pixels.rows)};
    const auto value = [&](size_t index, bool color) {
        if (!bytes) {
            float stored;
            std::memcpy(&stored, pixels.bytes.data() + index * sizeof(float), sizeof(float));
            return stored;
        }
        const float stored = float(uint8_t(pixels.bytes[index])) / 255.0f;
        return color && encoding == ImageEncoding::Srgb ? DecodeSrgb(stored) : stored;
    };

    for (size_t i = 0; i < image.pixels.size(); ++i) {
        const size_t base = i * size_t(pixels.channels);
        if (pixels.channels == 1) {
            const float gray = value(base, true);
            image.pixels[i] = {gray, gray, gray, 1};
        } else {
            image.pixels[i] = {value(base, true), value(base + 1, true), value(base + 2, true),
                               pixels.channels == 4 ? value(base + 3, false) : 1};
        }
    }
    return image;
}

std::vector<uint8_t> Encoded(const Pixels& pixels, const char* format) {
    const int64_t channels = pixels.channels;
    if (pixels.type == device::DType::UInt8 && channels >= 1 && channels <= 4) {
        std::vector<uint8_t> bytes(pixels.bytes.size());
        std::memcpy(bytes.data(), pixels.bytes.data(), bytes.size());
        return bytes;
    }
    if (pixels.type == device::DType::UInt32) {
        throw Error("{} images can't hold uint32 ids, colorize them with ColorizeIds or write .exr",
                    format);
    }
    if (pixels.type == device::DType::Float32 && channels == 1) {
        throw Error("{} images can't hold float32 single channel data, colorize it with "
                    "ColorizeScalars or write .exr",
                    format);
    }
    if (pixels.type != device::DType::Float32 || (channels != 3 && channels != 4)) {
        throw Error("{} images are uint8 (rows, cols[, 1 to 4]) or float32 linear color (rows, "
                    "cols, 3 or 4) pixels",
                    format);
    }

    const std::vector<uint8_t> rgba =
        ToRgba8(ToImage(pixels, ImageEncoding::Linear), ImageEncoding::Srgb);
    const size_t count = size_t(pixels.rows) * size_t(pixels.cols);
    std::vector<uint8_t> bytes(count * size_t(channels));
    for (size_t i = 0; i < count; ++i) {
        for (size_t c = 0; c < size_t(channels); ++c) {
            bytes[i * channels + c] = rgba[4 * i + c];
        }
    }
    return bytes;
}

void WritePngBytes(const std::filesystem::path& path, const uint8_t* bytes, int rows, int cols,
                   int channels) {
    if (stbi_write_png(path.string().c_str(), cols, rows, channels, bytes, cols * channels) == 0) {
        throw Error("Cannot write {}", path.string());
    }
}

void WriteJpegBytes(const std::filesystem::path& path, const uint8_t* bytes, int rows, int cols,
                    int channels, int quality) {
    if (quality < 1 || quality > 100) {
        throw Error("JPEG quality is from 1 to 100, not {}", quality);
    }
    if (stbi_write_jpg(path.string().c_str(), cols, rows, channels, bytes, quality) == 0) {
        throw Error("Cannot write {}", path.string());
    }
}

void WriteExrChannels(const std::filesystem::path& path, char* base, int rows, int cols,
                      int channels, Imf::PixelType pixel) {
    static constexpr std::array<const char*, 4> kRgba{"R", "G", "B", "A"};
    const auto name = [&](int c) -> std::string {
        if (channels == 1) {
            return "Y";
        }
        return channels <= 4 ? kRgba[size_t(c)] : std::to_string(c);
    };

    try {
        Imf::Header header(cols, rows);
        header.compression() = Imf::ZIP_COMPRESSION;
        Imf::FrameBuffer frame;
        const size_t texel = size_t(channels) * sizeof(float);
        for (int c = 0; c < channels; ++c) {
            header.channels().insert(name(c), Imf::Channel(pixel));
            frame.insert(name(c), Imf::Slice(pixel, base + size_t(c) * sizeof(float), texel,
                                             texel * size_t(cols)));
        }

        Imf::OutputFile file(path.string().c_str(), header);
        file.setFrameBuffer(frame);
        file.writePixels(rows);
    } catch (const std::exception& error) {
        throw Error("Cannot write {}: {}", path.string(), error.what());
    }
}

}

Rgba Image::Mean() const {
    if (pixels.empty()) {
        return {};
    }
    double sum[4] = {};
    for (const Rgba& p : pixels) {
        sum[0] += p.r;
        sum[1] += p.g;
        sum[2] += p.b;
        sum[3] += p.a;
    }
    const double count = double(pixels.size());
    return {float(sum[0] / count), float(sum[1] / count), float(sum[2] / count),
            float(sum[3] / count)};
}

namespace {

std::vector<char> ReadBytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw Error("Cannot read {}: can't open it", path.string());
    }
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

}

Image LoadImageFile(const std::filesystem::path& path, ImageEncoding encoding) {
    const std::vector<char> text = ReadBytes(path);
    return LoadImageBytes(std::as_bytes(std::span(text)), path.string(), encoding);
}

Rgba8Image LoadRgba8File(const std::filesystem::path& path) {
    const std::vector<char> text = ReadBytes(path);
    return LoadRgba8Bytes(std::as_bytes(std::span(text)), path.string());
}

Rgba8Image LoadRgba8Bytes(Span<const std::byte> bytes, std::string_view name) {
    ERNIE_ZONE("IO/LoadImage");
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* pixels = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(bytes.data()),
                                            int(bytes.size()), &width, &height, &channels, 4);
    if (pixels == nullptr) {
        throw Error("Cannot read {}: {}", name, stbi_failure_reason());
    }
    Rgba8Image image{uint32_t(width), uint32_t(height), {}};
    image.texels.resize(size_t(width) * size_t(height));
    std::memcpy(image.texels.data(), pixels, image.texels.size() * sizeof(uint32_t));
    stbi_image_free(pixels);
    return image;
}

Image LoadImageBytes(Span<const std::byte> bytes, std::string_view name, ImageEncoding encoding) {
    ERNIE_ZONE("IO/LoadImage");
    const std::string named(name);
    if (Lowercase(std::filesystem::path(named).extension().string()) == ".exr") {
        return LoadExr(bytes, named);
    }
    if (IsPngOrJpeg(named)) {
        return LoadStb(bytes, named, encoding);
    }
    throw Error("Unsupported image format: {}", named);
}

bool IsPngOrJpeg(const std::filesystem::path& path) {
    const std::string extension = Lowercase(path.extension().string());
    return extension == ".png" || extension == ".jpg" || extension == ".jpeg";
}

Pixels::Pixels(const device::Tensor& tensor) : type(tensor.Type()) {
    const Span<const int64_t> shape = tensor.Shape();
    if ((shape.size() != 2 && shape.size() != 3) || shape[0] < 1 || shape[1] < 1) {
        throw Error("Pixels come from (rows, cols[, channels]) tensors, not {} dimensions",
                    shape.size());
    }
    rows = shape[0];
    cols = shape[1];
    channels = shape.size() == 3 ? shape[2] : 1;
    bytes = tensor.ToHostBytes();
}

Pixels::Pixels(Span<const std::byte> data, device::DType elementType, int64_t rowCount,
               int64_t colCount, int64_t channelCount)
    : rows(rowCount), cols(colCount), channels(channelCount), type(elementType),
      bytes(data.begin(), data.end()) {
    if (rows < 1 || cols < 1 || channels < 1) {
        throw Error("Pixels are at least 1 by 1 by 1, not {} by {} by {}", rows, cols, channels);
    }
    const uint64_t expected =
        uint64_t(rows) * uint64_t(cols) * uint64_t(channels) * device::ElementBytes(type);
    if (data.size() != expected) {
        throw Error("{} by {} by {} pixels are {} bytes, not {}", rows, cols, channels, expected,
                    data.size());
    }
}

Image ImageFromTensor(const device::Tensor& tensor, ImageEncoding encoding) {
    return ToImage(Pixels(tensor), encoding);
}

std::vector<uint8_t> ToRgba8(const Image& image, ImageEncoding encoding) {
    std::vector<uint8_t> bytes(image.pixels.size() * 4);
    const auto quantize = [](float value) {
        return uint8_t(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
    };
    const auto color = [&](float value) {
        return quantize(encoding == ImageEncoding::Srgb ? EncodeSrgb(value) : value);
    };

    for (size_t i = 0; i < image.pixels.size(); ++i) {
        const Rgba& p = image.pixels[i];
        bytes[4 * i] = color(p.r);
        bytes[4 * i + 1] = color(p.g);
        bytes[4 * i + 2] = color(p.b);
        bytes[4 * i + 3] = quantize(p.a);
    }
    return bytes;
}

void WritePng(const std::filesystem::path& path, const Image& image) {
    ERNIE_ZONE("IO/WritePng");
    const std::vector<uint8_t> bytes = ToRgba8(image, ImageEncoding::Srgb);
    WritePngBytes(path, bytes.data(), int(image.height), int(image.width), 4);
}

void WritePng(const std::filesystem::path& path, const Pixels& image) {
    ERNIE_ZONE("IO/WritePng");
    const std::vector<uint8_t> bytes = Encoded(image, "PNG");
    WritePngBytes(path, bytes.data(), int(image.rows), int(image.cols), int(image.channels));
}

void WriteJpeg(const std::filesystem::path& path, const Image& image, int quality) {
    const std::vector<uint8_t> bytes = ToRgba8(image, ImageEncoding::Srgb);
    WriteJpegBytes(path, bytes.data(), int(image.height), int(image.width), 4, quality);
}

void WriteJpeg(const std::filesystem::path& path, const Pixels& image, int quality) {
    const std::vector<uint8_t> bytes = Encoded(image, "JPEG");
    WriteJpegBytes(path, bytes.data(), int(image.rows), int(image.cols), int(image.channels),
                   quality);
}

void WriteExr(const std::filesystem::path& path, const Image& image) {
    ERNIE_ZONE("IO/WriteExr");
    std::vector<Rgba> pixels = image.pixels;
    WriteExrChannels(path, reinterpret_cast<char*>(pixels.data()), int(image.height),
                     int(image.width), 4, Imf::FLOAT);
}

void WriteExr(const std::filesystem::path& path, const Pixels& image) {
    ERNIE_ZONE("IO/WriteExr");
    if (image.type != device::DType::Float32 && image.type != device::DType::UInt32 &&
        image.type != device::DType::UInt8) {
        throw Error("EXR images are float32, uint32 or uint8 pixels");
    }

    std::vector<std::byte> bytes = image.bytes;
    std::vector<float> floats;
    if (image.type == device::DType::UInt8) {
        floats.resize(bytes.size());
        for (size_t i = 0; i < floats.size(); ++i) {
            floats[i] = float(uint8_t(bytes[i]));
        }
    }
    char* base = floats.empty() ? reinterpret_cast<char*>(bytes.data())
                                : reinterpret_cast<char*>(floats.data());
    WriteExrChannels(path, base, int(image.rows), int(image.cols), int(image.channels),
                     image.type == device::DType::UInt32 ? Imf::UINT : Imf::FLOAT);
}

std::filesystem::path WriteImage(std::filesystem::path path, const Pixels& image) {
    if (!path.has_extension()) {
        path += image.type == device::DType::UInt8 ? ".png" : ".exr";
    }
    const std::string extension = Lowercase(path.extension().string());
    if (extension == ".png") {
        WritePng(path, image);
    } else if (extension == ".jpg" || extension == ".jpeg") {
        WriteJpeg(path, image);
    } else if (extension == ".exr") {
        WriteExr(path, image);
    } else {
        throw Error("Cannot write {}: images are .png, .jpg, .jpeg or .exr", path.string());
    }
    return path;
}

namespace {

device::Tensor ToTensor(const device::Tensor& like, const Pixels& pixels) {
    device::Device& device = like.GetDevice();
    const device::BufferUsage usage =
        device.SharedDevice() ? device::BufferUsage::Interop : device::BufferUsage::None;
    Ref<device::Buffer> buffer =
        device.CreateBuffer({.usage = usage, .label = "colorized"}, pixels.bytes);
    return device::Tensor(std::move(buffer), 0, pixels.type,
                          {pixels.rows, pixels.cols, pixels.channels});
}

Pixels Packed(const Pixels& like, const std::vector<uint32_t>& rgba) {
    Pixels packed;
    packed.rows = like.rows;
    packed.cols = like.cols;
    packed.channels = 4;
    packed.bytes.resize(rgba.size() * sizeof(uint32_t));
    std::memcpy(packed.bytes.data(), rgba.data(), packed.bytes.size());
    return packed;
}

void RequireScalars(device::DType type, int64_t channels, const ColorizeOptions& options) {
    if (type != device::DType::Float32) {
        throw Error("Scalars are colorized from float32 (rows, cols[, channels]) pixels");
    }
    if (int64_t(options.channel) >= channels) {
        throw Error("Colorizing channel {} of pixels with only {} channels", options.channel,
                    channels);
    }
    if (options.logarithmic && options.centered) {
        throw Error("Scalars are colorized logarithmically or centered on 0, not both");
    }
}

uint32_t ScalarFlags(const ColorizeOptions& options) {
    return (options.minimum ? kFixedMinimum : 0) | (options.maximum ? kFixedMaximum : 0) |
           (options.logarithmic ? kLogarithmic : 0) | (options.reversed ? kReversed : 0) |
           (options.centered ? kCentered : 0);
}

void RequireHdr(device::DType type, int64_t channels) {
    if (type != device::DType::Float32 || (channels != 3 && channels != 4)) {
        throw Error("HDR color is colorized from float32 (rows, cols, 3 or 4) pixels");
    }
}

void RequireNormals(device::DType type, int64_t channels) {
    if (type != device::DType::Float32 || (channels != 3 && channels != 4)) {
        throw Error("Normals are colorized from float32 (rows, cols, 3 or 4) pixels");
    }
}

void RequireIds(device::DType type, int64_t channels) {
    if (type != device::DType::UInt32 || (channels != 1 && channels != 4)) {
        throw Error("Ids are colorized from uint32 (rows, cols[, 1 or 4]) pixels");
    }
}

uint32_t PixelId(Span<const uint32_t> words) {
    uint32_t key = words.empty() ? 0 : words[0];
    bool empty = key == 0;
    for (size_t c = 1; c < words.size(); ++c) {
        key = MixIdWord(key, words[c]);
        empty = empty && words[c] == 0;
    }
    return empty ? 0 : IdRgba(key);
}

struct Layout {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t channels = 1;
    bool compact = false;
};

Layout Describe(const device::Tensor& tensor) {
    const Span<const int64_t> shape = tensor.Shape();
    if ((shape.size() != 2 && shape.size() != 3) || shape[0] < 1 || shape[1] < 1) {
        throw Error("Pixels come from (rows, cols[, channels]) tensors, not {} dimensions",
                    shape.size());
    }
    Layout layout{shape[0], shape[1], shape.size() == 3 ? shape[2] : 1, true};
    const Span<const int64_t> strides = tensor.Strides();
    int64_t expected = 1;
    for (size_t axis = shape.size(); axis-- > 0;) {
        layout.compact = layout.compact && (shape[axis] <= 1 || strides[axis] == expected);
        expected *= shape[axis];
    }
    return layout;
}

device::Tensor Launch(const std::vector<const char*>& entries, const device::Tensor& input,
                      ColorizeParams params, const Ref<device::Buffer>& range) {
    const Layout layout = Describe(input);
    const uint64_t count = uint64_t(layout.rows) * uint64_t(layout.cols);
    if (count > std::numeric_limits<uint32_t>::max()) {
        throw Error("Colorizing {} pixels; it takes at most {}", count,
                    std::numeric_limits<uint32_t>::max());
    }
    device::Device& device = input.GetDevice();
    const Ref<device::Program> program =
        device.CreateProgram({.modules = {"ernie.io.kernels.colorize"},
                              .library = &kIoKernels,
                              .entries = {entries.begin(), entries.end()},
                              .label = "colorize"});
    device::Tensor output =
        device.CreateTensor(device::DType::UInt8, {layout.rows, layout.cols, 4}, "colorized");
    const uint64_t words = count * uint64_t(layout.channels);
    params.input = device::RequireRange<uint32_t>(
        *program, "Colorize", input.GetBuffer().Slice(input.Offset(), words * 4), words);
    params.output = {output.Address()};
    params.count = uint32_t(count);
    params.channels = uint32_t(layout.channels);

    device::Queue& queue = device.GetQueue();
    device::CommandEncoder encoder = queue.CreateEncoder();
    const device::Size3 groups = device::FoldGroups(device::GroupCount(count, kColorizeThreads));
    for (const char* entry : entries) {
        encoder.Barrier();
        encoder.Dispatch(program->Entry(entry), params, groups);
    }
    const device::Operation done = queue.Submit(encoder.Finish());
    queue.Retire(std::make_shared<
                     std::tuple<Ref<device::Program>, Ref<device::Buffer>, Ref<device::Buffer>>>(
                     program, Ref<device::Buffer>(&input.GetBuffer()), range),
                 done);
    return output;
}

ScalarBounds HostBounds(const Pixels& values, const ColorizeOptions& options) {
    const Span<const float> all = values.Values<float>();
    const uint32_t flags = ScalarFlags(options);
    const size_t channels = size_t(values.channels);
    const size_t count = size_t(values.rows) * size_t(values.cols);
    float low = std::numeric_limits<float>::infinity();
    float high = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < count; ++i) {
        const float value = all[i * channels + options.channel];
        if (ScalarShown(value, flags)) {
            low = std::min(low, value);
            high = std::max(high, value);
        }
    }
    return ResolveBounds(low, high, options.minimum.value_or(0), options.maximum.value_or(0),
                         flags);
}

}

std::array<uint8_t, 4> ColormapColor(Colormap colormap, float position) {
    const uint32_t rgba = ColormapRgba(uint32_t(colormap), position);
    return {uint8_t(rgba), uint8_t(rgba >> 8), uint8_t(rgba >> 16), uint8_t(rgba >> 24)};
}

Pixels ColorizeScalars(const Pixels& values, const ColorizeOptions& options) {
    RequireScalars(values.type, values.channels, options);
    const Span<const float> all = values.Values<float>();
    const uint32_t flags = ScalarFlags(options);
    const size_t channels = size_t(values.channels);
    const size_t count = size_t(values.rows) * size_t(values.cols);
    const ScalarBounds bounds = HostBounds(values, options);
    std::vector<uint32_t> rgba(count);
    for (size_t i = 0; i < count; ++i) {
        rgba[i] = ScalarRgba(all[i * channels + options.channel], bounds, flags,
                             uint32_t(options.colormap));
    }
    return Packed(values, rgba);
}

Pixels ColorizeHdr(const Pixels& color, float exposure) {
    RequireHdr(color.type, color.channels);
    const Span<const float> values = color.Values<float>();
    const float scale = std::exp2(exposure);
    std::vector<uint32_t> rgba(size_t(color.rows) * size_t(color.cols));
    for (size_t i = 0; i < rgba.size(); ++i) {
        const float* pixel = values.data() + i * size_t(color.channels);
        rgba[i] =
            HdrRgba(pixel[0], pixel[1], pixel[2], color.channels == 4 ? pixel[3] : 1.0f, scale);
    }
    return Packed(color, rgba);
}

Pixels ColorizeNormals(const Pixels& normals) {
    RequireNormals(normals.type, normals.channels);
    const Span<const float> values = normals.Values<float>();
    std::vector<uint32_t> rgba(size_t(normals.rows) * size_t(normals.cols));
    for (size_t i = 0; i < rgba.size(); ++i) {
        const float* normal = values.data() + i * size_t(normals.channels);
        rgba[i] = NormalRgba(normal[0], normal[1], normal[2]);
    }
    return Packed(normals, rgba);
}

std::array<uint8_t, 4> IdColor(Span<const uint32_t> words) {
    const uint32_t rgba = PixelId(words);
    return {uint8_t(rgba), uint8_t(rgba >> 8), uint8_t(rgba >> 16), uint8_t(rgba >> 24)};
}

Pixels ColorizeIds(const Pixels& ids) {
    RequireIds(ids.type, ids.channels);
    const Span<const uint32_t> words = ids.Values<uint32_t>();
    const size_t channels = size_t(ids.channels);
    std::vector<uint32_t> rgba(size_t(ids.rows) * size_t(ids.cols));
    for (size_t i = 0; i < rgba.size(); ++i) {
        rgba[i] = PixelId(Span<const uint32_t>(words.data() + i * channels, channels));
    }
    return Packed(ids, rgba);
}

namespace {

device::Tensor ScalarTensor(const device::Tensor& values, const ColorizeOptions& options,
                            device::Tensor* bounds) {
    const Layout layout = Describe(values);
    RequireScalars(values.Type(), layout.channels, options);
    if (!layout.compact) {
        const Pixels pixels(values);
        if (bounds) {
            const ScalarBounds used = HostBounds(pixels, options);
            const float words[2] = {used.low, used.high};
            *bounds = device::Tensor(values.GetDevice().CreateBuffer({.label = "colorize bounds"},
                                                                     Span<const float>(words)),
                                     0, device::DType::Float32, {2});
        }
        return ToTensor(values, ColorizeScalars(pixels, options));
    }
    ColorizeParams params{};
    params.channel = options.channel;
    params.colormap = uint32_t(options.colormap);
    params.flags = ScalarFlags(options);
    params.minimum = options.minimum.value_or(0);
    params.maximum = options.maximum.value_or(0);
    if (bounds) {
        *bounds = values.GetDevice().CreateTensor(device::DType::Float32, {2}, "colorize bounds");
        params.bounds = {bounds->Address()};
        params.flags |= kWriteBounds;
    }
    std::vector<const char*> entries = {"ColorizeScalars"};
    const uint32_t initial[2] = {kNoRange, 0};
    const Ref<device::Buffer> range =
        values.GetDevice().CreateBuffer({.label = "colorize range"}, Span<const uint32_t>(initial));
    params.range = {range->Address()};
    if (!options.minimum || !options.maximum) {
        entries.insert(entries.begin(), "ScalarRange");
    }
    return Launch(entries, values, params, range);
}

}

device::Tensor ColorizeScalars(const device::Tensor& values, const ColorizeOptions& options) {
    return ScalarTensor(values, options, nullptr);
}

device::Tensor ColorizeScalars(const device::Tensor& values, const ColorizeOptions& options,
                               device::Tensor& bounds) {
    return ScalarTensor(values, options, &bounds);
}

device::Tensor ColorizeHdr(const device::Tensor& color, float exposure) {
    const Layout layout = Describe(color);
    RequireHdr(color.Type(), layout.channels);
    if (!layout.compact) {
        return ToTensor(color, ColorizeHdr(Pixels(color), exposure));
    }
    ColorizeParams params{};
    params.scale = std::exp2(exposure);
    return Launch({"ColorizeHdr"}, color, params, {});
}

device::Tensor ColorizeNormals(const device::Tensor& normals) {
    const Layout layout = Describe(normals);
    RequireNormals(normals.Type(), layout.channels);
    if (!layout.compact) {
        return ToTensor(normals, ColorizeNormals(Pixels(normals)));
    }
    return Launch({"ColorizeNormals"}, normals, {}, {});
}

device::Tensor ColorizeIds(const device::Tensor& ids) {
    const Layout layout = Describe(ids);
    RequireIds(ids.Type(), layout.channels);
    if (!layout.compact) {
        return ToTensor(ids, ColorizeIds(Pixels(ids)));
    }
    return Launch({"ColorizeIds"}, ids, {}, {});
}

}
