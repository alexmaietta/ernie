// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/io/image.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/device.h"
#include "ernie/io/kernels/encoding.h"
#include "ernie/util/error.h"
#include "support/devices.h"
#include "support/support.h"

namespace ernie {
namespace {

std::filesystem::path Temporary(const char* name) {
    static const test::TemporaryDirectory directory;
    return directory.Path() / name;
}

Image Gradient() {
    Image image(4, 3);
    for (uint32_t y = 0; y < image.height; ++y) {
        for (uint32_t x = 0; x < image.width; ++x) {
            image.Pixel(x, y) = {float(x) / 3, float(y) / 2, 0.25f, 1};
        }
    }
    return image;
}

TEST(ImageTest, PngsRoundTripThroughSrgb) {
    const Image image = Gradient();
    const std::filesystem::path path = Temporary("ernie_io_gradient.png");
    WritePng(path, image);
    const Image loaded = LoadImageFile(path, ImageEncoding::Srgb);
    ASSERT_EQ(loaded.width, 4u);
    ASSERT_EQ(loaded.height, 3u);
    for (size_t i = 0; i < image.pixels.size(); ++i) {
        EXPECT_NEAR(loaded.pixels[i].r, image.pixels[i].r, 0.01f) << i;
        EXPECT_NEAR(loaded.pixels[i].g, image.pixels[i].g, 0.01f) << i;
        EXPECT_NEAR(loaded.pixels[i].b, image.pixels[i].b, 0.01f) << i;
        EXPECT_EQ(loaded.pixels[i].a, 1.0f) << i;
    }
    EXPECT_TRUE(IsPngOrJpeg("a.PNG"));
    EXPECT_FALSE(IsPngOrJpeg("a.exr"));
    std::filesystem::remove(path);
}

TEST(ImageTest, ExrsRoundTripExactly) {
    const Image image = Gradient();
    const std::filesystem::path path = Temporary("ernie_io_gradient.exr");
    WriteExr(path, image);
    const Image loaded = LoadImageFile(path, ImageEncoding::Linear);
    ASSERT_EQ(loaded.pixels.size(), image.pixels.size());
    for (size_t i = 0; i < image.pixels.size(); ++i) {
        EXPECT_NEAR(loaded.pixels[i].r, image.pixels[i].r, 1e-3f) << i;
        EXPECT_NEAR(loaded.pixels[i].g, image.pixels[i].g, 1e-3f) << i;
    }
    EXPECT_NEAR(loaded.Mean().b, 0.25f, 1e-3f);
    std::filesystem::remove(path);
    EXPECT_THROW(LoadImageFile(Temporary("ernie_io_missing.exr"), ImageEncoding::Linear), Error);
    EXPECT_THROW(LoadImageFile("image.bmp", ImageEncoding::Linear), Error);
}

std::vector<std::byte> ReadBytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    const std::vector<char> text((std::istreambuf_iterator<char>(file)),
                                 std::istreambuf_iterator<char>());
    const auto bytes = std::as_bytes(std::span(text));
    return {bytes.begin(), bytes.end()};
}

TEST(ImageTest, BytesDecodeLikeTheirFiles) {
    const Image image = Gradient();
    for (const char* name : {"ernie_io_bytes.png", "ernie_io_bytes.exr"}) {
        const std::filesystem::path path = Temporary(name);
        std::string_view(name).ends_with(".png") ? WritePng(path, image) : WriteExr(path, image);
        const std::vector<std::byte> bytes = ReadBytes(path);
        const Image fromBytes = LoadImageBytes(bytes, name, ImageEncoding::Linear);
        const Image fromFile = LoadImageFile(path, ImageEncoding::Linear);
        EXPECT_EQ(fromBytes.width, fromFile.width) << name;
        EXPECT_EQ(fromBytes.height, fromFile.height) << name;
        EXPECT_EQ(fromBytes.pixels, fromFile.pixels) << name;
        std::filesystem::remove(path);
    }
    const std::vector<std::byte> junk(16, std::byte{7});
    EXPECT_THROW(LoadImageBytes(junk, "junk.png", ImageEncoding::Linear), Error);
    EXPECT_THROW(LoadImageBytes(junk, "junk.exr", ImageEncoding::Linear), Error);
    EXPECT_THROW(LoadImageBytes(junk, "junk.tga", ImageEncoding::Linear), Error);
}

TEST(ImageTest, GpuTensorsAreWrittenAndRead) {
    test::ForEachDevice([](device::Device& device) {
        std::vector<uint8_t> bytes;
        for (uint32_t camera = 0; camera < 2; ++camera) {
            for (uint32_t pixel = 0; pixel < 6; ++pixel) {
                bytes.insert(bytes.end(), {uint8_t(camera * 100 + pixel), 0, 255, 255});
            }
        }
        const Ref<device::Buffer> buffer =
            device.CreateBuffer({.label = "pixels"}, std::as_bytes(std::span(bytes)));
        const device::Tensor cameras(buffer, 0, device::DType::UInt8, {2, 2, 3, 4});
        const device::Tensor second = cameras.Select(1);
        EXPECT_EQ(std::vector<int64_t>(second.Shape().begin(), second.Shape().end()),
                  (std::vector<int64_t>{2, 3, 4}));
        EXPECT_THROW(cameras.Select(2), Error);

        const std::filesystem::path path = Temporary("ernie_io_camera.png");
        WritePng(path, second);
        const Image loaded = LoadImageFile(path, ImageEncoding::Linear);
        ASSERT_EQ(loaded.width, 3u);
        ASSERT_EQ(loaded.height, 2u);
        EXPECT_NEAR(loaded.Pixel(2, 1).r, 105.0f / 255, 1e-6f);
        const Rgba direct = ImageFromTensor(second, ImageEncoding::Linear).Pixel(2, 1);
        EXPECT_NEAR(direct.r, loaded.Pixel(2, 1).r, 1e-6f);
        EXPECT_NEAR(direct.b, 1.0f, 1e-6f);
        std::filesystem::remove(path);
    });
}

TEST(ImageTest, WriteImagePicksTheFormat) {
    test::ForEachDevice([](device::Device& device) {
        const std::vector<uint8_t> bytes(2 * 3 * 4, 128);
        const std::vector<float> floats(2 * 3, 4.5f);
        const device::Tensor color(
            device.CreateBuffer({.label = "color"}, std::as_bytes(std::span(bytes))), 0,
            device::DType::UInt8, {2, 3, 4});
        const device::Tensor depth(
            device.CreateBuffer({.label = "depth"}, std::as_bytes(std::span(floats))), 0,
            device::DType::Float32, {2, 3});

        const std::filesystem::path png = WriteImage(Temporary("ernie_io_any"), color);
        EXPECT_EQ(png.extension(), ".png");
        EXPECT_EQ(LoadImageFile(png, ImageEncoding::Linear).width, 3u);
        const std::filesystem::path exr = WriteImage(Temporary("ernie_io_any"), depth);
        EXPECT_EQ(exr.extension(), ".exr");
        EXPECT_NEAR(LoadImageFile(exr, ImageEncoding::Linear).Pixel(1, 1).r, 4.5f, 1e-6f);
        const std::filesystem::path jpeg = Temporary("ernie_io_any.JPG");
        EXPECT_EQ(WriteImage(jpeg, color), jpeg);
        EXPECT_TRUE(std::filesystem::exists(jpeg));
        EXPECT_THROW(WriteImage(Temporary("ernie_io_any.bmp"), color), Error);
        for (const std::filesystem::path& path : {png, exr, jpeg}) {
            std::filesystem::remove(path);
        }
    });
}

template <class T>
device::Tensor Upload(device::Device& device, const std::vector<T>& values,
                      std::vector<int64_t> shape, device::DType type) {
    return device::Tensor(
        device.CreateBuffer({.label = "values"}, std::as_bytes(std::span(values))), 0, type,
        std::move(shape));
}

TEST(ImageTest, DataIsColorizedForViewing) {
    test::ForEachDevice([](device::Device& device) {
        const float inf = std::numeric_limits<float>::infinity();
        const device::Tensor depth =
            Upload(device, std::vector<float>{1, 10, 100, inf}, {2, 2}, device::DType::Float32);
        const std::vector<uint8_t> shades = ColorizeScalars(depth, kDepthColors).ToHost<uint8_t>();
        ASSERT_EQ(shades.size(), 16u);
        EXPECT_EQ(shades[0], 255);
        EXPECT_NEAR(shades[4], 128, 1);
        EXPECT_EQ(shades[8], 0);
        EXPECT_EQ(shades[12], 0);
        EXPECT_EQ(shades[15], 255);
        EXPECT_EQ(
            ColorizeScalars(
                depth, {.minimum = 10.0f, .maximum = 100.0f, .logarithmic = true, .reversed = true})
                .ToHost<uint8_t>()[0],
            255);

        const device::Tensor normals =
            Upload(device, std::vector<float>{1, 0, -1, 0, 0, 1, 0, 0, 0}, {1, 3, 3},
                   device::DType::Float32);
        EXPECT_EQ(ColorizeNormals(normals).ToHost<uint8_t>(),
                  (std::vector<uint8_t>{255, 128, 0, 255, 128, 128, 255, 255, 0, 0, 0, 0}));

        const device::Tensor ids =
            Upload(device, std::vector<uint32_t>{0, 1, 2, 1}, {2, 2}, device::DType::UInt32);
        const std::vector<uint8_t> colors = ColorizeIds(ids).ToHost<uint8_t>();
        EXPECT_EQ(colors[3], 0);
        EXPECT_EQ(colors[7], 255);
        EXPECT_EQ(std::vector<uint8_t>(colors.begin() + 4, colors.begin() + 8),
                  std::vector<uint8_t>(colors.begin() + 12, colors.end()));
        EXPECT_NE(std::vector<uint8_t>(colors.begin() + 4, colors.begin() + 8),
                  std::vector<uint8_t>(colors.begin() + 8, colors.begin() + 12));

        EXPECT_THROW(ColorizeScalars(ids), Error);
        EXPECT_THROW(ColorizeScalars(depth, {.channel = 1}), Error);
        EXPECT_THROW(ColorizeScalars(depth, {.logarithmic = true, .centered = true}), Error);
        EXPECT_THROW(ColorizeHdr(depth), Error);
        EXPECT_THROW(ColorizeNormals(depth), Error);
        EXPECT_THROW(ColorizeIds(depth), Error);
        EXPECT_THROW(WritePng(Temporary("ernie_io_ids.png"), ids), Error);
        EXPECT_THROW(WritePng(Temporary("ernie_io_depth.png"), depth), Error);
    });
}

TEST(ImageTest, HostValuesAreColorizedAndWritten) {
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> depth = {1, 10, 100, inf};
    const Pixels shades = ColorizeScalars(Pixels(depth, 2, 2), kDepthColors);
    EXPECT_EQ(shades.channels, 4);
    EXPECT_EQ(shades.Values<uint8_t>()[0], 255);
    EXPECT_EQ(shades.Values<uint8_t>()[12], 0);
    EXPECT_THROW(shades.Values<float>(), std::invalid_argument);
    EXPECT_THROW(Pixels(depth, 3, 2), Error);

    const std::vector<uint32_t> ids = {0, 1, 2, 1};
    const std::filesystem::path png =
        WriteImage(Temporary("ernie_io_ids"), ColorizeIds(Pixels(ids, 2, 2)));
    EXPECT_EQ(png.extension(), ".png");
    EXPECT_EQ(LoadImageFile(png, ImageEncoding::Linear).width, 2u);
    const std::filesystem::path exr = WriteImage(Temporary("ernie_io_depth"), Pixels(depth, 2, 2));
    EXPECT_EQ(exr.extension(), ".exr");
    EXPECT_NEAR(LoadImageFile(exr, ImageEncoding::Linear).Pixel(1, 0).r, 10.0f, 1e-6f);
    for (const std::filesystem::path& path : {png, exr}) {
        std::filesystem::remove(path);
    }
}

TEST(ImageTest, ScalarsFollowTheirOptions) {
    const std::vector<float> values = {-2, 0, 1, 2};
    const Pixels linear = ColorizeScalars(Pixels(values, 2, 2));
    EXPECT_EQ(linear.Values<uint8_t>()[0], 0);
    EXPECT_EQ(linear.Values<uint8_t>()[4], 128);
    EXPECT_EQ(linear.Values<uint8_t>()[12], 255);

    const Pixels centered =
        ColorizeScalars(Pixels(std::vector<float>{-1, 0, 0.5f, 4}, 2, 2), {.centered = true});
    EXPECT_EQ(centered.Values<uint8_t>()[0], 96);
    EXPECT_EQ(centered.Values<uint8_t>()[4], 128);
    EXPECT_EQ(centered.Values<uint8_t>()[12], 255);
    const Pixels flat = ColorizeScalars(Pixels(std::vector<float>{0, 0}, 1, 2), {.centered = true});
    EXPECT_EQ(flat.Values<uint8_t>()[0], 128);

    const Pixels fixed =
        ColorizeScalars(Pixels(values, 2, 2), {.minimum = 0.0f, .maximum = 1.0f, .reversed = true});
    EXPECT_EQ(fixed.Values<uint8_t>()[0], 255);
    EXPECT_EQ(fixed.Values<uint8_t>()[8], 0);

    const Pixels channel =
        ColorizeScalars(Pixels(std::vector<float>{5, 0, 5, 1}, 1, 2, 2), {.channel = 1});
    EXPECT_EQ(channel.Values<uint8_t>()[0], 0);
    EXPECT_EQ(channel.Values<uint8_t>()[4], 255);

    const Pixels turbo = ColorizeScalars(Pixels(values, 2, 2), {.colormap = Colormap::Turbo});
    const std::array<uint8_t, 4> low = ColormapColor(Colormap::Turbo, 0);
    EXPECT_EQ(
        std::vector<uint8_t>(turbo.Values<uint8_t>().begin(), turbo.Values<uint8_t>().begin() + 4),
        std::vector<uint8_t>(low.begin(), low.end()));
    EXPECT_NE(ColormapColor(Colormap::Viridis, 0.5f), ColormapColor(Colormap::Turbo, 0.5f));
    EXPECT_EQ(ColormapColor(Colormap::Grayscale, 2), (std::array<uint8_t, 4>{255, 255, 255, 255}));

    const float inf = std::numeric_limits<float>::infinity();
    const Pixels missing =
        ColorizeScalars(Pixels(std::vector<float>{inf, 1}, 1, 2), {.colormap = Colormap::Viridis});
    EXPECT_EQ(missing.Values<uint8_t>()[0], 0);
    EXPECT_EQ(missing.Values<uint8_t>()[3], 255);
}

TEST(ImageTest, HdrColorIsExposedLikeLdrColor) {
    const std::vector<float> color = {0.5f, 0.25f, 2, 1, 0.1f, 0, 0, 0};
    const Pixels shown = ColorizeHdr(Pixels(color, 1, 2, 4));
    const Span<const uint8_t> bytes = shown.Values<uint8_t>();
    EXPECT_EQ(bytes[0], EncodeUnorm8(EncodeSrgb(0.5f)));
    EXPECT_EQ(bytes[1], EncodeUnorm8(EncodeSrgb(0.25f)));
    EXPECT_EQ(bytes[2], 255);
    EXPECT_EQ(bytes[3], 255);
    EXPECT_EQ(bytes[7], 0);
    const Pixels brighter = ColorizeHdr(Pixels(color, 1, 2, 4), 1);
    EXPECT_EQ(brighter.Values<uint8_t>()[0], 255);
    EXPECT_EQ(brighter.Values<uint8_t>()[4], EncodeUnorm8(EncodeSrgb(0.2f)));
    EXPECT_EQ(ColorizeHdr(Pixels(std::vector<float>{0, 0, 0}, 1, 1, 3)).Values<uint8_t>()[3], 255);
}

TEST(ImageTest, GpuColorizingMatchesTheHost) {
    test::ForEachDevice([](device::Device& device) {
        constexpr int64_t kRows = 300;
        constexpr int64_t kCols = 211;
        std::vector<float> depth(size_t(kRows * kCols));
        std::vector<float> normals(depth.size() * 4);
        std::vector<uint32_t> ids(depth.size() * 4);
        for (size_t i = 0; i < depth.size(); ++i) {
            depth[i] = i % 17 == 0 ? std::numeric_limits<float>::infinity() : 0.5f + float(i % 997);
            const float angle = float(i) * 0.01f;
            normals[4 * i] = std::cos(angle);
            normals[4 * i + 1] = std::sin(angle);
            normals[4 * i + 2] = -0.25f;
            for (uint32_t c = 0; c < 4; ++c) {
                ids[4 * i + c] = i % 5 == 0 ? 0 : uint32_t(i % 41) * (c + 1) + c;
            }
        }
        const auto matches = [](const std::vector<uint8_t>& gpu, const Pixels& host) {
            const Span<const uint8_t> expected = host.Values<uint8_t>();
            ASSERT_EQ(gpu.size(), expected.size());
            for (size_t i = 0; i < gpu.size(); ++i) {
                ASSERT_LE(std::abs(int(gpu[i]) - int(expected[i])), 1) << i;
            }
        };
        const device::Tensor depths = Upload(device, depth, {kRows, kCols}, device::DType::Float32);
        const ColorizeOptions options[] = {
            kDepthColors,
            {.minimum = 2.0f, .logarithmic = true, .reversed = true},
            {.minimum = 2.0f, .maximum = 50.0f, .logarithmic = true, .reversed = true},
            {.colormap = Colormap::Turbo},
            {.colormap = Colormap::Viridis, .maximum = 300.0f, .reversed = true},
            {.colormap = Colormap::Turbo, .centered = true},
        };
        for (const ColorizeOptions& each : options) {
            matches(ColorizeScalars(depths, each).ToHost<uint8_t>(),
                    ColorizeScalars(Pixels(depth, kRows, kCols), each));
        }
        std::vector<float> signedValues(depth.size());
        for (size_t i = 0; i < depth.size(); ++i) {
            signedValues[i] = float(int(i % 997) - 400) * 0.25f;
        }
        matches(
            ColorizeScalars(Upload(device, signedValues, {kRows, kCols}, device::DType::Float32),
                            {.colormap = Colormap::Turbo, .centered = true})
                .ToHost<uint8_t>(),
            ColorizeScalars(Pixels(signedValues, kRows, kCols),
                            {.colormap = Colormap::Turbo, .centered = true}));
        const device::Tensor hdr =
            Upload(device, normals, {kRows, kCols, 4}, device::DType::Float32);
        matches(ColorizeHdr(hdr, 1.5f).ToHost<uint8_t>(),
                ColorizeHdr(Pixels(normals, kRows, kCols, 4), 1.5f));
        matches(ColorizeScalars(hdr, {.channel = 1}).ToHost<uint8_t>(),
                ColorizeScalars(Pixels(normals, kRows, kCols, 4), {.channel = 1}));
        matches(ColorizeNormals(Upload(device, normals, {kRows, kCols, 4}, device::DType::Float32))
                    .ToHost<uint8_t>(),
                ColorizeNormals(Pixels(normals, kRows, kCols, 4)));
        const Pixels hostIds = ColorizeIds(Pixels(ids, kRows, kCols, 4));
        EXPECT_EQ(ColorizeIds(Upload(device, ids, {kRows, kCols, 4}, device::DType::UInt32))
                      .ToHost<uint8_t>(),
                  std::vector<uint8_t>(hostIds.Values<uint8_t>().begin(),
                                       hostIds.Values<uint8_t>().end()));
    });
}

}
}
