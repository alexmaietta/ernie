// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/buffer.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/raster/gpu_block_encoder.h"
#include "ernie/raster/material.h"
#include "ernie/util/texture_bake.h"
#include "support/vulkan_fixture.h"

#include <bc7decomp.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

using ernie::GpuBlockEncoder;
using ernie::TextureFormat;

namespace {

class GpuBlockEncoderTest : public ernie::testing::VulkanTest {};

/// An RGBA8 image of smooth gradients and sharper detail, as textures
/// have.
std::vector<std::uint8_t> Picture(std::uint32_t width,
                                  std::uint32_t height) {
    std::vector<std::uint8_t> rgba;
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const double s = std::sin(x * 0.21) * std::cos(y * 0.13);
            rgba.push_back(
                static_cast<std::uint8_t>(x * 255 / width));
            rgba.push_back(
                static_cast<std::uint8_t>(y * 255 / height));
            rgba.push_back(
                static_cast<std::uint8_t>(127.5 + 127.0 * s));
            rgba.push_back(static_cast<std::uint8_t>(
                (x / 8 + y / 8) % 2 == 0 ? 255 : 160));
        }
    }
    return rgba;
}

/// The peak signal-to-noise ratio of BC7 `blocks` decoded against
/// `rgba` (`width` x `height`), over every channel, in decibels.
double Psnr(const std::vector<std::uint8_t>& rgba,
            std::uint32_t width, std::uint32_t height,
            const std::vector<std::byte>& blocks) {
    const std::uint32_t across = (width + 3) / 4;
    double squared = 0.0;
    std::size_t count = 0;
    for (std::uint32_t by = 0; by < (height + 3) / 4; ++by) {
        for (std::uint32_t bx = 0; bx < across; ++bx) {
            bc7decomp::color_rgba decoded[16];
            EXPECT_TRUE(bc7decomp::unpack_bc7(
                blocks.data() + (std::size_t{by} * across + bx) * 16,
                decoded));
            for (std::uint32_t i = 0; i < 16; ++i) {
                const std::uint32_t x = bx * 4 + i % 4;
                const std::uint32_t y = by * 4 + i / 4;
                if (x >= width || y >= height) {
                    continue;
                }
                for (int c = 0; c < 4; ++c) {
                    const double d =
                        double(decoded[i].m_comps[c]) -
                        double(rgba[(std::size_t{y} * width + x) * 4 +
                                    static_cast<std::size_t>(c)]);
                    squared += d * d;
                    ++count;
                }
            }
        }
    }
    const double mse = squared / static_cast<double>(count);
    return 10.0 * std::log10(255.0 * 255.0 / std::max(mse, 1e-12));
}

/// BC7 by DirectXTex's compute encoder (the fixture
/// third_party/directxtex/bc7_encode.slang) as BCDirectCompute.cpp
/// dispatches it: modes 4, 5 and 6 into the first errors, modes 1, 3
/// and 7 back and forth, then the best into the blocks. All of `rgba`
/// in one batch (at most 65 532 blocks).
std::vector<std::byte>
DirectXTexBc7(const ernie::VulkanContext& context,
              const std::vector<std::uint8_t>& rgba,
              std::uint32_t width, std::uint32_t height) {
    struct Constants {
        std::uint32_t texWidth;
        std::uint32_t blocksAcross;
        std::uint32_t format;
        std::uint32_t mode;
        std::uint32_t firstBlock;
        std::uint32_t blocks;
        float alphaWeight;
        std::uint32_t texHeight;
    };
    const std::uint32_t across = (width + 3) / 4;
    const std::uint32_t blocks = across * ((height + 3) / 4);
    EXPECT_LE(blocks, 65532u);
    std::ifstream file(std::filesystem::path(ERNIE_TEST_FIXTURES) /
                       "third_party" / "directxtex" /
                       "bc7_encode.slang");
    const std::string shader((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
    EXPECT_FALSE(shader.empty());
    ernie::ShaderCompiler compiler;
    const std::vector<ernie::ShaderModuleSource> modules{
        {"bc7_encode", shader}};
    const auto pipeline = [&](const char* entry) {
        return std::make_unique<ernie::ComputePipeline>(
            context, compiler.CompileSpirv(modules, entry), 4);
    };
    const auto tryMode456 = pipeline("TryMode456CS");
    const auto tryMode137 = pipeline("TryMode137CS");
    const auto encodeBlock = pipeline("EncodeBlockCS");

    constexpr VkBufferUsageFlags kStorage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const auto buffer = [&](VkDeviceSize bytes) {
        return std::make_unique<ernie::GpuBuffer>(
            context, bytes, kStorage, ernie::BufferMemory::Device);
    };
    const auto source = buffer(rgba.size());
    source->Write(std::as_bytes(std::span(rgba)));
    // Four-block groups run past the last block by up to three.
    const VkDeviceSize blockBytes = VkDeviceSize{blocks + 4} * 16;
    const auto errors1 = buffer(blockBytes);
    const auto errors2 = buffer(blockBytes);
    const auto encoded = buffer(blockBytes);
    constexpr std::array<std::uint32_t, 5> kModes{0, 1, 3, 7, 0};
    const VkDeviceSize stride = std::max<VkDeviceSize>(
        context.Limits().minStorageBufferOffsetAlignment,
        sizeof(Constants));
    std::vector<std::byte> constants(stride * kModes.size());
    for (std::size_t pass = 0; pass < kModes.size(); ++pass) {
        const Constants c{width, across, 98,   kModes[pass],
                          0,     blocks, 1.0f, height};
        std::memcpy(constants.data() + stride * pass, &c, sizeof(c));
    }
    const auto parameters = buffer(constants.size());
    parameters->Write(constants);

    context.Submit([&](VkCommandBuffer commands) {
        const auto dispatch = [&](const ernie::ComputePipeline& p,
                                  std::size_t pass,
                                  const ernie::GpuBuffer& in,
                                  const ernie::GpuBuffer& output,
                                  std::uint32_t groups) {
            const ernie::DescriptorResource resources[] = {
                ernie::DescriptorResource::Of(*source),
                ernie::DescriptorResource::Of(in),
                ernie::DescriptorResource::Of(output),
                ernie::DescriptorResource::Of(
                    *parameters, stride * pass, sizeof(Constants))};
            p.Dispatch(commands, resources, groups);
            ernie::ComputeBarrier(commands);
        };
        const std::uint32_t quads = (blocks + 3) / 4;
        dispatch(*tryMode456, 0, *errors2, *errors1, quads);
        dispatch(*tryMode137, 1, *errors1, *errors2, blocks);
        dispatch(*tryMode137, 2, *errors2, *errors1, blocks);
        dispatch(*tryMode137, 3, *errors1, *errors2, blocks);
        dispatch(*encodeBlock, 4, *errors2, *encoded, quads);
    });
    std::vector<std::byte> out(VkDeviceSize{blocks} * 16);
    encoded->Read(std::span(out));
    return out;
}

}  // namespace

TEST_F(GpuBlockEncoderTest, EncodesBc7AsWellAsTheReferences) {
    // 61 x 37: blocks over the right and bottom edges. The GPU's BC7
    // decodes to the source as closely as the CPU's (bc7enc, effort
    // 1) and DirectXTex's (modes 1 and 3 to 7), within a quarter
    // decibel (in fact about 0.8 dB closer than the CPU's).
    constexpr std::uint32_t kWidth = 61;
    constexpr std::uint32_t kHeight = 37;
    const std::vector<std::uint8_t> rgba = Picture(kWidth, kHeight);
    const std::size_t bytes =
        ernie::ImageBytes(TextureFormat::Bc7, kWidth, kHeight);
    ernie::ShaderCompiler compiler;
    GpuBlockEncoder encoder(*context_, compiler);
    EXPECT_TRUE(encoder.Encodes(TextureFormat::Bc7Srgb));
    EXPECT_FALSE(encoder.Encodes(TextureFormat::Bc5));
    std::vector<std::byte> gpu(bytes);
    encoder.Encode(rgba, kWidth, kHeight, TextureFormat::Bc7,
                   gpu.data());

    ernie::Image image(kWidth, kHeight);
    for (std::size_t i = 0; i < image.pixels.size(); ++i) {
        for (int c = 0; c < 4; ++c) {
            image.pixels[i][c] =
                rgba[i * 4 + static_cast<std::size_t>(c)] / 255.0f;
        }
    }
    const ernie::TextureData cpu =
        ernie::EncodeTexture(std::span(&image, 1), TextureFormat::Bc7,
                             ernie::ImageEncoding::Linear);
    const std::vector<std::byte> cpuBlocks(cpu.Mip(0, 0).begin(),
                                           cpu.Mip(0, 0).end());
    const double gpuPsnr = Psnr(rgba, kWidth, kHeight, gpu);
    const double cpuPsnr = Psnr(rgba, kWidth, kHeight, cpuBlocks);
    // DirectXTex's encoder needs integer dot products.
    const double directXTexPsnr =
        context_->SupportsIntegerDotProduct()
            ? Psnr(rgba, kWidth, kHeight,
                   DirectXTexBc7(*context_, rgba, kWidth, kHeight))
            : 0.0;
    std::printf("[ info ] PSNR: GPU %.2f dB, CPU %.2f dB, DirectXTex "
                "%.2f dB\n",
                gpuPsnr, cpuPsnr, directXTexPsnr);
    EXPECT_GT(gpuPsnr, 35.0);
    EXPECT_GT(gpuPsnr, cpuPsnr - 0.25);
    EXPECT_GT(gpuPsnr, directXTexPsnr - 0.25);
}

TEST_F(GpuBlockEncoderTest, LargeImagesEncodeInBatches) {
    // 1100 x 1100: 75 625 blocks, more than a batch; every block
    // decodes close to its source.
    constexpr std::uint32_t kSide = 1100;
    const std::vector<std::uint8_t> rgba = Picture(kSide, kSide);
    ernie::ShaderCompiler compiler;
    GpuBlockEncoder encoder(*context_, compiler);
    std::vector<std::byte> blocks(
        ernie::ImageBytes(TextureFormat::Bc7, kSide, kSide));
    const auto start = std::chrono::steady_clock::now();
    encoder.Encode(rgba, kSide, kSide, TextureFormat::Bc7,
                   blocks.data());
    const double seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start)
            .count();
    EXPECT_GT(Psnr(rgba, kSide, kSide, blocks), 35.0);
    std::printf("[ info ] 1100 x 1100 BC7 on the GPU: %.3f s\n",
                seconds);
}

TEST_F(GpuBlockEncoderTest, CompressionInPartsMatchesItWhole) {
    // A 300 x 200 RGBA8 texture and its mips compressed whole, and a
    // part of some 5000 texels at a time (strips of whole block rows,
    // as frames spread it): the same blocks.
    constexpr std::uint32_t kWidth = 300;
    constexpr std::uint32_t kHeight = 200;
    ernie::PreparedTexture prepared;
    prepared.width = kWidth;
    prepared.height = kHeight;
    prepared.format = VK_FORMAT_R8G8B8A8_SRGB;
    for (std::uint32_t w = kWidth, h = kHeight;;
         w = std::max(1u, w / 2), h = std::max(1u, h / 2)) {
        prepared.mips.push_back(Picture(w, h));
        if (w == 1 && h == 1) {
            break;
        }
    }
    ernie::ShaderCompiler compiler;
    GpuBlockEncoder encoder(*context_, compiler);
    ernie::PreparedTexture whole = prepared;
    ernie::CompressPreparedTexture(encoder, whole);
    ernie::TextureCompression parts(prepared);
    int steps = 0;
    while (!parts.Step(encoder, 5000)) {
        ++steps;
        EXPECT_GT(parts.Remaining(), 0u);
    }
    EXPECT_EQ(parts.Remaining(), 0u);
    EXPECT_GT(steps, 5);
    const ernie::PreparedTexture compressed = parts.Take();
    EXPECT_EQ(compressed.format, VK_FORMAT_BC7_SRGB_BLOCK);
    EXPECT_EQ(compressed.mips, whole.mips);
}
