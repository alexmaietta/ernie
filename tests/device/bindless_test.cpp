// SPDX-License-Identifier: BSD-3-Clause
/// @file bindless_test.cpp
/// @brief Tests of images and the bindless heap: one kernel paints a storage image by its index,
/// another samples it as a texture, under validation.
#include "ernie/device/arena.h"
#include "ernie/device/bindless.h"
#include "ernie/device/kernel.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/device/staging.h"

#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <glm/vec4.hpp>

#include <cstring>
#include <vector>

namespace ernie {
namespace {

using test::ExpectClean;

const ShaderModule kBindless{"bindless", R"(
[[vk::binding(0, 0)]] public Texture2D textures[];
[[vk::binding(1, 0)]] public RWTexture2D<float4> images[];
[[vk::binding(2, 0)]] public SamplerState samplers[];
)"};

const ShaderModule kPaint{"paint", R"(
import bindless;
struct Parameters { uint image; uint width; uint height; };
[[vk::push_constant]] ConstantBuffer<Parameters> parameters;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x < parameters.width && id.y < parameters.height) {
        images[NonUniformResourceIndex(parameters.image)][id.xy] = float4(id.x, id.y, 1, 1);
    }
}
)"};

const ShaderModule kSample{"sample", R"(
import bindless;
struct Parameters { float4* out; uint texture; uint sampler; uint width; uint height; };
[[vk::push_constant]] ConstantBuffer<Parameters> parameters;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x < parameters.width && id.y < parameters.height) {
        float2 uv = (float2(id.xy) + 0.5) / float2(parameters.width, parameters.height);
        parameters.out[id.y * parameters.width + id.x] =
            textures[NonUniformResourceIndex(parameters.texture)]
                .SampleLevel(samplers[NonUniformResourceIndex(parameters.sampler)], uv, 0);
    }
}
)"};

struct PaintParameters {
    std::uint32_t image, width, height;
};
struct SampleParameters {
    VkDeviceAddress out;
    std::uint32_t texture, sampler, width, height;
};

/// Orders one dispatch's writes before the next's reads.
void ComputeBarrier(VkCommandBuffer commands) {
    const VkMemoryBarrier2 barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                                   .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                                   .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                   .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT};
    const VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                      .memoryBarrierCount = 1,
                                      .pMemoryBarriers = &barrier};
    vkCmdPipelineBarrier2(commands, &dependency);
}

TEST(BindlessHeap, KernelsPaintAndSampleImagesByIndex) {
    ERNIE_CONTEXT(context);
    const ShaderCompiler compiler;
    BindlessHeap heap(*context);
    const std::vector<ShaderModule> paintModules{kBindless, kPaint};
    const std::vector<ShaderModule> sampleModules{kBindless, kSample};
    const ComputeKernel paint(*context, compiler.Spirv(paintModules, "main"), "paint",
                              sizeof(PaintParameters), &heap);
    const ComputeKernel sample(*context, compiler.Spirv(sampleModules, "main"), "sample",
                               sizeof(SampleParameters), &heap);

    const std::uint32_t width = 16;
    const std::uint32_t height = 8;
    const Image image(*context,
                      {.format = ImageFormat::Rgba32Float,
                       .extent = {width, height, 1},
                       .usage = ImageUsage::Sampled | ImageUsage::Storage},
                      "canvas");
    const Sampler nearest(*context, {.filter = Filter::Nearest, .address = Address::Clamp},
                          "nearest");
    const BindlessIndex storage = heap.AddStorageImage(image);
    const BindlessIndex texture = heap.AddTexture(image);
    const BindlessIndex sampler = heap.AddSampler(nearest);

    Arenas arenas(*context, MemoryUse::Device, "out", 1 << 16);
    const BufferRange out = arenas.Allocate(width * height * sizeof(glm::vec4));
    const Ticket ready = image.Ready();
    const Ticket done = context->Main().Submit(
        [&](VkCommandBuffer commands) {
            paint.Dispatch(commands, {width / 8, height / 8, 1},
                           PaintParameters{storage.value, width, height});
            ComputeBarrier(commands);
            sample.Dispatch(
                commands, {width / 8, height / 8, 1},
                SampleParameters{arenas.Address(out), texture.value, sampler.value, width, height});
        },
        std::span(&ready, 1));

    Readback readback(*context, context->Main(), arenas.Storage(out), out.offset, out.size,
                      std::span(&done, 1));
    std::vector<glm::vec4> texels(width * height);
    std::memcpy(texels.data(), readback.Bytes().data(), out.size);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            ASSERT_EQ(texels[y * width + x], glm::vec4(x, y, 1, 1)) << x << ", " << y;
        }
    }
    ExpectClean(*context);
}

TEST(BindlessHeap, SlotsAreReusedOnceTheGpuIsDone) {
    ERNIE_CONTEXT(context);
    BindlessHeap heap(*context);
    const Image image(*context, {.format = ImageFormat::R32Float, .extent = {4, 4, 1}}, "small");
    const BindlessIndex first = heap.AddTexture(image);
    EXPECT_NE(heap.AddTexture(image), first);
    heap.RemoveTexture(first);
    context->WaitIdle();
    EXPECT_EQ(heap.AddTexture(image), first) << "a freed slot is taken first";
    ExpectClean(*context);
}

}  // namespace
}  // namespace ernie
