// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/gpu_block_encoder.h"

#include "ernie/device/buffer.h"
#include "ernie/kernel/embedded_shaders.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ernie {

namespace {

/// bc7.slang's `Constants`.
struct Constants {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t blocksAcross;
    std::uint32_t blocks;
    std::uint32_t firstBlock;
    std::uint32_t pad[3];
};
static_assert(sizeof(Constants) == 32);

/// Blocks a dispatch encodes, 64 a workgroup.
constexpr std::uint32_t kBatch = 1u << 20;
/// Dispatches recorded into one submission, so no submission runs
/// long.
constexpr std::uint32_t kBatchesPerSubmit = 16;

}  // namespace

GpuBlockEncoder::GpuBlockEncoder(const VulkanContext& context,
                                 const ShaderCompiler& compiler)
    : context_(context) {
    const std::vector<ShaderModuleSource> modules{
        {"bc7", shaders::kBc7Slang}};
    pipeline_ = std::make_unique<ComputePipeline>(
        context, compiler.CompileSpirv(modules, "main"), 3);
}

GpuBlockEncoder::~GpuBlockEncoder() = default;

void GpuBlockEncoder::Encode(std::span<const std::uint8_t> rgba,
                             std::uint32_t width,
                             std::uint32_t height,
                             TextureFormat format, std::byte* out) {
    if (!Encodes(format)) {
        throw std::invalid_argument(
            "The GPU encoder encodes BC7 alone");
    }
    const std::uint32_t across = (width + 3) / 4;
    const std::uint32_t blocks = across * ((height + 3) / 4);
    const std::uint32_t batches = (blocks + kBatch - 1) / kBatch;
    constexpr VkBufferUsageFlags kStorage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    GpuBuffer source(context_, rgba.size(), kStorage,
                     BufferMemory::Device);
    source.Write(std::as_bytes(rgba));
    const VkDeviceSize blockBytes = VkDeviceSize{blocks} * 16;
    GpuBuffer encoded(context_, blockBytes, kStorage,
                      BufferMemory::Device);

    // Every dispatch's constants, each in its own aligned slice.
    const VkDeviceSize alignment = std::max<VkDeviceSize>(
        context_.Limits().minStorageBufferOffsetAlignment,
        sizeof(Constants));
    const VkDeviceSize stride =
        (sizeof(Constants) + alignment - 1) / alignment * alignment;
    std::vector<std::byte> constants(stride * batches);
    for (std::uint32_t batch = 0; batch < batches; ++batch) {
        const Constants c{width,  height,         across,
                          blocks, batch * kBatch, {}};
        std::memcpy(constants.data() + stride * batch, &c, sizeof(c));
    }
    GpuBuffer parameters(context_, constants.size(), kStorage,
                         BufferMemory::Device);
    parameters.Write(constants);

    for (std::uint32_t first = 0; first < batches;
         first += kBatchesPerSubmit) {
        const std::uint32_t last =
            std::min(batches, first + kBatchesPerSubmit);
        context_.Submit([&](VkCommandBuffer commands) {
            for (std::uint32_t batch = first; batch < last; ++batch) {
                const std::uint32_t count =
                    std::min(kBatch, blocks - batch * kBatch);
                const DescriptorResource resources[] = {
                    DescriptorResource::Of(source),
                    DescriptorResource::Of(encoded),
                    DescriptorResource::Of(parameters, stride * batch,
                                           sizeof(Constants))};
                pipeline_->Dispatch(commands, resources,
                                    (count + 63) / 64);
            }
        });
    }
    encoded.Read(std::span(out, blockBytes));
}

}  // namespace ernie
