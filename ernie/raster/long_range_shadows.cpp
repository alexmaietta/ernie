// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/long_range_shadows.h"

#include "ernie/device/vulkan_context.h"
#include "ernie/kernel/embedded_shaders.h"

#include <algorithm>
#include <bit>
#include <cstring>

namespace ernie {

namespace {

constexpr VkShaderStageFlags kCompute = VK_SHADER_STAGE_COMPUTE_BIT;

/// long_range.slang's bindings: each pass reads one image and writes
/// another.
constexpr DescriptorBinding kCoverageBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
};
constexpr DescriptorBinding kMipBindings[] = {
    {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
};
constexpr DescriptorBinding kVisibilityBindings[] = {
    {4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
};

/// long_range.slang's `LongRangeConstants`.
struct Constants {
    std::uint32_t size[4];
    float params[4];
    float wavelengths[4];
    float region[4];
};
static_assert(sizeof(Constants) == 64,
              "Constants must match long_range.slang");

std::uint32_t Groups(std::uint32_t size) {
    return (size + 7) / 8;
}

/// A visibility map of `layers` layers `texels` across, cleared to 1:
/// no long-range shadow until a view writes it.
std::unique_ptr<GpuImage>
VisibilityImage(const VulkanContext& context, std::uint32_t layers,
                std::uint32_t texels) {
    auto image = std::make_unique<GpuImage>(
        context, ImageDesc{texels, texels, layers, 1,
                           VK_FORMAT_R16G16B16A16_SFLOAT,
                           VK_IMAGE_USAGE_STORAGE_BIT |
                               VK_IMAGE_USAGE_SAMPLED_BIT |
                               VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                           false, true});
    context.Submit([&](VkCommandBuffer commands) {
        image->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
        const VkClearColorValue one{{1.0f, 1.0f, 1.0f, 1.0f}};
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT,
                                            0, 1, 0, layers};
        vkCmdClearColorImage(commands, image->Handle(),
                             VK_IMAGE_LAYOUT_GENERAL, &one, 1,
                             &range);
        image->Transition(commands,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    return image;
}

}  // namespace

LongRangeShadows::LongRangeShadows(const VulkanContext& context,
                                   const ShaderCompiler& compiler)
    : context_(context),
      coverageLayout_(context, kCoverageBindings, sizeof(Constants)),
      mipLayout_(context, kMipBindings, sizeof(Constants)),
      visibilityLayout_(context, kVisibilityBindings,
                        sizeof(Constants)) {
    const ShaderModuleSource modules[] = {
        {"long_range", shaders::kLongRangeSlang}};
    coverage_ = std::make_unique<ComputePipeline>(
        context, coverageLayout_,
        compiler.CompileSpirv(modules, "coverageMain"));
    mips_ = std::make_unique<ComputePipeline>(
        context, mipLayout_,
        compiler.CompileSpirv(modules, "coverageMipMain"));
    integrate_ = std::make_unique<ComputePipeline>(
        context, visibilityLayout_,
        compiler.CompileSpirv(modules, "visibilityMain"));
    visibility_ = VisibilityImage(context, 2, 1);
}

void LongRangeShadows::Resize(std::uint32_t layers,
                              std::uint32_t texels) {
    const std::uint32_t half = kVisibilityTexels;
    if (coverageImage_ && coverageImage_->Desc().width == texels &&
        visibility_->Desc().width == half &&
        visibility_->Desc().layers >= layers) {
        return;
    }
    // Replaced only once no frame still reads them.
    vkDeviceWaitIdle(context_.Device());
    const auto mips = static_cast<std::uint32_t>(
        std::bit_width(std::max(texels, 1u)));
    coverageImage_ = std::make_unique<GpuImage>(
        context_,
        ImageDesc{texels, texels, 1, mips, VK_FORMAT_R32G32_SFLOAT,
                  VK_IMAGE_USAGE_STORAGE_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT});
    visibility_ =
        VisibilityImage(context_, std::max(layers, 1u), half);
}

void LongRangeShadows::Prepare(std::span<const View> views) {
    if (views.empty()) {
        return;
    }
    std::uint32_t texels = 1;
    for (const View& view : views) {
        texels = std::max(texels, view.texels);
    }
    Resize(2 * static_cast<std::uint32_t>(views.size()), texels);
}

void LongRangeShadows::Record(VkCommandBuffer commands,
                              const GpuImage& shadowMaps,
                              std::span<const View> views) const {
    if (views.empty() || !coverageImage_) {
        return;
    }
    const std::uint32_t mips = coverageImage_->Desc().mips;
    const std::uint32_t half = visibility_->Desc().width;
    visibility_->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    for (std::size_t v = 0; v < views.size(); ++v) {
        const View& view = views[v];
        Constants constants{};
        constants.size[0] = view.texels;
        constants.size[1] = view.layer;
        constants.size[2] = half;
        constants.size[3] = mips;
        constants.params[0] = view.tanUv;
        constants.params[1] = view.receiversNear;
        constants.params[3] = view.receiversFar;
        constants.params[2] = static_cast<float>(v);
        std::memcpy(constants.wavelengths, kWavelengths.data(),
                    sizeof(constants.wavelengths));
        constants.region[0] = view.region.x;
        constants.region[1] = view.region.y;
        constants.region[2] = view.region.z;
        // The casters' coverage, then its mips.
        coverageImage_->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
        {
            const DescriptorResource resources[] = {
                DescriptorResource::Of(shadowMaps.View()),
                DescriptorResource::Of(coverageImage_->MipView(0),
                                       VK_IMAGE_LAYOUT_GENERAL)};
            coverageLayout_.Push(
                commands, VK_PIPELINE_BIND_POINT_COMPUTE, resources);
            coverageLayout_.PushConstants(commands, &constants,
                                          sizeof(constants));
            vkCmdBindPipeline(commands,
                              VK_PIPELINE_BIND_POINT_COMPUTE,
                              coverage_->Handle());
            vkCmdDispatch(commands, Groups(view.texels),
                          Groups(view.texels), 1);
            ComputeBarrier(commands);
        }
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                          mips_->Handle());
        for (std::uint32_t mip = 1; mip < mips; ++mip) {
            const DescriptorResource resources[] = {
                DescriptorResource::Of(
                    coverageImage_->MipView(mip - 1),
                    VK_IMAGE_LAYOUT_GENERAL),
                DescriptorResource::Of(coverageImage_->MipView(mip),
                                       VK_IMAGE_LAYOUT_GENERAL)};
            mipLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                            resources);
            Constants level = constants;
            level.size[0] = std::max(view.texels >> (mip - 1), 1u);
            mipLayout_.PushConstants(commands, &level, sizeof(level));
            const std::uint32_t size =
                std::max(view.texels >> mip, 1u);
            vkCmdDispatch(commands, Groups(size), Groups(size), 1);
            ComputeBarrier(commands);
        }
        // The light's disc integrated over it.
        {
            const DescriptorResource resources[] = {
                DescriptorResource::Of(coverageImage_->View(),
                                       VK_IMAGE_LAYOUT_GENERAL),
                DescriptorResource::Of(visibility_->View(),
                                       VK_IMAGE_LAYOUT_GENERAL)};
            visibilityLayout_.Push(
                commands, VK_PIPELINE_BIND_POINT_COMPUTE, resources);
            visibilityLayout_.PushConstants(commands, &constants,
                                            sizeof(constants));
            vkCmdBindPipeline(commands,
                              VK_PIPELINE_BIND_POINT_COMPUTE,
                              integrate_->Handle());
            // At the receivers' nearest and farthest depths.
            vkCmdDispatch(commands, Groups(half), Groups(half), 2);
            ComputeBarrier(commands);
        }
    }
    visibility_->Transition(commands,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

}  // namespace ernie
