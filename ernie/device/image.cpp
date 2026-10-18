// SPDX-License-Identifier: BSD-3-Clause
/// @file image.cpp
/// @brief Allocating images through VMA, their views and samplers.
#include "ernie/device/image.h"

#include "ernie/util/on_failure.h"

#include <array>

namespace ernie {
namespace {

struct FormatInfo {
    VkFormat vk;
    std::uint32_t bytes;
};

FormatInfo Info(ImageFormat format) {
    static constexpr std::array kFormats{
        FormatInfo{VK_FORMAT_R8G8B8A8_UNORM, 4},
        FormatInfo{VK_FORMAT_R8G8B8A8_SRGB, 4},
        FormatInfo{VK_FORMAT_R32_SFLOAT, 4},
        FormatInfo{VK_FORMAT_R32G32_SFLOAT, 8},
        FormatInfo{VK_FORMAT_R16G16B16A16_SFLOAT, 8},
        FormatInfo{VK_FORMAT_R32G32B32A32_SFLOAT, 16},
        FormatInfo{VK_FORMAT_R32_UINT, 4},
    };
    return kFormats[static_cast<std::size_t>(format)];
}

VkImageUsageFlags ToVkUsage(ImageUsage usage) {
    VkImageUsageFlags flags = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (Any(usage, ImageUsage::Sampled)) {
        flags |= VK_IMAGE_USAGE_SAMPLED_BIT;
    }
    if (Any(usage, ImageUsage::Storage)) {
        flags |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    if (Any(usage, ImageUsage::Attachment)) {
        flags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    }
    return flags;
}

}  // namespace

std::uint32_t TexelBytes(ImageFormat format) {
    return Info(format).bytes;
}

VkFormat ToVkFormat(ImageFormat format) {
    return Info(format).vk;
}

DeviceImage::DeviceImage(DeviceContext& context, const ImageDescription& description,
                         std::string_view name)
    : context_(context), description_(description) {
    const bool volume = description.extent.z > 1;
    const VkImageCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = volume ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,
        .format = ToVkFormat(description.format),
        .extent = {description.extent.x, description.extent.y, description.extent.z},
        .mipLevels = description.mips,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = ToVkUsage(description.usage),
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    const VkPhysicalDeviceImageFormatInfo2 query{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
        .format = info.format,
        .type = info.imageType,
        .tiling = info.tiling,
        .usage = info.usage};
    VkImageFormatProperties2 supported{.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
    const VkImageFormatProperties& limits = supported.imageFormatProperties;
    if (description.mips == 0 ||
        vkGetPhysicalDeviceImageFormatProperties2(context.Physical(), &query, &supported) !=
            VK_SUCCESS ||
        description.mips > limits.maxMipLevels || description.extent.x > limits.maxExtent.width ||
        description.extent.y > limits.maxExtent.height ||
        description.extent.z > limits.maxExtent.depth) {
        throw DeviceError(DeviceErrorKind::Unsupported,
                          context.Info().name + " has no such image as " + std::string(name));
    }
    const VmaAllocationCreateInfo allocation{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    CheckVk(vmaCreateImage(context.Allocator(), &info, &allocation, &image_, &allocation_, nullptr),
            "Allocating image " + std::string(name));
    const OnFailure freeImage([&] { vmaDestroyImage(context.Allocator(), image_, allocation_); });
    const VkImageSubresourceRange all{
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = description.mips, .layerCount = 1};
    const VkImageViewCreateInfo view{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                     .image = image_,
                                     .viewType =
                                         volume ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D,
                                     .format = info.format,
                                     .subresourceRange = all};
    CheckVk(vkCreateImageView(context.Device(), &view, nullptr, &view_),
            "Viewing image " + std::string(name));
    const OnFailure freeView([&] { vkDestroyImageView(context.Device(), view_, nullptr); });
    context.Name(VK_OBJECT_TYPE_IMAGE, reinterpret_cast<std::uint64_t>(image_), name);

    ready_ = context.Main().Submit([&](VkCommandBuffer commands) {
        const VkImageMemoryBarrier2 barrier{.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT |
                                                             VK_ACCESS_2_MEMORY_WRITE_BIT,
                                            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                            .image = image_,
                                            .subresourceRange = all};
        const VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                          .imageMemoryBarrierCount = 1,
                                          .pImageMemoryBarriers = &barrier};
        vkCmdPipelineBarrier2(commands, &dependency);
    });
}

DeviceImage::~DeviceImage() {
    vkDestroyImageView(context_.Device(), view_, nullptr);
    vmaDestroyImage(context_.Allocator(), image_, allocation_);
}

Sampler::Sampler(DeviceContext& context, const SamplerDescription& description,
                 std::string_view name)
    : device_(context.Device()) {
    const VkFilter filter =
        description.filter == Filter::Linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    static constexpr std::array kAddress{VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                         VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                         VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT};
    const VkSamplerAddressMode address = kAddress[static_cast<std::size_t>(description.address)];
    const VkSamplerCreateInfo info{.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                   .magFilter = filter,
                                   .minFilter = filter,
                                   .mipmapMode = description.filter == Filter::Linear
                                                     ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                                     : VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                   .addressModeU = address,
                                   .addressModeV = address,
                                   .addressModeW = address,
                                   .maxLod = VK_LOD_CLAMP_NONE};
    CheckVk(vkCreateSampler(device_, &info, nullptr, &sampler_), "Creating a sampler");
    context.Name(VK_OBJECT_TYPE_SAMPLER, reinterpret_cast<std::uint64_t>(sampler_), name);
}

Sampler::~Sampler() {
    vkDestroySampler(device_, sampler_, nullptr);
}

}  // namespace ernie
