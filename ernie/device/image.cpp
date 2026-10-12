// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/image.h"

#include "ernie/device/buffer.h"
#include "ernie/device/memory_stats.h"
#include "ernie/device/pipeline.h"

#include <algorithm>
#include <bit>
#include <stdexcept>
#include <string>
#include <utility>

namespace ernie {

VkImageAspectFlags AspectOf(VkFormat format) {
    switch (format) {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
        return VK_IMAGE_ASPECT_DEPTH_BIT;
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return VK_IMAGE_ASPECT_DEPTH_BIT |
               VK_IMAGE_ASPECT_STENCIL_BIT;
    default:
        return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

std::uint32_t GpuImage::TexelBytes(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8_UNORM:
        return 1;
    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R16_SFLOAT:
        return 2;
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_D32_SFLOAT:
        return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R32G32_SFLOAT:
        return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_R32G32B32A32_UINT:
        return 16;
    default:
        throw std::invalid_argument("Unsupported texel format");
    }
}

std::uint32_t GpuImage::BlockBytes(VkFormat format) {
    switch (format) {
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
    case VK_FORMAT_BC4_UNORM_BLOCK:
    case VK_FORMAT_BC4_SNORM_BLOCK:
        return 8;
    case VK_FORMAT_BC2_UNORM_BLOCK:
    case VK_FORMAT_BC2_SRGB_BLOCK:
    case VK_FORMAT_BC3_UNORM_BLOCK:
    case VK_FORMAT_BC3_SRGB_BLOCK:
    case VK_FORMAT_BC5_UNORM_BLOCK:
    case VK_FORMAT_BC5_SNORM_BLOCK:
    case VK_FORMAT_BC6H_UFLOAT_BLOCK:
    case VK_FORMAT_BC6H_SFLOAT_BLOCK:
    case VK_FORMAT_BC7_UNORM_BLOCK:
    case VK_FORMAT_BC7_SRGB_BLOCK:
        return 16;
    default:
        return 0;
    }
}

VkDeviceSize GpuImage::MipBytes(VkFormat format, std::uint32_t width,
                                std::uint32_t height) {
    if (const std::uint32_t block = BlockBytes(format); block > 0) {
        return VkDeviceSize{(width + 3) / 4} * ((height + 3) / 4) *
               block;
    }
    return VkDeviceSize{width} * height * TexelBytes(format);
}

GpuImage::GpuImage(const VulkanContext& context,
                   const ImageDesc& desc)
    : context_(&context), desc_(desc) {
    if (desc.groupLayers > 0 && desc.layers % desc.groupLayers != 0) {
        throw std::invalid_argument(
            "An image's layers must be a whole number of groups");
    }
    if (desc.linear && (desc.layers != 1 || desc.mips != 1)) {
        throw std::invalid_argument(
            "A linear image has one layer and one mip");
    }
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = desc.format;
    info.extent = {desc.width, desc.height, 1};
    info.mipLevels = desc.mips;
    info.arrayLayers = desc.layers;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = desc.linear ? VK_IMAGE_TILING_LINEAR
                              : VK_IMAGE_TILING_OPTIMAL;
    info.usage = desc.usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (desc.shared) {
        if (!context.SupportsExternalMemory()) {
            throw std::runtime_error(
                "The Vulkan device cannot export image memory");
        }
        VkExternalMemoryImageCreateInfo external{
            VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
        external.handleTypes = kExternalMemoryHandleType;
        info.pNext = &external;
        CheckVk(
            vkCreateImage(context.Device(), &info, nullptr, &image_),
            "vkCreateImage");
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(context.Device(), image_,
                                     &requirements);
        VkMemoryDedicatedAllocateInfo dedicated{
            VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
        dedicated.image = image_;
        VkExportMemoryAllocateInfo exported{
            VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
        exported.handleTypes = kExternalMemoryHandleType;
        exported.pNext = &dedicated;
        VkMemoryAllocateInfo allocate{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocate.pNext = &exported;
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = context.DeviceLocalMemoryType(
            requirements.memoryTypeBits);
        const VkResult result = vkAllocateMemory(
            context.Device(), &allocate, nullptr, &sharedMemory_);
        if (result != VK_SUCCESS) {
            vkDestroyImage(context.Device(), image_, nullptr);
            image_ = VK_NULL_HANDLE;
            CheckVk(result, "vkAllocateMemory");
        }
        CheckVk(vkBindImageMemory(context.Device(), image_,
                                  sharedMemory_, 0),
                "vkBindImageMemory");
        bytes_ = requirements.size;
    } else {
        VmaAllocationCreateInfo allocation{};
        allocation.usage = VMA_MEMORY_USAGE_AUTO;
        VmaAllocationInfo result{};
        CheckVk(vmaCreateImage(context.Allocator(), &info,
                               &allocation, &image_, &allocation_,
                               &result),
                "vmaCreateImage");
        bytes_ = result.size;
    }
    if (desc.linear) {
        const VkImageSubresource texels{AspectOf(desc.format), 0, 0};
        VkSubresourceLayout layout{};
        vkGetImageSubresourceLayout(context.Device(), image_, &texels,
                                    &layout);
        rowPitch_ = layout.rowPitch;
        texelOffset_ = layout.offset;
    }
    const char* category = "Textures";
    if (desc.shared) {
        category = "Shared with CUDA";
    } else if ((AspectOf(desc.format) & VK_IMAGE_ASPECT_DEPTH_BIT) !=
               0) {
        category = "Depth targets";
    } else if ((desc.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) !=
               0) {
        category = "Render targets";
    } else if ((desc.usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0) {
        category = "Storage images";
    }
    std::string detail = std::to_string(desc.width) + " x " +
                         std::to_string(desc.height);
    if (desc.layers > 1) {
        detail += ", " + std::to_string(desc.layers) + " layers";
    }
    if (desc.mips > 1) {
        detail += ", " + std::to_string(desc.mips) + " mips";
    }
    memory::Track(std::bit_cast<std::uint64_t>(image_),
                  {category, std::move(detail), bytes_, false});

    VkImageViewCreateInfo view{
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = image_;
    const bool array = desc.layers > 1 || desc.array;
    view.viewType =
        array ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    view.format = desc.format;
    view.components = desc.components;
    view.subresourceRange = {AspectOf(desc.format), 0, desc.mips, 0,
                             desc.layers};
    CheckVk(
        vkCreateImageView(context.Device(), &view, nullptr, &view_),
        "vkCreateImageView");
    if (array) {
        for (std::uint32_t layer = 0; layer < desc.layers; ++layer) {
            view.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view.subresourceRange = {AspectOf(desc.format), 0, 1,
                                     layer, 1};
            VkImageView layerView = VK_NULL_HANDLE;
            CheckVk(vkCreateImageView(context.Device(), &view,
                                      nullptr, &layerView),
                    "vkCreateImageView");
            layerViews_.push_back(layerView);
        }
    }
    if (desc.mips > 1) {
        for (std::uint32_t mip = 0; mip < desc.mips; ++mip) {
            view.viewType = array ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                                  : VK_IMAGE_VIEW_TYPE_2D;
            view.subresourceRange = {AspectOf(desc.format), mip, 1, 0,
                                     desc.layers};
            VkImageView mipView = VK_NULL_HANDLE;
            CheckVk(vkCreateImageView(context.Device(), &view,
                                      nullptr, &mipView),
                    "vkCreateImageView");
            mipViews_.push_back(mipView);
        }
    }
    const std::uint32_t group = desc.groupLayers;
    if (desc.layers > 1 && group > 0 && group < desc.layers) {
        for (std::uint32_t first = 0; first < desc.layers;
             first += group) {
            view.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            view.subresourceRange = {AspectOf(desc.format), 0, 1,
                                     first, group};
            VkImageView groupView = VK_NULL_HANDLE;
            CheckVk(vkCreateImageView(context.Device(), &view,
                                      nullptr, &groupView),
                    "vkCreateImageView");
            groupViews_.push_back(groupView);
        }
    }
}

GpuImage::~GpuImage() {
    Release();
}

GpuImage::GpuImage(GpuImage&& other) noexcept
    : context_(other.context_), desc_(other.desc_),
      image_(std::exchange(other.image_, VK_NULL_HANDLE)),
      allocation_(std::exchange(other.allocation_, VK_NULL_HANDLE)),
      sharedMemory_(
          std::exchange(other.sharedMemory_, VK_NULL_HANDLE)),
      attached_(std::move(other.attached_)),
      view_(std::exchange(other.view_, VK_NULL_HANDLE)),
      layerViews_(std::move(other.layerViews_)),
      groupViews_(std::move(other.groupViews_)),
      mipViews_(std::move(other.mipViews_)), layout_(other.layout_),
      bytes_(other.bytes_), rowPitch_(other.rowPitch_),
      texelOffset_(other.texelOffset_) {}

GpuImage& GpuImage::operator=(GpuImage&& other) noexcept {
    if (this != &other) {
        Release();
        context_ = other.context_;
        desc_ = other.desc_;
        image_ = std::exchange(other.image_, VK_NULL_HANDLE);
        allocation_ =
            std::exchange(other.allocation_, VK_NULL_HANDLE);
        sharedMemory_ =
            std::exchange(other.sharedMemory_, VK_NULL_HANDLE);
        attached_ = std::move(other.attached_);
        view_ = std::exchange(other.view_, VK_NULL_HANDLE);
        layerViews_ = std::move(other.layerViews_);
        groupViews_ = std::move(other.groupViews_);
        mipViews_ = std::move(other.mipViews_);
        layout_ = other.layout_;
        bytes_ = other.bytes_;
        rowPitch_ = other.rowPitch_;
        texelOffset_ = other.texelOffset_;
    }
    return *this;
}

void GpuImage::Release() {
    if (image_ == VK_NULL_HANDLE) {
        return;
    }
    memory::Untrack(std::bit_cast<std::uint64_t>(image_));
    for (VkImageView view : layerViews_) {
        vkDestroyImageView(context_->Device(), view, nullptr);
    }
    layerViews_.clear();
    for (VkImageView view : groupViews_) {
        vkDestroyImageView(context_->Device(), view, nullptr);
    }
    groupViews_.clear();
    for (VkImageView view : mipViews_) {
        vkDestroyImageView(context_->Device(), view, nullptr);
    }
    mipViews_.clear();
    attached_.reset();
    vkDestroyImageView(context_->Device(), view_, nullptr);
    if (sharedMemory_ != VK_NULL_HANDLE) {
        vkDestroyImage(context_->Device(), image_, nullptr);
        vkFreeMemory(context_->Device(), sharedMemory_, nullptr);
        sharedMemory_ = VK_NULL_HANDLE;
    } else {
        vmaDestroyImage(context_->Allocator(), image_, allocation_);
    }
    image_ = VK_NULL_HANDLE;
    view_ = VK_NULL_HANDLE;
}

ExternalHandle GpuImage::ExportHandle() const {
    if (sharedMemory_ == VK_NULL_HANDLE) {
        throw std::logic_error(
            "Only shared images export their memory");
    }
    return context_->ExportMemory(sharedMemory_);
}

VkImageView GpuImage::LayerView(std::uint32_t layer) const {
    if (layer >= desc_.layers) {
        throw std::out_of_range(
            "GpuImage::LayerView past the last layer");
    }
    return layerViews_.empty() ? view_ : layerViews_[layer];
}

VkImageView GpuImage::MipView(std::uint32_t mip) const {
    if (mip >= desc_.mips) {
        throw std::out_of_range(
            "GpuImage::MipView past the last mip");
    }
    return mipViews_.empty() ? view_ : mipViews_[mip];
}

VkImageView GpuImage::GroupView(std::uint32_t group) const {
    if (groupViews_.empty()) {
        if (group != 0) {
            throw std::out_of_range(
                "GpuImage::GroupView past the last group");
        }
        return view_;
    }
    if (group >= groupViews_.size()) {
        throw std::out_of_range(
            "GpuImage::GroupView past the last group");
    }
    return groupViews_[group];
}

void GpuImage::Transition(VkCommandBuffer commands,
                          VkImageLayout layout) {
    TransitionImage(commands, image_, AspectOf(desc_.format), layout_,
                    layout);
    layout_ = layout;
}

void GpuImage::Upload(
    std::span<const std::span<const std::byte>> mipData) {
    const std::uint32_t mips = desc_.mips;
    if (mipData.size() != mips &&
        mipData.size() != std::size_t{mips} * desc_.layers) {
        throw std::invalid_argument(
            "GpuImage::Upload needs data for every mip");
    }
    std::vector<VkBufferImageCopy> regions;
    VkDeviceSize total = 0;
    for (std::size_t i = 0; i < mipData.size(); ++i) {
        const auto mip = static_cast<std::uint32_t>(i % mips);
        const auto layer = static_cast<std::uint32_t>(i / mips);
        const std::uint32_t w = std::max(1u, desc_.width >> mip);
        const std::uint32_t h = std::max(1u, desc_.height >> mip);
        if (mipData[i].size() != MipBytes(desc_.format, w, h)) {
            throw std::invalid_argument(
                "GpuImage::Upload got the wrong size for a mip");
        }
        VkBufferImageCopy region{};
        region.bufferOffset = total;
        region.imageSubresource = {AspectOf(desc_.format), mip, layer,
                                   1};
        region.imageExtent = {w, h, 1};
        regions.push_back(region);
        total += mipData[i].size();
    }
    GpuBuffer staging(*context_, total, 0, BufferMemory::Upload);
    for (std::size_t i = 0; i < mipData.size(); ++i) {
        staging.Write(mipData[i], regions[i].bufferOffset);
    }
    context_->Submit([&](VkCommandBuffer commands) {
        Transition(commands, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkCmdCopyBufferToImage(
            commands, staging.Handle(), image_,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            static_cast<std::uint32_t>(regions.size()),
            regions.data());
        Transition(commands,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
}

Sampler::Sampler(const VulkanContext& context,
                 const SamplerDesc& desc)
    : context_(context) {
    VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    info.magFilter = desc.filter;
    info.minFilter = desc.filter;
    info.mipmapMode = desc.mipmapMode;
    info.addressModeU = desc.addressMode;
    info.addressModeV = desc.addressModeV.value_or(desc.addressMode);
    info.addressModeW = info.addressModeV;
    info.borderColor = desc.borderColor;
    info.maxLod = VK_LOD_CLAMP_NONE;
    const float limit = context.Limits().maxSamplerAnisotropy;
    if (desc.maxAnisotropy > 1.0f && context.SupportsAnisotropy() &&
        limit > 1.0f) {
        info.anisotropyEnable = VK_TRUE;
        info.maxAnisotropy = std::min(desc.maxAnisotropy, limit);
    }
    if (desc.compare) {
        info.compareEnable = VK_TRUE;
        info.compareOp = VK_COMPARE_OP_LESS;
    }
    CheckVk(
        vkCreateSampler(context.Device(), &info, nullptr, &sampler_),
        "vkCreateSampler");
}

Sampler::~Sampler() {
    vkDestroySampler(context_.Device(), sampler_, nullptr);
}

}  // namespace ernie
