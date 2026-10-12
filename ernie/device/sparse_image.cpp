// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/sparse_image.h"

#include "ernie/device/buffer.h"
#include "ernie/device/image.h"
#include "ernie/device/memory_stats.h"
#include "ernie/device/pipeline.h"

#include <algorithm>
#include <bit>
#include <stdexcept>
#include <string>
#include <vector>

namespace ernie {

namespace {

/// Device-local memory for `count` pieces of `size` bytes each,
/// aligned as `requirements` says.
std::vector<VmaAllocation>
AllocatePieces(const VulkanContext& context,
               const VkMemoryRequirements& requirements,
               VkDeviceSize size, std::size_t count) {
    std::vector<VmaAllocation> pieces(count, VK_NULL_HANDLE);
    if (count == 0) {
        return pieces;
    }
    VkMemoryRequirements piece = requirements;
    piece.size = size;
    VmaAllocationCreateInfo create{};
    create.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    CheckVk(vmaAllocateMemoryPages(context.Allocator(), &piece,
                                   &create, count, pieces.data(),
                                   nullptr),
            "vmaAllocateMemoryPages");
    return pieces;
}

/// Submits `info` on the queue and waits for it.
void BindAndWait(const VulkanContext& context,
                 const VkBindSparseInfo& info) {
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    CheckVk(
        vkCreateFence(context.Device(), &fenceInfo, nullptr, &fence),
        "vkCreateFence");
    const VkResult bound =
        vkQueueBindSparse(context.Queue(), 1, &info, fence);
    if (bound == VK_SUCCESS) {
        vkWaitForFences(context.Device(), 1, &fence, VK_TRUE,
                        UINT64_MAX);
    }
    vkDestroyFence(context.Device(), fence, nullptr);
    CheckVk(bound, "vkQueueBindSparse");
}

}  // namespace

SparseImage::SparseImage(const VulkanContext& context,
                         const SparseImageDesc& desc)
    : context_(context), desc_(desc) {
    if (!context.SupportsSparseResidency()) {
        throw std::runtime_error(
            "The Vulkan device has no sparse image residency");
    }
    // Copied from too, as `GpuImage`s are, to read tiles back.
    const VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    std::uint32_t count = 0;
    vkGetPhysicalDeviceSparseImageFormatProperties(
        context.PhysicalDevice(), desc.format, VK_IMAGE_TYPE_2D,
        VK_SAMPLE_COUNT_1_BIT, usage, VK_IMAGE_TILING_OPTIMAL, &count,
        nullptr);
    if (count == 0) {
        throw std::runtime_error("The Vulkan device cannot make "
                                 "images of this format sparse");
    }

    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                 VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = desc.format;
    info.extent = {desc.width, desc.height, 1};
    info.mipLevels = desc.mips;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    CheckVk(vkCreateImage(context.Device(), &info, nullptr, &image_),
            "vkCreateImage");
    aspect_ = AspectOf(desc.format);
    vkGetImageMemoryRequirements(context.Device(), image_,
                                 &requirements_);
    tileMemory_ = requirements_.alignment;

    std::uint32_t aspects = 0;
    vkGetImageSparseMemoryRequirements(context.Device(), image_,
                                       &aspects, nullptr);
    std::vector<VkSparseImageMemoryRequirements> sparse(aspects);
    vkGetImageSparseMemoryRequirements(context.Device(), image_,
                                       &aspects, sparse.data());
    // The colour aspect's tiles and tail, and the metadata some
    // implementations keep beside them, bound whole with the tail.
    std::vector<VkSparseMemoryBind> tailBinds;
    std::vector<VkDeviceSize> tailSizes;
    for (const VkSparseImageMemoryRequirements& r : sparse) {
        const bool metadata = (r.formatProperties.aspectMask &
                               VK_IMAGE_ASPECT_METADATA_BIT) != 0;
        if (!metadata) {
            const VkExtent3D granularity =
                r.formatProperties.imageGranularity;
            tileTexels_ = {granularity.width, granularity.height};
            tailMip_ = std::min(r.imageMipTailFirstLod, desc.mips);
        }
        if (metadata || r.imageMipTailFirstLod < desc.mips) {
            VkSparseMemoryBind bind{};
            bind.resourceOffset = r.imageMipTailOffset;
            bind.size = r.imageMipTailSize;
            bind.flags =
                metadata ? VK_SPARSE_MEMORY_BIND_METADATA_BIT : 0;
            tailBinds.push_back(bind);
        }
    }
    if (tileTexels_.width == 0 || tileTexels_.height == 0) {
        vkDestroyImage(context.Device(), image_, nullptr);
        throw std::runtime_error("Sparse image has no colour tiles");
    }
    // One allocation holds the tail and any metadata, each piece at
    // an aligned offset.
    for (VkSparseMemoryBind& bind : tailBinds) {
        bind.memoryOffset = tailBytes_;
        tailBytes_ +=
            (bind.size + tileMemory_ - 1) / tileMemory_ * tileMemory_;
    }
    if (tailBytes_ > 0) {
        tail_ =
            AllocatePieces(context, requirements_, tailBytes_, 1)[0];
        VmaAllocationInfo memory{};
        vmaGetAllocationInfo(context.Allocator(), tail_, &memory);
        for (VkSparseMemoryBind& bind : tailBinds) {
            bind.memory = memory.deviceMemory;
            bind.memoryOffset += memory.offset;
        }
        VkSparseImageOpaqueMemoryBindInfo opaque{};
        opaque.image = image_;
        opaque.bindCount =
            static_cast<std::uint32_t>(tailBinds.size());
        opaque.pBinds = tailBinds.data();
        VkBindSparseInfo bindInfo{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
        bindInfo.imageOpaqueBindCount = 1;
        bindInfo.pImageOpaqueBinds = &opaque;
        BindAndWait(context, bindInfo);
    }
    context.Submit([&](VkCommandBuffer commands) {
        TransitionImage(commands, image_, aspect_,
                        VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkImageViewCreateInfo view{
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = image_;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = desc.format;
    view.components = desc.components;
    view.subresourceRange = {aspect_, 0, desc.mips, 0, 1};
    CheckVk(
        vkCreateImageView(context.Device(), &view, nullptr, &view_),
        "vkCreateImageView");
    Track();
}

SparseImage::~SparseImage() {
    if (view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(context_.Device(), view_, nullptr);
    }
    vkDestroyImage(context_.Device(), image_, nullptr);
    for (const auto& [tile, memory] : tiles_) {
        vmaFreeMemory(context_.Allocator(), memory);
    }
    if (tail_ != VK_NULL_HANDLE) {
        vmaFreeMemory(context_.Allocator(), tail_);
    }
    memory::Untrack(std::bit_cast<std::uint64_t>(image_));
}

VkExtent2D SparseImage::TileCount(std::uint32_t mip) const {
    if (mip >= tailMip_) {
        throw std::out_of_range("Sparse image mip " +
                                std::to_string(mip) +
                                " has no tiles");
    }
    const std::uint32_t width = std::max(1u, desc_.width >> mip);
    const std::uint32_t height = std::max(1u, desc_.height >> mip);
    return {(width + tileTexels_.width - 1) / tileTexels_.width,
            (height + tileTexels_.height - 1) / tileTexels_.height};
}

void SparseImage::Check(const SparseTile& tile) const {
    const VkExtent2D count = TileCount(tile.mip);
    if (tile.x >= count.width || tile.y >= count.height) {
        throw std::out_of_range("Sparse image has no tile (" +
                                std::to_string(tile.x) + ", " +
                                std::to_string(tile.y) + ") in mip " +
                                std::to_string(tile.mip));
    }
}

VkExtent2D SparseImage::TileExtent(const SparseTile& tile) const {
    Check(tile);
    const std::uint32_t width = std::max(1u, desc_.width >> tile.mip);
    const std::uint32_t height =
        std::max(1u, desc_.height >> tile.mip);
    return {std::min(tileTexels_.width,
                     width - tile.x * tileTexels_.width),
            std::min(tileTexels_.height,
                     height - tile.y * tileTexels_.height)};
}

VkDeviceSize
SparseImage::TileDataBytes(const SparseTile& tile) const {
    const VkExtent2D extent = TileExtent(tile);
    return GpuImage::MipBytes(desc_.format, extent.width,
                              extent.height);
}

VkDeviceSize SparseImage::Bytes() const {
    return tailBytes_ + tileMemory_ * tiles_.size();
}

void SparseImage::BindTiles(std::span<const SparseTile> tiles,
                            std::span<const VmaAllocation> memory) {
    if (tiles.empty()) {
        return;
    }
    std::vector<VkSparseImageMemoryBind> binds;
    binds.reserve(tiles.size());
    for (std::size_t i = 0; i < tiles.size(); ++i) {
        const SparseTile& tile = tiles[i];
        const VkExtent2D extent = TileExtent(tile);
        VkSparseImageMemoryBind bind{};
        bind.subresource = {aspect_, tile.mip, 0};
        bind.offset = {
            static_cast<std::int32_t>(tile.x * tileTexels_.width),
            static_cast<std::int32_t>(tile.y * tileTexels_.height),
            0};
        bind.extent = {extent.width, extent.height, 1};
        if (memory[i] != VK_NULL_HANDLE) {
            VmaAllocationInfo allocation{};
            vmaGetAllocationInfo(context_.Allocator(), memory[i],
                                 &allocation);
            bind.memory = allocation.deviceMemory;
            bind.memoryOffset = allocation.offset;
        }
        binds.push_back(bind);
    }
    VkSparseImageMemoryBindInfo image{};
    image.image = image_;
    image.bindCount = static_cast<std::uint32_t>(binds.size());
    image.pBinds = binds.data();
    VkBindSparseInfo info{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO};
    info.imageBindCount = 1;
    info.pImageBinds = &image;
    BindAndWait(context_, info);
}

void SparseImage::Write(
    std::span<const SparseTile> tiles,
    std::span<const std::span<const std::byte>> texels) {
    if (tiles.size() != texels.size()) {
        throw std::invalid_argument(
            "SparseImage::Write needs data for every tile");
    }
    std::vector<SparseTile> unbound;
    for (std::size_t i = 0; i < tiles.size(); ++i) {
        if (texels[i].size() != TileDataBytes(tiles[i])) {
            throw std::invalid_argument(
                "SparseImage::Write got the wrong size for a tile");
        }
        if (!Bound(tiles[i]) &&
            std::find(unbound.begin(), unbound.end(), tiles[i]) ==
                unbound.end()) {
            unbound.push_back(tiles[i]);
        }
    }
    const std::vector<VmaAllocation> memory = AllocatePieces(
        context_, requirements_, tileMemory_, unbound.size());
    BindTiles(unbound, memory);
    for (std::size_t i = 0; i < unbound.size(); ++i) {
        tiles_[unbound[i]] = memory[i];
    }

    std::vector<VkBufferImageCopy> regions;
    VkDeviceSize total = 0;
    for (std::size_t i = 0; i < tiles.size(); ++i) {
        const VkExtent2D extent = TileExtent(tiles[i]);
        VkBufferImageCopy region{};
        region.bufferOffset = total;
        region.imageSubresource = {aspect_, tiles[i].mip, 0, 1};
        region.imageOffset = {
            static_cast<std::int32_t>(tiles[i].x * tileTexels_.width),
            static_cast<std::int32_t>(tiles[i].y *
                                      tileTexels_.height),
            0};
        region.imageExtent = {extent.width, extent.height, 1};
        regions.push_back(region);
        total += texels[i].size();
    }
    if (total == 0) {
        Track();
        return;
    }
    GpuBuffer staging(context_, total, 0, BufferMemory::Upload);
    for (std::size_t i = 0; i < tiles.size(); ++i) {
        staging.Write(texels[i], regions[i].bufferOffset);
    }
    context_.Submit([&](VkCommandBuffer commands) {
        TransitionImage(commands, image_, aspect_, layout_,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkCmdCopyBufferToImage(
            commands, staging.Handle(), image_,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            static_cast<std::uint32_t>(regions.size()),
            regions.data());
        TransitionImage(commands, image_, aspect_,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    Track();
}

void SparseImage::WriteTail(
    std::span<const std::span<const std::byte>> mips) {
    if (mips.size() != desc_.mips - tailMip_) {
        throw std::invalid_argument(
            "SparseImage::WriteTail needs every mip of the tail");
    }
    std::vector<VkBufferImageCopy> regions;
    VkDeviceSize total = 0;
    for (std::uint32_t i = 0; i < mips.size(); ++i) {
        const std::uint32_t mip = tailMip_ + i;
        const std::uint32_t width = std::max(1u, desc_.width >> mip);
        const std::uint32_t height =
            std::max(1u, desc_.height >> mip);
        if (mips[i].size() !=
            GpuImage::MipBytes(desc_.format, width, height)) {
            throw std::invalid_argument("SparseImage::WriteTail got "
                                        "the wrong size for a mip");
        }
        VkBufferImageCopy region{};
        region.bufferOffset = total;
        region.imageSubresource = {aspect_, mip, 0, 1};
        region.imageExtent = {width, height, 1};
        regions.push_back(region);
        total += mips[i].size();
    }
    if (total == 0) {
        return;
    }
    GpuBuffer staging(context_, total, 0, BufferMemory::Upload);
    for (std::size_t i = 0; i < mips.size(); ++i) {
        staging.Write(mips[i], regions[i].bufferOffset);
    }
    context_.Submit([&](VkCommandBuffer commands) {
        TransitionImage(commands, image_, aspect_, layout_,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkCmdCopyBufferToImage(
            commands, staging.Handle(), image_,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            static_cast<std::uint32_t>(regions.size()),
            regions.data());
        TransitionImage(commands, image_, aspect_,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
}

void SparseImage::Release(std::span<const SparseTile> tiles) {
    std::vector<SparseTile> bound;
    for (const SparseTile& tile : tiles) {
        if (Bound(tile) && std::find(bound.begin(), bound.end(),
                                     tile) == bound.end()) {
            bound.push_back(tile);
        }
    }
    if (bound.empty()) {
        return;
    }
    vkDeviceWaitIdle(context_.Device());
    const std::vector<VmaAllocation> none(bound.size(),
                                          VK_NULL_HANDLE);
    BindTiles(bound, none);
    for (const SparseTile& tile : bound) {
        vmaFreeMemory(context_.Allocator(), tiles_.at(tile));
        tiles_.erase(tile);
    }
    Track();
}

void SparseImage::Track() const {
    memory::Track(std::bit_cast<std::uint64_t>(image_),
                  {"Sparse textures",
                   std::to_string(desc_.width) + " x " +
                       std::to_string(desc_.height) + ", " +
                       std::to_string(tiles_.size()) + " tiles bound",
                   Bytes(), false});
}

}  // namespace ernie
