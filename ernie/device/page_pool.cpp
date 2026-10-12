// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/page_pool.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ernie {

PagePool::PagePool(const VulkanContext& context,
                   const SparseImageDesc& desc, std::uint32_t pages)
    : context_(context), desc_(desc),
      pageCount_(
          std::min(pages, context.Limits().maxImageArrayLayers)) {
    if (pageCount_ == 0) {
        throw std::invalid_argument("A page pool needs a page");
    }
    // The tail: mips of at most one tile, from the first whose sides
    // both fit.
    while (tailMip_ + 1 < desc.mips &&
           std::max(desc.width >> tailMip_, desc.height >> tailMip_) >
               kTileTexels) {
        ++tailMip_;
    }
    if (std::max(desc.width >> tailMip_, desc.height >> tailMip_) >
        kTileTexels) {
        throw std::invalid_argument(
            "A page pool's texture needs mips down to a tile");
    }
    ImageDesc pageDesc{kPageTexels, kPageTexels,
                       pageCount_,  1,
                       desc.format, VK_IMAGE_USAGE_SAMPLED_BIT};
    pageDesc.array = true;
    pageDesc.components = desc.components;
    pages_ = std::make_unique<GpuImage>(context, pageDesc);
    ImageDesc tailDesc{std::max(1u, desc.width >> tailMip_),
                       std::max(1u, desc.height >> tailMip_),
                       1,
                       desc.mips - tailMip_,
                       desc.format,
                       VK_IMAGE_USAGE_SAMPLED_BIT};
    tailDesc.components = desc.components;
    tail_ = std::make_unique<GpuImage>(context, tailDesc);
    // Pages are handed out from the front.
    for (std::uint32_t page = pageCount_; page-- > 0;) {
        free_.push_back(page);
    }
    context.Submit([&](VkCommandBuffer commands) {
        pages_->Transition(commands,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
}

PagePool::~PagePool() {
    for (const auto& pending : staging_) {
        context_.WaitForSubmission(pending.first);
    }
}

VkExtent2D PagePool::TileCount(std::uint32_t mip) const {
    if (mip >= tailMip_) {
        throw std::out_of_range(
            "Page pool mip " + std::to_string(mip) + " has no tiles");
    }
    const std::uint32_t width = std::max(1u, desc_.width >> mip);
    const std::uint32_t height = std::max(1u, desc_.height >> mip);
    return {(width + kTileTexels - 1) / kTileTexels,
            (height + kTileTexels - 1) / kTileTexels};
}

VkExtent2D PagePool::TileExtent(const SparseTile& tile) const {
    const VkExtent2D count = TileCount(tile.mip);
    if (tile.x >= count.width || tile.y >= count.height) {
        throw std::out_of_range("Page pool has no tile (" +
                                std::to_string(tile.x) + ", " +
                                std::to_string(tile.y) + ") in mip " +
                                std::to_string(tile.mip));
    }
    const std::uint32_t width = std::max(1u, desc_.width >> tile.mip);
    const std::uint32_t height =
        std::max(1u, desc_.height >> tile.mip);
    return {std::min(kTileTexels, width - tile.x * kTileTexels),
            std::min(kTileTexels, height - tile.y * kTileTexels)};
}

VkDeviceSize PagePool::TileDataBytes(const SparseTile& tile) const {
    TileExtent(tile);  // Throws for a tile the texture lacks.
    return TileMemoryBytes();
}

VkDeviceSize PagePool::TileMemoryBytes() const {
    return GpuImage::MipBytes(desc_.format, kPageTexels, kPageTexels);
}

VkDeviceSize PagePool::Bytes() const {
    return pages_->Bytes() + tail_->Bytes();
}

void PagePool::Write(
    std::span<const SparseTile> tiles,
    std::span<const std::span<const std::byte>> texels) {
    if (tiles.size() != texels.size()) {
        throw std::invalid_argument(
            "PagePool::Write needs data for every tile");
    }
    for (std::size_t i = 0; i < tiles.size(); ++i) {
        if (texels[i].size() != TileDataBytes(tiles[i])) {
            throw std::invalid_argument(
                "PagePool::Write got the wrong size for a page");
        }
    }
    std::erase_if(staging_, [&](const auto& pending) {
        return context_.WaitForSubmission(pending.first, 0);
    });
    std::vector<VkBufferImageCopy> regions;
    auto staging = std::make_unique<GpuBuffer>(
        context_,
        TileMemoryBytes() * std::max<std::size_t>(tiles.size(), 1), 0,
        BufferMemory::Upload);
    for (std::size_t i = 0; i < tiles.size(); ++i) {
        auto [it, added] = tiles_.try_emplace(tiles[i], 0);
        if (added) {
            if (free_.empty()) {
                tiles_.erase(it);
                throw std::runtime_error(
                    "The page pool has no page free");
            }
            it->second = free_.back();
            free_.pop_back();
        }
        VkBufferImageCopy region{};
        region.bufferOffset = TileMemoryBytes() * i;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                   it->second, 1};
        region.imageExtent = {kPageTexels, kPageTexels, 1};
        regions.push_back(region);
        staging->Write(texels[i], region.bufferOffset);
    }
    if (regions.empty()) {
        return;
    }
    GpuImage& pages = *pages_;
    const GpuBuffer& source = *staging;
    const std::uint64_t submission =
        context_.SubmitAsync([&](VkCommandBuffer commands) {
            pages.Transition(commands,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            vkCmdCopyBufferToImage(
                commands, source.Handle(), pages.Handle(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                static_cast<std::uint32_t>(regions.size()),
                regions.data());
            pages.Transition(
                commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });
    staging_.emplace_back(submission, std::move(staging));
}

void PagePool::WriteTail(
    std::span<const std::span<const std::byte>> mips) {
    if (mips.size() != desc_.mips - tailMip_) {
        throw std::invalid_argument(
            "PagePool::WriteTail needs every mip of the tail");
    }
    tail_->Upload(mips);
}

void PagePool::Release(std::span<const SparseTile> tiles) {
    for (const SparseTile& tile : tiles) {
        if (const auto it = tiles_.find(tile); it != tiles_.end()) {
            free_.push_back(it->second);
            tiles_.erase(it);
        }
    }
}

}  // namespace ernie
