// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/virtual_texture.h"

#include "ernie/device/gpu_profile.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ernie {

namespace {

std::uint32_t FullMipCount(std::uint32_t width,
                           std::uint32_t height) {
    std::uint32_t count = 1;
    for (std::uint32_t side = std::max(width, height); side > 1;
         side >>= 1) {
        ++count;
    }
    return count;
}

}  // namespace

VirtualTexture::VirtualTexture(
    const VulkanContext& context,
    std::shared_ptr<const TextureData> source, VkFormat format,
    VkComponentMapping components, VkDeviceSize budget)
    : context_(context), source_(std::move(source)) {
    ERNIE_ZONE("Make virtual texture");
    if (source_->mips !=
        FullMipCount(source_->width, source_->height)) {
        throw std::invalid_argument(
            "A virtual texture needs every mip down to 1 x 1");
    }
    const SparseImageDesc desc{source_->width, source_->height,
                               source_->mips, format, components};
    // A sparse image where the device has sparse residency and the
    // texture fits its images; else pages, of which the budget holds
    // as many as the device's image layers allow.
    const std::uint32_t maxSide =
        context.Limits().maxImageDimension2D;
    if (context.SupportsSparseResidency() &&
        source_->width <= maxSide && source_->height <= maxSide) {
        try {
            auto sparse =
                std::make_unique<SparseImage>(context, desc);
            sparse_ = sparse.get();
            store_ = std::move(sparse);
        } catch (const std::runtime_error&) {
            // A format the device cannot make sparse: pages instead.
        }
    }
    if (store_ == nullptr) {
        const VkDeviceSize page = GpuImage::MipBytes(
            format, PagePool::kPageTexels, PagePool::kPageTexels);
        auto pool = std::make_unique<PagePool>(
            context, desc,
            static_cast<std::uint32_t>(std::clamp<VkDeviceSize>(
                budget / page, 1, VkDeviceSize{UINT32_MAX})));
        pool_ = pool.get();
        store_ = std::move(pool);
    }
    std::vector<std::span<const std::byte>> tail;
    for (std::uint32_t mip = store_->TailMip(); mip < source_->mips;
         ++mip) {
        tail.push_back(source_->Mip(0, mip));
    }
    store_->WriteTail(tail);

    std::size_t flags = 0;
    for (std::uint32_t mip = 0; mip < store_->TailMip(); ++mip) {
        mipFlags_.push_back(flags);
        const VkExtent2D count = store_->TileCount(mip);
        flags += std::size_t{count.width} * count.height;
    }
    mipFlags_.push_back(flags);
    // Pages' tables follow the flags in the same buffers.
    const std::vector<std::uint32_t> zeros(
        std::max<std::size_t>(flags, 1) *
            (store_->TileBorder() > 0 ? 2 : 1),
        0);
    for (auto& requests : requests_) {
        requests = std::make_unique<GpuBuffer>(
            context, zeros.size() * sizeof(std::uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            BufferMemory::Readback);
        requests->Write(std::as_bytes(std::span(zeros)));
    }
    maxTiles_ = std::max<std::size_t>(
        1,
        static_cast<std::size_t>(budget / store_->TileMemoryBytes()));
    if (pool_ != nullptr) {
        maxTiles_ = std::min<std::size_t>(maxTiles_, pool_->Pages());
        for (std::uint32_t slot = 0; slot < FrameUploads::kSlots;
             ++slot) {
            WritePageTable(slot);
        }
        return;
    }
    const VkExtent2D count = store_->TailMip() > 0
                                 ? store_->TileCount(0)
                                 : VkExtent2D{1, 1};
    for (std::uint32_t slot = 0; slot < FrameUploads::kSlots;
         ++slot) {
        minLods_[slot] = std::make_unique<GpuImage>(
            context, ImageDesc{count.width, count.height, 1, 1,
                               VK_FORMAT_R32_SFLOAT,
                               VK_IMAGE_USAGE_SAMPLED_BIT});
        minLodStaging_[slot] = std::make_unique<GpuBuffer>(
            context,
            VkDeviceSize{count.width} * count.height * sizeof(float),
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, BufferMemory::Upload);
        WriteMinLod(slot);
    }
}

std::size_t VirtualTexture::FlagIndex(const SparseTile& tile) const {
    const VkExtent2D count = store_->TileCount(tile.mip);
    return mipFlags_[tile.mip] + std::size_t{tile.y} * count.width +
           tile.x;
}

void VirtualTexture::Want(const SparseTile& tile) {
    for (std::uint32_t mip = tile.mip; mip < store_->TailMip();
         ++mip) {
        const std::uint32_t shift = mip - tile.mip;
        wanted_.insert({mip, tile.x >> shift, tile.y >> shift});
    }
}

void VirtualTexture::Request(const SparseTile& tile) {
    store_->TileExtent(tile);  // Throws for a tile the image lacks.
    Want(tile);
}

std::vector<std::byte>
VirtualTexture::TileData(const SparseTile& tile) const {
    const TextureFormatInfo info = InfoOf(source_->format);
    const VkExtent2D texels = store_->TileTexels();
    const std::uint32_t border = store_->TileBorder();
    const std::uint32_t mipWidth =
        std::max(1u, source_->width >> tile.mip);
    const std::uint32_t mipHeight =
        std::max(1u, source_->height >> tile.mip);
    const auto blocks = [&](std::uint32_t texels) {
        return (texels + info.blockSize - 1) / info.blockSize;
    };
    const std::uint32_t blocksAcrossMip = blocks(mipWidth);
    const std::uint32_t blocksDownMip = blocks(mipHeight);
    // A bordered page is whole; a sparse tile is cut at the mip's
    // edge. Tile origins and borders fall on whole blocks.
    const VkExtent2D extent =
        border > 0 ? VkExtent2D{texels.width + 2 * border,
                                texels.height + 2 * border}
                   : store_->TileExtent(tile);
    const std::uint32_t across = blocks(extent.width);
    const std::uint32_t down = blocks(extent.height);
    const auto firstColumn =
        static_cast<std::int64_t>(tile.x * texels.width /
                                  info.blockSize) -
        static_cast<std::int64_t>(border / info.blockSize);
    const auto firstRow =
        static_cast<std::int64_t>(tile.y * texels.height /
                                  info.blockSize) -
        static_cast<std::int64_t>(border / info.blockSize);
    const std::span<const std::byte> mip = source_->Mip(0, tile.mip);
    std::vector<std::byte> cut(std::size_t{across} * down *
                               info.blockBytes);
    for (std::uint32_t row = 0; row < down; ++row) {
        // Blocks past the mip's edges repeat its last ones.
        const auto sourceRow =
            static_cast<std::size_t>(std::clamp<std::int64_t>(
                firstRow + row, 0, blocksDownMip - 1));
        for (std::uint32_t column = 0; column < across; ++column) {
            const auto sourceColumn =
                static_cast<std::size_t>(std::clamp<std::int64_t>(
                    firstColumn + column, 0, blocksAcrossMip - 1));
            std::memcpy(
                cut.data() + (std::size_t{row} * across + column) *
                                 info.blockBytes,
                mip.data() +
                    (sourceRow * blocksAcrossMip + sourceColumn) *
                        info.blockBytes,
                info.blockBytes);
        }
    }
    return cut;
}

void VirtualTexture::BeginFrame(std::uint32_t slot,
                                std::uint64_t frame) {
    if (frame == frame_) {
        return;
    }
    frame_ = frame;
    ERNIE_ZONE("Virtual texture frame");
    GatherRequests(slot);
    Serve(frame, kTilesPerFrame);
    // This slot's map, if it shows an older residency.
    if (mapVersions_[slot] != version_) {
        if (pool_ != nullptr) {
            WritePageTable(slot);
        } else {
            WriteMinLod(slot);
        }
    }
}

bool VirtualTexture::Settle(std::uint32_t slot) {
    ERNIE_ZONE("Settle virtual texture");
    GatherRequests(slot);
    const std::size_t loaded =
        Serve(frame_, std::numeric_limits<std::size_t>::max());
    for (std::uint32_t each = 0; each < FrameUploads::kSlots;
         ++each) {
        if (mapVersions_[each] != version_) {
            if (pool_ != nullptr) {
                WritePageTable(each);
            } else {
                WriteMinLod(each);
            }
        }
    }
    return loaded > 0;
}

void VirtualTexture::GatherRequests(std::uint32_t slot) {
    // The flags this slot's last frame set, which has finished.
    GpuBuffer& requests = *requests_[slot];
    std::vector<std::uint32_t> flags(mipFlags_.back());
    if (!flags.empty()) {
        ERNIE_ZONE("Read tile requests");
        requests.Read(std::as_writable_bytes(std::span(flags)));
    }
    bool any = false;
    for (std::size_t i = 0; i < flags.size(); ++i) {
        if (flags[i] == 0) {
            continue;
        }
        any = true;
        const auto next =
            std::upper_bound(mipFlags_.begin(), mipFlags_.end(), i);
        const auto mip =
            static_cast<std::uint32_t>(next - mipFlags_.begin() - 1);
        const std::size_t index = i - mipFlags_[mip];
        const std::uint32_t across = store_->TileCount(mip).width;
        Want({mip, static_cast<std::uint32_t>(index % across),
              static_cast<std::uint32_t>(index / across)});
    }
    if (any) {
        const std::vector<std::uint32_t> zeros(flags.size(), 0);
        requests.Write(std::as_bytes(std::span(zeros)));
    }
}

std::size_t VirtualTexture::Serve(std::uint64_t frame,
                                  std::size_t limit) {
    // Evicted tiles go once every frame whose map still showed them
    // has finished: the frames before the one they left the maps in,
    // the last of which ran in the other slot.
    bool changed = false;
    std::vector<SparseTile> unbind;
    for (auto it = retiring_.begin(); it != retiring_.end();) {
        if (it->second + FrameUploads::kSlots - 1 <= frame) {
            unbind.push_back(it->first);
            it = retiring_.erase(it);
        } else {
            ++it;
        }
    }
    store_->Release(unbind);

    // Wanted tiles not resident, coarsest first so detail sharpens in
    // order; the rest wait for later frames. Evicted tiles wanted
    // again come back as they are.
    std::vector<SparseTile> load;
    for (const SparseTile& tile : wanted_) {
        lastWanted_[tile] = frame;
        if (retiring_.erase(tile) > 0) {
            changed = true;
        } else if (!store_->Bound(tile)) {
            load.push_back(tile);
        }
    }
    std::stable_sort(load.begin(), load.end(),
                     [](const SparseTile& a, const SparseTile& b) {
                         return a.mip > b.mip;
                     });
    if (load.size() > limit) {
        load.resize(limit);
    }
    // Past the budget, the least recently wanted resident tiles go,
    // finest first; never those wanted this frame.
    std::vector<SparseTile> evict;
    const std::size_t resident = store_->BoundTiles();
    if (resident + load.size() > maxTiles_) {
        std::vector<std::pair<std::uint64_t, SparseTile>> candidates;
        for (const auto& [tile, last] : lastWanted_) {
            if (Resident(tile) && !wanted_.contains(tile)) {
                candidates.emplace_back(last, tile);
            }
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) {
                      return a.first != b.first
                                 ? a.first < b.first
                                 : a.second.mip < b.second.mip;
                  });
        const std::size_t excess = resident + load.size() - maxTiles_;
        for (std::size_t i = 0;
             i < std::min(excess, candidates.size()); ++i) {
            evict.push_back(candidates[i].second);
        }
        const std::size_t room =
            maxTiles_ - (resident - evict.size());
        if (load.size() > room) {
            load.resize(room);
        }
    }
    for (const SparseTile& tile : load) {
        wanted_.erase(tile);
    }
    // Tiles wanted this frame are asked for again by the shaders that
    // still sample them; those loaded leave the list.
    for (auto it = wanted_.begin(); it != wanted_.end();) {
        it = Resident(*it) ? wanted_.erase(it) : std::next(it);
    }
    for (const SparseTile& tile : evict) {
        retiring_[tile] = frame;
        lastWanted_.erase(tile);
        changed = true;
    }
    if (!load.empty()) {
        ERNIE_ZONE("Load tiles");
        std::vector<std::vector<std::byte>> data;
        std::vector<std::span<const std::byte>> spans;
        {
            // From the mapped file: the disk reads, the first time.
            ERNIE_ZONE("Cut tiles");
            for (const SparseTile& tile : load) {
                data.push_back(TileData(tile));
            }
        }
        for (const auto& bytes : data) {
            spans.push_back(bytes);
        }
        store_->Write(load, spans);
        changed = true;
    }
    ERNIE_PLOT("Virtual texture tiles loaded",
               static_cast<std::int64_t>(load.size()));
    ERNIE_PLOT("Virtual texture tiles resident",
               static_cast<std::int64_t>(store_->BoundTiles()));
    if (changed) {
        ++version_;
    }
    return load.size();
}

void VirtualTexture::WritePageTable(std::uint32_t slot) {
    ERNIE_ZONE("Write page table");
    std::vector<std::uint32_t> table(
        std::max<std::size_t>(mipFlags_.back(), 1), 0);
    for (std::uint32_t mip = 0; mip < store_->TailMip(); ++mip) {
        const VkExtent2D count = store_->TileCount(mip);
        for (std::uint32_t y = 0; y < count.height; ++y) {
            for (std::uint32_t x = 0; x < count.width; ++x) {
                const SparseTile tile{mip, x, y};
                if (Resident(tile)) {
                    table[FlagIndex(tile)] = pool_->PageOf(tile) + 1;
                }
            }
        }
    }
    requests_[slot]->Write(
        std::as_bytes(std::span(table)),
        std::max<std::size_t>(mipFlags_.back(), 1) *
            sizeof(std::uint32_t));
    mapVersions_[slot] = version_;
}

VirtualTexture::~VirtualTexture() {
    for (const std::uint64_t copy : mapCopies_) {
        context_.WaitForSubmission(copy);
    }
}

bool VirtualTexture::Resident(const SparseTile& tile) const {
    return store_->Bound(tile) && !retiring_.contains(tile);
}

std::uint32_t VirtualTexture::FinestResident(std::uint32_t x,
                                             std::uint32_t y) const {
    std::uint32_t finest = store_->TailMip();
    for (std::uint32_t mip = store_->TailMip(); mip-- > 0;) {
        if (!Resident({mip, x >> mip, y >> mip})) {
            break;
        }
        finest = mip;
    }
    return finest;
}

void VirtualTexture::WriteMinLod(std::uint32_t slot) {
    ERNIE_ZONE("Write min-LOD map");
    const VkExtent2D count = store_->TailMip() > 0
                                 ? store_->TileCount(0)
                                 : VkExtent2D{1, 1};
    std::vector<float> finest(std::size_t{count.width} *
                              count.height);
    for (std::uint32_t y = 0; y < count.height; ++y) {
        for (std::uint32_t x = 0; x < count.width; ++x) {
            finest[std::size_t{y} * count.width + x] =
                static_cast<float>(
                    store_->TailMip() > 0 ? FinestResident(x, y) : 0);
        }
    }
    // A filter's footprint reaches into the neighbouring tiles,
    // across the edges for repeating textures: each texel takes the
    // coarsest of its own and its eight neighbours'.
    std::vector<float> clamp(finest.size());
    for (std::uint32_t y = 0; y < count.height; ++y) {
        for (std::uint32_t x = 0; x < count.width; ++x) {
            float coarsest = 0.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const std::uint32_t nx =
                        (x + count.width +
                         static_cast<std::uint32_t>(dx)) %
                        count.width;
                    const std::uint32_t ny =
                        (y + count.height +
                         static_cast<std::uint32_t>(dy)) %
                        count.height;
                    coarsest = std::max(
                        coarsest,
                        finest[std::size_t{ny} * count.width + nx]);
                }
            }
            clamp[std::size_t{y} * count.width + x] = coarsest;
        }
    }
    // The last copy from this staging has almost always finished with
    // the slot's last frame (not when the texture was made
    // mid-frame); this copy runs ahead of the slot's next frame.
    context_.WaitForSubmission(mapCopies_[slot]);
    GpuBuffer& staging = *minLodStaging_[slot];
    staging.Write(std::as_bytes(std::span(clamp)));
    GpuImage& map = *minLods_[slot];
    mapCopies_[slot] =
        context_.SubmitAsync([&](VkCommandBuffer commands) {
            ERNIE_GPU_ZONE(context_, commands, "Min-LOD map copy");
            map.Transition(commands,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                       0, 1};
            region.imageExtent = {count.width, count.height, 1};
            vkCmdCopyBufferToImage(
                commands, staging.Handle(), map.Handle(),
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            map.Transition(commands,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });
    mapVersions_[slot] = version_;
}

VkDeviceSize VirtualTexture::Bytes() const {
    VkDeviceSize bytes = store_->Bytes();
    for (const auto& map : minLods_) {
        bytes += map != nullptr ? map->Bytes() : 0;
    }
    for (const auto& requests : requests_) {
        bytes += requests->Size();
    }
    return bytes;
}

}  // namespace ernie
