// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/virtual_texture.h"

#include <algorithm>
#include <cstring>
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
    if (source_->mips !=
        FullMipCount(source_->width, source_->height)) {
        throw std::invalid_argument(
            "A virtual texture needs every mip down to 1 x 1");
    }
    const std::uint32_t maxSide =
        context.Limits().maxImageDimension2D;
    if (source_->width > maxSide || source_->height > maxSide) {
        throw std::invalid_argument("Virtual texture exceeds the "
                                    "device's maximum image size (" +
                                    std::to_string(maxSide) + ")");
    }
    image_ = std::make_unique<SparseImage>(
        context, SparseImageDesc{source_->width, source_->height,
                                 source_->mips, format, components});
    std::vector<std::span<const std::byte>> tail;
    for (std::uint32_t mip = image_->TailMip(); mip < source_->mips;
         ++mip) {
        tail.push_back(source_->Mip(0, mip));
    }
    image_->WriteTail(tail);

    std::size_t flags = 0;
    for (std::uint32_t mip = 0; mip < image_->TailMip(); ++mip) {
        mipFlags_.push_back(flags);
        const VkExtent2D count = image_->TileCount(mip);
        flags += std::size_t{count.width} * count.height;
    }
    mipFlags_.push_back(flags);
    const std::vector<std::uint32_t> zeros(
        std::max<std::size_t>(flags, 1), 0);
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
        static_cast<std::size_t>(budget / image_->TileMemoryBytes()));
    WriteMinLod();
}

std::size_t VirtualTexture::FlagIndex(const SparseTile& tile) const {
    const VkExtent2D count = image_->TileCount(tile.mip);
    return mipFlags_[tile.mip] + std::size_t{tile.y} * count.width +
           tile.x;
}

void VirtualTexture::Want(const SparseTile& tile) {
    for (std::uint32_t mip = tile.mip; mip < image_->TailMip();
         ++mip) {
        const std::uint32_t shift = mip - tile.mip;
        wanted_.insert({mip, tile.x >> shift, tile.y >> shift});
    }
}

void VirtualTexture::Request(const SparseTile& tile) {
    image_->TileExtent(tile);  // Throws for a tile the image lacks.
    Want(tile);
}

std::vector<std::byte>
VirtualTexture::TileData(const SparseTile& tile) const {
    const TextureFormatInfo info = InfoOf(source_->format);
    const VkExtent2D texels = image_->TileTexels();
    const VkExtent2D extent = image_->TileExtent(tile);
    const std::uint32_t mipWidth =
        std::max(1u, source_->width >> tile.mip);
    const std::size_t blocksAcrossMip =
        (mipWidth + info.blockSize - 1) / info.blockSize;
    const std::size_t across =
        (extent.width + info.blockSize - 1) / info.blockSize;
    const std::size_t down =
        (extent.height + info.blockSize - 1) / info.blockSize;
    // Tile origins fall on whole blocks: tiles span whole blocks.
    const std::size_t firstColumn =
        tile.x * texels.width / info.blockSize;
    const std::size_t firstRow =
        tile.y * texels.height / info.blockSize;
    const std::span<const std::byte> mip = source_->Mip(0, tile.mip);
    std::vector<std::byte> data(across * down * info.blockBytes);
    for (std::size_t row = 0; row < down; ++row) {
        std::memcpy(data.data() + row * across * info.blockBytes,
                    mip.data() + ((firstRow + row) * blocksAcrossMip +
                                  firstColumn) *
                                     info.blockBytes,
                    across * info.blockBytes);
    }
    return data;
}

void VirtualTexture::BeginFrame(std::uint32_t slot,
                                std::uint64_t frame) {
    if (frame == frame_) {
        return;
    }
    frame_ = frame;
    // The flags this slot's last frame set, which has finished.
    GpuBuffer& requests = *requests_[slot];
    std::vector<std::uint32_t> flags(mipFlags_.back());
    if (!flags.empty()) {
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
        const std::uint32_t across = image_->TileCount(mip).width;
        Want({mip, static_cast<std::uint32_t>(index % across),
              static_cast<std::uint32_t>(index / across)});
    }
    if (any) {
        const std::vector<std::uint32_t> zeros(flags.size(), 0);
        requests.Write(std::as_bytes(std::span(zeros)));
    }

    // Wanted tiles not resident, coarsest first so detail sharpens in
    // order; the rest wait for later frames.
    std::vector<SparseTile> load;
    for (const SparseTile& tile : wanted_) {
        lastWanted_[tile] = frame;
        if (!image_->Bound(tile)) {
            load.push_back(tile);
        }
    }
    std::stable_sort(load.begin(), load.end(),
                     [](const SparseTile& a, const SparseTile& b) {
                         return a.mip > b.mip;
                     });
    if (load.size() > kTilesPerFrame) {
        load.resize(kTilesPerFrame);
    }
    // Past the budget, the least recently wanted resident tiles go,
    // finest first; never those wanted this frame.
    std::vector<SparseTile> evict;
    const std::size_t resident = image_->BoundTiles();
    if (resident + load.size() > maxTiles_) {
        std::vector<std::pair<std::uint64_t, SparseTile>> candidates;
        for (const auto& [tile, last] : lastWanted_) {
            if (image_->Bound(tile) && !wanted_.contains(tile)) {
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
        it = image_->Bound(*it) ? wanted_.erase(it) : std::next(it);
    }
    if (load.empty() && evict.empty()) {
        return;
    }
    vkDeviceWaitIdle(context_.Device());
    image_->Release(evict);
    for (const SparseTile& tile : evict) {
        lastWanted_.erase(tile);
    }
    std::vector<std::vector<std::byte>> data;
    std::vector<std::span<const std::byte>> spans;
    for (const SparseTile& tile : load) {
        data.push_back(TileData(tile));
    }
    for (const auto& bytes : data) {
        spans.push_back(bytes);
    }
    image_->Write(load, spans);
    WriteMinLod();
}

std::uint32_t VirtualTexture::FinestResident(std::uint32_t x,
                                             std::uint32_t y) const {
    std::uint32_t finest = image_->TailMip();
    for (std::uint32_t mip = image_->TailMip(); mip-- > 0;) {
        if (!image_->Bound({mip, x >> mip, y >> mip})) {
            break;
        }
        finest = mip;
    }
    return finest;
}

void VirtualTexture::WriteMinLod() {
    const VkExtent2D count = image_->TailMip() > 0
                                 ? image_->TileCount(0)
                                 : VkExtent2D{1, 1};
    std::vector<float> finest(std::size_t{count.width} *
                              count.height);
    for (std::uint32_t y = 0; y < count.height; ++y) {
        for (std::uint32_t x = 0; x < count.width; ++x) {
            finest[std::size_t{y} * count.width + x] =
                static_cast<float>(
                    image_->TailMip() > 0 ? FinestResident(x, y) : 0);
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
    auto map = std::make_unique<GpuImage>(
        context_,
        ImageDesc{count.width, count.height, 1, 1,
                  VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT});
    const std::span<const std::byte> mips[] = {
        std::as_bytes(std::span(clamp))};
    map->Upload(mips);
    minLod_ = std::move(map);
}

VkDeviceSize VirtualTexture::Bytes() const {
    VkDeviceSize bytes = image_->Bytes() + minLod_->Bytes();
    for (const auto& requests : requests_) {
        bytes += requests->Size();
    }
    return bytes;
}

}  // namespace ernie
