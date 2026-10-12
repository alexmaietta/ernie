// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/buffer.h"
#include "ernie/device/image.h"
#include "ernie/device/sparse_image.h"
#include "ernie/device/tile_store.h"

#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace ernie {

/// The software path of virtual textures, for devices without sparse
/// residency and textures larger than the device's images: resident
/// tiles live in pages, the layers of one 2D array image, which
/// shaders find through a page table (`VirtualTexture`) and filter
/// themselves (bilinear within a page, trilinear across two mips).
///
/// A page is `kPageTexels` square: a `kTileTexels` tile with a border
/// of `kBorder` texels from its neighbours (clamped at the texture's
/// edges), so bilinear filtering near a tile's edge stays in its
/// page. Every size is a whole number of 4 x 4 blocks, so BC tiles
/// are cut from BC sources as they are. Mips of at most one tile form
/// the tail, a plain image with mips.
///
/// Writes are copied in submissions the host does not wait for,
/// ordered before work submitted after them.
///
/// Reference: J. M. P. van Waveren, "Software Virtual Textures", id
/// Software, 2012.
class PagePool : public TileStore {
  public:
    static constexpr std::uint32_t kPageTexels = 128;
    static constexpr std::uint32_t kBorder = 4;
    static constexpr std::uint32_t kTileTexels =
        kPageTexels - 2 * kBorder;

    /// A pool of `pages` pages (at most the device's image layers)
    /// for a texture as `desc` describes it, of any size.
    /// @throws std::invalid_argument for no pages.
    PagePool(const VulkanContext& context,
             const SparseImageDesc& desc, std::uint32_t pages);
    ~PagePool() override;
    PagePool(const PagePool&) = delete;
    PagePool& operator=(const PagePool&) = delete;

    VkExtent2D TileTexels() const override {
        return {kTileTexels, kTileTexels};
    }

    std::uint32_t TileBorder() const override {
        return kBorder;
    }

    std::uint32_t TailMip() const override {
        return tailMip_;
    }

    VkExtent2D TileCount(std::uint32_t mip) const override;
    VkExtent2D TileExtent(const SparseTile& tile) const override;

    /// A whole page, its border included.
    VkDeviceSize TileDataBytes(const SparseTile& tile) const override;

    VkDeviceSize TileMemoryBytes() const override;

    /// Puts each tile not yet resident in a free page, and writes its
    /// page (`TileDataBytes`, rows top to bottom).
    /// @throws std::invalid_argument for a tile the texture does not
    /// have or data of the wrong size; std::runtime_error with no
    /// page free.
    void Write(
        std::span<const SparseTile> tiles,
        std::span<const std::span<const std::byte>> texels) override;

    void WriteTail(
        std::span<const std::span<const std::byte>> mips) override;
    void Release(std::span<const SparseTile> tiles) override;

    bool Bound(const SparseTile& tile) const override {
        return tiles_.contains(tile);
    }

    std::size_t BoundTiles() const override {
        return tiles_.size();
    }

    VkDeviceSize Bytes() const override;

    /// The page holding `tile`.
    /// @throws std::out_of_range for a tile not resident.
    std::uint32_t PageOf(const SparseTile& tile) const {
        return tiles_.at(tile);
    }

    std::uint32_t Pages() const {
        return pageCount_;
    }

    /// The pages: a 2D array, a page a layer.
    const GpuImage& PageImage() const {
        return *pages_;
    }

    /// The tail: mip `TailMip()` and those below it.
    const GpuImage& TailImage() const {
        return *tail_;
    }

  private:
    const VulkanContext& context_;
    SparseImageDesc desc_;
    std::uint32_t tailMip_ = 0;
    std::uint32_t pageCount_ = 0;
    std::unique_ptr<GpuImage> pages_;
    std::unique_ptr<GpuImage> tail_;
    std::vector<std::uint32_t> free_;
    std::map<SparseTile, std::uint32_t> tiles_;
    /// Staging of copies in flight, by their submission.
    std::vector<std::pair<std::uint64_t, std::unique_ptr<GpuBuffer>>>
        staging_;
};

}  // namespace ernie
