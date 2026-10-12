// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/buffer.h"
#include "ernie/device/tile_store.h"
#include "ernie/device/vulkan_context.h"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace ernie {

/// Description of a `SparseImage`: one 2D layer, sampled.
struct SparseImageDesc {
    std::uint32_t width = 1;
    std::uint32_t height = 1;
    std::uint32_t mips = 1;
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    /// How the image's view maps its channels.
    VkComponentMapping components{};
};

/// A 2D image whose memory is bound tile by tile (Vulkan sparse
/// residency), the hardware path of virtual textures: a texture far
/// larger than device memory holds only the tiles in view, and
/// filtering spans tiles in hardware. Shaders keep their samples to
/// bound tiles by sampling no finer than the finest mip resident
/// there (`VirtualTexture`'s min-LOD map).
///
/// Tiles have the format's standard block shape (64 KiB each: 128 x
/// 128 texels of 32-bit texels, 256 x 256 of BC7). Mips smaller than
/// a tile share the mip tail, bound at creation and written with
/// `WriteTail`. Binding waits for the bind alone; texels are copied
/// in submissions the host does not wait for, ordered before work
/// submitted after them. Releasing unbinds at once: callers keep work
/// in flight off the tiles they release (`VirtualTexture` lets a
/// frame pass first).
///
/// Requires `VulkanContext::SupportsSparseResidency`.
///
/// Reference: Khronos, "Sparse Resources", Vulkan specification,
/// chapter 33.
/// https://docs.vulkan.org/spec/latest/chapters/sparsemem.html
class SparseImage : public TileStore {
  public:
    /// @throws std::runtime_error without sparse residency, or for a
    /// format the device cannot make sparse.
    SparseImage(const VulkanContext& context,
                const SparseImageDesc& desc);
    ~SparseImage() override;
    SparseImage(const SparseImage&) = delete;
    SparseImage& operator=(const SparseImage&) = delete;

    VkImage Handle() const {
        return image_;
    }

    /// View of every mip, sampled in
    /// `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`.
    VkImageView View() const {
        return view_;
    }

    const SparseImageDesc& Desc() const {
        return desc_;
    }

    /// Texels along a tile's sides.
    VkExtent2D TileTexels() const override {
        return tileTexels_;
    }

    /// None: hardware filtering spans tiles.
    std::uint32_t TileBorder() const override {
        return 0;
    }

    /// The first mip in the mip tail; `Desc().mips` when there is
    /// none.
    std::uint32_t TailMip() const override {
        return tailMip_;
    }

    /// Tiles across and down mip `mip` (below `TailMip`).
    /// @throws std::out_of_range for a mip in the tail or beyond.
    VkExtent2D TileCount(std::uint32_t mip) const override;

    /// Texels across and down `tile`, a whole tile but where it
    /// overhangs its mip's right or bottom edge.
    /// @throws std::out_of_range for a tile the image does not have.
    VkExtent2D TileExtent(const SparseTile& tile) const override;

    /// Bytes of tightly packed texels (or blocks) of `tile`.
    VkDeviceSize TileDataBytes(const SparseTile& tile) const override;

    /// Device memory a tile takes.
    VkDeviceSize TileMemoryBytes() const override {
        return tileMemory_;
    }

    /// Binds memory to the tiles not yet bound among `tiles` and
    /// writes `texels[i]` (`TileDataBytes(tiles[i])` bytes, rows top
    /// to bottom) into tile `i`, for work submitted after it to read.
    /// @throws std::invalid_argument for a tile the image does not
    /// have, or data of the wrong size.
    void Write(
        std::span<const SparseTile> tiles,
        std::span<const std::span<const std::byte>> texels) override;

    /// Writes the mip tail: `mips[i]` holds mip `TailMip() + i`,
    /// tightly packed, for work submitted after it to read.
    /// @throws std::invalid_argument for the wrong number of mips or
    /// a mip of the wrong size.
    void WriteTail(
        std::span<const std::span<const std::byte>> mips) override;

    /// Unbinds `tiles` and frees their memory now; no work in flight
    /// may read them. Tiles not bound are skipped.
    void Release(std::span<const SparseTile> tiles) override;

    bool Bound(const SparseTile& tile) const override {
        return tiles_.contains(tile);
    }

    std::size_t BoundTiles() const override {
        return tiles_.size();
    }

    /// Device memory bound: the mip tail and every bound tile.
    VkDeviceSize Bytes() const override;

  private:
    void Check(const SparseTile& tile) const;
    /// Binds (or, with null memory, unbinds) `tiles` on the queue and
    /// waits for it.
    void BindTiles(std::span<const SparseTile> tiles,
                   std::span<const VmaAllocation> memory);
    void Track() const;
    /// Copies `data[i]` into `regions[i]` (whose buffer offsets lay
    /// the data out one after another) in a submission the host does
    /// not wait for, keeping the staging until the device is done.
    void Copy(std::span<const VkBufferImageCopy> regions,
              std::span<const std::span<const std::byte>> data);

    const VulkanContext& context_;
    SparseImageDesc desc_;
    VkImage image_ = VK_NULL_HANDLE;
    VkImageView view_ = VK_NULL_HANDLE;
    VkImageAspectFlags aspect_ = VK_IMAGE_ASPECT_COLOR_BIT;
    VkExtent2D tileTexels_{};
    std::uint32_t tailMip_ = 0;
    VkMemoryRequirements requirements_{};
    VkDeviceSize tileMemory_ = 0;
    VmaAllocation tail_ = VK_NULL_HANDLE;
    VkDeviceSize tailBytes_ = 0;
    std::map<SparseTile, VmaAllocation> tiles_;
    VkImageLayout layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    /// Staging of copies in flight, by their submission.
    std::vector<std::pair<std::uint64_t, std::unique_ptr<GpuBuffer>>>
        staging_;
};

}  // namespace ernie
