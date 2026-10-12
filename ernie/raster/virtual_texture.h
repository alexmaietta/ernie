// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/buffer.h"
#include "ernie/device/frame_uploads.h"
#include "ernie/device/image.h"
#include "ernie/device/sparse_image.h"
#include "ernie/util/texture_file.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace ernie {

/// A texture far larger than device memory, of which only the tiles
/// in view are resident: the hardware path of virtual texturing, on a
/// `SparseImage`, so filtering spans tiles in hardware.
///
/// Each frame, shaders sampling it mark the tiles they want, at the
/// mip their footprint asks for, in a request buffer (one flag per
/// tile of every mip above the tail, mip after mip, rows top to
/// bottom); `BeginFrame` reads the flags its frame slot's last frame
/// wrote, loads the wanted tiles from the source coarsest first (each
/// with its coarser ancestors), evicts the least recently wanted past
/// the budget, and rewrites the slot's min-LOD map: per mip-0 tile,
/// the finest mip resident there and around it (one tile of margin,
/// wrapping), which shaders clamp the level of detail to, so no
/// sample reaches an unbound tile. The mip tail is always resident.
///
/// Nothing waits for the device: new tiles are bound before any frame
/// that maps them, their texels and each slot's map copied in queue
/// order ahead of the frame; an evicted tile first leaves the maps
/// and is unbound only once the frames that could still sample it
/// have finished (a frame later), so the budget may run over by a
/// frame's evictions.
///
/// Shaders: surface.slang's `virtualTexture`.
///
/// References:
/// - J. M. P. van Waveren, "Software Virtual Textures", id Software,
///   2012.
/// - Khronos, "Sparse Resources", Vulkan specification, chapter 33.
///   https://docs.vulkan.org/spec/latest/chapters/sparsemem.html
class VirtualTexture {
  public:
    /// Tiles loaded per `BeginFrame` at most.
    static constexpr std::size_t kTilesPerFrame = 64;

    /// A virtual texture of `source` (every mip down to 1 x 1),
    /// stored in `format` (`source`'s, or its sRGB or plain variant)
    /// and viewed through `components`, holding at most `budget`
    /// bytes of tiles.
    /// @throws std::runtime_error without sparse residency, or for a
    /// format the device cannot make sparse; std::invalid_argument
    /// for a source without every mip, or larger than the device's
    /// images.
    VirtualTexture(const VulkanContext& context,
                   std::shared_ptr<const TextureData> source,
                   VkFormat format,
                   VkComponentMapping components = {},
                   VkDeviceSize budget = VkDeviceSize{1} << 30);
    /// Waits for its maps' last copies.
    ~VirtualTexture();
    VirtualTexture(const VirtualTexture&) = delete;
    VirtualTexture& operator=(const VirtualTexture&) = delete;

    /// Serves the requests frame slot `slot` (below
    /// `FrameUploads::kSlots`) gathered in its last frame, which has
    /// finished, and clears them for this frame; does nothing more
    /// for a `frame` it has seen. `frame` counts up by one each
    /// frame, whatever its slot.
    void BeginFrame(std::uint32_t slot, std::uint64_t frame);

    /// Marks `tile` (and its coarser ancestors) wanted, as a shader's
    /// request would; served by the next `BeginFrame`.
    void Request(const SparseTile& tile);

    const SparseImage& Image() const {
        return *image_;
    }

    /// Frame slot `slot`'s min-LOD map: one R32 float per mip-0 tile.
    const GpuImage& MinLodMap(std::uint32_t slot) const {
        return *minLods_[slot];
    }

    /// The address of slot `slot`'s request flags.
    VkDeviceAddress RequestAddress(std::uint32_t slot) const {
        return requests_[slot]->Address();
    }

    /// Tiles resident now, the tail's and those evicted but still
    /// bound not counted.
    std::size_t ResidentTiles() const {
        return image_->BoundTiles() - retiring_.size();
    }

    /// The finest mip resident at mip-0 tile (`x`, `y`), before the
    /// margin: what the min-LOD map holds there without its
    /// neighbours.
    std::uint32_t FinestResident(std::uint32_t x,
                                 std::uint32_t y) const;

    /// Device memory held: the sparse image's, the map's and the
    /// request buffers'.
    VkDeviceSize Bytes() const;

  private:
    /// Where `tile`'s flag sits in the request buffers.
    std::size_t FlagIndex(const SparseTile& tile) const;
    /// `tile`'s blocks cut from the source, rows top to bottom.
    std::vector<std::byte> TileData(const SparseTile& tile) const;
    void Want(const SparseTile& tile);
    /// Whether `tile` is bound and not on its way out.
    bool Resident(const SparseTile& tile) const;
    /// Rewrites slot `slot`'s min-LOD map, copied ahead of its frame.
    void WriteMinLod(std::uint32_t slot);

    const VulkanContext& context_;
    std::shared_ptr<const TextureData> source_;
    std::unique_ptr<SparseImage> image_;
    std::array<std::unique_ptr<GpuImage>, FrameUploads::kSlots>
        minLods_;
    std::array<std::unique_ptr<GpuBuffer>, FrameUploads::kSlots>
        minLodStaging_;
    /// Residency changes so far, and the count each slot's map shows.
    std::uint64_t version_ = 0;
    std::array<std::uint64_t, FrameUploads::kSlots> mapVersions_{};
    /// Each slot's last map copy, by its submission (0 for none).
    std::array<std::uint64_t, FrameUploads::kSlots> mapCopies_{};
    std::array<std::unique_ptr<GpuBuffer>, FrameUploads::kSlots>
        requests_;
    /// The first flag of each mip above the tail, and the total.
    std::vector<std::size_t> mipFlags_;
    std::size_t maxTiles_ = 0;
    std::uint64_t frame_ = ~std::uint64_t{0};
    /// Tiles wanted since the last `BeginFrame`, and the frame each
    /// resident tile was last wanted in.
    std::set<SparseTile> wanted_;
    std::map<SparseTile, std::uint64_t> lastWanted_;
    /// Evicted tiles still bound, by the frame they left the maps in.
    std::map<SparseTile, std::uint64_t> retiring_;
};

}  // namespace ernie
