// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/vulkan_context.h"

#include <compare>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ernie {

/// A tile of a virtual texture: mip `mip`'s tile at (`x`, `y`),
/// counted in tiles from the top left.
struct SparseTile {
    std::uint32_t mip = 0;
    std::uint32_t x = 0;
    std::uint32_t y = 0;

    auto operator<=>(const SparseTile&) const = default;
};

/// Where a virtual texture's resident tiles live on the device: a
/// sparse image's bound tiles (`SparseImage`, filtered in hardware)
/// or a pool of pages (`PagePool`, paged in software). Mips of at
/// most one tile form the tail, held whole.
class TileStore {
  public:
    virtual ~TileStore() = default;

    /// Texels along a tile's sides.
    virtual VkExtent2D TileTexels() const = 0;

    /// Texels of the neighbouring tiles written around each tile
    /// (`TileDataBytes` counts them), so filtering stays inside it.
    virtual std::uint32_t TileBorder() const = 0;

    /// The first mip in the tail.
    virtual std::uint32_t TailMip() const = 0;

    /// Tiles across and down mip `mip` (below `TailMip`).
    /// @throws std::out_of_range for a mip in the tail or beyond.
    virtual VkExtent2D TileCount(std::uint32_t mip) const = 0;

    /// Texels across and down `tile`, a whole tile but where it
    /// overhangs its mip's right or bottom edge.
    /// @throws std::out_of_range for a tile the texture does not
    /// have.
    virtual VkExtent2D TileExtent(const SparseTile& tile) const = 0;

    /// Bytes of the texels (or blocks) `Write` takes for `tile`.
    virtual VkDeviceSize
    TileDataBytes(const SparseTile& tile) const = 0;

    /// Device memory a tile takes.
    virtual VkDeviceSize TileMemoryBytes() const = 0;

    /// Makes `tiles` resident, writing `texels[i]` into tile `i`, for
    /// work submitted after it to read.
    virtual void
    Write(std::span<const SparseTile> tiles,
          std::span<const std::span<const std::byte>> texels) = 0;

    /// Writes the tail: `mips[i]` holds mip `TailMip() + i`.
    virtual void
    WriteTail(std::span<const std::span<const std::byte>> mips) = 0;

    /// Frees `tiles` now; no work in flight may read them.
    virtual void Release(std::span<const SparseTile> tiles) = 0;

    virtual bool Bound(const SparseTile& tile) const = 0;
    virtual std::size_t BoundTiles() const = 0;

    /// Device memory held.
    virtual VkDeviceSize Bytes() const = 0;
};

}  // namespace ernie
