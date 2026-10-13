// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/buffer.h"
#include "ernie/device/frame_uploads.h"
#include "ernie/geometry/planet_heights.h"
#include "ernie/util/raster_source.h"
#include "ernie/util/worker_pool.h"

#include <glm/glm.hpp>

#include <compare>
#include <cstdint>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <vector>

namespace ernie {

class VulkanContext;

/// A planet's heights finer than its base faces, paged in from a
/// latitude-longitude raster as views need them: virtual heights,
/// for elevation models too large for one texture a face (LOLA's and
/// MOLA's at 128 pixels a degree are 6 x 16385^2 heights).
///
/// The planet's faces are grids of `baseCells` quads; depth j of the
/// pages is the grid 2^j times finer, up to `Depth()`. A page is 256
/// quads square of one depth (`PlanetTileOf`, its heights unfiltered
/// so every depth agrees where its points meet, and its slopes for
/// normals), aligned to multiples of 256 there, so a LOD patch whose
/// vertices are a depth's points lies in one page. Pages live in one
/// storage buffer of `Capacity()` slots, the least recently used
/// replaced; they are made on threads of their own, a frame's patches
/// reading the deepest page made so far and asking for the next
/// depth, so detail arrives coarse to fine.
///
/// References:
/// - J. M. P. van Waveren, "Software Virtual Textures", id Software,
///   2012 (pages requested by what is drawn, the coarser page drawn
///   until the finer arrives).
class PlanetPages {
  public:
    static constexpr std::uint32_t kPageCells = 256;
    static constexpr std::uint32_t kPageSide = kPageCells + 1;
    /// A `Ref` slot reading the base faces.
    static constexpr std::uint32_t kNone = ~0u;

    /// A page: its depth (1 to `Depth()`), face, and column and row
    /// among that depth's pages.
    struct Key {
        std::uint32_t depth = 0;
        std::uint32_t face = 0;
        std::uint32_t x = 0;
        std::uint32_t y = 0;
        auto operator<=>(const Key&) const = default;
    };

    /// What a patch reads: the page's slot (`kNone` for the base
    /// faces), its depth, and its first point's column and row in its
    /// depth's grid; as planet.slang's `patchPages`.
    struct Ref {
        std::uint32_t slot = kNone;
        std::uint32_t depth = 0;
        std::uint32_t x = 0;
        std::uint32_t y = 0;
    };

    /// Pages of `source` (read as `options` say) over the ellipsoid
    /// of semi-axes `radii`, its base faces `baseCells` quads across
    /// (a multiple of `kPageCells`), `depth` levels finer, in a pool
    /// of `capacity` pages (at most as many as there are).
    /// @throws std::invalid_argument for base faces not of whole
    /// pages, or no depth or capacity.
    PlanetPages(const VulkanContext& context,
                std::shared_ptr<const RasterSource> source,
                const PlanetHeightsOptions& options,
                const glm::dvec3& radii, std::uint32_t baseCells,
                std::uint32_t depth, std::uint32_t capacity = 512);
    /// Waits for pages being made.
    ~PlanetPages();
    PlanetPages(const PlanetPages&) = delete;
    PlanetPages& operator=(const PlanetPages&) = delete;

    std::uint32_t Depth() const {
        return depth_;
    }
    std::uint32_t BaseCells() const {
        return baseCells_;
    }
    std::uint32_t Capacity() const {
        return capacity_;
    }

    /// Begins frame `frame`: pages made since the last frame are
    /// written into the pool through `uploads`, at most `limit`.
    /// Returns how many.
    std::size_t BeginFrame(std::uint64_t frame, FrameUploads& uploads,
                           std::size_t limit = 16);

    /// The page a patch of face `face` whose vertices are depth
    /// `depth`'s points, from point (`x`, `y`) there, reads: the
    /// deepest resident page at or above `depth` holding it, else the
    /// base faces. Asks for the shallowest page missing on the way.
    Ref Find(std::uint32_t face, std::uint32_t depth, std::uint32_t x,
             std::uint32_t y);

    /// Starts making the pages asked for since the last call,
    /// shallowest first, keeping at most `inFlight` being made.
    void Submit(std::size_t inFlight = 32);

    /// Makes every page asked for, waiting for them, and writes them
    /// through `uploads`: for frames that may not show coarser pages.
    /// Returns whether any page became resident.
    bool Complete(FrameUploads& uploads);

    /// Pages asked for or being made.
    std::size_t Pending() const {
        return requested_.size() + loading_.size();
    }
    std::size_t Resident() const {
        return resident_.size();
    }

    const GpuBuffer& Pool() const {
        return pool_;
    }

    /// Bytes of one page in the pool: per point its height's float
    /// bits and its slopes as two half floats.
    static constexpr std::size_t PageBytes() {
        return std::size_t{kPageSide} * kPageSide * 8;
    }

  private:
    /// The page's points packed as the pool holds them.
    std::vector<glm::uvec2> Packed(const PlanetTile& tile) const;
    /// A slot for a new page: free, else the least recently used one
    /// no frame in flight reads; none if every slot is in use.
    std::optional<std::uint32_t> FreeSlot();
    /// Writes the page into a slot; false if none is free.
    bool Place(const Key& key, const PlanetTile& tile,
               FrameUploads& uploads);

    std::shared_ptr<const RasterSource> source_;
    PlanetHeightsOptions options_;
    glm::dvec3 radii_;
    std::uint32_t baseCells_;
    std::uint32_t depth_;
    std::uint32_t capacity_;
    GpuBuffer pool_;

    struct Slot {
        Key key;
        std::uint64_t used = 0;
        bool full = false;
    };
    std::vector<Slot> slots_;
    std::map<Key, std::uint32_t> resident_;
    std::set<Key> requested_;
    std::map<Key, std::future<PlanetTile>> loading_;
    /// Pages that could not be made, never asked for again.
    std::set<Key> failed_;
    std::uint64_t frame_ = 0;
    /// Rasters are read by one thread at a time.
    std::mutex reading_;
    WorkerPool workers_;
};

}  // namespace ernie
