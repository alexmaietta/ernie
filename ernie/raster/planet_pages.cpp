// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/planet_pages.h"

#include "ernie/geometry/cube_sphere_lod.h"
#include "ernie/util/log.h"
#include "ernie/util/profile.h"

#include <glm/gtc/packing.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ernie {

namespace {

/// Pages there are at depths 1 to `depth` over faces of `baseCells`
/// quads.
std::uint64_t PagesIn(std::uint32_t baseCells, std::uint32_t depth) {
    std::uint64_t pages = 0;
    for (std::uint32_t j = 1; j <= depth; ++j) {
        const std::uint64_t across =
            (std::uint64_t{baseCells} << j) / PlanetPages::kPageCells;
        pages += CubeSphereLod::kFaces * across * across;
    }
    return pages;
}

}  // namespace

PlanetPages::PlanetPages(const VulkanContext& context,
                         std::shared_ptr<const RasterSource> source,
                         const PlanetHeightsOptions& options,
                         const glm::dvec3& radii,
                         std::uint32_t baseCells, std::uint32_t depth,
                         std::uint32_t capacity)
    : source_(std::move(source)), options_(options), radii_(radii),
      baseCells_(baseCells), depth_(depth),
      capacity_(static_cast<std::uint32_t>(std::min<std::uint64_t>(
          capacity, PagesIn(baseCells, depth)))),
      pool_(context,
            std::max<VkDeviceSize>(
                VkDeviceSize{capacity_} * PageBytes(), 16),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device),
      workers_(
          std::clamp(std::thread::hardware_concurrency() / 2, 1u, 4u),
          "Height pages") {
    if (baseCells == 0 || baseCells % kPageCells != 0 || depth == 0 ||
        capacity_ == 0) {
        throw std::invalid_argument("Planet pages need base faces of "
                                    "whole pages, a depth and "
                                    "room");
    }
    slots_.resize(capacity_);
}

PlanetPages::~PlanetPages() {
    for (auto& [key, loading] : loading_) {
        if (loading.valid()) {
            loading.wait();
        }
    }
}

std::vector<glm::uvec2>
PlanetPages::Packed(const PlanetTile& tile) const {
    std::vector<glm::uvec2> packed(tile.heights.size());
    for (std::size_t i = 0; i < packed.size(); ++i) {
        packed[i] =
            glm::uvec2(std::bit_cast<std::uint32_t>(tile.heights[i]),
                       glm::packHalf2x16(tile.slopes[i]));
    }
    return packed;
}

std::optional<std::uint32_t> PlanetPages::FreeSlot() {
    std::optional<std::uint32_t> oldest;
    for (std::uint32_t slot = 0; slot < capacity_; ++slot) {
        const Slot& s = slots_[slot];
        if (!s.full) {
            return slot;
        }
        // Frames still on the device may read it until it is a frame
        // older than they are.
        if (s.used + FrameUploads::kSlots < frame_ &&
            (!oldest || s.used < slots_[*oldest].used)) {
            oldest = slot;
        }
    }
    return oldest;
}

bool PlanetPages::Place(const Key& key, const PlanetTile& tile,
                        FrameUploads& uploads) {
    const std::optional<std::uint32_t> slot = FreeSlot();
    if (!slot) {
        return false;
    }
    Slot& s = slots_[*slot];
    if (s.full) {
        resident_.erase(s.key);
    }
    s = Slot{key, frame_, true};
    resident_[key] = *slot;
    const std::vector<glm::uvec2> packed = Packed(tile);
    uploads.Write(pool_, std::as_bytes(std::span(packed)),
                  VkDeviceSize{*slot} * PageBytes());
    return true;
}

std::size_t PlanetPages::BeginFrame(std::uint64_t frame,
                                    FrameUploads& uploads,
                                    std::size_t limit) {
    ERNIE_ZONE("Planet pages begin frame");
    frame_ = frame;
    std::size_t placed = 0;
    for (auto it = loading_.begin();
         it != loading_.end() && placed < limit;) {
        if (it->second.wait_for(std::chrono::seconds(0)) !=
            std::future_status::ready) {
            ++it;
            continue;
        }
        try {
            // Made while the pool was full of this frame's pages:
            // made again when next asked for.
            placed +=
                Place(it->first, it->second.get(), uploads) ? 1 : 0;
        } catch (const std::exception& error) {
            failed_.insert(it->first);
            log::Info("Planet height page not made: " +
                      std::string(error.what()));
        }
        it = loading_.erase(it);
    }
    ERNIE_PLOT("Planet pages resident",
               static_cast<std::int64_t>(resident_.size()));
    return placed;
}

PlanetPages::Ref PlanetPages::Find(std::uint32_t face,
                                   std::uint32_t depth,
                                   std::uint32_t x, std::uint32_t y) {
    Ref found;
    depth = std::min(depth, depth_);
    for (std::uint32_t j = 1; j <= depth; ++j) {
        const std::uint32_t shift = depth - j;
        const Key key{j, face, (x >> shift) / kPageCells,
                      (y >> shift) / kPageCells};
        const auto it = resident_.find(key);
        if (it == resident_.end()) {
            if (!loading_.contains(key) && !failed_.contains(key)) {
                requested_.insert(key);
            }
            break;
        }
        slots_[it->second].used = frame_;
        found = Ref{it->second, j, key.x * kPageCells,
                    key.y * kPageCells};
    }
    return found;
}

void PlanetPages::Submit(std::size_t inFlight) {
    // `requested_` orders by depth first: the shallowest go first.
    for (auto it = requested_.begin();
         it != requested_.end() && loading_.size() < inFlight;) {
        const Key key = *it;
        loading_.emplace(key, workers_.Submit([this, key] {
            return PlanetTileOf(
                *source_, options_, radii_, baseCells_ << key.depth,
                key.face, key.x * kPageCells, key.y * kPageCells,
                kPageCells, false, &reading_);
        }));
        it = requested_.erase(it);
    }
}

bool PlanetPages::Complete(FrameUploads& uploads) {
    ERNIE_ZONE("Planet pages complete");
    Submit(std::numeric_limits<std::size_t>::max());
    for (auto& [key, loading] : loading_) {
        loading.wait();
    }
    return BeginFrame(frame_, uploads,
                      std::numeric_limits<std::size_t>::max()) > 0;
}

}  // namespace ernie
