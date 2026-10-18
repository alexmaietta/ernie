// SPDX-License-Identifier: BSD-3-Clause
/// @file extent_cache.h
/// @brief Prims' extents kept per bucket and refreshed for the rows that changed.
#pragma once

#include "ernie/geometry/bounds.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/storage.h"
#include "ernie/util/bytes.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace ernie {

/// @brief The extents of every bucket with an extent function, a box per row. `Update` replays
/// the buckets' row changes and recomputes only the rows whose values or places changed.
class ExtentCache {
  public:
    /// Tracks a bucket, computing its extents with `compute`; null for a bucket without them.
    void Track(BucketId bucket, ExtentFunction compute);
    /// Brings every tracked bucket up to the store's version.
    void Update(const PrimStore& store);
    /// @return A row's extent, or none when its bucket changed since `Update` or has none.
    std::optional<Box> Find(const PrimStore& store, BucketId bucket, std::uint32_t row) const;

  private:
    struct Entry {
        ExtentFunction compute = nullptr;
        std::vector<Box> boxes;
        BitSet stale;
    };

    static void Replay(Entry& entry, const StructuralChange& change);
    static void Resize(Entry& entry, std::uint32_t rows);
    static void Refresh(Entry& entry, const PrimStore& store, BucketId bucket);

    std::vector<Entry> buckets_;
    std::uint64_t cursor_ = 0;
};

}  // namespace ernie
