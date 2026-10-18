// SPDX-License-Identifier: BSD-3-Clause
/// @file extent_cache.cpp
/// @brief Replaying row changes into the extent boxes and recomputing stale runs.
#include "ernie/scene/extent_cache.h"

#include <algorithm>

namespace ernie {

void ExtentCache::Track(BucketId bucket, ExtentFunction compute) {
    buckets_.resize(std::max<std::size_t>(buckets_.size(), bucket + 1));
    buckets_[bucket].compute = compute;
}

void ExtentCache::Update(const PrimStore& store) {
    const ChangeSet changes = store.ChangesSince(cursor_);
    const auto tracked = [&](BucketId bucket) {
        return bucket < buckets_.size() && buckets_[bucket].compute != nullptr;
    };
    for (const BucketId bucket : changes.resets) {
        if (tracked(bucket)) {
            Resize(buckets_[bucket], 0);
        }
    }
    for (const ChangeSet::Rows& rows : changes.rows) {
        if (tracked(rows.bucket)) {
            Replay(buckets_[rows.bucket], rows.change);
        }
    }
    for (const ChangeSet::Values& values : changes.values) {
        if (tracked(values.bucket)) {
            for (const RowRange range : values.rows) {
                buckets_[values.bucket].stale.SetRange(range.begin, range.end);
            }
        }
    }
    for (BucketId bucket = 0; bucket < buckets_.size(); ++bucket) {
        if (tracked(bucket) && store.GetBucket(bucket).version > cursor_) {
            Refresh(buckets_[bucket], store, bucket);
        }
    }
    cursor_ = changes.version;
}

std::optional<Box> ExtentCache::Find(const PrimStore& store, BucketId bucket,
                                     std::uint32_t row) const {
    if (bucket >= buckets_.size() || buckets_[bucket].compute == nullptr ||
        store.GetBucket(bucket).version > cursor_ || row >= buckets_[bucket].boxes.size()) {
        return std::nullopt;
    }
    return buckets_[bucket].boxes[row];
}

void ExtentCache::Replay(Entry& entry, const StructuralChange& change) {
    using Kind = StructuralChange::Kind;
    if (change.kind == Kind::Added) {
        Resize(entry, change.row + change.count);
        entry.stale.SetRange(change.row, change.row + change.count);
    } else if (change.kind == Kind::Moved) {
        // Recomputed, not copied: the moved row's own value edits are reported at its old place.
        entry.stale.Set(change.row);
        Resize(entry, change.from);
    } else if (change.row + 1 == entry.boxes.size()) {
        // Removing the last row moves nothing; otherwise the Moved that follows shrinks it.
        Resize(entry, change.row);
    }
}

void ExtentCache::Resize(Entry& entry, std::uint32_t rows) {
    entry.boxes.resize(rows);
    entry.stale.Resize(rows);
}

void ExtentCache::Refresh(Entry& entry, const PrimStore& store, BucketId bucket) {
    const std::uint32_t rows = store.GetBucket(bucket).Rows();
    if (entry.boxes.size() != rows) {
        // Too far behind to replay (a reset): recompute the bucket whole.
        Resize(entry, rows);
        entry.stale.SetRange(0, rows);
    }
    const ExtentRows columns{store, bucket};
    entry.stale.ForEachRun([&](std::size_t begin, std::size_t end) {
        entry.compute(columns, {static_cast<std::uint32_t>(begin), static_cast<std::uint32_t>(end)},
                      std::span(entry.boxes).subspan(begin, end - begin));
    });
    entry.stale.Clear();
}

}  // namespace ernie
