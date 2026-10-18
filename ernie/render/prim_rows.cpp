// SPDX-License-Identifier: BSD-3-Clause
/// @file prim_rows.cpp
/// @brief Replaying bucket row logs.
#include "ernie/render/prim_rows.h"

#include "ernie/util/profiler.h"

#include <algorithm>

namespace ernie {
namespace {

/// A row added since the last replay, whose prim is read from the bucket afterwards.
bool IsPlaceholder(PrimId prim) {
    return prim.slot == PrimId().slot;
}

}  // namespace

PrimRows::Changes PrimRows::Follow(const PrimStore& store, const ChangeSet& changes) {
    ERNIE_ZONE("Follow the buckets' rows");
    rows_.resize(store.BucketCount());
    Changes result;
    std::vector<BucketId> touched;
    for (const ChangeSet::Rows& entry : changes.rows) {
        std::vector<PrimId>& rows = rows_[entry.bucket];
        const StructuralChange& change = entry.change;
        touched.push_back(entry.bucket);
        switch (change.kind) {
        case StructuralChange::Kind::Added:
            rows.resize(change.row + change.count, PrimId());
            break;
        case StructuralChange::Kind::Removed:
            if (!IsPlaceholder(rows[change.row])) {
                result.removed.push_back(rows[change.row]);
            }
            // Unless it was the last, a Moved change brings the last row here next.
            if (change.row + 1 == rows.size()) {
                rows.pop_back();
            }
            break;
        case StructuralChange::Kind::Moved:
            rows[change.row] = rows[change.from];
            rows.pop_back();
            if (!IsPlaceholder(rows[change.row])) {
                result.moved.push_back(rows[change.row]);
            }
            break;
        case StructuralChange::Kind::Reset:
            for (const PrimId prim : rows) {
                if (!IsPlaceholder(prim)) {
                    result.removed.push_back(prim);
                }
            }
            rows.assign(change.count, PrimId());
            break;
        }
    }
    std::ranges::sort(touched);
    touched.erase(std::ranges::unique(touched).begin(), touched.end());
    for (const BucketId bucket : touched) {
        std::vector<PrimId>& rows = rows_[bucket];
        const std::vector<PrimId>& now = store.GetBucket(bucket).prims;
        for (std::size_t row = 0; row < rows.size(); ++row) {
            if (IsPlaceholder(rows[row])) {
                rows[row] = now[row];
                result.added.push_back(now[row]);
            }
        }
    }
    // Parents before children: ancestors' paths are interned first.
    std::ranges::sort(result.added, {}, [&](PrimId prim) { return store.PrimPath(prim); });
    return result;
}

}  // namespace ernie
