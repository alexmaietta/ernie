// SPDX-License-Identifier: BSD-3-Clause
/// @file prim_rows.h
/// @brief Each bucket's prims by row as a reader last saw them, kept in step by replaying the
/// buckets' row logs, so the device scene's managers learn which prims came, went and moved
/// without scanning the scene.
#pragma once

#include "ernie/scene/storage.h"

#include <span>
#include <vector>

namespace ernie {

class PrimRows {
  public:
    /// @brief The prims a change set brought, took and moved to other rows.
    struct Changes {
        std::vector<PrimId> removed;
        /// Sorted by path, so parents come before their children.
        std::vector<PrimId> added;
        std::vector<PrimId> moved;
    };

    /// @brief Replays the change set's row logs onto the rows last seen.
    Changes Follow(const PrimStore& store, const ChangeSet& changes);

    /// @return A bucket's prims by row as last followed; empty for a bucket not seen yet.
    std::span<const PrimId> Bucket(BucketId bucket) const {
        return bucket < rows_.size() ? std::span<const PrimId>(rows_[bucket])
                                     : std::span<const PrimId>();
    }

  private:
    std::vector<std::vector<PrimId>> rows_;
};

}  // namespace ernie
