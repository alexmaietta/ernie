// SPDX-License-Identifier: BSD-3-Clause
/// @file holder_index.cpp
/// @brief Re-indexing changed relationship cells, and finding a target's live holders.
#include "ernie/scene/holder_index.h"

#include <algorithm>
#include <unordered_set>

namespace ernie {
namespace {

bool IsRelationship(ColumnType type) {
    return type == ColumnType::Relationship || type == ColumnType::Relationships;
}

/// @return The targets of a relationship cell, without the unset one.
std::vector<PathId> Targets(const PrimStore& store, BucketId bucket, const Column& column,
                            std::uint32_t row) {
    std::vector<PathId> targets;
    if (column.values->Type() == ColumnType::Relationship) {
        targets.push_back(store.Read<Target>(bucket, column.attribute)[row].path);
    } else {
        for (const Target target : store.Read<std::vector<Target>>(bucket, column.attribute)[row]) {
            targets.push_back(target.path);
        }
    }
    std::erase(targets, kNoPath);
    return targets;
}

}  // namespace

void HolderIndex::Update(const PrimStore& store) {
    const ChangeSet changes = store.ChangesSince(cursor_);
    for (const ChangeSet::Values& values : changes.values) {
        const Bucket& bucket = store.GetBucket(values.bucket);
        const Column& column = *bucket.Find(values.attribute);
        if (!IsRelationship(column.values->Type())) {
            continue;
        }
        for (const RowRange range : values.rows) {
            for (std::uint32_t row = range.begin; row < std::min(range.end, bucket.Rows()); ++row) {
                Index(bucket.prims[row], values.attribute,
                      Targets(store, values.bucket, column, row));
            }
        }
    }
    // A bucket too far behind to replay reports every row of every column, as a fresh one does.
    for (const BucketId id : changes.resets) {
        const Bucket& bucket = store.GetBucket(id);
        for (const Column& column : bucket.columns) {
            for (std::uint32_t row = 0;
                 IsRelationship(column.values->Type()) && row < bucket.Rows(); ++row) {
                Index(bucket.prims[row], column.attribute, Targets(store, id, column, row));
            }
        }
    }
    cursor_ = changes.version;
}

void HolderIndex::Index(PrimId prim, AttributeId attribute, std::vector<PathId> targets) {
    const Cell cell = Key(prim, attribute);
    const auto found = targets_.find(cell);
    if (found != targets_.end()) {
        for (const PathId old : found->second) {
            holders_[old].erase(cell);
        }
    }
    for (const PathId target : targets) {
        holders_[target][cell] = prim.generation;
    }
    if (!targets.empty()) {
        targets_[cell] = std::move(targets);
    } else if (found != targets_.end()) {
        targets_.erase(found);
    }
}

std::vector<PrimId> HolderIndex::Find(const PrimStore& store, PathId target) const {
    std::vector<PrimId> found;
    const auto holders = holders_.find(target);
    if (holders == holders_.end()) {
        return found;
    }
    std::unordered_set<std::uint32_t> slots;
    for (const auto& [cell, generation] : holders->second) {
        const PrimId prim{static_cast<std::uint32_t>(cell >> 32), generation};
        // Cells of removed prims, or of prims since moved to a bucket without the attribute, stay
        // until their slot's next prim writes them.
        if (Holds(store, prim, static_cast<AttributeId>(cell), target) &&
            slots.insert(prim.slot).second) {
            found.push_back(prim);
        }
    }
    return found;
}

bool HolderIndex::Holds(const PrimStore& store, PrimId prim, AttributeId attribute, PathId target) {
    if (!store.Alive(prim)) {
        return false;
    }
    const BucketId bucket = store.PrimBucket(prim);
    const Column* column = store.GetBucket(bucket).Find(attribute);
    if (column == nullptr || !IsRelationship(column->values->Type())) {
        return false;
    }
    const std::vector<PathId> targets = Targets(store, bucket, *column, store.PrimRow(prim));
    return std::ranges::find(targets, target) != targets.end();
}

}  // namespace ernie
