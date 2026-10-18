// SPDX-License-Identifier: BSD-3-Clause
/// @file storage.cpp
/// @brief Buckets, rows, versions and change sets of the columnar prim store.
#include "ernie/scene/storage.h"

#include "ernie/util/random.h"

#include <algorithm>

namespace ernie {

namespace {

constexpr std::uint32_t kNoSlot = std::numeric_limits<std::uint32_t>::max();
/// Row changes kept per bucket; a reader further behind resyncs the bucket.
constexpr std::size_t kLogLimit = std::size_t{1} << 16;

}  // namespace

std::size_t PrimStore::KeyHash::operator()(const BucketKey& key) const {
    std::uint64_t hash = Mix64(key.type);
    for (const Token schema : key.apiSchemas) {
        hash = HashCombine(hash, schema);
    }
    return static_cast<std::size_t>(hash);
}

BucketKey BucketKey::Normalized() const {
    BucketKey sorted = *this;
    std::ranges::sort(sorted.apiSchemas);
    const auto [last, end] = std::ranges::unique(sorted.apiSchemas);
    sorted.apiSchemas.erase(last, end);
    return sorted;
}

std::optional<BucketId> PrimStore::FindBucket(const BucketKey& key) const {
    const auto found = bucketIndex_.find(key.Normalized());
    return found != bucketIndex_.end() ? std::optional(found->second) : std::nullopt;
}

BucketId PrimStore::AddBucket(const BucketKey& key, std::span<const ColumnLayout> layout) {
    BucketKey sorted = key.Normalized();
    if (const auto found = bucketIndex_.find(sorted); found != bucketIndex_.end()) {
        return found->second;
    }
    auto bucket = std::make_unique<Bucket>();
    bucket->key = sorted;
    for (const ColumnLayout& column : layout) {
        AddColumn(*bucket, column.attribute, column.prototype->Clone());
    }
    const auto id = static_cast<BucketId>(buckets_.size());
    buckets_.push_back(std::move(bucket));
    bucketIndex_.emplace(std::move(sorted), id);
    return id;
}

Column& PrimStore::AddColumn(Bucket& bucket, AttributeId attribute,
                             std::unique_ptr<ColumnValues> values) {
    values->Grow(bucket.Rows());
    Column column;
    column.attribute = attribute;
    column.values = std::move(values);
    column.Resize(bucket.Rows());
    bucket.columnIndex.emplace(attribute, static_cast<std::uint32_t>(bucket.columns.size()));
    return bucket.columns.emplace_back(std::move(column));
}

const PrimStore::SlotData& PrimStore::Slot(PrimId prim) const {
    if (!Alive(prim)) {
        throw std::invalid_argument("The prim has been removed");
    }
    return slots_[prim.slot];
}

bool PrimStore::Alive(PrimId prim) const {
    return prim.slot < slots_.size() && slots_[prim.slot].alive &&
           slots_[prim.slot].generation == prim.generation;
}

std::optional<PrimId> PrimStore::Find(PathId path) const {
    if (path >= pathSlots_.size() || pathSlots_[path] == kNoSlot) {
        return std::nullopt;
    }
    const std::uint32_t slot = pathSlots_[path];
    return PrimId{slot, slots_[slot].generation};
}

PrimId PrimStore::AddSlot(PathId path, BucketId bucket, std::uint32_t row) {
    std::uint32_t slot;
    if (free_.empty()) {
        slot = static_cast<std::uint32_t>(slots_.size());
        slots_.emplace_back();
    } else {
        slot = free_.back();
        free_.pop_back();
    }
    SlotData& data = slots_[slot];
    data.bucket = bucket;
    data.row = row;
    data.path = path;
    data.alive = true;
    if (pathSlots_.size() <= path) {
        pathSlots_.resize(std::max<std::size_t>(paths_.Size(), path + 1), kNoSlot);
    }
    pathSlots_[path] = slot;
    return {slot, data.generation};
}

PrimId PrimStore::Define(PathId path, BucketId bucket) {
    if (const std::optional<PrimId> existing = Find(path)) {
        if (PrimBucket(*existing) != bucket) {
            throw std::invalid_argument("Prim " + paths_.ToPath(path).String() +
                                        " already exists with another type");
        }
        return *existing;
    }
    return DefineMany(std::span(&path, 1), bucket).front();
}

std::vector<PrimId> PrimStore::DefineMany(std::span<const PathId> paths, BucketId bucket) {
    for (const PathId path : paths) {
        if (Find(path)) {
            throw std::invalid_argument("Prim " + paths_.ToPath(path).String() + " already exists");
        }
    }
    const auto count = static_cast<std::uint32_t>(paths.size());
    const std::uint32_t first = AddRows(bucket, count);
    Bucket& b = *buckets_[bucket];
    std::vector<PrimId> prims;
    prims.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        prims.push_back(AddSlot(paths[i], bucket, first + i));
        b.prims[first + i] = prims.back();
    }
    return prims;
}

std::uint32_t PrimStore::AddRows(BucketId bucket, std::uint32_t count) {
    Bucket& b = *buckets_[bucket];
    const std::uint32_t first = b.Rows();
    for (Column& column : b.columns) {
        column.values->Grow(count);
        column.Resize(first + count);
    }
    b.prims.resize(first + count);
    Log(b, {Next(), StructuralChange::Kind::Added, first, count, 0});
    return first;
}

void PrimStore::Remove(PrimId prim) {
    SlotData& slot = const_cast<SlotData&>(Slot(prim));
    DropRow(slot);
    pathSlots_[slot.path] = kNoSlot;
    slot.alive = false;
    ++slot.generation;
    free_.push_back(prim.slot);
}

void PrimStore::Move(PrimId prim, BucketId bucket) {
    SlotData& slot = const_cast<SlotData&>(Slot(prim));
    if (slot.bucket == bucket) {
        return;
    }
    const std::uint32_t row = AddRows(bucket, 1);
    Bucket& to = *buckets_[bucket];
    const Bucket& from = *buckets_[slot.bucket];
    to.prims[row] = prim;
    const std::uint64_t version = Next();
    for (const Column& source : from.columns) {
        const auto found = to.columnIndex.find(source.attribute);
        if (found == to.columnIndex.end() || !source.authored.Test(slot.row)) {
            continue;
        }
        Column& column = to.columns[found->second];
        if (column.values->Type() == source.values->Type()) {
            column.values->CopyRow(row, *source.values, slot.row);
            Stamp(bucket, column, row, row + 1, version);
        }
    }
    DropRow(slot);
    slot.bucket = bucket;
    slot.row = row;
}

void PrimStore::DropRow(const SlotData& slot) {
    Bucket& b = *buckets_[slot.bucket];
    const std::uint32_t row = slot.row;
    const std::uint32_t last = b.Rows() - 1;
    const std::uint64_t version = Next();
    Log(b, {version, StructuralChange::Kind::Removed, row, 1, 0});
    if (row != last) {
        b.prims[row] = b.prims[last];
        slots_[b.prims[row].slot].row = row;
        Log(b, {version, StructuralChange::Kind::Moved, row, 1, last});
    }
    b.prims.pop_back();
    for (Column& column : b.columns) {
        column.values->SwapRemove(row);
        column.authored.Set(row, column.authored.Test(last));
        column.Resize(last);
    }
}

void PrimStore::Clear(PrimId prim, AttributeId attribute) {
    const SlotData& slot = Slot(prim);
    Bucket& b = *buckets_[slot.bucket];
    const auto found = b.columnIndex.find(attribute);
    if (found == b.columnIndex.end() || !b.columns[found->second].authored.Test(slot.row)) {
        return;
    }
    Column& column = b.columns[found->second];
    column.values->Reset(slot.row);
    Stamp(slot.bucket, column, slot.row, slot.row + 1, Next());
    column.authored.Set(slot.row, false);
}

void PrimStore::Log(Bucket& bucket, StructuralChange change) {
    bucket.version = change.version;
    bucket.log.push_back(change);
    if (bucket.log.size() > kLogLimit) {
        bucket.logStart = bucket.log.front().version;
        bucket.log.pop_front();
    }
}

void PrimStore::Stamp(BucketId bucket, Column& column, std::uint32_t begin, std::uint32_t end,
                      std::uint64_t version) {
    if (begin == end) {
        return;
    }
    column.authored.SetRange(begin, end);
    std::fill(column.chunkVersions.begin() + begin / kChunkRows,
              column.chunkVersions.begin() + (end - 1) / kChunkRows + 1, version);
    column.version = version;
    buckets_[bucket]->version = version;
}

ChangeSet PrimStore::ChangesSince(std::uint64_t cursor) const {
    ChangeSet changes;
    changes.version = version_;
    for (BucketId id = 0; id < buckets_.size(); ++id) {
        const Bucket& bucket = *buckets_[id];
        if (bucket.version <= cursor) {
            continue;
        }
        if (cursor < bucket.logStart) {
            changes.resets.push_back(id);
            continue;
        }
        const auto newer = std::upper_bound(
            bucket.log.begin(), bucket.log.end(), cursor,
            [](std::uint64_t at, const StructuralChange& change) { return at < change.version; });
        for (auto change = newer; change != bucket.log.end(); ++change) {
            changes.rows.push_back({id, *change});
        }
        for (const Column& column : bucket.columns) {
            if (column.version <= cursor) {
                continue;
            }
            ChangeSet::Values values{id, column.attribute, {}};
            for (std::uint32_t chunk = 0; chunk < column.chunkVersions.size(); ++chunk) {
                if (column.chunkVersions[chunk] <= cursor) {
                    continue;
                }
                const std::uint32_t begin = chunk * kChunkRows;
                const std::uint32_t end = std::min(begin + kChunkRows, bucket.Rows());
                if (!values.rows.empty() && values.rows.back().end == begin) {
                    values.rows.back().end = end;
                } else {
                    values.rows.push_back({begin, end});
                }
            }
            if (!values.rows.empty()) {
                changes.values.push_back(std::move(values));
            }
        }
    }
    return changes;
}

}  // namespace ernie
