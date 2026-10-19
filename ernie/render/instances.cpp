// SPDX-License-Identifier: BSD-3-Clause
/// @file instances.cpp
/// @brief Keeping instance rows dense as prims come and go, and uploading the rows that changed.
#include "ernie/render/instances.h"

#include "ernie/util/profiler.h"

#include <algorithm>
#include <limits>

namespace ernie {
namespace {

constexpr std::uint32_t kNoRow = std::numeric_limits<std::uint32_t>::max();

}  // namespace

InstanceTable::InstanceTable(DeviceContext& context, Arenas& arenas)
    : context_(context), uploader_(context, context.Main()) {
    const auto column = [&](std::string_view name, DType type, std::vector<std::int64_t> element) {
        return std::make_unique<DeviceColumn>(arenas, name, type, std::move(element), 0);
    };
    nodes_ = column("instance nodes", DType::UInt32, {});
    geometries_ = column("instance geometries", DType::UInt32, {});
    shapes_ = column("instance shapes", DType::Float32, {3, 4});
    colours_ = column("instance colours", DType::Float32, {3});
    sources_ = column("top-level sources", DType::UInt32, {});
}

std::optional<std::uint32_t> InstanceTable::Row(PrimId prim) const {
    if (prim.slot >= rows_.size() || rows_[prim.slot] == kNoRow ||
        !(prims_[rows_[prim.slot]] == prim)) {
        return std::nullopt;
    }
    return rows_[prim.slot];
}

void InstanceTable::Set(std::uint32_t row, const PrimGeometry& placed) {
    prims_[row] = placed.prim;
    hostNodes_[row] = placed.prim.slot;
    hostGeometries_[row] = placed.geometry;
    hostShapes_[row] = ToAffineRows(placed.shape);
    hostColours_[row] = placed.colour;
    relayout_ = relayout_ || hostWorlds_[row] != placed.world;
    hostWorlds_[row] = placed.world;
    rows_[placed.prim.slot] = row;
    dirty_.push_back(row);
}

void InstanceTable::Remove(PrimId prim) {
    const std::optional<std::uint32_t> row = Row(prim);
    if (!row) {
        return;
    }
    // The last row moves into the gap.
    const auto last = static_cast<std::uint32_t>(prims_.size() - 1);
    if (*row != last) {
        ForEachHostColumn([&](auto& column) { column[*row] = column[last]; });
        rows_[prims_[*row].slot] = *row;
        dirty_.push_back(*row);
    }
    rows_[prim.slot] = kNoRow;
    ForEachHostColumn([](auto& column) { column.pop_back(); });
    relayout_ = true;
}

void InstanceTable::LayOutWorlds() {
    std::int32_t highest = 0;
    std::vector<std::uint32_t> shared;
    for (std::uint32_t row = 0; row < hostWorlds_.size(); ++row) {
        highest = std::max(highest, hostWorlds_[row]);
        if (hostWorlds_[row] < 0) {
            shared.push_back(row);
        }
    }
    const std::size_t count =
        std::max<std::size_t>(static_cast<std::size_t>(highest) + 1, reserved_);
    std::vector<std::vector<std::uint32_t>> own(count);
    for (std::uint32_t row = 0; row < hostWorlds_.size(); ++row) {
        if (hostWorlds_[row] >= 0) {
            own[static_cast<std::size_t>(hostWorlds_[row])].push_back(row);
        }
    }
    worlds_.assign(count, {});
    hostSources_.clear();
    for (std::size_t world = 0; world < count; ++world) {
        worlds_[world] = {.first = static_cast<std::uint32_t>(hostSources_.size()),
                          .count = static_cast<std::uint32_t>(shared.size() + own[world].size())};
        hostSources_.insert(hostSources_.end(), shared.begin(), shared.end());
        hostSources_.insert(hostSources_.end(), own[world].begin(), own[world].end());
    }
}

void InstanceTable::Apply(const GeometryPool::Changes& changes) {
    ERNIE_ZONE("Update instances");
    for (const PrimId prim : changes.removed) {
        Remove(prim);
    }
    for (const PrimGeometry& placed : changes.placed) {
        if (placed.prim.slot >= rows_.size()) {
            rows_.resize(placed.prim.slot + 1, kNoRow);
        }
        const std::optional<std::uint32_t> row = Row(placed.prim);
        if (row) {
            Set(*row, placed);
            continue;
        }
        ForEachHostColumn([](auto& column) { column.emplace_back(); });
        relayout_ = true;
        Set(static_cast<std::uint32_t>(prims_.size() - 1), placed);
    }
}

void InstanceTable::AddUploads(graph::Graph& graph) {
    const std::size_t rows = prims_.size();
    for (DeviceColumn* column : {nodes_.get(), geometries_.get(), shapes_.get(), colours_.get()}) {
        column->Resize(rows, context_.Main());
    }
    StageRows(uploader_, *nodes_, hostNodes_, dirty_);
    StageRows(uploader_, *geometries_, hostGeometries_, dirty_);
    StageRows(uploader_, *shapes_, hostShapes_, dirty_);
    StageRows(uploader_, *colours_, hostColours_, dirty_);
    dirty_.clear();
    if (relayout_) {
        LayOutWorlds();
        sources_->Resize(hostSources_.size(), context_.Main());
        uploader_.Write(sources_->Storage(), sources_->Range().offset,
                        std::span<const std::uint32_t>(hostSources_));
        relayout_ = false;
    }
    std::array columns{graph.Import(*nodes_), graph.Import(*geometries_), graph.Import(*shapes_),
                       graph.Import(*colours_), graph.Import(*sources_)};
    graph::AddUploadPass(graph, "Upload instances", uploader_, columns);
}

}  // namespace ernie
