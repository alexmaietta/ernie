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

InstanceTable::Shape RowMajor(const glm::dmat4& m) {
    InstanceTable::Shape rows{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            rows[static_cast<std::size_t>(r * 4 + c)] = static_cast<float>(m[c][r]);
        }
    }
    return rows;
}

}  // namespace

InstanceTable::InstanceTable(DeviceContext& context, Arenas& arenas)
    : context_(context), uploader_(context, context.Main()) {
    const auto column = [&](std::string_view name, DType type, std::vector<std::int64_t> element) {
        return std::make_unique<DeviceColumn>(arenas, name, type, std::move(element), 0);
    };
    nodes_ = column("instance nodes", DType::UInt32, {});
    geometries_ = column("instance geometries", DType::UInt32, {});
    shapes_ = column("instance shapes", DType::Float32, {3, 4});
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
    hostShapes_[row] = RowMajor(placed.shape);
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
        prims_[*row] = prims_[last];
        hostNodes_[*row] = hostNodes_[last];
        hostGeometries_[*row] = hostGeometries_[last];
        hostShapes_[*row] = hostShapes_[last];
        rows_[prims_[*row].slot] = *row;
        dirty_.push_back(*row);
    }
    rows_[prim.slot] = kNoRow;
    prims_.pop_back();
    hostNodes_.pop_back();
    hostGeometries_.pop_back();
    hostShapes_.pop_back();
}

void InstanceTable::Apply(const GeometryPool::Changes& changes) {
    ERNIE_ZONE("Update instances");
    for (const PrimId prim : changes.removed) {
        Remove(prim);
    }
    for (const PrimGeometry& placed : changes.placed) {
        if (placed.prim.slot >= rows_.size()) {
            rows_.resize(std::max<std::size_t>(placed.prim.slot + 1, 2 * rows_.size()), kNoRow);
        }
        const std::optional<std::uint32_t> row = Row(placed.prim);
        if (row) {
            Set(*row, placed);
            continue;
        }
        prims_.emplace_back();
        hostNodes_.emplace_back();
        hostGeometries_.emplace_back();
        hostShapes_.emplace_back();
        Set(static_cast<std::uint32_t>(prims_.size() - 1), placed);
    }
}

void InstanceTable::AddUploads(graph::Graph& graph) {
    const std::size_t rows = prims_.size();
    for (DeviceColumn* column : {nodes_.get(), geometries_.get(), shapes_.get()}) {
        column->Resize(rows, context_.Main());
    }
    std::ranges::sort(dirty_);
    dirty_.erase(std::ranges::unique(dirty_).begin(), dirty_.end());
    std::erase_if(dirty_, [&](std::uint32_t row) { return row >= rows; });
    if (dirty_.empty()) {
        return;
    }
    const auto upload = [&](const DeviceColumn& column, const auto& host) {
        uploader_.WriteRows(column.Storage(), column.Range().offset, column.RowBytes(),
                            std::as_bytes(std::span(host)), dirty_);
    };
    upload(*nodes_, hostNodes_);
    upload(*geometries_, hostGeometries_);
    upload(*shapes_, hostShapes_);
    dirty_.clear();
    std::array columns{graph.Import(*nodes_, nodes_->Name()),
                       graph.Import(*geometries_, geometries_->Name()),
                       graph.Import(*shapes_, shapes_->Name())};
    graph.Add(
        "Upload instances", graph::Kind::Copy,
        [&](graph::PassBuilder& pass) {
            for (graph::Buffer& column : columns) {
                column = pass.Write(column, graph::Use::CopyDst);
            }
        },
        [this](graph::PassContext& pass) {
            for (std::shared_ptr<const Buffer>& staging : uploader_.Record(pass.commands)) {
                pass.Keep(std::move(staging));
            }
        });
}

}  // namespace ernie
