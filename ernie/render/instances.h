// SPDX-License-Identifier: BSD-3-Clause
/// @file instances.h
/// @brief The device scene's instances: a row for each prim drawn, naming its transform node,
/// its geometry and its shape transform, kept dense on the device for kernels to read.
#pragma once

#include "ernie/device/column.h"
#include "ernie/device/staging.h"
#include "ernie/graph/graph.h"
#include "ernie/render/geometry.h"
#include "ernie/util/glm.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ernie {

/// @brief A row a drawn prim, in device columns: `Nodes` its transform graph slot, `Geometries` its
/// geometry, `Shapes` the 3x4 row-major transform from that geometry to the prim, and `Colours` its
/// linear RGB albedo. Rows are kept dense by swap-removal, so a row is no stable id; a prim's row
/// is found with `Row`.
///
/// Each row is in a batch world, or in every world. The top levels' records, a world after
/// another, are laid out by `Sources`: each world's records are the shared rows, then its own.
/// The layout changes only as rows come, go or change world.
class InstanceTable {
  public:
    using Shape = std::array<glm::vec4, 3>;
    /// A world's records: the first, and how many.
    struct WorldRecords {
        std::uint32_t first = 0;
        std::uint32_t count = 0;
    };

    /// @param arenas Where the columns live.
    InstanceTable(DeviceContext& context, Arenas& arenas);
    InstanceTable(const InstanceTable&) = delete;
    InstanceTable& operator=(const InstanceTable&) = delete;

    /// @brief Takes in what a geometry update placed and removed.
    void Apply(const GeometryPool::Changes& changes);
    /// @brief Adds the pass that uploads changed rows, before the passes that read the columns.
    void AddUploads(graph::Graph& graph);

    std::uint32_t Count() const {
        return static_cast<std::uint32_t>(prims_.size());
    }
    /// @return The prim's row, or none when it draws nothing.
    std::optional<std::uint32_t> Row(PrimId prim) const;
    const DeviceColumn& Nodes() const {
        return *nodes_;
    }
    const DeviceColumn& Geometries() const {
        return *geometries_;
    }
    const DeviceColumn& Shapes() const {
        return *shapes_;
    }
    const DeviceColumn& Colours() const {
        return *colours_;
    }
    /// @brief Keeps at least `count` worlds, for views of worlds with no prims of their own.
    void ReserveWorlds(std::uint32_t count) {
        relayout_ = relayout_ || count > reserved_;
        reserved_ = std::max(reserved_, count);
    }
    /// @return Each world's records: as many worlds as the highest world authored plus one, or as
    /// reserved, and at least one.
    std::span<const WorldRecords> Worlds() const {
        return worlds_;
    }
    /// @return Each top-level record's instance row, a uint32 a record, world after world.
    const DeviceColumn& Sources() const {
        return *sources_;
    }
    /// @return The records of every world.
    std::uint32_t Records() const {
        return static_cast<std::uint32_t>(hostSources_.size());
    }

  private:
    /// Lays the records out again, world by world.
    void LayOutWorlds();
    void Set(std::uint32_t row, const PrimGeometry& placed);
    void Remove(PrimId prim);
    /// Calls `f` with each host array kept a row an instance, so they grow and shrink together.
    template <typename F> void ForEachHostColumn(F&& f) {
        f(prims_);
        f(hostNodes_);
        f(hostGeometries_);
        f(hostShapes_);
        f(hostColours_);
        f(hostWorlds_);
    }

    DeviceContext& context_;
    Uploader uploader_;
    // Host copies of the rows, by row.
    std::vector<PrimId> prims_;
    std::vector<std::uint32_t> hostNodes_;
    std::vector<std::uint32_t> hostGeometries_;
    std::vector<Shape> hostShapes_;
    std::vector<glm::vec3> hostColours_;
    std::vector<std::int32_t> hostWorlds_;
    /// Each prim's row by its id's slot.
    std::vector<std::uint32_t> rows_;
    std::vector<std::uint32_t> dirty_;
    /// Whether rows came, went, moved or changed world since the layout.
    bool relayout_ = false;
    std::uint32_t reserved_ = 1;
    std::vector<WorldRecords> worlds_{1};
    std::vector<std::uint32_t> hostSources_;
    std::unique_ptr<DeviceColumn> sources_;
    std::unique_ptr<DeviceColumn> nodes_;
    std::unique_ptr<DeviceColumn> geometries_;
    std::unique_ptr<DeviceColumn> shapes_;
    std::unique_ptr<DeviceColumn> colours_;
};

}  // namespace ernie
