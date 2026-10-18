// SPDX-License-Identifier: BSD-3-Clause
/// @file instances.h
/// @brief The device scene's instances: a row for each prim drawn, naming its transform node,
/// its geometry and its shape transform, kept dense on the device for kernels to read.
#pragma once

#include "ernie/device/column.h"
#include "ernie/device/staging.h"
#include "ernie/graph/graph.h"
#include "ernie/render/geometry.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace ernie {

/// @brief A row a drawn prim, in device columns: `Nodes` its transform graph slot, `Geometries` its
/// geometry, `Shapes` the 3x4 row-major transform from that geometry to the prim, and `Colours` its
/// linear RGB albedo. Rows are kept dense by swap-removal, so a row is no stable id; a prim's row
/// is found with `Row`.
class InstanceTable {
  public:
    using Shape = std::array<float, 12>;

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

  private:
    void Set(std::uint32_t row, const PrimGeometry& placed);
    void Remove(PrimId prim);

    DeviceContext& context_;
    Uploader uploader_;
    // Host copies of the rows, by row.
    std::vector<PrimId> prims_;
    std::vector<std::uint32_t> hostNodes_;
    std::vector<std::uint32_t> hostGeometries_;
    std::vector<Shape> hostShapes_;
    std::vector<glm::vec3> hostColours_;
    /// Each prim's row by its id's slot.
    std::vector<std::uint32_t> rows_;
    std::vector<std::uint32_t> dirty_;
    std::unique_ptr<DeviceColumn> nodes_;
    std::unique_ptr<DeviceColumn> geometries_;
    std::unique_ptr<DeviceColumn> shapes_;
    std::unique_ptr<DeviceColumn> colours_;
};

}  // namespace ernie
