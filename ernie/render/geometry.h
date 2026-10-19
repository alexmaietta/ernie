// SPDX-License-Identifier: BSD-3-Clause
/// @file geometry.h
/// @brief The geometry the device scene traces: meshes as triangles, and UsdGeom's implicit
/// shapes as unit shapes in procedural boxes, intersected exactly at any size.
#pragma once

#include "ernie/device/arena.h"
#include "ernie/device/column.h"
#include "ernie/device/staging.h"
#include "ernie/graph/graph.h"
#include "ernie/render/prim_rows.h"
#include "ernie/scene/scene.h"
#include "ernie/shaders/scene/geometry.slang"

#include <glm/glm.hpp>

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ernie {

/// Triangles, or the unit shapes procedural primitives stand for: a unit sphere and cube, a unit
/// cylinder and cone about +z with z in [-1, 1] (the cone's apex at +1), a capsule of radius 1
/// about a segment of half-length `Geometry::halfLength`, and a unit square in z = 0 facing +z
/// (planes, and ground planes 200 km across).
enum class GeometryKind : std::uint8_t {
    Triangles = shaders::kTriangles,
    Sphere = shaders::kSphere,
    Cube = shaders::kCube,
    Cylinder = shaders::kCylinder,
    Cone = shaders::kCone,
    Capsule = shaders::kCapsule,
    Plane = shaders::kPlane,
};

using GeometryId = std::uint32_t;

/// @brief A piece of geometry on the device, shared by every prim and world that draws it.
struct Geometry {
    GeometryKind kind = GeometryKind::Triangles;
    /// Triangles' float x, y, z positions and uint32 indices; or the one procedural box.
    BufferRange vertices;
    BufferRange indices;
    BufferRange box;
    std::uint32_t vertexCount = 0;
    std::uint32_t triangleCount = 0;
    /// A capsule's segment half-length over its radius.
    float halfLength = 0.0f;
};

/// @brief A geometry as kernels read it from `GeometryPool::Records`, as geometry.slang defines it.
using GeometryRecord = shaders::GeometryRecord;

/// @brief What a prim draws: its geometry, and the transform from the geometry's frame to the
/// prim's, which sizes and turns a unit shape.
struct PrimGeometry {
    PrimId prim;
    GeometryId geometry = 0;
    glm::dmat4 shape{1.0};
    /// Its albedo in linear RGB, from `primvars:displayColor`, until materials bind one.
    glm::vec3 colour{0.5f};
    /// The batch world it is in, from its resolved `ernie:world`; -1 for every world.
    std::int32_t world = -1;
};

/// @brief The scene's gprims as device geometry. Implicit shapes share one unit geometry a kind
/// (capsules one each, as their caps stay round), so a thousand spheres need one bottom level.
/// Updates follow the scene's changes and cost what changed: new points of the same count are
/// rewritten in place for a refit, new topology is a rebuild, and shape attributes change only
/// the prim's shape transform.
class GeometryPool {
  public:
    /// @brief What an update changed, for the bottom levels and instances that follow it.
    struct Changes {
        /// Prims drawn anew, or whose geometry, shape or colour changed.
        std::vector<PrimGeometry> placed;
        /// Prims removed, or meshes left with no triangles.
        std::vector<PrimId> removed;
        /// Geometries new or of new topology, whose bottom levels are built anew.
        std::vector<GeometryId> rebuilt;
        /// Geometries whose points moved in place, whose bottom levels may be refitted.
        std::vector<GeometryId> refitted;
        /// Geometries no prim draws any more.
        std::vector<GeometryId> freed;
    };

    /// @param arenas Where the geometry lives.
    GeometryPool(DeviceContext& context, Arenas& arenas, const Scene& scene);

    /// @brief Follows the scene's gprims since the last update, staging their data's uploads.
    /// @throws std::invalid_argument for a mesh with malformed topology, or points not on the host.
    Changes Update();
    /// @brief Adds the pass that uploads what the last update staged, before the passes that
    /// import `Buffers` to build from them.
    void AddUploads(graph::Graph& graph);

    const Geometry& Get(GeometryId geometry) const {
        return geometries_[geometry];
    }
    /// @return What the prim draws, or none for a prim that is no gprim Ernie traces.
    std::optional<PrimGeometry> Find(PrimId prim) const;
    /// @return The buffers geometry lives in, as a ranges' arena indexes them.
    const Buffer& Storage(const BufferRange& range) const {
        return arenas_.Storage(range);
    }
    /// @brief Imports every buffer geometry lives in, for passes that read it.
    std::vector<graph::Buffer> Import(graph::Graph& graph) const;
    /// @return Each geometry's `GeometryRecord`, by geometry id, in the buffers `Import` imports.
    const DeviceColumn& Records() const {
        return *records_;
    }

  private:
    /// Places a prim: its geometry, built or shared, its shape transform and its colour. A mesh's
    /// points are uploaded when it is new or `points`, its triangles when `topology`.
    void Place(PrimId prim, bool points, bool topology, Changes& changes);
    /// @return The unit shape geometry of a kind, made the first time it is asked for.
    GeometryId UnitShape(GeometryKind kind);
    /// @return A new or recycled geometry id.
    GeometryId Allocate(const Geometry& geometry);
    void Free(GeometryId geometry, Changes& changes);
    void UploadBox(const Geometry& geometry, const glm::vec3& low, const glm::vec3& high);
    /// Uploads a mesh's points, and its triangles when `topology`; a refit when only points move.
    void UploadMesh(PrimId prim, GeometryId geometry, bool topology, Changes& changes);
    /// Marks a geometry's record for the next upload.
    void Touch(GeometryId geometry);

    DeviceContext& context_;
    Arenas& arenas_;
    const Scene& scene_;
    Uploader uploader_;
    std::uint64_t cursor_ = 0;
    PrimRows rows_;
    std::vector<Geometry> geometries_;
    std::vector<GeometryId> unused_;
    /// What each prim draws, by its id's slot.
    std::vector<std::optional<PrimGeometry>> prims_;
    std::vector<std::optional<GeometryId>> unitShapes_;
    /// The records' host copies, and the geometries whose records changed.
    std::vector<GeometryRecord> hostRecords_;
    std::vector<std::uint32_t> dirtyRecords_;
    std::unique_ptr<DeviceColumn> records_;
};

}  // namespace ernie
