// SPDX-License-Identifier: BSD-3-Clause
/// @file geometry.cpp
/// @brief Following the scene's gprims: meshes uploaded as triangles, implicit shapes placed on
/// shared unit shapes.
#include "ernie/render/geometry.h"

#include "ernie/geometry/triangulation.h"
#include "ernie/scene/convert.h"
#include "ernie/scene/schema/geom.h"
#include "ernie/scene/schema/mesh.h"
#include "ernie/scene/schema/planet.h"

#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <stdexcept>
#include <string>

namespace ernie {
namespace {

/// Metres from a ground plane's centre to its edges: far past any horizon a camera near it sees.
constexpr double kGroundHalfWidth = 1e5;

std::optional<GeometryKind> KindHeld(const Scene& scene, PrimId prim) {
    const SchemaDef* schema = scene.Schema(prim);
    const std::array<std::pair<const SchemaDef*, GeometryKind>, 8> kinds{{
        {&Mesh::Definition(), GeometryKind::Triangles},
        {&Sphere::Definition(), GeometryKind::Sphere},
        {&Cube::Definition(), GeometryKind::Cube},
        {&Cylinder::Definition(), GeometryKind::Cylinder},
        {&Cone::Definition(), GeometryKind::Cone},
        {&Capsule::Definition(), GeometryKind::Capsule},
        {&Plane::Definition(), GeometryKind::Plane},
        {&GroundPlane::Definition(), GeometryKind::Plane},
    }};
    for (const auto& [definition, kind] : kinds) {
        if (IsA(schema, *definition)) {
            return kind;
        }
    }
    return std::nullopt;
}

double Number(const Scene& scene, PrimId prim, std::string_view attribute) {
    const std::optional<Value> value = scene.GetOrFallback(prim, attribute);
    const std::optional<double> number = value ? ToNumber(*value) : std::nullopt;
    if (!number) {
        throw std::invalid_argument("'" + scene.PrimPath(prim).String() + "' has no number " +
                                    std::string(attribute));
    }
    return *number;
}

/// @return The prim's display colour, authored in double or float precision.
glm::vec3 DisplayColour(const Scene& scene, PrimId prim) {
    const std::optional<Value> value = scene.GetOrFallback(prim, Gprim::kDisplayColor);
    if (value) {
        if (const auto* colour = value->Get<glm::dvec3>()) {
            return glm::vec3(*colour);
        }
        if (const auto* colour = value->Get<glm::vec3>()) {
            return *colour;
        }
    }
    return glm::vec3(0.5f);
}

/// @return The turn taking a unit shape's +z to the prim's axis, cyclically, keeping handedness.
glm::dmat4 AxisTurn(const Scene& scene, PrimId prim) {
    const std::optional<Value> axis = scene.GetOrFallback(prim, Cylinder::kAxis);
    const auto* name = axis ? axis->Get<std::string>() : nullptr;
    const char letter = name != nullptr && !name->empty() ? name->front() : 'Z';
    glm::dmat4 turn(0.0);
    turn[3][3] = 1.0;
    // Columns are the images of the unit x, y and z axes.
    const int z = letter == 'X' ? 0 : letter == 'Y' ? 1 : 2;
    turn[0][(z + 1) % 3] = 1.0;
    turn[1][(z + 2) % 3] = 1.0;
    turn[2][z] = 1.0;
    return turn;
}

/// @return The transform from a kind's unit shape to the prim's shape.
glm::dmat4 ShapeTransform(const Scene& scene, PrimId prim, GeometryKind kind) {
    const auto scaled = [](const glm::dvec3& halves) {
        return glm::scale(glm::dmat4(1.0), halves);
    };
    switch (kind) {
    case GeometryKind::Sphere:
        return scaled(glm::dvec3(Number(scene, prim, Sphere::kRadius)));
    case GeometryKind::Cube:
        return scaled(glm::dvec3(0.5 * Number(scene, prim, Cube::kSize)));
    case GeometryKind::Cylinder:
    case GeometryKind::Cone: {
        const double radius = Number(scene, prim, Cylinder::kRadius);
        return AxisTurn(scene, prim) *
               scaled({radius, radius, 0.5 * Number(scene, prim, Cylinder::kHeight)});
    }
    case GeometryKind::Capsule:
        return AxisTurn(scene, prim) * scaled(glm::dvec3(Number(scene, prim, Capsule::kRadius)));
    case GeometryKind::Plane: {
        if (IsA(scene.Schema(prim), GroundPlane::Definition())) {
            return scaled({kGroundHalfWidth, kGroundHalfWidth, 1.0});
        }
        // Width along x, or z for an X normal; length along y, or z for a Y normal: under the
        // cyclic turn, the unit square's x carries the length unless the normal is z.
        const double width = 0.5 * Number(scene, prim, Plane::kWidth);
        const double length = 0.5 * Number(scene, prim, Plane::kLength);
        const glm::dmat4 turn = AxisTurn(scene, prim);
        const bool normalZ = turn[2][2] == 1.0;
        return turn *
               scaled(normalZ ? glm::dvec3(width, length, 1.0) : glm::dvec3(length, width, 1.0));
    }
    case GeometryKind::Triangles:
        break;
    }
    return glm::dmat4(1.0);
}

/// @return Whether a changed attribute reshapes a gprim, moves a mesh's points, or remakes its
/// topology.
struct Touched {
    bool shape = false;
    bool points = false;
    bool topology = false;
    bool colour = false;

    bool Any() const {
        return shape || points || topology || colour;
    }
};

Touched Classify(std::string_view attribute) {
    if (attribute == Gprim::kDisplayColor) {
        return {.colour = true};
    }
    if (attribute == Mesh::kPoints) {
        return {.points = true};
    }
    if (attribute == Mesh::kFaceVertexCounts || attribute == Mesh::kFaceVertexIndices) {
        return {.topology = true};
    }
    for (const std::string_view name : {Sphere::kRadius, Cube::kSize, Cylinder::kHeight,
                                        Cylinder::kAxis, Plane::kWidth, Plane::kLength}) {
        if (attribute == name) {
            return {.shape = true};
        }
    }
    return {};
}

}  // namespace

GeometryPool::GeometryPool(DeviceContext& context, Arenas& arenas, const Scene& scene)
    : context_(context), arenas_(arenas), scene_(scene), uploader_(context, context.Main()),
      unitShapes_(static_cast<std::size_t>(GeometryKind::Plane) + 1),
      records_(std::make_unique<DeviceColumn>(
          arenas, "geometry records", DType::UInt32,
          std::vector<std::int64_t>{sizeof(GeometryRecord) / sizeof(std::uint32_t)}, 0)) {}

void GeometryPool::Touch(GeometryId geometry) {
    dirtyRecords_.push_back(geometry);
}

std::optional<PrimGeometry> GeometryPool::Find(PrimId prim) const {
    if (prim.slot >= prims_.size() || !prims_[prim.slot] || !(prims_[prim.slot]->prim == prim)) {
        return std::nullopt;
    }
    return prims_[prim.slot];
}

GeometryId GeometryPool::Allocate(const Geometry& geometry) {
    GeometryId id = 0;
    if (!unused_.empty()) {
        id = unused_.back();
        unused_.pop_back();
        geometries_[id] = geometry;
    } else {
        id = static_cast<GeometryId>(geometries_.size());
        geometries_.push_back(geometry);
    }
    Touch(id);
    return id;
}

void GeometryPool::Free(GeometryId id, Changes& changes) {
    Geometry& geometry = geometries_[id];
    for (const BufferRange* range : {&geometry.vertices, &geometry.indices, &geometry.box}) {
        if (range->size > 0) {
            arenas_.Free(*range);
        }
    }
    geometry = {};
    unused_.push_back(id);
    Touch(id);
    changes.freed.push_back(id);
}

void GeometryPool::UploadBox(const Geometry& geometry, const glm::vec3& low,
                             const glm::vec3& high) {
    const std::array<float, 6> box{low.x, low.y, low.z, high.x, high.y, high.z};
    uploader_.Write(arenas_.Storage(geometry.box), geometry.box.offset,
                    std::span<const float>(box));
}

GeometryId GeometryPool::UnitShape(GeometryKind kind) {
    std::optional<GeometryId>& shared = unitShapes_[static_cast<std::size_t>(kind)];
    if (!shared) {
        Geometry geometry{.kind = kind, .box = arenas_.Allocate(6 * sizeof(float))};
        // The unit square is flat: its box has no depth.
        UploadBox(geometry, glm::vec3(-1, -1, kind == GeometryKind::Plane ? 0 : -1), glm::vec3(1));
        shared = Allocate(geometry);
    }
    return *shared;
}

void GeometryPool::UploadMesh(PrimId prim, GeometryId id, bool topology, Changes& changes) {
    const std::string path = scene_.PrimPath(prim).String();
    const Tensor* points = scene_.Read(prim, Mesh::kPoints);
    const std::vector<float> positions =
        points != nullptr ? CopyFloats(*points, 3, path + "'s points") : std::vector<float>();
    const auto vertexCount = static_cast<std::uint32_t>(positions.size() / 3);
    Geometry& geometry = geometries_[id];
    const bool resized = vertexCount != geometry.vertexCount;
    if (topology || resized) {
        const auto indices = [&](std::string_view attribute) {
            const Tensor* tensor = scene_.Read(prim, attribute);
            return tensor != nullptr ? CopyInts(*tensor, path + "'s " + std::string(attribute))
                                     : std::vector<std::int64_t>();
        };
        const std::vector<std::uint32_t> triangles = TriangulateFaces(
            indices(Mesh::kFaceVertexCounts), indices(Mesh::kFaceVertexIndices), vertexCount);
        for (BufferRange* range : {&geometry.vertices, &geometry.indices}) {
            if (range->size > 0) {
                arenas_.Free(*range);
                *range = {};
            }
        }
        geometry.vertexCount = vertexCount;
        geometry.triangleCount = static_cast<std::uint32_t>(triangles.size() / 3);
        if (!triangles.empty()) {
            geometry.vertices = arenas_.Allocate(positions.size() * sizeof(float));
            geometry.indices = arenas_.Allocate(triangles.size() * sizeof(std::uint32_t));
            uploader_.Write(arenas_.Storage(geometry.indices), geometry.indices.offset,
                            std::span<const std::uint32_t>(triangles));
        }
        Touch(id);
        changes.rebuilt.push_back(id);
    } else {
        changes.refitted.push_back(id);
    }
    if (geometry.triangleCount > 0) {
        uploader_.Write(arenas_.Storage(geometry.vertices), geometry.vertices.offset,
                        std::span<const float>(positions));
    }
}

void GeometryPool::Place(PrimId prim, bool points, bool topology, Changes& changes) {
    const std::optional<GeometryKind> kind = KindHeld(scene_, prim);
    if (!kind) {
        return;
    }
    if (prim.slot >= prims_.size()) {
        prims_.resize(std::max<std::size_t>(prim.slot + 1, 2 * prims_.size()));
    }
    std::optional<PrimGeometry>& placed = prims_[prim.slot];
    const bool known = placed && placed->prim == prim;
    PrimGeometry next{.prim = prim};
    if (*kind == GeometryKind::Triangles) {
        next.geometry = known ? placed->geometry : Allocate({.kind = GeometryKind::Triangles});
        if (!known || points || topology) {
            UploadMesh(prim, next.geometry, topology || !known, changes);
        }
    } else if (*kind == GeometryKind::Capsule) {
        const double radius = Number(scene_, prim, Capsule::kRadius);
        const auto halfLength =
            static_cast<float>(0.5 * Number(scene_, prim, Capsule::kHeight) / radius);
        next.geometry = known ? placed->geometry
                              : Allocate({.kind = GeometryKind::Capsule,
                                          .box = arenas_.Allocate(6 * sizeof(float))});
        Geometry& geometry = geometries_[next.geometry];
        if (!known || geometry.halfLength != halfLength) {
            geometry.halfLength = halfLength;
            UploadBox(geometry, glm::vec3(-1, -1, -1 - halfLength),
                      glm::vec3(1, 1, 1 + halfLength));
            Touch(next.geometry);
            changes.rebuilt.push_back(next.geometry);
        }
    } else {
        next.geometry = UnitShape(*kind);
        if (!known) {
            changes.rebuilt.push_back(next.geometry);
        }
    }
    next.shape = ShapeTransform(scene_, prim, *kind);
    next.colour = DisplayColour(scene_, prim);
    if (!known || placed->geometry != next.geometry || placed->shape != next.shape ||
        placed->colour != next.colour) {
        changes.placed.push_back(next);
    }
    placed = next;
}

GeometryPool::Changes GeometryPool::Update() {
    ERNIE_ZONE("Update geometry");
    const PrimStore& store = scene_.Store();
    const ChangeSet changeSet = store.ChangesSince(cursor_);
    cursor_ = changeSet.version;
    const PrimRows::Changes followed = rows_.Follow(store, changeSet);
    Changes changes;
    for (const PrimId prim : followed.removed) {
        if (const std::optional<PrimGeometry> placed = Find(prim)) {
            const Geometry& geometry = geometries_[placed->geometry];
            if (geometry.kind == GeometryKind::Triangles ||
                geometry.kind == GeometryKind::Capsule) {
                Free(placed->geometry, changes);
            }
            prims_[prim.slot].reset();
            changes.removed.push_back(prim);
        }
    }
    for (const PrimId prim : followed.added) {
        Place(prim, true, true, changes);
    }
    // Prims just added were placed from all their values: none of them is an edit.
    std::vector<bool> fresh(prims_.size(), false);
    for (const PrimId prim : followed.added) {
        if (prim.slot < fresh.size()) {
            fresh[prim.slot] = true;
        }
    }
    // Values the host authored: shapes, points, topology and colours.
    for (const ChangeSet::Values& values : changeSet.values) {
        const Touched touched = Classify(store.Tokens().Text(values.attribute));
        if (!touched.Any()) {
            continue;
        }
        const Bucket& bucket = store.GetBucket(values.bucket);
        for (const RowRange& range : values.rows) {
            for (std::uint32_t row = range.begin; row < std::min(range.end, bucket.Rows()); ++row) {
                const PrimId prim = bucket.prims[row];
                if (Find(prim) && !fresh[prim.slot]) {
                    Place(prim, touched.points, touched.topology, changes);
                }
            }
        }
    }
    return changes;
}

void GeometryPool::AddUploads(graph::Graph& graph) {
    if (!dirtyRecords_.empty()) {
        hostRecords_.resize(geometries_.size());
        records_->Resize(geometries_.size(), context_.Main());
        std::ranges::sort(dirtyRecords_);
        dirtyRecords_.erase(std::ranges::unique(dirtyRecords_).begin(), dirtyRecords_.end());
        const auto address = [&](const BufferRange& range) {
            return range.size > 0 ? arenas_.Address(range) : std::uint64_t{0};
        };
        for (const std::uint32_t id : dirtyRecords_) {
            const Geometry& geometry = geometries_[id];
            hostRecords_[id] = {.kind = static_cast<std::uint32_t>(geometry.kind),
                                .halfLength = geometry.halfLength,
                                .vertexCount = geometry.vertexCount,
                                .triangleCount = geometry.triangleCount,
                                .vertices = address(geometry.vertices),
                                .indices = address(geometry.indices)};
        }
        uploader_.WriteRows(records_->Storage(), records_->Range().offset, sizeof(GeometryRecord),
                            std::as_bytes(std::span(hostRecords_)), dirtyRecords_);
        dirtyRecords_.clear();
    }
    if (!uploader_.Staged()) {
        return;
    }
    std::vector<graph::Buffer> buffers = Import(graph);
    graph.Add(
        "Upload geometry", graph::Kind::Copy,
        [&](graph::PassBuilder& pass) {
            for (graph::Buffer& buffer : buffers) {
                buffer = pass.Write(buffer, graph::Use::CopyDst);
            }
        },
        [this](graph::PassContext& pass) {
            for (std::shared_ptr<const Buffer>& staging : uploader_.Record(pass.commands)) {
                pass.Keep(std::move(staging));
            }
        });
}

std::vector<graph::Buffer> GeometryPool::Import(graph::Graph& graph) const {
    std::vector<graph::Buffer> buffers;
    for (std::uint32_t arena = 0; arena < arenas_.Count(); ++arena) {
        const Buffer& buffer = arenas_.Storage(BufferRange{.arena = arena});
        buffers.push_back(graph.Import(buffer, 0, buffer.Size(), "geometry"));
    }
    return buffers;
}

}  // namespace ernie
