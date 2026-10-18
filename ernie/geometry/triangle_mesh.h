// SPDX-License-Identifier: BSD-3-Clause
/// @file triangle_mesh.h
/// @brief Triangle meshes: UsdGeom's implicit primitives tessellated, and polygons triangulated.
///
/// Primitives are centred on the origin with outward normals and counter-clockwise triangles.
/// Curved surfaces share vertices; faces meeting at an edge have their own. Shapes are built
/// around +z and turned cyclically to `axis`, keeping handedness. Builders throw
/// std::invalid_argument for sizes not positive and finite (a capsule's height may be 0), or under
/// 3 slices or 2 stacks.
///
/// References:
/// - Pixar, "UsdGeom: USD Geometry Schema" (Cube, Sphere, Cylinder, Cone, Capsule and Plane).
///   https://openusd.org/release/api/usd_geom_page_front.html
#pragma once

#include <glm/glm.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace ernie {

/// @brief Triangles in a mesh's own frame.
struct TriangleMesh {
    std::vector<glm::vec3> points;
    /// One a point; empty when the mesh has none.
    std::vector<glm::vec3> normals;
    /// Three point indices a triangle.
    std::vector<std::uint32_t> indices;
};

/// @brief The axis a primitive is built around (UsdGeom's `axis`).
enum class Axis : std::uint8_t { X, Y, Z };

/// @return The axis "X", "Y" or "Z" names; none for anything else.
std::optional<Axis> ParseAxis(std::string_view name);

/// @brief A box; UsdGeomCube is one of equal edges, 2 by default.
TriangleMesh BoxMesh(const glm::vec3& size);
/// @brief A UV sphere (UsdGeomSphere, radius 1 by default).
/// @param slices Divisions around.
/// @param stacks Divisions from pole to pole.
TriangleMesh SphereMesh(float radius, std::uint32_t slices = 48, std::uint32_t stacks = 24);
/// @brief A capped cylinder (UsdGeomCylinder, radius 1 and height 2 by default).
TriangleMesh CylinderMesh(float radius, float height, Axis axis = Axis::Z,
                          std::uint32_t slices = 48);
/// @brief A capped cone, its apex at +height / 2 (UsdGeomCone, radius 1 and height 2 by default).
TriangleMesh ConeMesh(float radius, float height, Axis axis = Axis::Z, std::uint32_t slices = 48);
/// @brief A capsule (UsdGeomCapsule, radius 0.5 and height 1 by default).
/// @param height The cylinder's length; the capsule's is height + 2 radius.
/// @param stacks Sphere divisions; each cap gets half.
TriangleMesh CapsuleMesh(float radius, float height, Axis axis = Axis::Z, std::uint32_t slices = 48,
                         std::uint32_t stacks = 24);
/// @brief A one-sided rectangle facing +`axis` (UsdGeomPlane, 2 by 2 by default).
/// @param width  Along x (along z for X).
/// @param length Along y (along z for Y).
TriangleMesh PlaneMesh(float width, float length, Axis axis = Axis::Z);

/// @brief Triangulates USD polygons into fans.
/// @param faceVertexCounts  Vertices a face.
/// @param faceVertexIndices Point indices, face by face.
/// @param points            How many points the indices may name.
/// @return Three point indices a triangle.
/// @throws std::invalid_argument for a face of under 3 vertices, counts that do not add up to the
/// indices, or an index out of range.
std::vector<std::uint32_t> TriangulateFaces(std::span<const std::int64_t> faceVertexCounts,
                                            std::span<const std::int64_t> faceVertexIndices,
                                            std::size_t points);

}  // namespace ernie
