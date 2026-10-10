// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/base_schemas.h"
#include "ernie/scene/prim_type.h"
#include "ernie/scene/tokens.h"

#include <vector>

/// Geometry as reflected schemas (`prim_type.h`): UsdGeom's implicit shapes (Cube, Sphere, and
/// along an axis Cylinder, Cone, Capsule and Plane), its point-based Mesh and Points, the
/// PointInstancer and UsdVol's Volume, and Ernie's GroundPlane. Each declaration's fallback is the
/// value Ernie draws an unset attribute with (UsdGeom's own). Bulk arrays (points, topology,
/// instance transforms) are tensors, written with `Scene::Write`; primvars other than the display
/// colour are an open namespace (`primvars:<name>`), read by name.
///
/// References:
/// - Pixar, UsdGeom, https://openusd.org/release/api/usd_geom_page_front.html
/// - Pixar, UsdVol, https://openusd.org/release/api/usd_vol_page_front.html

// clang-format off
#define ERNIE_GEOM_SHAPE_RANGE .group = "Shape", .min = 0.0, .max = 1000.0

#define ERNIE_CUBE_ATTRIBUTES(X)                                                                   \
    X(double, Size, ::ernie::tokens::kSize, 2.0,                                                   \
      (::ernie::AttributeMeta{.doc = "Edge length, metres.", ERNIE_GEOM_SHAPE_RANGE}))

#define ERNIE_SPHERE_ATTRIBUTES(X)                                                                 \
    X(double, Radius, ::ernie::tokens::kGeomRadius, 1.0,                                           \
      (::ernie::AttributeMeta{.doc = "Metres.", ERNIE_GEOM_SHAPE_RANGE}))

#define ERNIE_AXIS_SHAPE_ATTRIBUTES(X)                                                             \
    X(std::string, Axis, ::ernie::tokens::kAxis, "Z",                                              \
      (::ernie::AttributeMeta{.doc = "The axis of its height, or a plane's normal.",               \
                              .group = "Shape", .options = {"X", "Y", "Z"}}))

// A cylinder's and a cone's; a capsule's have other fallbacks.
#define ERNIE_CYLINDER_ATTRIBUTES(X)                                                               \
    X(double, Radius, ::ernie::tokens::kGeomRadius, 1.0,                                           \
      (::ernie::AttributeMeta{.doc = "Metres.", ERNIE_GEOM_SHAPE_RANGE}))                          \
    X(double, Height, ::ernie::tokens::kGeomHeight, 2.0,                                           \
      (::ernie::AttributeMeta{.doc = "Metres along the axis.", ERNIE_GEOM_SHAPE_RANGE}))

#define ERNIE_CAPSULE_ATTRIBUTES(X)                                                                \
    X(double, Radius, ::ernie::tokens::kGeomRadius, 0.5,                                           \
      (::ernie::AttributeMeta{.doc = "Metres, of the cylinder and its end caps.",                  \
                              ERNIE_GEOM_SHAPE_RANGE}))                                            \
    X(double, Height, ::ernie::tokens::kGeomHeight, 1.0,                                           \
      (::ernie::AttributeMeta{.doc = "Metres along the axis of the cylinder between the caps.",    \
                              ERNIE_GEOM_SHAPE_RANGE}))

#define ERNIE_PLANE_ATTRIBUTES(X)                                                                  \
    X(double, Width, ::ernie::tokens::kGeomWidth, 2.0,                                             \
      (::ernie::AttributeMeta{.doc = "Metres along x, or z when the axis is X.",                   \
                              ERNIE_GEOM_SHAPE_RANGE}))                                            \
    X(double, Length, ::ernie::tokens::kGeomLength, 2.0,                                           \
      (::ernie::AttributeMeta{.doc = "Metres along y, or z when the axis is Y.",                   \
                              ERNIE_GEOM_SHAPE_RANGE}))

#define ERNIE_POINT_BASED_ATTRIBUTES(X)                                                            \
    X(::ernie::Tensor, Points, ::ernie::tokens::kPoints, nullptr,                                  \
      (::ernie::AttributeMeta{.doc = "float32 or float64 (n, 3) positions in the prim's frame.",   \
                              .group = "Geometry"}))

#define ERNIE_MESH_ATTRIBUTES(X)                                                                   \
    X(::ernie::Tensor, FaceVertexCounts, ::ernie::tokens::kFaceVertexCounts, nullptr,              \
      (::ernie::AttributeMeta{.doc = "Integer number of vertices of each face.",                   \
                              .group = "Geometry"}))                                               \
    X(::ernie::Tensor, FaceVertexIndices, ::ernie::tokens::kFaceVertexIndices, nullptr,            \
      (::ernie::AttributeMeta{.doc = "Integer point indices of the faces' vertices, in order.",    \
                              .group = "Geometry"}))                                               \
    X(::ernie::Tensor, Normals, ::ernie::tokens::kNormals, nullptr,                                \
      (::ernie::AttributeMeta{.doc = "(n, 3) normals, one per point; ignored when of another "     \
                                     "count.",                                                     \
                              .group = "Geometry"}))

#define ERNIE_POINTS_ATTRIBUTES(X)                                                                 \
    X(std::vector<float>, Widths, ::ernie::tokens::kWidths, std::vector<float>{1.0f},             \
      (::ernie::AttributeMeta{.doc = "Diameters: one for every point, or one per point. Many "     \
                                     "points' widths may be written as an array instead.",         \
                              .group = "Geometry", .min = 0.0, .max = 1000.0}))

#define ERNIE_POINT_INSTANCER_ATTRIBUTES(X)                                                        \
    X(::ernie::Tensor, Positions, ::ernie::tokens::kPositions, nullptr,                            \
      (::ernie::AttributeMeta{.doc = "(n, 3) instance positions in the instancer's frame; the "    \
                                     "number of instances.",                                       \
                              .group = "Instances"}))                                              \
    X(::ernie::Tensor, Orientations, ::ernie::tokens::kOrientations, nullptr,                      \
      (::ernie::AttributeMeta{.doc = "(n, 4) instance rotations as quaternions.",                  \
                              .group = "Instances"}))                                              \
    X(::ernie::Tensor, Scales, ::ernie::tokens::kScales, nullptr,                                  \
      (::ernie::AttributeMeta{.doc = "(n, 3) instance scales.", .group = "Instances"}))            \
    X(::ernie::Tensor, ProtoIndices, ::ernie::tokens::kProtoIndices, nullptr,                      \
      (::ernie::AttributeMeta{.doc = "Integer prototype of each instance, of the prototypes "      \
                                     "relationship's targets; the first for all when unset.",      \
                              .group = "Instances"}))                                              \
    X(std::vector<::ernie::Path>, Prototypes, ::ernie::tokens::kPrototypes,                        \
      std::vector<::ernie::Path>{},                                                                \
      (::ernie::AttributeMeta{.doc = "The prims instanced, indexed by the proto indices.",         \
                              .group = "Instances"}))                                              \
    X(::ernie::Tensor, Worlds, ::ernie::tokens::kWorlds, nullptr,                                  \
      (::ernie::AttributeMeta{.doc = "int (n,): each instance's world, in place of the "           \
                                     "instancer's world.",                                         \
                              .group = "Instances"}))

#define ERNIE_GROUND_PLANE_ATTRIBUTES(X)                                                           \
    X(double, Height, ::ernie::tokens::kGroundHeight, 0.0,                                         \
      (::ernie::AttributeMeta{.doc = "World z", .group = "Ground", .min = -1e4, .max = 1e4}))      \
    X(bool, Checker, ::ernie::tokens::kChecker, false,                                             \
      (::ernie::AttributeMeta{.label = "Checkerboard",                                             \
                              .doc = "Multiplies the material's albedo by checker squares.",       \
                              .group = "Ground"}))                                                 \
    X(double, CheckerSize, ::ernie::tokens::kCheckerSize, 1.0,                                     \
      (::ernie::AttributeMeta{.doc = "Edge of a square, metres; positive.", .group = "Ground",     \
                              .min = 0.001, .max = 1000.0}))                                       \
    X(glm::dvec3, CheckerColor0, ::ernie::tokens::kCheckerColor0, glm::dvec3(1.0),                 \
      (::ernie::AttributeMeta{.label = "Checker colour 0", .doc = "Linear RGB.",                   \
                              .group = "Ground", .color = true}))                                  \
    X(glm::dvec3, CheckerColor1, ::ernie::tokens::kCheckerColor1, glm::dvec3(0.5),                 \
      (::ernie::AttributeMeta{.label = "Checker colour 1", .doc = "Linear RGB.",                   \
                              .group = "Ground", .color = true}))                                  \
    X(::ernie::Path, MaterialBinding, ::ernie::tokens::kMaterialBinding, ::ernie::Path(),          \
      (::ernie::AttributeMeta{.label = "Material", .doc = "The Material prim it is drawn with.",   \
                              .group = "Ground", .options = {::ernie::tokens::kMaterial}}))

// Every attribute of each type, its bases' first, for its accessors.
#define ERNIE_CUBE_ALL(X) ERNIE_GPRIM_ALL(X) ERNIE_CUBE_ATTRIBUTES(X)
#define ERNIE_SPHERE_ALL(X) ERNIE_GPRIM_ALL(X) ERNIE_SPHERE_ATTRIBUTES(X)
#define ERNIE_AXIS_SHAPE_ALL(X) ERNIE_GPRIM_ALL(X) ERNIE_AXIS_SHAPE_ATTRIBUTES(X)
#define ERNIE_CYLINDER_ALL(X) ERNIE_AXIS_SHAPE_ALL(X) ERNIE_CYLINDER_ATTRIBUTES(X)
#define ERNIE_CAPSULE_ALL(X) ERNIE_AXIS_SHAPE_ALL(X) ERNIE_CAPSULE_ATTRIBUTES(X)
#define ERNIE_PLANE_ALL(X) ERNIE_AXIS_SHAPE_ALL(X) ERNIE_PLANE_ATTRIBUTES(X)
#define ERNIE_POINT_BASED_ALL(X) ERNIE_GPRIM_ALL(X) ERNIE_POINT_BASED_ATTRIBUTES(X)
#define ERNIE_MESH_ALL(X) ERNIE_POINT_BASED_ALL(X) ERNIE_MESH_ATTRIBUTES(X)
#define ERNIE_POINTS_ALL(X) ERNIE_POINT_BASED_ALL(X) ERNIE_POINTS_ATTRIBUTES(X)
#define ERNIE_POINT_INSTANCER_ALL(X) ERNIE_BOUNDABLE_ALL(X) ERNIE_POINT_INSTANCER_ATTRIBUTES(X)
#define ERNIE_GROUND_PLANE_ALL(X) ERNIE_IMAGEABLE_ATTRIBUTES(X) ERNIE_GROUND_PLANE_ATTRIBUTES(X)
// clang-format on

namespace ernie {

/// A box centred on its origin, `size` along each axis.
class Cube : public Gprim {
    ERNIE_SCHEMA_BODY(Cube, Gprim, ERNIE_CUBE_ALL)
    ERNIE_SCHEMA_DEFINE(Cube)
};

/// A sphere centred on its origin.
class Sphere : public Gprim {
    ERNIE_SCHEMA_BODY(Sphere, Gprim, ERNIE_SPHERE_ALL)
    ERNIE_SCHEMA_DEFINE(Sphere)
};

/// An implicit shape centred on its origin along one of its axes: a cylinder, cone, capsule or
/// plane. Not a UsdGeom type: the base of those that have an `axis`.
class AxisShape : public Gprim {
    ERNIE_SCHEMA_BODY(AxisShape, Gprim, ERNIE_AXIS_SHAPE_ALL)
};

/// A capped cylinder along its axis.
class Cylinder : public AxisShape {
    ERNIE_SCHEMA_BODY(Cylinder, AxisShape, ERNIE_CYLINDER_ALL)
    ERNIE_SCHEMA_DEFINE(Cylinder)
};

/// A capped cone along its axis, its apex at +height / 2.
class Cone : public AxisShape {
    ERNIE_SCHEMA_BODY(Cone, AxisShape, ERNIE_CYLINDER_ALL)
    ERNIE_SCHEMA_DEFINE(Cone)
};

/// A cylinder with hemispherical caps along its axis.
class Capsule : public AxisShape {
    ERNIE_SCHEMA_BODY(Capsule, AxisShape, ERNIE_CAPSULE_ALL)
    ERNIE_SCHEMA_DEFINE(Capsule)
};

/// A one-sided rectangle facing along its axis.
class Plane : public AxisShape {
    ERNIE_SCHEMA_BODY(Plane, AxisShape, ERNIE_PLANE_ALL)
    ERNIE_SCHEMA_DEFINE(Plane)
};

/// Geometry from an array of points: a mesh or a point cloud.
class PointBased : public Gprim {
    ERNIE_SCHEMA_BODY(PointBased, Gprim, ERNIE_POINT_BASED_ALL)
};

/// A polygon mesh: points and the faces over them.
class Mesh : public PointBased {
    ERNIE_SCHEMA_BODY(Mesh, PointBased, ERNIE_MESH_ALL)
    ERNIE_SCHEMA_DEFINE(Mesh)
};

/// Points drawn as spheres `widths` across, coloured by their display colour.
class Points : public PointBased {
    ERNIE_SCHEMA_BODY(Points, PointBased, ERNIE_POINTS_ALL)
    ERNIE_SCHEMA_DEFINE(Points)
};

/// Instances of its prototypes (the `prototypes` relationship's targets) at rows of positions,
/// orientations and scales.
class PointInstancer : public Boundable {
    ERNIE_SCHEMA_BODY(PointInstancer, Boundable, ERNIE_POINT_INSTANCER_ALL)
    ERNIE_SCHEMA_DEFINE(PointInstancer)
};

/// A volume (UsdVol); Ernie does not draw volumes yet.
class Volume : public Gprim {
    ERNIE_SCHEMA_BODY(Volume, Gprim, ERNIE_GPRIM_ALL)
    ERNIE_SCHEMA_DEFINE(Volume)
};

/// Ernie: an infinite horizontal floor facing +z at a world height, drawn with its material,
/// optionally as a checkerboard. Its height is in world space; it has no transform.
class GroundPlane : public Imageable {
    ERNIE_SCHEMA_BODY(GroundPlane, Imageable, ERNIE_GROUND_PLANE_ALL)
    ERNIE_SCHEMA_DEFINE(GroundPlane)
};

/// This file's types' registry entries, bases first (`PrimTypes`).
std::vector<const PrimTypeDef*> GeometrySchemaTypes();

}  // namespace ernie
