// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/base_schemas.h"
#include "ernie/scene/prim_type.h"
#include "ernie/scene/tokens.h"

#include <vector>

/// Geometry as reflected schemas (`prim_type.h`): UsdGeom's implicit
/// shapes (Cube, Sphere, and along an axis Cylinder, Cone, Capsule
/// and Plane), its point-based Mesh and Points, the PointInstancer,
/// and Ernie's GroundPlane (UsdVol's Volume is in
/// `volume_schemas.h`). Each declaration's
/// fallback is the value Ernie draws an unset attribute with
/// (UsdGeom's own). Bulk arrays (points, topology, instance
/// transforms) are tensors, written with `Scene::Write`; primvars
/// other than the display colour are an open namespace
/// (`primvars:<name>`), read by name.
///
/// References:
/// - Pixar, UsdGeom,
///   https://openusd.org/release/api/usd_geom_page_front.html
/// - Pixar, UsdVol,
///   https://openusd.org/release/api/usd_vol_page_front.html

// clang-format off
#define ERNIE_GEOM_SHAPE_RANGE .group = "Shape", .min = 0.0, .max = 1000.0

#define ERNIE_CUBE_ATTRIBUTES(X)                                                                   \
    X(double, Size, ::ernie::tokens::kSize, 2.0,                                                   \
      (::ernie::AttributeMeta{.doc = "Edge length, metres.", ERNIE_GEOM_SHAPE_RANGE}))

#define ERNIE_SPHERE_ATTRIBUTES(X)                                                                 \
    X(double, Radius, ::ernie::tokens::kGeomRadius, 1.0,                                           \
      (::ernie::AttributeMeta{.doc = "Metres.", ERNIE_GEOM_SHAPE_RANGE}))                          \
    X(std::string, Tessellation, ::ernie::tokens::kTessellation, ::ernie::tokens::kUvSphere,       \
      (::ernie::AttributeMeta{.doc = "uv: latitude rings and meridians; icosphere: a divided "     \
                                     "icosahedron; cubeSphere: a cube's faces as planet "          \
                                     "Terrains'; healpix: HEALPix's equal-area pixels; adaptive: " \
                                     "a planet's cube-sphere LOD, finer towards the eye, exact "   \
                                     "at any scale (raster; a cube sphere elsewhere). Each has "   \
                                     "equirectangular texture coordinates.",                       \
                              .group = "Shape",                                                    \
                              .options = {::ernie::tokens::kUvSphere, ::ernie::tokens::kIcosphere, \
                                          ::ernie::tokens::kCubeSphere, ::ernie::tokens::kHealpix, \
                                          ::ernie::tokens::kAdaptiveSphere}}))                     \
    X(int, TessellationLevel, ::ernie::tokens::kTessellationLevel, 0,                              \
      (::ernie::AttributeMeta{.label = "Level", .doc = "UV stacks, icosphere subdivisions, cube "  \
                                     "quads a face edge or HEALPix N_side; 0 for some 2,000 to "   \
                                     "6,000 triangles.",                                           \
                              .group = "Shape", .max = 1024.0}))

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
                              .group = "Geometry", .dirty = ::ernie::Dirty::Points}))

#define ERNIE_MESH_ATTRIBUTES(X)                                                                   \
    X(::ernie::Tensor, FaceVertexCounts, ::ernie::tokens::kFaceVertexCounts, nullptr,              \
      (::ernie::AttributeMeta{.doc = "Integer number of vertices of each face.",                   \
                              .group = "Geometry", .integer = true,                                \
                              .dirty = ::ernie::Dirty::Topology}))                                 \
    X(::ernie::Tensor, FaceVertexIndices, ::ernie::tokens::kFaceVertexIndices, nullptr,            \
      (::ernie::AttributeMeta{.doc = "Integer point indices of the faces' vertices, in order.",    \
                              .group = "Geometry", .integer = true,                                \
                              .dirty = ::ernie::Dirty::Topology}))                                 \
    X(::ernie::Tensor, Normals, ::ernie::tokens::kNormals, nullptr,                                \
      (::ernie::AttributeMeta{.doc = "(n, 3) normals: one per point, or per face vertex, face or "  \
                                     "mesh (normals:interpolation as USD's, else by count) for "   \
                                     "hard edges; primvars:normals, perhaps indexed, in their "    \
                                     "place; ignored when of another count. Without them, smooth " \
                                     "ones from the triangles, computed on the GPU.",              \
                              .group = "Geometry", .dirty = ::ernie::Dirty::Points}))              \
    X(::ernie::Tensor, ActiveCounts, ::ernie::tokens::kActiveCounts, nullptr,                      \
      (::ernie::AttributeMeta{.doc = "int32 (n >= 2,): the points and triangle indices in use, "   \
                                     "first, of those the points and triangle indices hold, for "  \
                                     "topology that changes on the GPU within their capacity, "    \
                                     "as simulators count them (more entries are ignored); read "  \
                                     "on the GPU, the rest neither drawn nor bounded.",            \
                              .group = "Geometry", .integer = true}))                              \
    X(std::string, SubdivisionScheme, ::ernie::tokens::kSubdivisionScheme, ::ernie::tokens::kNone, \
      (::ernie::AttributeMeta{.label = "Subdivision",                                              \
                              .doc = "The surface the faces are the control mesh of, refined by "  \
                                     "OpenSubdiv; none draws the faces themselves (USD's "         \
                                     "fallback is catmullClark, which import keeps). Normals are " \
                                     "the surface's, not the authored ones. Points written on "    \
                                     "the GPU draw the control mesh.",                             \
                              .group = "Subdivision",                                              \
                              .options = {::ernie::tokens::kNone, ::ernie::tokens::kCatmullClark,  \
                                          ::ernie::tokens::kLoop, ::ernie::tokens::kBilinear},     \
                              .dirty = ::ernie::Dirty::Topology}))                                 \
    X(int, RefineLevel, ::ernie::tokens::kRefineLevel, -1,                                         \
      (::ernie::AttributeMeta{.label = "Refine level",                                             \
                              .doc = "2^level segments along each control edge; -1 for the "       \
                                     "stage's ernie:subdivision:refineLevel.",                     \
                              .group = "Subdivision", .min = -1.0, .max = 8.0,                     \
                              .dirty = ::ernie::Dirty::Topology}))                                 \
    X(std::string, InterpolateBoundary, ::ernie::tokens::kInterpolateBoundary,                     \
      ::ernie::tokens::kEdgeAndCorner,                                                             \
      (::ernie::AttributeMeta{.doc = "How boundary edges and corners are interpolated.",           \
                              .group = "Subdivision",                                              \
                              .options = {::ernie::tokens::kNone, ::ernie::tokens::kEdgeOnly,      \
                                          ::ernie::tokens::kEdgeAndCorner},                        \
                              .dirty = ::ernie::Dirty::Topology}))                                 \
    X(std::string, FaceVaryingLinearInterpolation,                                                 \
      ::ernie::tokens::kFaceVaryingLinearInterpolation, ::ernie::tokens::kCornersPlus1,            \
      (::ernie::AttributeMeta{.doc = "Where texture coordinates stay linear, not smoothed: none "  \
                                     "smooths them everywhere, all keeps them linear.",            \
                              .group = "Subdivision",                                              \
                              .options = {::ernie::tokens::kNone, ::ernie::tokens::kCornersOnly,   \
                                          ::ernie::tokens::kCornersPlus1,                          \
                                          ::ernie::tokens::kCornersPlus2,                          \
                                          ::ernie::tokens::kBoundaries, ::ernie::tokens::kAll},    \
                              .dirty = ::ernie::Dirty::Topology}))                                 \
    X(std::string, TriangleSubdivisionRule, ::ernie::tokens::kTriangleSubdivisionRule,             \
      ::ernie::tokens::kCatmullClark,                                                              \
      (::ernie::AttributeMeta{.doc = "Catmull-Clark's weights for triangles, or smoother ones.",   \
                              .group = "Subdivision",                                              \
                              .options = {::ernie::tokens::kCatmullClark,                          \
                                          ::ernie::tokens::kSmooth},                               \
                              .dirty = ::ernie::Dirty::Topology}))                                 \
    X(::ernie::Tensor, CreaseIndices, ::ernie::tokens::kCreaseIndices, nullptr,                    \
      (::ernie::AttributeMeta{.doc = "Integer point indices of crease chains, one after another.", \
                              .group = "Subdivision", .integer = true,                             \
                              .dirty = ::ernie::Dirty::Topology}))                                 \
    X(::ernie::Tensor, CreaseLengths, ::ernie::tokens::kCreaseLengths, nullptr,                    \
      (::ernie::AttributeMeta{.doc = "Integer number of points of each crease chain.",             \
                              .group = "Subdivision", .integer = true,                             \
                              .dirty = ::ernie::Dirty::Topology}))                                 \
    X(::ernie::Tensor, CreaseSharpnesses, ::ernie::tokens::kCreaseSharpnesses, nullptr,            \
      (::ernie::AttributeMeta{.doc = "Sharpness of each crease, or of each crease edge; 10 or "    \
                                     "more is infinitely sharp.",                                  \
                              .group = "Subdivision", .dirty = ::ernie::Dirty::Topology}))         \
    X(::ernie::Tensor, CornerIndices, ::ernie::tokens::kCornerIndices, nullptr,                    \
      (::ernie::AttributeMeta{.doc = "Integer indices of sharp corner points.",                    \
                              .group = "Subdivision", .integer = true,                             \
                              .dirty = ::ernie::Dirty::Topology}))                                 \
    X(::ernie::Tensor, CornerSharpnesses, ::ernie::tokens::kCornerSharpnesses, nullptr,            \
      (::ernie::AttributeMeta{.doc = "Sharpness of each corner.", .group = "Subdivision",          \
                              .dirty = ::ernie::Dirty::Topology}))                                 \
    X(::ernie::Tensor, HoleIndices, ::ernie::tokens::kHoleIndices, nullptr,                        \
      (::ernie::AttributeMeta{.doc = "Integer indices of faces left out of the surface.",          \
                              .group = "Subdivision", .integer = true,                             \
                              .dirty = ::ernie::Dirty::Topology}))                                 \
    X(::ernie::Path, Skeleton, ::ernie::tokens::kSkelSkeleton, ::ernie::Path(),                    \
      (::ernie::AttributeMeta{.doc = "The Skeleton skinning it; unset, it is not skinned.",        \
                              .group = "Skinning", .options = {::ernie::tokens::kSkeleton},        \
                              .dirty = ::ernie::Dirty::Topology,                                   \
                              .targetDirty = ::ernie::Dirty::Points}))                                 \
    X(std::vector<std::string>, SkelJoints, ::ernie::tokens::kSkelJoints,                          \
      std::vector<std::string>{},                                                                  \
      (::ernie::AttributeMeta{.doc = "The Skeleton's joints its joint indices count, in their "    \
                                     "order; unset, all of them in the Skeleton's order.",         \
                              .group = "Skinning", .dirty = ::ernie::Dirty::Topology}))            \
    X(::ernie::Tensor, JointIndices, ::ernie::tokens::kSkelJointIndices, nullptr,                  \
      (::ernie::AttributeMeta{.doc = "Integer (points, k) joints influencing each point, or "      \
                                     "(1, k) for all points (rigid).",                             \
                              .group = "Skinning", .integer = true,                                \
                              .dirty = ::ernie::Dirty::Topology}))                                 \
    X(::ernie::Tensor, JointWeights, ::ernie::tokens::kSkelJointWeights, nullptr,                  \
      (::ernie::AttributeMeta{.doc = "(points, k) or (1, k) weights of the joint indices.",        \
                              .group = "Skinning", .dirty = ::ernie::Dirty::Topology}))            \
    X(glm::dmat4, GeomBindTransform, ::ernie::tokens::kSkelGeomBindTransform, glm::dmat4(1.0),     \
      (::ernie::AttributeMeta{.doc = "Places its points in the Skeleton's space when bound.",      \
                              .group = "Skinning", .dirty = ::ernie::Dirty::Topology}))            \
    X(std::string, SkinningMethod, ::ernie::tokens::kSkelSkinningMethod,                           \
      ::ernie::tokens::kClassicLinear,                                                             \
      (::ernie::AttributeMeta{.doc = "How joint transforms blend: linearly, or as dual "           \
                                     "quaternions, which keep volume at twisting joints.",         \
                              .group = "Skinning",                                                 \
                              .options = {::ernie::tokens::kClassicLinear,                         \
                                          ::ernie::tokens::kDualQuaternion},                       \
                              .dirty = ::ernie::Dirty::Topology}))                                 \
    X(std::vector<std::string>, SkelBlendShapes, ::ernie::tokens::kSkelBlendShapes,                \
      std::vector<std::string>{},                                                                  \
      (::ernie::AttributeMeta{.doc = "Its names for its blend shapes, as animations weight them.", \
                              .group = "Skinning", .dirty = ::ernie::Dirty::Topology}))            \
    X(std::vector<::ernie::Path>, BlendShapeTargets, ::ernie::tokens::kSkelBlendShapeTargets,      \
      std::vector<::ernie::Path>{},                                                                \
      (::ernie::AttributeMeta{.doc = "The BlendShape prim of each name.", .group = "Skinning",     \
                              .dirty = ::ernie::Dirty::Topology}))                                 \
    X(::ernie::Tensor, ThermalTemperatures, ::ernie::tokens::kThermalTemperatures, nullptr,        \
      (::ernie::AttributeMeta{.label = "Temperatures",                                             \
                              .doc = "float32 (n,) kelvin, one per point, on the CPU or a CUDA "   \
                                     "device, as a thermal solver writes them back each step: "    \
                                     "with thermal emission on, in place of the prim's own and "   \
                                     "its material's, interpolated across its faces; 0 at a "      \
                                     "point for those.",                                           \
                              .group = "Thermal", .dirty = ::ernie::Dirty::Points}))              \
    X(::ernie::Tensor, ThermalNodeIds, ::ernie::tokens::kThermalNodeIds, nullptr,                  \
      (::ernie::AttributeMeta{.label = "Node ids",                                                 \
                              .doc = "int32 (n,), each point's node in a thermal network, whose "  \
                                     "temperatures the node temperatures give; a face takes its "  \
                                     "own node where its points are its own.",                     \
                              .group = "Thermal", .integer = true,                                 \
                              .dirty = ::ernie::Dirty::Points}))                                   \
    X(::ernie::Tensor, ThermalNodeTemperatures, ::ernie::tokens::kThermalNodeTemperatures,         \
      nullptr,                                                                                     \
      (::ernie::AttributeMeta{.label = "Node temperatures",                                        \
                              .doc = "float32 (nodes,) kelvin per node of the node ids, on the "   \
                                     "CPU or a CUDA device, as a thermal network's solver writes " \
                                     "them each step; the points' own temperatures, when given, "  \
                                     "come first.",                                                \
                              .group = "Thermal", .dirty = ::ernie::Dirty::Points}))

#define ERNIE_POINTS_ATTRIBUTES(X)                                                                 \
    X(::ernie::OrArray<std::vector<float>>, Widths, ::ernie::tokens::kWidths,                      \
      std::vector<float>{1.0f},                                                                    \
      (::ernie::AttributeMeta{.doc = "Diameters: one for every point, or one per point. Many "     \
                                     "points' widths may be written as an array instead.",         \
                              .group = "Geometry", .min = 0.0, .max = 1000.0}))

#define ERNIE_POINT_INSTANCER_ATTRIBUTES(X)                                                        \
    X(::ernie::Tensor, Positions, ::ernie::tokens::kPositions, nullptr,                            \
      (::ernie::AttributeMeta{.doc = "(n, 3) instance positions in the instancer's frame; the "    \
                                     "number of instances.",                                       \
                              .group = "Instances", .dirty = ::ernie::Dirty::Points}))             \
    X(::ernie::Tensor, Orientations, ::ernie::tokens::kOrientations, nullptr,                      \
      (::ernie::AttributeMeta{.doc = "(n, 4) instance rotations as quaternions.",                  \
                              .group = "Instances", .dirty = ::ernie::Dirty::Points}))             \
    X(::ernie::Tensor, Scales, ::ernie::tokens::kScales, nullptr,                                  \
      (::ernie::AttributeMeta{.doc = "(n, 3) instance scales.", .group = "Instances",              \
                              .dirty = ::ernie::Dirty::Points}))                                   \
    X(::ernie::Tensor, ProtoIndices, ::ernie::tokens::kProtoIndices, nullptr,                      \
      (::ernie::AttributeMeta{.doc = "Integer prototype of each instance, of the prototypes "      \
                                     "relationship's targets; the first for all when unset.",      \
                              .group = "Instances", .integer = true,                               \
                              .dirty = ::ernie::Dirty::Topology}))                                 \
    X(std::vector<::ernie::Path>, Prototypes, ::ernie::tokens::kPrototypes,                        \
      std::vector<::ernie::Path>{},                                                                \
      (::ernie::AttributeMeta{.doc = "The prims instanced, indexed by the proto indices.",         \
                              .group = "Instances", .dirty = ::ernie::Dirty::Topology}))           \
    X(::ernie::Tensor, Worlds, ::ernie::tokens::kWorlds, nullptr,                                  \
      (::ernie::AttributeMeta{.doc = "int (n,): each instance's world, in place of the "           \
                                     "instancer's world.",                                         \
                              .group = "Instances"}))                                              \
    X(::ernie::Tensor, DisplayColor, ::ernie::tokens::kDisplayColor, nullptr,                      \
      (::ernie::AttributeMeta{.doc = "(n, 3) linear RGB of each instance, in place of its "        \
                                     "prototype's displayColor (USD's per-instance primvar).",     \
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
                              .group = "Ground", .options = {::ernie::tokens::kMaterial},          \
                              .dirty = ::ernie::Dirty::Material,                                   \
                              .targetDirty = ::ernie::Dirty::Material}))                                 \
    X(double, ThermalTemperature, ::ernie::tokens::kThermalTemperature, 0.0,                       \
      (::ernie::AttributeMeta{.label = "Temperature",                                              \
                              .doc = "Kelvin, in place of its material's; 0 for the "              \
                                     "material's own.",                                            \
                              .group = "Thermal", .max = 5000.0}))

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

/// An implicit shape centred on its origin along one of its axes: a
/// cylinder, cone, capsule or plane. Not a UsdGeom type: the base of
/// those that have an `axis`.
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

/// Points drawn as spheres `widths` across, coloured by their display
/// colour.
class Points : public PointBased {
    ERNIE_SCHEMA_BODY(Points, PointBased, ERNIE_POINTS_ALL)
    ERNIE_SCHEMA_DEFINE(Points)
};

/// Instances of its prototypes (the `prototypes` relationship's
/// targets) at rows of positions, orientations and scales.
class PointInstancer : public Boundable {
    ERNIE_SCHEMA_BODY(PointInstancer, Boundable,
                      ERNIE_POINT_INSTANCER_ALL)
    ERNIE_SCHEMA_DEFINE(PointInstancer)
};

/// Ernie: an infinite horizontal floor facing +z at a world height,
/// drawn with its material, optionally as a checkerboard. Its height
/// is in world space; it has no transform.
class GroundPlane : public Imageable {
    ERNIE_SCHEMA_BODY(GroundPlane, Imageable, ERNIE_GROUND_PLANE_ALL)
    ERNIE_SCHEMA_DEFINE(GroundPlane)
};

/// This file's types' registry entries, bases first (`PrimTypes`).
std::vector<const PrimTypeDef*> GeometrySchemaTypes();

}  // namespace ernie
