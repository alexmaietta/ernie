// SPDX-License-Identifier: BSD-3-Clause
/// @file mesh.h
/// @brief UsdGeom's point-based geometry: meshes and point clouds.
///
/// References:
/// - Pixar, UsdGeomMesh and UsdGeomPoints, https://openusd.org/release/api/usd_geom_page_front.html
#pragma once

#include "ernie/scene/schemas/geom.h"

// clang-format off
#define ERNIE_POINT_BASED_ATTRIBUTES(X)                                                            \
    X(Tensor, Points, "points", Tensor(), "float32 or float64 (n, 3) positions in its own space.", \
      .group = "Geometry", .components = 3, .dirty = Dirty::Points)                                \
    X(Tensor, Velocities, "velocities", Tensor(), "(n, 3) metres per second of each point.",       \
      .group = "Motion", .components = 3, .dirty = Dirty::Points)                                  \
    X(Tensor, Accelerations, "accelerations", Tensor(), "(n, 3) metres per second squared.",       \
      .group = "Motion", .components = 3, .dirty = Dirty::Points)                                  \
    X(Tensor, Normals, "normals", Tensor(), "(n, 3) normals; computed from the faces when unset.", \
      .group = "Geometry", .components = 3, .dirty = Dirty::Points)

#define ERNIE_MESH_INDICES(X, Name, usdName, text)                                                 \
    X(Tensor, Name, usdName, Tensor(), text, .group = "Topology", .integer = true,                 \
      .dirty = Dirty::Topology)

#define ERNIE_MESH_ATTRIBUTES(X)                                                                   \
    ERNIE_MESH_INDICES(X, FaceVertexCounts, "faceVertexCounts",                                    \
                       "The number of vertices of each face.")                                     \
    ERNIE_MESH_INDICES(X, FaceVertexIndices, "faceVertexIndices",                                  \
                       "The point indices of the faces' vertices, in order.")                      \
    X(std::string, SubdivisionScheme, "subdivisionScheme", "catmullClark",                         \
      "The surface the faces are the control mesh of; none draws the faces themselves.",           \
      .label = "Subdivision", .group = "Subdivision",                                              \
      .options = {"catmullClark", "loop", "bilinear", "none"}, .dirty = Dirty::Topology)           \
    X(std::string, InterpolateBoundary, "interpolateBoundary", "edgeAndCorner",                    \
      "How boundary edges and corners are interpolated.", .group = "Subdivision",                  \
      .options = {"none", "edgeOnly", "edgeAndCorner"}, .dirty = Dirty::Topology)                  \
    X(std::string, FaceVaryingLinearInterpolation, "faceVaryingLinearInterpolation",               \
      "cornersPlus1", "Where face-varying primvars stay linear rather than smoothed.",             \
      .group = "Subdivision",                                                                      \
      .options = {"none", "cornersOnly", "cornersPlus1", "cornersPlus2", "boundaries", "all"},     \
      .dirty = Dirty::Topology)                                                                    \
    X(std::string, TriangleSubdivisionRule, "triangleSubdivisionRule", "catmullClark",             \
      "Catmull-Clark's weights for triangles, or smoother ones.", .group = "Subdivision",          \
      .options = {"catmullClark", "smooth"}, .dirty = Dirty::Topology)                             \
    ERNIE_MESH_INDICES(X, HoleIndices, "holeIndices", "Faces left out of the surface.")            \
    ERNIE_MESH_INDICES(X, CornerIndices, "cornerIndices", "Points that are sharp corners.")        \
    X(Tensor, CornerSharpnesses, "cornerSharpnesses", Tensor(), "The sharpness of each corner.",   \
      .group = "Subdivision", .dirty = Dirty::Topology)                                            \
    ERNIE_MESH_INDICES(X, CreaseIndices, "creaseIndices",                                          \
                       "Point indices of creases, one after another.")                             \
    ERNIE_MESH_INDICES(X, CreaseLengths, "creaseLengths", "The number of points of each crease.") \
    X(Tensor, CreaseSharpnesses, "creaseSharpnesses", Tensor(),                                    \
      "The sharpness of each crease, or of each crease edge.", .group = "Subdivision",             \
      .dirty = Dirty::Topology)

#define ERNIE_POINTS_ATTRIBUTES(X)                                                                 \
    X(Tensor, Widths, "widths", Tensor(), "Diameters in metres: one per point, or one for all.",   \
      .group = "Geometry", .dirty = Dirty::Points)                                                 \
    X(Tensor, Ids, "ids", Tensor(), "int64 ids that follow points across time samples.",           \
      .group = "Geometry", .integer = true, .dirty = Dirty::Topology)
// clang-format on

namespace ernie {

ERNIE_SCHEMA(PointBased, Gprim, ERNIE_POINT_BASED_ATTRIBUTES, "Geometry from an array of points.");
ERNIE_CONCRETE_SCHEMA(Mesh, PointBased, ERNIE_MESH_ATTRIBUTES,
                      "A polygon mesh, or the control mesh of a subdivision surface.");
ERNIE_CONCRETE_SCHEMA(Points, PointBased, ERNIE_POINTS_ATTRIBUTES,
                      "A point cloud of spheres, each its own width.");

}  // namespace ernie
