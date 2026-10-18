// SPDX-License-Identifier: BSD-3-Clause
/// @file mesh.h
/// @brief UsdGeom's point-based geometry: meshes, point clouds and curves.
///
/// References:
/// - Pixar, UsdGeomMesh, UsdGeomPoints and UsdGeomCurves,
///   https://openusd.org/release/api/usd_geom_page_front.html
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
      .dirty = Dirty::Topology)                                                                    \
    X(int, SubdivisionRefineLevel, "ernie:subdivision:refineLevel", 1,                             \
      "Uniform refinement of the surface: 2^level segments along each control edge.",              \
      .label = "Refine level", .group = "Subdivision", .lower = AtLeast(0.0),                      \
      .upper = AtMost(8.0), .dirty = Dirty::Topology)

#define ERNIE_WIDTHS_ATTRIBUTE(X, text)                                                            \
    X(Tensor, Widths, "widths", Tensor(), text, .group = "Geometry", .dirty = Dirty::Points)

#define ERNIE_POINTS_ATTRIBUTES(X)                                                                 \
    ERNIE_WIDTHS_ATTRIBUTE(X, "Diameters in metres: one per point, or one for all.")               \
    X(Tensor, Ids, "ids", Tensor(), "int64 ids that follow points across time samples.",           \
      .group = "Geometry", .integer = true, .dirty = Dirty::Topology)

#define ERNIE_CURVES_ATTRIBUTES(X)                                                                 \
    ERNIE_MESH_INDICES(X, CurveVertexCounts, "curveVertexCounts",                                  \
                       "The number of points of each curve.")                                      \
    ERNIE_WIDTHS_ATTRIBUTE(X, "Diameters in metres: per point, per segment end, or one for all.")

#define ERNIE_BASIS_CURVES_ATTRIBUTES(X)                                                           \
    X(std::string, Type, "type", "cubic", "Linear segments, or cubic ones by the basis.",          \
      .group = "Topology", .options = {"linear", "cubic"}, .dirty = Dirty::Topology)               \
    X(std::string, Basis, "basis", "bezier", "The cubic basis.", .group = "Topology",              \
      .options = {"bezier", "bspline", "catmullRom"}, .dirty = Dirty::Topology)                    \
    X(std::string, Wrap, "wrap", "nonperiodic",                                                    \
      "Open curves, closed ones, or open ones through their end points.", .group = "Topology",     \
      .options = {"nonperiodic", "periodic", "pinned"}, .dirty = Dirty::Topology)

#define ERNIE_NURBS_CURVES_ATTRIBUTES(X)                                                           \
    ERNIE_MESH_INDICES(X, Order, "order", "The order (degree + 1) of each curve.")                 \
    X(Tensor, Knots, "knots", Tensor(), "Each curve's knots, curveVertexCount + order of them.",   \
      .group = "Topology", .dirty = Dirty::Topology)                                               \
    X(Tensor, Ranges, "ranges", Tensor(), "(n, 2) parametric range drawn of each curve.",          \
      .group = "Topology", .components = 2, .dirty = Dirty::Topology)                              \
    X(Tensor, PointWeights, "pointWeights", Tensor(), "Rational weights, one per point.",          \
      .group = "Geometry", .dirty = Dirty::Points)

#define ERNIE_HERMITE_CURVES_ATTRIBUTES(X)                                                         \
    X(Tensor, Tangents, "tangents", Tensor(), "(n, 3) tangent of each point.",                     \
      .group = "Geometry", .components = 3, .dirty = Dirty::Points)
// clang-format on

namespace ernie {

ERNIE_SCHEMA(PointBased, Gprim, ERNIE_POINT_BASED_ATTRIBUTES, "Geometry from an array of points.");
ERNIE_CONCRETE_SCHEMA(Mesh, PointBased, ERNIE_MESH_ATTRIBUTES,
                      "A polygon mesh, or the control mesh of a subdivision surface.");
ERNIE_CONCRETE_SCHEMA(Points, PointBased, ERNIE_POINTS_ATTRIBUTES,
                      "A point cloud of spheres, each its own width.");
ERNIE_SCHEMA(Curves, PointBased, ERNIE_CURVES_ATTRIBUTES,
             "Batched curves, each a run of points, swept by its widths.");
ERNIE_CONCRETE_SCHEMA(BasisCurves, Curves, ERNIE_BASIS_CURVES_ATTRIBUTES,
                      "Linear or cubic curves: hair, fur, cables and wires.");
ERNIE_CONCRETE_SCHEMA(NurbsCurves, Curves, ERNIE_NURBS_CURVES_ATTRIBUTES,
                      "Rational B-spline curves of any order.");
ERNIE_CONCRETE_SCHEMA(HermiteCurves, Curves, ERNIE_HERMITE_CURVES_ATTRIBUTES,
                      "Cubic curves through their points with a tangent at each.");

}  // namespace ernie
