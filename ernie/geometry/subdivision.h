// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/geometry/mesh_data.h"

#include <cstdint>
#include <span>

namespace ernie {

/// USD's `subdivisionScheme`.
enum class SubdivisionScheme { None, CatmullClark, Loop, Bilinear };

/// USD's `interpolateBoundary`.
enum class BoundaryInterpolation { None, EdgeOnly, EdgeAndCorner };

/// USD's `faceVaryingLinearInterpolation`: where face-varying data
/// (such as texture coordinates) stays linear rather than smoothed.
enum class FaceVaryingInterpolation {
    None,  ///< Smooth everywhere.
    CornersOnly,
    CornersPlus1,
    CornersPlus2,
    Boundaries,
    All,  ///< Linear everywhere.
};

/// USD's `triangleSubdivisionRule`.
enum class TriangleSubdivision { CatmullClark, Smooth };

/// A subdivision surface's control mesh and tags, as USD's Mesh holds
/// them. Spans are borrowed for the call.
struct SubdivisionSurface {
    SubdivisionScheme scheme = SubdivisionScheme::CatmullClark;
    BoundaryInterpolation boundary =
        BoundaryInterpolation::EdgeAndCorner;
    FaceVaryingInterpolation faceVarying =
        FaceVaryingInterpolation::CornersPlus1;
    TriangleSubdivision triangles = TriangleSubdivision::CatmullClark;
    std::span<const float> points;  ///< x, y, z per control point.
    std::span<const std::int64_t> faceVertexCounts;
    std::span<const std::int64_t> faceVertexIndices;
    /// Crease chains of vertices, `creaseLengths` long each, with a
    /// sharpness per chain or per edge of the chains (10 or more is
    /// infinitely sharp).
    std::span<const std::int64_t> creaseIndices;
    std::span<const std::int64_t> creaseLengths;
    std::span<const float> creaseSharpnesses;
    std::span<const std::int64_t> cornerIndices;
    std::span<const float> cornerSharpnesses;
    /// Faces left out of the surface.
    std::span<const std::int64_t> holeIndices;
    /// Empty, or texture coordinates s, t per face corner (per entry
    /// of `faceVertexIndices`); equal values at a point join across
    /// faces.
    std::span<const float> cornerUvs;
    /// Empty, or linear r, g, b per control point.
    std::span<const float> colors;
};

/// The limit surface of `surface` tessellated uniformly into
/// triangles, `2^level` segments along each control edge (level 0
/// evaluates only the control faces' corners on the limit surface),
/// with the surface's normals, and texture coordinates and colours
/// when given. Points along shared edges and at shared vertices are
/// shared, so the result is watertight, except across infinitely
/// sharp creases, whose sides keep normals of their own. Above level
/// 0, the coarser levels down to 0 come as the mesh's levels of
/// detail: their points are among the finest's, so their triangles
/// index the same points.
///
/// References:
///   Pixar, OpenSubdiv's Bfr (base face representation),
///   https://graphics.pixar.com/opensubdiv/docs/bfr_overview.html
///   E. Catmull, J. Clark, "Recursively generated B-spline surfaces
///   on arbitrary topological meshes", Computer-Aided Design 10(6),
///   1978. C. Loop, "Smooth Subdivision Surfaces Based on Triangles",
///   MSc thesis, University of Utah, 1987.
/// @throws std::invalid_argument for a scheme of None, a level
/// outside [0, 8], counts and indices that disagree, an index out of
/// range, a face of fewer than 3 vertices, a Loop surface with other
/// than triangles, crease, corner or hole arrays that disagree, or
/// texture coordinates or colours of the wrong size.
MeshData Subdivide(const SubdivisionSurface& surface, int level);

/// Each point of a tessellation as a weighted sum of the control
/// points (the limit surface is linear in them), so control points
/// that move, as skinning moves them, refine again by these sums
/// alone: point p is the sum over k in [starts[p], starts[p + 1]) of
/// weights[k] times control point indices[k].
struct RefinementStencils {
    std::vector<std::uint32_t> starts;  ///< Points + 1.
    std::vector<std::uint32_t> indices;
    std::vector<float> weights;
};

/// `Subdivide`, with the stencils of its points in `stencils`.
MeshData Subdivide(const SubdivisionSurface& surface, int level,
                   RefinementStencils& stencils);

}  // namespace ernie
