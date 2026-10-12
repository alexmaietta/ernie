// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/geometry/frustum.h"
#include "ernie/geometry/lod_quadtree.h"
#include "ernie/util/raster_view.h"

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace ernie {

/// Region of space holding one node of a `CubeSphereLod`: every
/// direction within `angle` of `axis`, between radii `inner` and
/// `outer` from the body's centre. It contains the node's vertices
/// and the flat triangles drawn between them.
struct CubeSphereRegion {
    glm::dvec3 axis{0.0, 0.0,
                    1.0};  ///< Unit direction of the node's centre.
    double angle = 0.0;  ///< Largest angle from `axis` to the node's
                         ///< corners, radians.
    double inner = 0.0;
    double outer = 0.0;

    /// Whether the region holds no point, as for nodes without valid
    /// heights.
    bool Empty() const {
        return inner > outer;
    }

    /// Distance from `point` to the nearest point of the region; 0
    /// inside it.
    double Distance(const glm::dvec3& point) const;

    /// Axis-aligned bounds (low, high) of the region.
    std::pair<glm::dvec3, glm::dvec3> Box() const;
};

/// CDLOD terrain over a whole planet: six `LodQuadtree` height
/// pyramids, one per face of a cube projected onto a sphere of radius
/// `Radius()` (a quadrilateralized or "cubed" sphere).
///
/// Coordinates are the body-fixed frame: z towards the north pole and
/// x towards longitude 0, in the same units as the radius. Each face
/// is a grid of `FaceCells()` x `FaceCells()` quads whose
/// `FaceCells() + 1` squared vertices carry heights above the radius;
/// vertex (`col`, `row`) of face `f` lies in direction
/// `FaceDirection(f, s, t)` with s = 2 col / FaceCells() - 1 and t =
/// 2 row / FaceCells() - 1. Vertices on an edge or corner of the cube
/// appear on every face sharing it and must carry the same height
/// there.
///
/// Faces 0 to 3 straddle the equator, centred on longitudes 0, 90,
/// 180 and 270 degrees east, with columns running east and rows
/// running south. Face 4 is centred on the north pole and face 5 on
/// the south pole. On every face, column x row points into the body,
/// so all faces wind the same way. Adjacent faces may run their grids
/// in opposite directions along a shared edge.
///
/// The projection is equiangular (Ronchi et al. 1996): s and t map to
/// the face plane through tan(pi s / 4), so cells span nearly equal
/// angles and areas differ by at most a factor of about 1.4 over a
/// face, against 5.2 for a plain gnomonic cube. Grid lines are great
/// circles.
///
/// `Select` runs Strugar's algorithm on all six trees, using
/// distances to each node's `CubeSphereRegion`. Neighbouring patches,
/// across faces too, differ by at most one level when the first range
/// is at least `MinFirstRange`. Odd vertices of a patch morph in 3D
/// towards the midpoint of their two neighbours on the next coarser
/// level: along the patch's edges these are the edge's own
/// neighbours, and for vertices odd in both directions those on the
/// quad diagonal from (1, 0) to (0, 1). Unlike the flat terrain's
/// slide towards lower indices, this does not depend on which way a
/// face's grid runs, so edges between faces match, and a fully
/// morphed patch is exactly the coarser mesh.
///
/// Refinement (`SetRefinement`) adds levels finer than the data:
/// with k extra levels each face is a grid of `FineCells()` = 2^k
/// `FaceCells()` quads, the data's vertices every 2^k of its lines
/// and heights between them interpolated, so the surface keeps its
/// shape (the sphere's curvature and the data's heights) down to
/// cells 2^k times smaller than the data's near the eye. Levels then
/// count from the finest, a data level l being level l + k, and
/// patches' origins are on the fine grid. A refined node's heights
/// are its data leaf's.
///
/// Positions on a planet need more precision than single floats.
/// Selection works in doubles, and its frustum is relative to the
/// eye: build it from a view-projection matrix whose view has the eye
/// at the origin and the body frame's axes.
///
/// References:
/// - C. Ronchi, R. Iacono and P. S. Paolucci, "The 'Cubed Sphere': A
///   New Method for the Solution of Partial Differential Equations in
///   Spherical Geometry", Journal of Computational Physics
///   124(1):93-114, 1996. https://doi.org/10.1006/jcph.1996.0047
/// - F. Strugar, "Continuous Distance-Dependent Level of Detail for
///   Rendering Heightmaps", Journal of Graphics, GPU, and Game Tools
///   14(4):57-74, 2009.
///   https://doi.org/10.1080/2151237X.2009.10129287
class CubeSphereLod {
  public:
    static constexpr std::uint32_t kFaces = 6;

    CubeSphereLod() = default;

    /// Builds the height pyramids of six face rasters of `FaceCells()
    /// + 1` squared samples, in face order. Samples that are not
    ///   finite or equal a raster's nodata value are ignored, and
    ///   nodes without valid samples are never selected. The tree
    ///   keeps no reference to the rasters; their origins and cell
    ///   sizes are not used.
    /// @param leafCells Grid quads along a level-0 node; a power of
    /// two of at least 4.
    /// @throws std::invalid_argument if the radius is not positive
    /// and finite, the rasters are not square with a power of two of
    /// at least `leafCells` quads along each side, their sizes
    /// differ, or a face would need more than
    /// `LodQuadtree::kMaxLevels` levels.
    CubeSphereLod(double radius,
                  std::span<const RasterView, kFaces> faces,
                  std::uint32_t leafCells);

    /// A sphere of faces `faceCells` quads across whose leaves'
    /// heights come later (`SetLeaves`), such as from the GPU; until
    /// then no node is selected.
    /// @throws std::invalid_argument as the other constructor.
    CubeSphereLod(double radius, std::uint32_t faceCells,
                  std::uint32_t leafCells);

    /// Unit direction of the point (`s`, `t`) of face `face`, both in
    /// [-1, 1]. Points on a shared edge or corner give bit-identical
    /// directions from every face.
    static glm::dvec3 FaceDirection(std::uint32_t face, double s,
                                    double t);

    /// The face holding `direction`, which need not be unit length,
    /// and its (s, t) there. Ties on edges go to the lowest face
    /// index.
    static std::pair<std::uint32_t, glm::dvec2>
    FacePoint(const glm::dvec3& direction);

    /// Unit direction of vertex (`col`, `row`) of face `face`'s data
    /// grid.
    glm::dvec3 VertexDirection(std::uint32_t face, std::uint32_t col,
                               std::uint32_t row) const;

    /// Unit direction of vertex (`col`, `row`) of face `face`'s fine
    /// grid (`FineCells()`), where patches lie.
    glm::dvec3 FineDirection(std::uint32_t face, std::uint32_t col,
                             std::uint32_t row) const;

    /// Adds `levels` levels finer than the data's.
    /// @throws std::invalid_argument when the fine grid would exceed
    /// 2^30 quads along a face or the levels `kMaxLevels`.
    void SetRefinement(std::uint32_t levels);

    std::uint32_t Refinement() const {
        return refinement_;
    }

    /// Quads along each side of a face's fine grid: `FaceCells()`
    /// times 2^`Refinement()`.
    std::uint32_t FineCells() const {
        return faceCells_ << refinement_;
    }

    /// The most levels, the data's and the refinement's together.
    static constexpr std::uint32_t kMaxLevels = 30;

    bool Empty() const {
        return faceCells_ == 0;
    }

    double Radius() const {
        return radius_;
    }

    /// Grid quads along each side of a face.
    std::uint32_t FaceCells() const {
        return faceCells_;
    }

    std::uint32_t LeafCells() const {
        return trees_[0].LeafCells();
    }

    /// Grid quads along each patch side, half of `LeafCells()`.
    std::uint32_t PatchCells() const {
        return trees_[0].PatchCells();
    }

    /// Levels in all: the data's and the refinement's.
    std::uint32_t Levels() const {
        return trees_[0].Levels() + refinement_;
    }

    /// Level-0 nodes along each side of a face: `SetLeaves` takes
    /// their square, row-major.
    std::uint32_t LeavesPerSide() const {
        return trees_[0].LeavesPerSide();
    }

    /// Replaces the (min, max) height of every leaf of face `face`,
    /// min above max for a leaf without valid samples, and rebuilds
    /// that face's levels above.
    /// @throws std::invalid_argument for a count other than
    /// `LeavesPerSide()` squared or a face out of range.
    void SetLeaves(std::uint32_t face,
                   std::span<const glm::vec2> leaves);

    /// The (min, max) height over every face, min above max without
    /// valid samples.
    glm::vec2 HeightRange() const;

    /// Region of the node at (`x`, `y`) on `level` of face `face`
    /// (`LeafCells()` 2^`level` fine quads across); empty for nodes
    /// without valid samples.
    CubeSphereRegion NodeRegion(std::uint32_t face,
                                std::uint32_t level, std::uint32_t x,
                                std::uint32_t y,
                                float exaggeration) const;

    /// Radius of a sphere inside every triangle drawn, at any level:
    /// the lowest height on the body, less the sag of the coarsest
    /// level's flat triangles below the curved surface. Nothing
    /// behind it is visible.
    double OccluderRadius(float exaggeration) const;

    /// Smallest first range for which selection keeps neighbours
    /// within one level of each other and every shared edge is fully
    /// morphed on the finer side and unmorphed on the coarser side:
    /// the diameter of every level-L node's region is at most
    /// `morphStart * range[L]`.
    /// @param morphStart Fraction of each level's distance band
    /// before morphing starts, in (0, 1).
    double MinFirstRange(float exaggeration, float morphStart) const;

    /// Per-level ranges as `LodQuadtree::Ranges`; the roots' range is
    /// infinite.
    std::vector<LodRange> Ranges(double firstRange,
                                 float morphStart) const;

    /// Appends the patches to draw for an eye at `eye` in the body
    /// frame, with each patch's face in `LodPatch::face`. Nodes
    /// outside `frustum`, which is relative to the eye, are skipped;
    /// pass nullptr to keep everything in range. With `cullHorizon`,
    /// nodes hidden behind a sphere of `OccluderRadius` are skipped
    /// too.
    void Select(const glm::dvec3& eye,
                std::span<const LodRange> ranges,
                const Frustum* frustum, float exaggeration,
                bool cullHorizon, std::vector<LodPatch>& out) const;

    /// Appends level-0 patches covering every face, for drawing at
    /// full resolution.
    void SelectAll(std::vector<LodPatch>& out) const;

    /// Triangles drawn for `patches`.
    std::uint64_t
    TriangleCount(std::span<const LodPatch> patches) const;

  private:
    /// Checks the shape and sets up the faces' trees, every node
    /// empty.
    void Shape(double radius, std::uint32_t faceCells,
               std::uint32_t leafCells);

    bool Hidden(const CubeSphereRegion& region, const glm::dvec3& eye,
                const Frustum* frustum, double occluder) const;
    bool SelectNode(std::uint32_t face, std::uint32_t level,
                    std::uint32_t x, std::uint32_t y,
                    const glm::dvec3& eye,
                    std::span<const LodRange> ranges,
                    const Frustum* frustum, float exaggeration,
                    double occluder,
                    std::vector<LodPatch>& out) const;
    void EmitQuarters(std::uint32_t face, std::uint32_t level,
                      std::uint32_t x, std::uint32_t y,
                      const glm::dvec3& eye, const Frustum* frustum,
                      float exaggeration, double occluder,
                      std::vector<LodPatch>& out) const;
    void EmitChildArea(std::uint32_t face, std::uint32_t level,
                       std::uint32_t childX, std::uint32_t childY,
                       const glm::dvec3& eye, const Frustum* frustum,
                       float exaggeration, double occluder,
                       std::vector<LodPatch>& out) const;

    /// The (min, max) height of node (`x`, `y`) on `level`: a
    /// refined node's, its data leaf's.
    const glm::vec2& NodeHeights(std::uint32_t face,
                                 std::uint32_t level, std::uint32_t x,
                                 std::uint32_t y) const;

    double radius_ = 0.0;
    std::uint32_t faceCells_ = 0;
    std::uint32_t refinement_ = 0;
    std::array<LodQuadtree, kFaces> trees_;
};

}  // namespace ernie
