// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/buffer.h"
#include "ernie/device/image.h"
#include "ernie/device/pipeline.h"
#include "ernie/geometry/cube_sphere_lod.h"
#include "ernie/geometry/frustum.h"
#include "ernie/raster/drawable.h"
#include "ernie/raster/renderer.h"
#include "ernie/raster/terrain_drawable.h"
#include "ernie/util/raster_view.h"

#include <glm/glm.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace ernie {

/// A whole planet's terrain drawn as a lit, shadowed triangle mesh: a
/// `RasterDrawable` that renders six cube-face heightfields projected
/// onto a sphere (`CubeSphereLod`).
///
/// The body is centred at the world's origin, with its body-fixed
/// frame the world's (z north, x towards longitude 0). Heights live
/// in a six-layer `R32_SFLOAT` image the vertex shader reads; each
/// frame `CubeSphereLod` selects patches at continuous levels of
/// detail, culled by the frustum and the horizon, and each patch is
/// one instance of a shared grid.
///
/// Planets need more precision than single floats hold, so the host
/// computes, in double, each patch's anchor (the radius-high point
/// above its first vertex) relative to the frame's origin
/// (`SetOrigin`), and the vertex shader adds the vertex's small
/// offset from it, computed from the difference of their grid angles.
/// Normals come from a per-vertex gradient texture with mips, so
/// shading is independent of the mesh level, and the material's
/// detail textures are projected triplanar in world space. Its albedo
/// map, a planar raster, is not applied.
///
/// Options are `TerrainParams`, as for flat terrain.
class PlanetDrawable : public RasterDrawable {
  public:
    /// Grid quads along a level-0 LOD node; patches are half this.
    static constexpr std::uint32_t kLodLeafCells = 64;

    /// A planet drawn by `renderer`, which must outlive it.
    explicit PlanetDrawable(RasterRenderer& renderer);
    ~PlanetDrawable() override;
    PlanetDrawable(const PlanetDrawable&) = delete;
    PlanetDrawable& operator=(const PlanetDrawable&) = delete;

    /// Uploads a body of radius `radius` whose six faces
    /// (`CubeSphereLod`, in face order) carry heights above it; the
    /// data is copied. Cells that are not finite or equal a face's
    /// nodata value draw nothing where they are the only valid
    /// heights of a LOD node, and otherwise read as 0.
    /// @throws std::invalid_argument as `CubeSphereLod`, with fewer
    /// than `kLodLeafCells` quads along a face, or for faces larger
    /// than the device's maximum texture size.
    void SetHeights(
        double radius,
        std::span<const RasterView, CubeSphereLod::kFaces> faces);

    /// As above on the ellipsoid of semi-axes `radii` along x, y and
    /// z (`CubeSphereLod::SetRadii`), heights along its normals.
    void SetHeights(
        const glm::dvec3& radii,
        std::span<const RasterView, CubeSphereLod::kFaces> faces);

    /// Drops the heights, so nothing is drawn. Waits for the device
    /// to go idle.
    void ClearHeights();

    bool HasHeights() const {
        return heights_ != nullptr;
    }

    void SetParams(const TerrainParams& params) {
        params_ = params;
    }

    const TerrainParams& Params() const {
        return params_;
    }

    const CubeSphereLod& Lod() const {
        return lod_;
    }

    /// Triangles drawn in every view by the last frame.
    std::uint64_t TriangleCount() const {
        return triangles_;
    }

    /// LOD patches drawn in view 0 by the last frame.
    std::size_t PatchCount() const {
        return mainRanges_.empty() ? 0 : mainRanges_.front().count;
    }

    /// First LOD range used by the last frame, in world units; 0 with
    /// LOD off.
    double LodFirstRange() const {
        return firstRange_;
    }

    /// GPU memory held for the heights and the drawable's buffers, in
    /// bytes.
    std::uint64_t GpuBytes() const;

    void SetOrigin(const glm::dvec3& origin) override {
        origin_ = origin;
    }

    /// The frame's origin relative to the planet's centre (its
    /// transform's translation, `TerrainParams::planetCentre`): the
    /// planet's points less it are the frame's.
    glm::dvec3 Offset() const {
        return origin_ - params_.planetCentre;
    }

    const char* ProfileName() const override {
        return "Planets";
    }

    /// The bounding box of the whole body at the current
    /// exaggeration, relative to the origin, for framing it; none
    /// without heights.
    std::optional<Bounds3> BodyBounds() const;

    /// What of the body can cast the shadows a view sees, relative to
    /// the origin, for fitting shadow cascades: within the distance
    /// its highest ground can be seen from (past the horizon, over
    /// the relief) of the origin and the last frame's LOD eyes, and
    /// within `BodyBounds`. Near the ground that is kilometres, not
    /// the planet; none without heights.
    std::optional<Bounds3> Bounds() const override;

    /// Writes the planet uniforms of every view and selects the LOD
    /// patches for every view (each from its own eye and frustum) and
    /// every cascade (from view 0's eye).
    void Prepare(const FrameParams& params,
                 std::span<const ShadowCascade> cascades) override;

    void RecordShadow(VkCommandBuffer commands,
                      const RasterFrame& frame, std::size_t cascade,
                      const glm::mat4& lightViewProjection,
                      const ShadowCasters& casters) const override;
    void RecordColor(VkCommandBuffer commands,
                     const RasterFrame& frame, std::size_t view,
                     DrawPhase phase = DrawPhase::All) const override;

  private:
    struct DrawRange {
        std::uint32_t first = 0;
        std::uint32_t count = 0;
    };

    /// The G-buffer pipeline for a frame camera's views, or a
    /// pushbroom's.
    void CreateGBufferPipeline(bool pushbroom);
    void ReservePatches(std::size_t count);
    /// Appends the patches `lod_` selects and their anchors.
    void Select(const glm::dvec3& eye,
                std::span<const LodRange> ranges,
                const Frustum* frustum);
    std::vector<DescriptorResource>
    Resources(const RasterFrame& frame, std::size_t view) const;
    void DrawPatches(VkCommandBuffer commands, DrawRange range,
                     const glm::mat4& light, const RasterFrame& frame,
                     bool shadow = false) const;

    RasterRenderer& renderer_;
    PipelineLayout layout_;
    std::array<std::unique_ptr<GraphicsPipeline>, 2>
        gbufferPipelines_;
    bool pushbroom_ = false;
    std::unique_ptr<GraphicsPipeline> shadowPipeline_;
    TerrainParams params_;

    GpuBuffer uniforms_;
    std::unique_ptr<GpuImage> heights_;
    std::unique_ptr<GpuImage> gradients_;
    std::unique_ptr<GpuBuffer> patchBuffer_;
    std::unique_ptr<GpuBuffer> anchorBuffer_;

    CubeSphereLod lod_;
    std::vector<LodPatch> patches_;
    std::vector<glm::vec4> anchors_;
    std::vector<DrawRange> mainRanges_;
    std::vector<DrawRange> cascadeRanges_;
    std::uint64_t triangles_ = 0;
    double firstRange_ = 0.0;
    bool frozen_ = false;
    std::vector<glm::dvec3> lodEyes_;  ///< In world coordinates.
    std::vector<Frustum> lodFrusta_;   ///< Relative to each eye.
    glm::dvec3 origin_{0.0};
    bool prepared_ = false;
    float minHeight_ = 0.0f;
    float maxHeight_ = 0.0f;
};

}  // namespace ernie
