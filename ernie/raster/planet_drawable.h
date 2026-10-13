// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/bindless.h"
#include "ernie/device/buffer.h"
#include "ernie/device/image.h"
#include "ernie/device/pipeline.h"
#include "ernie/geometry/cube_sphere_lod.h"
#include "ernie/geometry/frustum.h"
#include "ernie/raster/drawable.h"
#include "ernie/raster/planet_pages.h"
#include "ernie/raster/renderer.h"
#include "ernie/raster/terrain_drawable.h"
#include "ernie/util/raster_view.h"

#include <glm/glm.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
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
/// The body is centred at `TerrainParams::planetCentre`, its
/// body-fixed frame (z north, x towards longitude 0) turned by
/// `TerrainParams::planetRotation`. Heights live
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
/// map, a planar raster, is not applied; its albedo texture is, in
/// latitude and longitude.
///
/// Heights finer than the faces can be paged in as views need them
/// (`SetPages`): patches finer than the faces read the deepest page
/// made so far for where they lie, or every page they want in frames
/// that wait for them.
///
/// Options are `TerrainParams`, as for flat terrain.
class PlanetDrawable : public RasterDrawable {
  public:
    /// Grid quads along a level-0 LOD node; patches are half this.
    static constexpr std::uint32_t kLodLeafCells = 64;

    /// A planet drawn by `renderer`, which must outlive it, with the
    /// pipelines of `sharing` if given (bodies of `AdaptiveSpheres`),
    /// else its own.
    explicit PlanetDrawable(RasterRenderer& renderer,
                            const PlanetDrawable* sharing = nullptr);
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
    /// Given `slopes` (per vertex as `PlanetTile::slopes`, face by
    /// face), the normals come from them rather than the heights'
    /// differences; given `ranges` (per quad as
    /// `PlanetTile::ranges`), the LOD's bounds widen to them
    /// (`CubeSphereLod::SetCellRanges`), as finer pages (`SetPages`)
    /// need.
    void SetHeights(
        const glm::dvec3& radii,
        std::span<const RasterView, CubeSphereLod::kFaces> faces,
        std::span<const glm::vec2> slopes = {},
        std::span<const glm::vec2> ranges = {});

    /// Draws patches finer than the heights from `pages` (none: from
    /// the heights, bilinear), whose base faces must be the heights'.
    /// @throws std::invalid_argument for pages of other faces.
    void SetPages(std::unique_ptr<PlanetPages> pages);

    const PlanetPages* Pages() const {
        return pages_.get();
    }

    /// Whether frames wait for every page their patches want
    /// (`PlanetPages::Complete`) rather than drawing coarser pages
    /// until those arrive.
    void SetWaitForPages(bool wait) {
        waitForPages_ = wait;
    }

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

    /// `Offset` in the body-fixed axes.
    glm::dvec3 BodyOffset() const {
        return glm::transpose(params_.planetRotation) * Offset();
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

    /// `Bounds` under its world (`TerrainParams::world`), for a
    /// batch's maps fitted to worlds; none for every world's.
    std::map<std::int32_t, Bounds3> WorldsBounds() const override;

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

    /// The layout and pipelines, shared by bodies of one shape.
    struct Pipelines {
        explicit Pipelines(const VulkanContext& context);
        PipelineLayout layout;
        std::array<std::unique_ptr<GraphicsPipeline>, 2> gbuffer;
        std::unique_ptr<GraphicsPipeline> shadow;
    };

    /// The G-buffer pipeline for a frame camera's views, or a
    /// pushbroom's.
    void CreateGBufferPipeline(bool pushbroom);
    /// The body's point `body` (body-fixed, about its centre)
    /// relative to the origin, world axes.
    glm::dvec3 WorldOf(const glm::dvec3& body) const;
    void ReservePatches(std::size_t count);
    /// Finds the page each patch reads (`pageRefs_`), asking for
    /// those missing.
    void FindPages();
    /// Appends the patches `lod_` selects and their anchors.
    void Select(const glm::dvec3& eye,
                std::span<const LodRange> ranges,
                const Frustum* frustum);
    /// `ranges` for shadow view `view`: no finer than its texels.
    std::vector<LodRange>
    ShadowRanges(std::span<const LodRange> ranges,
                 const ShadowCascade& view) const;
    std::vector<DescriptorResource>
    Resources(const RasterFrame& frame, std::size_t view) const;
    void DrawPatches(VkCommandBuffer commands, DrawRange range,
                     const glm::mat4& light, const RasterFrame& frame,
                     bool shadow = false) const;

    RasterRenderer& renderer_;
    std::shared_ptr<Pipelines> pipelines_;
    bool pushbroom_ = false;
    TerrainParams params_;

    GpuBuffer uniforms_;
    std::unique_ptr<GpuImage> heights_;
    std::unique_ptr<GpuImage> gradients_;
    /// Their bindless heap slots that the uniforms name, as of the
    /// last `Prepare`.
    std::vector<BindlessHeap::Slot> heapSlots_;
    std::unique_ptr<GpuBuffer> patchBuffer_;
    std::unique_ptr<GpuBuffer> anchorBuffer_;
    /// Per patch, the page it reads (`PlanetPages::Ref`).
    std::unique_ptr<GpuBuffer> pageRefBuffer_;
    std::vector<PlanetPages::Ref> pageRefs_;
    std::unique_ptr<PlanetPages> pages_;
    /// Bound in place of the pages' pool without pages.
    GpuBuffer emptyPool_;
    bool waitForPages_ = false;
    std::uint64_t frames_ = 0;

    CubeSphereLod lod_;
    std::vector<LodPatch> patches_;
    std::vector<glm::vec4> anchors_;
    std::vector<DrawRange> mainRanges_;
    std::vector<DrawRange> cascadeRanges_;
    std::uint64_t triangles_ = 0;
    double firstRange_ = 0.0;
    bool frozen_ = false;
    /// About the body's centre, body-fixed axes.
    std::vector<glm::dvec3> lodEyes_;
    std::vector<Frustum> lodFrusta_;  ///< Relative to each eye, body
                                      ///< axes.
    /// Per view, whether it sees the planet's world.
    std::vector<bool> lodSees_;
    glm::dvec3 origin_{0.0};
    bool prepared_ = false;
    float minHeight_ = 0.0f;
    float maxHeight_ = 0.0f;
};

}  // namespace ernie
