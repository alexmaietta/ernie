// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/buffer.h"
#include "ernie/device/compute_interop.h"
#include "ernie/device/pipeline.h"
#include "ernie/raster/drawable.h"
#include "ernie/raster/renderer.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ernie {

class MeshDrawable;

/// A point cloud on the host, as UsdGeomPoints gives it: a position
/// per point, and a width (the diameter of the sphere drawn) and
/// linear RGB colour per point or for all of them.
struct PointsData {
    std::vector<float> points;  ///< x, y, z per point.
    std::vector<float> widths;  ///< One per point, or empty for
                                ///< `width` for all of them.
    float width = 1.0f;
    std::vector<float>
        colors;  ///< r, g, b per point, or empty for the draw's tint.
};

/// A point cloud on the compute device (CUDA or HIP), as contiguous
/// float32 tensors: positions (n, 3), and widths (n) and colours (n,
/// 3), each empty for the constant ones.
struct DevicePointsData {
    Tensor points;
    Tensor widths;
    float width = 1.0f;
    Tensor colors;
    /// Bounds of the points in the prim's frame, widths included, as
    /// USD's `extent`; without them the points and widths are copied
    /// to the host to bound them.
    std::optional<std::pair<glm::vec3, glm::vec3>> extent;
    std::uintptr_t stream =
        1;  ///< The stream the arrays are written on.
};

/// One draw of a point cloud.
struct PointsItem {
    std::uint32_t points = 0;   ///< The cloud's handle.
    glm::dmat4 transform{1.0};  ///< World transform, in double; drawn
                                ///< relative to the origin.
    std::uint32_t material = RasterRenderer::kDefaultMaterial;
    glm::vec3 tint{0.5f};  ///< Every point's colour without colours
                           ///< of their own.
    std::uint32_t primId = 1;  ///< Written to the PrimId render var;
                               ///< the point's index to InstanceId.
    /// Its light link group (`LinkGroups`).
    std::uint32_t linkGroup = 0;
    std::int32_t world =
        -1;  ///< The world it belongs to (`tokens::kWorld`), -1 for
             ///< every one.
    /// Under an instance posed on the GPU (a cloud on a robot's
    /// link): its instancer in the meshes' drawable
    /// (`PointsDrawable::SetAnchors`), 0 for none, and row. The
    /// cloud's transform is then `transform`, the instance's pose,
    /// then `below`, composed on the GPU each frame.
    std::uint32_t anchorInstancer = 0;
    std::uint32_t anchorRow = 0;
    glm::dmat4 below{1.0};
    /// With an anchor: bounds of the cloud wherever the instance
    /// goes, in `transform`'s frame.
    std::pair<glm::vec3, glm::vec3> anchorBounds{glm::vec3(0.0f),
                                                 glm::vec3(0.0f)};
};

/// Point clouds drawn as spheres: a `RasterDrawable` for particles,
/// such as granular or MPM simulations, whose points can stay on the
/// GPU. Each point is a camera-facing quad covering its sphere's
/// silhouette, and the fragment shader intersects the pixel's ray
/// with the sphere, discarding misses and writing the hit's depth,
/// normal and ids (the point's index as the InstanceId var), so the
/// spheres are exact at any distance, sort against everything else
/// and are lit and shadowed by the renderer like any surface. Shadow
/// passes draw them as quads facing the light, likewise exact for the
/// sun's orthographic maps.
///
/// Positions are double-buffered like `MeshDrawable`'s instance
/// arrays: the first write of a frame goes into the other half, so
/// motion-blurred frames interpolate each point from where it was.
///
/// Reference: C. Sigg, T. Weyrich, M. Botsch and M. Gross, "GPU-Based
/// Ray-Casting of Quadratic Surfaces", Eurographics Symposium on
/// Point-Based Graphics, 59-65, 2006.
/// https://doi.org/10.2312/SPBG/SPBG06/059-065
class PointsDrawable : public RasterDrawable {
  public:
    /// Point clouds drawn by `renderer`, which must outlive the
    /// drawable.
    /// @throws std::runtime_error if its pipelines cannot be created.
    explicit PointsDrawable(RasterRenderer& renderer);
    ~PointsDrawable() override;
    PointsDrawable(const PointsDrawable&) = delete;
    PointsDrawable& operator=(const PointsDrawable&) = delete;

    /// Adds a cloud and returns its handle.
    /// @throws std::invalid_argument as `Set`.
    std::uint32_t Add(PointsData data);

    /// Replaces a cloud's arrays.
    /// @throws std::invalid_argument for an unknown cloud, points
    /// whose size is not a multiple of 3, widths or colours that do
    /// not match them, or a width that is not positive.
    void Set(std::uint32_t points, PointsData data);

    /// Replaces a cloud's arrays with tensors on the compute device,
    /// copied device to device at the next `Prepare`.
    /// @throws std::invalid_argument as `Set`, for tensors not on the
    /// compute device or without `SetComputeInterop`.
    void Set(std::uint32_t points, const DevicePointsData& data);

    /// Maps a cloud's positions for writing in place on the compute
    /// device, as `MeshDrawable::MapPoints`: a float32 (points, 3)
    /// tensor over the half of the drawable's own buffer the next
    /// frame draws, holding the positions as they are now; the device
    /// `Set` takes it back without copying.
    /// @throws std::invalid_argument for an unknown cloud or without
    /// `SetComputeInterop`.
    Tensor MapPositions(std::uint32_t points,
                        std::uintptr_t stream = 1);

    /// @throws std::invalid_argument for an unknown cloud.
    void Remove(std::uint32_t points);

    /// Enables clouds on the compute device; `compute` must outlive
    /// the drawable.
    void SetComputeInterop(ComputeInterop* compute) {
        compute_ = compute;
    }

    /// Sets the drawable whose instances items are posed under
    /// (`PointsItem::anchorInstancer`), which prepares before this
    /// one each frame and must outlive it.
    void SetAnchors(const MeshDrawable* meshes) {
        anchors_ = meshes;
    }

    /// Sets the draws of the next frames.
    /// @throws std::invalid_argument for an unknown cloud or
    /// material, or items posed under instances without `SetAnchors`.
    void SetItems(std::vector<PointsItem> items);

    /// Starts a frame's motion, as `MeshDrawable::SettleMotion`:
    /// clouds not written again before the next `Prepare` stand
    /// still.
    void SettleMotion();

    std::size_t CloudCount() const {
        return clouds_.size();
    }

    /// Points drawn per view.
    std::uint64_t PointCount() const;

    void SetOrigin(const glm::dvec3& origin) override {
        origin_ = origin;
    }
    const char* ProfileName() const override {
        return "Points";
    }
    std::optional<Bounds3> Bounds() const override;

    bool DrawsMinimal() const override {
        return true;
    }
    std::map<std::int32_t, Bounds3> WorldsBounds() const override;
    void Prepare(const FrameParams& params,
                 std::span<const ShadowCascade> cascades) override;
    /// Poses the items under instances
    /// (`PointsItem::anchorInstancer`) in their draws.
    void RecordPrepass(VkCommandBuffer commands) const override;
    /// With GPU culling, lists each cloud's chunks of points in each
    /// view and shadow view that sees them (points.slang's
    /// `cullMain`), which their draws then draw indirectly.
    void RecordCull(VkCommandBuffer commands) const override;
    /// The points the last finished frame's culling jobs kept to draw
    /// (whole chunks), in job order: per item, its views' and then
    /// its shadow views'. Reads the GPU back; for tests.
    std::vector<std::uint32_t> CulledPointCounts() const;
    void RecordShadow(VkCommandBuffer commands,
                      const RasterFrame& frame, std::size_t cascade,
                      const glm::mat4& lightViewProjection,
                      const ShadowCasters& casters) const override;
    void RecordColor(VkCommandBuffer commands,
                     const RasterFrame& frame, std::size_t view,
                     DrawPhase phase = DrawPhase::All) const override;
    /// Draws again the clouds view `view` drew that moved since the
    /// last frame (their transforms or positions), with the motion
    /// pipeline.
    void
    RecordMotion(VkCommandBuffer commands, const RasterFrame& frame,
                 std::size_t view,
                 const DescriptorResource& motion) const override;

  private:
    struct Cloud {
        PointsData data;  ///< Host arrays; for device clouds, what
                          ///< bounds them.
        DevicePointsData device;
        bool onDevice = false;
        std::size_t count = 0;
        glm::vec3 low{0.0f};  ///< Bounds of the points with their
                              ///< widths, in the prim's frame.
        glm::vec3 high{0.0f};
        /// Positions in two halves of `count` points (motion blur),
        /// widths and colours.
        std::shared_ptr<GpuBuffer> positions;
        std::shared_ptr<GpuBuffer> widths;
        std::shared_ptr<GpuBuffer> colors;
        std::uint32_t set = 0;
        bool moving = false;
        bool written = false;
        bool dirty = true;
        /// The view `MapPositions` last gave out, and whether the
        /// positions written are in it.
        Tensor mapped;
        bool inPlace = false;
    };

    void Check(const Cloud& cloud) const;
    /// Makes a cloud's buffers for its count, both halves of the
    /// positions holding them.
    void Allocate(Cloud& cloud);
    /// Writes a cloud's arrays into its buffers, the positions into
    /// the other half on the first write of a frame.
    void Write(Cloud& cloud);
    /// Draw `item`'s bounds relative to the origin. The G-buffer
    /// pipeline for a frame camera's views, or a pushbroom's.
    void CreateGBufferPipeline(bool pushbroom);
    /// Minimal shading's pipelines (`minimalPipelines_`), made the
    /// first time a frame with points shades minimally.
    void CreateMinimalPipelines();
    Bounds3 ItemBounds(std::size_t item) const;
    std::vector<DescriptorResource>
    Resources(const RasterFrame& frame, std::size_t item) const;
    /// Binding 15: the visible chunks of culled draws.
    DescriptorResource VisibleResource() const;

    RasterRenderer& renderer_;
    const VulkanContext& context_;
    ComputeInterop* compute_ = nullptr;
    PipelineLayout layout_;
    PipelineLayout
        motionLayout_;  ///< `layout_` and the MotionUniforms at 14.
    /// The MotionVector var's pipeline, made the first time a frame
    /// holds the var.
    mutable std::unique_ptr<GraphicsPipeline> motionPipeline_;
    /// The G-buffer pipelines for frame cameras and, once one is
    /// seen, pushbrooms.
    std::array<std::unique_ptr<GraphicsPipeline>, 2>
        gbufferPipelines_;
    bool pushbroom_ = false;  ///< Whether the last `Prepare`'s camera
                              ///< is a pushbroom.
    /// The last `Prepare`'s minimal shading, and its pipelines for
    /// frame cameras: shaded, then ids alone (`MinimalMode::None`).
    MinimalMode minimal_ = MinimalMode::Off;
    std::array<std::unique_ptr<GraphicsPipeline>, 2>
        minimalPipelines_;
    std::unique_ptr<GraphicsPipeline> shadowPipeline_;
    std::map<std::uint32_t, Cloud> clouds_;
    std::uint32_t nextHandle_ = 1;
    std::vector<PointsItem> items_;
    std::vector<glm::dmat4>
        previous_;  ///< Each item's transform at the last frame.
    glm::dvec3 origin_{0.0};
    std::unique_ptr<GpuBuffer>
        draws_;  ///< Per item, points.slang's PointsDraw.
    /// The items posed under instances: the drawable posing them, and
    /// this frame's records.
    const MeshDrawable* anchors_ = nullptr;
    std::unique_ptr<GpuBuffer> anchorRecords_;
    std::uint32_t anchored_ = 0;
    /// Per view of the last `Prepare`, the items it draws.
    std::vector<std::vector<std::uint32_t>> viewItems_;

    /// GPU culling: per item its jobs (points.slang's `cullJobs`),
    /// listed together, and per view and shadow view each item's job,
    /// -1 for a draw of every point. Clouds that move, pushbrooms and
    /// frames with motion blur draw every point.
    PipelineLayout cullLayout_;
    std::unique_ptr<ComputePipeline> cullPipeline_;
    std::unique_ptr<GpuBuffer> cullJobs_;
    std::unique_ptr<GpuBuffer> cullCommands_;
    std::unique_ptr<GpuBuffer> visibleChunks_;
    std::vector<glm::uvec4> jobs_;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> itemJobs_;
    std::vector<std::vector<std::int32_t>> viewJobs_;
    std::vector<std::vector<std::int32_t>> shadowJobs_;
    /// The draw constants' `instanceOffset` for job `job`: 1 + where
    /// its visible chunks start, 0 for -1.
    std::uint32_t InstanceOffset(std::int32_t job) const;
    /// Draws `item`'s points job `job` listed, or with -1 every one.
    void Draw(VkCommandBuffer commands, std::uint32_t item,
              std::int32_t job) const;
};

}  // namespace ernie
