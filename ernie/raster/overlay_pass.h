// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/buffer.h"
#include "ernie/device/compute_interop.h"
#include "ernie/device/frame_uploads.h"
#include "ernie/device/pipeline.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/device/vulkan_context.h"
#include "ernie/scene/overlay.h"

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ernie {

class MeshDrawable;

/// Where an overlay anchor (`OverlayStyle::anchor`) is this frame:
/// its world transform `frame`, then, for a prim posed on the GPU,
/// the pose of instance `slot` of the meshes' pools
/// (`MeshDrawable::AnchorSlot`, ~0 for none), then `below`.
struct OverlayAnchorPose {
    glm::dmat4 frame{1.0};
    std::uint32_t slot = 0xFFFFFFFFu;
    glm::dmat4 below{1.0};
    /// The pools the instance's pose fills
    /// (`MeshDrawable::AnchorOps`).
    std::uint32_t ops = 0;
    /// The prim's world (`tokens::kWorld`), the world of the items
    /// under it without worlds of their own; -1 for none.
    std::int32_t world = -1;
};

/// Poses the prim at a path for this frame (`OverlayAnchorPose`).
using OverlayAnchorResolver =
    std::function<OverlayAnchorPose(const Path&)>;

/// Draws overlays (`Overlay`: debug lines, triangles, markers, text
/// and panels) over a view's display image after the sensor filters
/// (`overlay.slang`): segments as quads of their width in pixels,
/// sprites at their size in pixels, each tested against the view's
/// depth by its `OverlayDepth`, blended over the image. Items posed
/// under prims are drawn by their anchors' transforms this frame,
/// those of prims posed on the GPU computed there. Screen items go
/// over the image as it is, after the lens of a cube camera.
class OverlayPass {
  public:
    /// Pipelines for LdrColor's format and `presentFormat`.
    /// @throws std::runtime_error if they cannot be created.
    OverlayPass(const VulkanContext& context,
                const ShaderCompiler& compiler, FrameUploads& uploads,
                VkFormat presentFormat);
    ~OverlayPass();
    OverlayPass(const OverlayPass&) = delete;
    OverlayPass& operator=(const OverlayPass&) = delete;

    /// Stages the visible items of `overlays`, relative to the
    /// frame's `origin`, for the next frame's `Record`s; uploads only
    /// when an overlay or the origin changed. Host items are grouped
    /// by world, so each view of a batched camera draws only the
    /// shared ones and its world's. Batches' arrays (`OverlayBatch`)
    /// are copied as they are into the batch buffer, device to device
    /// from the compute device (GPU ones every frame), and expanded
    /// into items as they are drawn. Batches with worlds are sorted
    /// by world on the GPU for `worlds` worlds (the frame's views'
    /// worlds are below it). Anchors are posed every frame by
    /// `resolve` (`SetAnchorResolver`). With `pushbroom`, first
    /// compiles the pipelines for a pushbroom camera's targets,
    /// unless it has.
    /// @throws std::invalid_argument for batch arrays on a GPU
    /// without `SetComputeInterop`, or on another device, or items
    /// posed under prims without a resolver.
    void Prepare(std::span<const Overlay* const> overlays,
                 const glm::dvec3& origin, bool pushbroom = false,
                 std::uint32_t worlds = 0);

    /// Connects the compute device batches' GPU arrays are copied
    /// from; the frames must wait for its copies
    /// (`ComputeInterop::VulkanWait`).
    void SetComputeInterop(ComputeInterop* compute);

    /// Poses the anchors through `resolve` (`PoseAnchors`), the
    /// meshes' anchor pass posing those under instances
    /// (`MeshDrawable::RecordAnchors`).
    void SetAnchorResolver(OverlayAnchorResolver resolve,
                           const MeshDrawable* meshes);

    /// Poses the prepared anchors for this frame, relative to its
    /// `origin`, once the meshes are prepared (their instances'
    /// slots, `MeshDrawable::AnchorSlot`): the host's transforms, and
    /// records for the first `Record` to pose those under instances.
    /// @throws std::invalid_argument for anchors without a resolver.
    void PoseAnchors(const glm::dvec3& origin);

    /// Array `array` (`Overlay::ReplaceBatchArray`'s names) of batch
    /// `batch` of `overlay`, as last prepared, as a tensor on the
    /// compute device over the batch buffer itself: writing it on
    /// `stream` (which first waits for the frames submitted) writes
    /// what the next frames draw, with no copy, once it replaces the
    /// batch's array. Map again after the overlay changes.
    /// @throws std::invalid_argument without compute interop, or for
    /// an overlay, batch or array not prepared.
    Tensor MapBatchArray(const Overlay& overlay, std::size_t batch,
                         std::string_view array,
                         std::uintptr_t stream = 1);

    /// Whether the last `Prepare` left nothing to draw.
    bool Empty() const {
        return groups_.empty() && batchDraws_.empty();
    }

    /// Whether the last `Prepare` left screen items (`Overlay::Text`)
    /// to draw.
    bool HasScreenItems() const {
        return hasScreen_;
    }

    /// Which of the prepared items one `Record` draws.
    enum class Layer : std::uint8_t {
        All,     ///< Every item.
        World,   ///< Items placed in the world, such as a cube
                 ///< camera's faces draw.
        Screen,  ///< Items placed on the image.
    };

    /// What one `Record` draws into: a colour attachment of `format`
    /// and `extent` holding the view's display image (LdrColor's
    /// layer, or a present target), the view's uniforms, its depth
    /// (`depth`, layer `depthLayer`, in the shader-read layout), its
    /// world and clipping range.
    struct Target {
        VkImageView color = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
        VkExtent2D extent{};
        DescriptorResource uniforms;
        VkImageView depth = VK_NULL_HANDLE;
        std::uint32_t depthLayer = 0;
        std::int32_t world = -1;
        float nearPlane = 0.1f;
        float farPlane = 1000.0f;
        bool orthographic = false;  ///< Whether the depth is linear
                                    ///< (an orthographic view).
        bool pushbroom =
            false;  ///< Whether the view is a pushbroom's strip.
    };

    /// Draws the prepared items of `layer` over `target`'s image,
    /// keeping what it holds.
    /// @throws std::invalid_argument for a format without a pipeline.
    void Record(VkCommandBuffer commands, const Target& target,
                Layer layer = Layer::All) const;

    std::uint64_t GpuBytes() const;

  private:
    /// Grows `buffer` to `bytes` if needed and stages `bytes` of
    /// `data` into it.
    void Upload(std::unique_ptr<GpuBuffer>& buffer,
                std::span<const std::byte> data);

    /// Compiles the pipelines of both attachment formats for a frame
    /// camera's or a pushbroom's views.
    void CreatePipelines(bool pushbroom);

    /// The index in the anchor table of the prim at `path`, added if
    /// new.
    std::uint32_t AnchorIndex(const Path& path);

    /// Stages the prepared overlays' batches: their arrays into
    /// `batchData_`, their records and draws.
    void PrepareBatches(const glm::dvec3& origin);

    /// Copies the batches' GPU arrays into `batchData_` again, but
    /// those mapped there (`MapBatchArray`).
    void CopyGpuArrays();

    /// Sorts the batches with worlds by world into `sortData_` and
    /// writes their draws' arguments, and those of the batches drawn
    /// by their counts (`sortPending_`).
    void RecordSort(VkCommandBuffer commands) const;

    /// Poses the items of batches posed under pose arrays into their
    /// anchor rows (`PoseJob`), and those under instances.
    void RecordAnchorPoses(VkCommandBuffer commands) const;

    const VulkanContext& context_;
    const ShaderCompiler& compiler_;
    FrameUploads& uploads_;
    VkFormat presentFormat_;
    PipelineLayout layout_;
    /// Per attachment format and whether for a pushbroom: the
    /// segment, triangle and sprite pipelines, then the batches',
    /// then the numbers'.
    std::map<std::pair<VkFormat, bool>,
             std::array<std::unique_ptr<GraphicsPipeline>, 7>>
        pipelines_;
    std::array<std::unique_ptr<GpuBuffer>, 3> buffers_;
    /// Host items of one primitive (0 segments, 1 triangles, 2
    /// sprites) of one world (-1 shared) on the screen or not: a run
    /// of that primitive's buffer.
    struct Group {
        std::size_t primitive = 0;
        std::int32_t world = -1;
        bool screen = false;
        std::uint32_t first = 0;
        std::uint32_t count = 0;
    };
    /// In order of drawing: triangles, segments, sprites, and by
    /// world within each.
    std::vector<Group> groups_;
    bool hasScreen_ = false;
    /// One draw of a batch: the batch, its primitive (0 segments, 1
    /// triangles, 2 sprites, 3 numbers) and vertex count; for a batch
    /// sorted by world, its draw arguments for the shared items (each
    /// world's follow, 2 apart).
    struct BatchDraw {
        std::uint32_t batch = 0;
        std::size_t primitive = 0;
        std::uint32_t vertices = 0;
        bool sorted = false;
        std::uint32_t args = 0;
        bool screen = false;
        /// Drawn by its count on the GPU (`OverlayBatch::count`),
        /// from its draw arguments at `args`.
        bool counted = false;
    };
    /// A batch drawn by its count and not sorted by world: its first
    /// draw arguments, items, and vertices per item of its first and
    /// second draws (overlay_sort.slang's `countArgsMain`).
    struct CountJob {
        std::uint32_t batch = 0;
        std::uint32_t args = 0;
        std::uint32_t count = 0;
        std::array<std::uint32_t, 2> vertices{};
    };
    std::vector<CountJob> countJobs_;
    /// A batch posed under a pose array (`OverlayBatch::poses`): its
    /// items' anchor rows from `row` (counted from the first after
    /// the prims'), its items, where its poses and pose indices (plus
    /// 1, 0 for none) are in `batchData_`, and the poses' frame
    /// (overlay_sort.slang's `posesMain`).
    struct PoseJob {
        std::uint32_t batch = 0;
        std::uint32_t row = 0;
        std::uint32_t count = 0;
        std::uint32_t poses = 0;
        std::uint32_t indices = 0;
        glm::dmat4 frame{1.0};
        /// 1 + where its poses' worlds are in `batchData_`, or 0.
        std::uint32_t worlds = 0;
    };
    std::vector<PoseJob> poseJobs_;
    /// Anchor rows of the pose jobs, after the prims'.
    std::uint32_t poseRows_ = 0;
    std::unique_ptr<ComputePipeline> countPipeline_;
    std::unique_ptr<ComputePipeline> posePipeline_;
    /// The frame's origin, for the poses' frames (`PoseAnchors`).
    glm::dvec3 anchorOrigin_{0.0};
    std::vector<BatchDraw> batchDraws_;
    /// A batch sorted by world (overlay_sort.slang): its slots, first
    /// word of sort data, first draw arguments, items, and vertices
    /// per item of its first and second draws.
    struct SortJob {
        std::uint32_t batch = 0;
        std::uint32_t slots = 0;
        std::uint32_t first = 0;
        std::uint32_t args = 0;
        std::uint32_t count = 0;
        std::array<std::uint32_t, 2> vertices{};
    };
    std::vector<SortJob> sortJobs_;
    /// Whether the next `Record` sorts the batches first.
    mutable bool sortPending_ = false;
    std::uint32_t worlds_ = 0;
    std::unique_ptr<GpuBuffer> sortData_;
    std::unique_ptr<GpuBuffer> drawArgs_;
    PipelineLayout sortLayout_;
    /// overlay_sort.slang's clear, count, scan and scatter.
    std::array<std::unique_ptr<ComputePipeline>, 4> sortPipelines_;
    /// The batches' arrays, shared with the compute device when
    /// connected, and their `Batch` records (overlay.slang).
    std::shared_ptr<GpuBuffer> batchData_;
    std::unique_ptr<GpuBuffer> batchInfos_;
    /// Where each batch array went in `batchData_`, in floats: its
    /// overlay, batch and name, and what filled it.
    struct Placed {
        const Overlay* overlay = nullptr;
        std::size_t batch = 0;
        std::string_view name;
        const Tensor* array = nullptr;
        Tensor owned;  ///< In place of `array`: numbers' text.
        std::uintptr_t stream = 1;
        std::uint32_t first = 0;

        const Tensor& Source() const {
            return array != nullptr ? *array : owned;
        }
    };
    std::vector<Placed> placed_;
    /// `batchData_`'s first byte on the compute device, once mapped,
    /// to tell arrays mapped there (`MapBatchArray`).
    const std::byte* batchDataBase_ = nullptr;
    /// The font (overlay.slang's `glyphs`).
    std::unique_ptr<GpuBuffer> glyphs_;
    ComputeInterop* compute_ = nullptr;
    /// The anchor table: its prims, their indices, and per anchor its
    /// row-major transform this frame, relative to the frame's
    /// origin; the records of those under instances
    /// (`MeshDrawable::AppendAnchorRecord`).
    std::vector<Path> anchorPaths_;
    std::unordered_map<Path, std::uint32_t> anchorIndices_;
    std::unique_ptr<GpuBuffer> anchorMatrices_;
    /// Per anchor row its world (int32, -1 for none): the prims' from
    /// the host, the poses' from `posesMain`.
    std::unique_ptr<GpuBuffer> anchorWorlds_;
    /// Whether a sorted batch's worlds come from its anchors, which
    /// may move between worlds: sorted again every frame.
    bool sortsByAnchors_ = false;
    std::unique_ptr<GpuBuffer> anchorRecords_;
    std::uint32_t gpuAnchors_ = 0;
    mutable bool anchorsPending_ = false;
    OverlayAnchorResolver resolve_;
    const MeshDrawable* meshes_ = nullptr;
    /// What the buffers hold: each overlay and its version, and the
    /// origin.
    std::vector<std::pair<const Overlay*, std::uint64_t>> prepared_;
    glm::dvec3 preparedOrigin_{0.0};
};

}  // namespace ernie
