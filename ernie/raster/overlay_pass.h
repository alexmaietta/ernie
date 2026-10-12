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
#include <map>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace ernie {

/// Draws overlays (`Overlay`: debug lines, triangles, markers and
/// labels) over a view's display image after the sensor filters
/// (`overlay.slang`): segments as quads of their width in pixels,
/// sprites at their size in pixels, each tested against the view's
/// depth by its `OverlayDepth`, blended over the image.
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
    /// when an overlay or the origin changed. Batches' arrays
    /// (`OverlayBatch`) are copied as they are into the batch buffer,
    /// device to device from the compute device, and expanded into
    /// items as they are drawn. With `pushbroom`, first compiles the
    /// pipelines for a pushbroom camera's targets, unless it has.
    /// @throws std::invalid_argument for batch arrays on a GPU
    /// without `SetComputeInterop`, or on another device.
    void Prepare(std::span<const Overlay* const> overlays,
                 const glm::dvec3& origin, bool pushbroom = false);

    /// Connects the compute device batches' GPU arrays are copied
    /// from; the frames must wait for its copies
    /// (`ComputeInterop::VulkanWait`).
    void SetComputeInterop(ComputeInterop* compute);

    /// Whether the last `Prepare` left nothing to draw.
    bool Empty() const {
        return counts_ == std::array<std::uint32_t, 3>{} &&
               batchDraws_.empty();
    }

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

    /// Draws the prepared items over `target`'s image, keeping what
    /// it holds.
    /// @throws std::invalid_argument for a format without a pipeline.
    void Record(VkCommandBuffer commands, const Target& target) const;

    std::uint64_t GpuBytes() const;

  private:
    /// Grows `buffer` to `bytes` if needed and stages `bytes` of
    /// `data` into it.
    void Upload(std::unique_ptr<GpuBuffer>& buffer,
                std::span<const std::byte> data);

    /// Compiles the pipelines of both attachment formats for a frame
    /// camera's or a pushbroom's views.
    void CreatePipelines(bool pushbroom);

    /// Stages the prepared overlays' batches: their arrays into
    /// `batchData_`, their records and draws.
    void PrepareBatches(const glm::dvec3& origin);

    const VulkanContext& context_;
    const ShaderCompiler& compiler_;
    FrameUploads& uploads_;
    VkFormat presentFormat_;
    PipelineLayout layout_;
    /// Per attachment format and whether for a pushbroom: the
    /// segment, triangle and sprite pipelines, then the batches'.
    std::map<std::pair<VkFormat, bool>,
             std::array<std::unique_ptr<GraphicsPipeline>, 6>>
        pipelines_;
    std::array<std::unique_ptr<GpuBuffer>, 3> buffers_;
    std::array<std::uint32_t, 3> counts_{};
    /// One draw of a batch: the batch, its primitive (0 segments, 1
    /// triangles, 2 sprites) and vertex count.
    struct BatchDraw {
        std::uint32_t batch = 0;
        std::size_t primitive = 0;
        std::uint32_t vertices = 0;
    };
    std::vector<BatchDraw> batchDraws_;
    /// The batches' arrays, shared with the compute device when
    /// connected, and their `Batch` records (overlay.slang).
    std::shared_ptr<GpuBuffer> batchData_;
    std::unique_ptr<GpuBuffer> batchInfos_;
    ComputeInterop* compute_ = nullptr;
    /// What the buffers hold: each overlay and its version, and the
    /// origin.
    std::vector<std::pair<const Overlay*, std::uint64_t>> prepared_;
    glm::dvec3 preparedOrigin_{0.0};
};

}  // namespace ernie
