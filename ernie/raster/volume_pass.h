// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/buffer.h"
#include "ernie/device/frame_uploads.h"
#include "ernie/device/image.h"
#include "ernie/device/pipeline.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/session/frame_params.h"
#include "ernie/util/nanovdb.h"
#include "ernie/util/spectrum.h"

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace ernie {

class LightPatternRegistry;
class RenderTargets;

/// The volume pass (volume.slang): after shading, each pixel's ray
/// marched through the Volumes its view sees, dimming the radiance
/// behind them and adding what they emit.
///
/// Grids load once per file and grid (again when the file changes)
/// into one buffer the GPU walks in place, shared by every volume and
/// instance naming them, so a batch of worlds each with its own
/// volume holds each grid once. Each view marches only the volumes
/// of its world and the shared ones that its frustum reaches.
class VolumePass {
  public:
    /// Where the views' blocks of the shading uniforms hold what the
    /// pass reads, in bytes.
    struct ViewLayout {
        std::uint32_t stride = 0;
        std::uint32_t invViewProj = 0;
        std::uint32_t strip = 0;
        std::uint32_t range = 0;
        std::uint32_t size = 0;  ///< Its w the light count.
        /// The light mask, the world's slot in a batch's distant
        /// light maps and the first light.
        std::uint32_t view = 0;
        std::uint32_t lightShadow = 0;  ///< Its z the maps' size.
        std::uint32_t atmosphere = 0;   ///< Two float4s.
    };

    /// What the march takes from the shading pass: its resources as
    /// it recorded them (shading.slang's ShadingResources: the views'
    /// uniforms, the lights, their shadow maps and the atmosphere's
    /// transmittance, by address and heap index), and the light
    /// patterns lasers draw with (light_pattern's data and registry).
    struct Shading {
        const GpuBuffer* resources = nullptr;
        const GpuBuffer* patternData = nullptr;
        const LightPatternRegistry* patterns = nullptr;
    };

    /// The most steps a ray takes through one volume.
    static constexpr std::uint32_t kMaxSteps = 256;
    /// Steps per voxel of the density grid, at most.
    static constexpr double kStepsPerVoxel = 2.0;

    VolumePass(const VulkanContext& context,
               const ShaderCompiler& compiler);

    /// Takes the frame's volumes: loads the grids they name that are
    /// not loaded yet, and uploads the volumes, their materials and,
    /// for each of the frame's `viewWorlds.size()` views, the volumes
    /// it sees: those of its world (`FrameParams::ViewWorld`) and the
    /// shared ones, and with `viewProjections` (one per view,
    /// relative to `origin`; empty when the poses are on the GPU)
    /// only those its frustum reaches.
    /// @throws std::runtime_error if a grid's file cannot be read.
    void Set(std::span<const VolumeParams> volumes,
             const glm::dvec3& origin, const SpectralBands& bands,
             std::span<const std::int32_t> viewWorlds,
             std::span<const glm::mat4> viewProjections,
             std::uint64_t frame, FrameUploads& uploads);

    /// Whether any volume is drawn this frame.
    bool Active() const {
        return active_;
    }

    /// Texels across a light shadow view's volume shadow map.
    static constexpr std::uint32_t kShadowSize = 256;

    /// Records what the march and shading read of the volumes before
    /// the views draw, once the light shadow views are known: the
    /// pass's buffers' addresses, then the volumes' shadow maps
    /// (volume.slang's `volumeShadowMain`), one per light shadow view
    /// of `shadowViews`, from the first `lightCount` lights of
    /// `lights` and their views' view-projections `lightViews` (with
    /// the light patterns its pipeline compiles with, as the
    /// march's).
    void Prepare(VkCommandBuffer commands, const GpuBuffer& lights,
                 const GpuBuffer& lightViews,
                 std::uint32_t lightCount,
                 const LightPatternRegistry& patterns,
                 const GpuBuffer& patternData,
                 std::uint32_t shadowViews) const;

    /// The volumes' shadow maps of the last `Prepare` (three layers
    /// per light shadow view), ready to sample; null without volumes
    /// or shadow views.
    const GpuImage* ShadowMap() const {
        return active_ && shadowViews_ > 0 ? shadowMap_.get()
                                           : nullptr;
    }

    /// Records the march over every view of `targets` (views
    /// `firstView` on in the shading uniforms), for a pinhole or a
    /// pushbroom camera; `sky` when the background holds radiance.
    /// Its pipelines compile again when light patterns are added.
    /// The caller adds the barrier after it.
    void Record(VkCommandBuffer commands, RenderTargets& targets,
                const Shading& shading, const ViewLayout& layout,
                std::uint32_t firstView, std::uint32_t width,
                std::uint32_t height, bool pushbroom, bool sky) const;

    /// The grids held, for tests: their count and bytes.
    std::size_t GridCount() const {
        return grids_.size();
    }
    std::size_t PoolBytes() const {
        return poolBytes_;
    }

  private:
    struct GridKey {
        std::string file;
        std::string name;
        int index = 0;
        auto operator<=>(const GridKey&) const = default;
    };
    struct LoadedGrid {
        NanoVdbGrid grid;
        std::filesystem::file_time_type written;
        std::uint64_t offset = 0;  ///< In the pool, bytes.
        bool used = false;
    };

    const LoadedGrid* Load(const VolumeFieldParams& field,
                           const std::string& prim);
    void RebuildPool();

    const VulkanContext& context_;
    const ShaderCompiler& compiler_;
    PipelineLayout layout_;
    mutable std::unique_ptr<ComputePipeline> pipelines_[2];
    /// The light patterns each pipeline was compiled with.
    mutable std::size_t pipelinePatterns_[2] = {};
    std::map<GridKey, LoadedGrid> grids_;
    bool poolStale_ = true;
    std::size_t poolBytes_ = 0;
    std::unique_ptr<GpuBuffer> pool_;
    std::unique_ptr<GpuBuffer> volumes_;
    std::unique_ptr<GpuBuffer> materials_;
    std::unique_ptr<GpuBuffer> runs_;
    std::unique_ptr<GpuBuffer> lists_;
    /// volume.slang's VolumeResources: the buffers above by address.
    std::unique_ptr<GpuBuffer> resources_;
    /// What `Set` gives `Prepare` to write into `resources_`: the
    /// buffers' addresses and the volumes' count.
    std::array<VkDeviceAddress, 4> addresses_{};
    std::uint32_t volumeCount_ = 0;
    PipelineLayout shadowLayout_;
    mutable std::unique_ptr<ComputePipeline> shadowPipeline_;
    mutable std::size_t shadowPipelinePatterns_ = 0;
    mutable std::unique_ptr<GpuImage> shadowMap_;
    mutable std::uint32_t shadowViews_ = 0;
    std::uint32_t views_ = 0;
    std::uint32_t seed_ = 0;
    bool active_ = false;
};

}  // namespace ernie
