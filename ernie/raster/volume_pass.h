// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/buffer.h"
#include "ernie/device/compute_interop.h"
#include "ernie/device/frame_uploads.h"
#include "ernie/device/image.h"
#include "ernie/device/pipeline.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/session/delegate.h"
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

    VolumePass(const VulkanContext& context,
               const ShaderCompiler& compiler);

    /// Takes the frame's volumes: loads the grids they name that are
    /// not loaded yet, and uploads the volumes, their materials and,
    /// for each of the frame's `viewWorlds.size()` views, the volumes
    /// it sees: those of its world (`FrameParams::ViewWorld`) and the
    /// shared ones, and with `viewProjections` (one per view,
    /// relative to `origin`; empty when the poses are on the GPU)
    /// only those its frustum reaches; and for each light shadow
    /// view, the world whose volumes it shadows with besides the
    /// shared ones (`lightViewWorlds`, `ShadowCasters::world`'s); as
    /// finely as `settings` say.
    /// @throws std::runtime_error if a grid's file cannot be read.
    void Set(std::span<const VolumeParams> volumes,
             const glm::dvec3& origin, const SpectralBands& bands,
             std::span<const std::int32_t> viewWorlds,
             std::span<const glm::mat4> viewProjections,
             std::span<const std::int32_t> lightViewWorlds,
             const VolumeSettings& settings, std::uint64_t frame,
             FrameUploads& uploads);

    /// The compute device whose arrays NanoVDBFields may hold, which
    /// copies their grids into the pool; Vulkan work reading the pool
    /// waits for it (`ComputeInterop::VulkanWait`).
    void SetComputeInterop(ComputeInterop* compute) {
        compute_ = compute;
    }

    /// Has `writer` write the values of the DenseField at `path` (its
    /// prim path), `shape` of them, every frame in `Prepare` before
    /// they are scattered into its grid; a null writer stops it.
    void SetFieldWriter(const std::string& path,
                        std::array<std::uint32_t, 3> shape,
                        FieldWriter writer);

    /// Whether any volume is drawn this frame.
    bool Active() const {
        return active_;
    }

    /// The union of the last `Set`'s volume instances' world boxes,
    /// relative to its origin; none without volumes. Volumes write no
    /// depth, so what fits to the depths seen (shadow cascades)
    /// widens by these.
    const std::optional<Bounds3>& Bounds() const {
        return bounds_;
    }

    /// Each of the last `Set`'s volume instances' world boxes,
    /// relative to its origin, which the virtual shadow map marks its
    /// pages through.
    const std::vector<Bounds3>& Boxes() const {
        return boxes_;
    }

    /// Texels across a light shadow view's volume shadow map, as the
    /// last `Set`'s settings say.
    std::uint32_t ShadowSize() const {
        return settings_.shadowResolution;
    }

    /// Records what the march and shading read of the volumes before
    /// the views draw, once the light shadow views are known: the
    /// pass's buffers' addresses, then the volumes' shadow maps
    /// (volume.slang's `volumeShadowMain`), one per light shadow view
    /// of `shadowViews`, from the first `lightCount` lights of
    /// `lights` and their views' view-projections `lightViews` (with
    /// the light patterns its pipeline compiles with, as the
    /// march's). Whether to build them again, and whether the march
    /// reprojects (temporal amortisation), the GPU decides from
    /// these, the lights' IES `profiles` and `timing` too.
    void Prepare(VkCommandBuffer commands, const GpuBuffer& lights,
                 const GpuBuffer& profiles, const GpuBuffer& timing,
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
        std::string file;  ///< Or a NanoVDBField's path, `array`.
        std::string name;
        int index = 0;
        bool array = false;
        auto operator<=>(const GridKey&) const = default;
    };
    struct LoadedGrid {
        NanoVdbGrid grid;  ///< Without its buffer when on a device.
        std::filesystem::file_time_type written;
        std::uint64_t version = 0;  ///< An array's.
        /// An array on a compute device: it, where the grid starts in
        /// it, the stream it was written on, and whether the grid
        /// still has to be copied into its place in the pool.
        Tensor device;
        std::uint64_t deviceOffset = 0;
        std::uintptr_t stream = 1;
        bool copy = false;
        /// A DenseField's on a device: `device` its values, the grid
        /// every voxel of their shape (`layout`), which a scatter on
        /// the GPU fills when `scatter` (`Prepare`).
        bool dense = false;
        double voxelSize = 1.0;
        DenseNanoVdbLayout layout;
        bool scatter = false;
        /// A DenseField a writer fills each frame (`SetFieldWriter`):
        /// its path, and its values' shape.
        std::string writer;
        glm::uvec3 shape{0};
        std::uint64_t offset = 0;  ///< In the pool, bytes.
        bool used = false;
        /// Its distance grid (volume.slang's `stepAfterEmptyBricks`):
        /// its first word in `distances_` (none for a grid without
        /// one), first brick and bricks along each axis, and whether
        /// it is due a build since the grid changed.
        std::uint32_t distances = 0xFFFFFFFFu;
        glm::ivec3 firstBrick{0};
        glm::ivec3 bricks{0};
        bool distanceStale = true;
    };

    const LoadedGrid* Load(const VolumeFieldParams& field,
                           const std::string& prim);
    void RebuildPool();
    void CopyDeviceGrid(LoadedGrid& loaded);
    /// Queues the dense grids due a scatter: their values copied on
    /// the device into `staging_`, for `Prepare` to scatter.
    void StageDenseGrids(FrameUploads& uploads);

    /// A dense grid's scatter from `staging_` into the pool, in
    /// 4-byte words: where its values start in `staging_`, its first
    /// leaf in the pool, the words of a leaf and of a leaf's values,
    /// its shape and its leaves along each axis.
    struct Scatter {
        std::uint32_t values = 0;
        std::uint32_t firstLeaf = 0;
        std::uint32_t leafWords = 0;
        std::uint32_t valuesWord = 0;
        glm::uvec3 shape{0};
        glm::uvec3 leaves{0};
        /// The field's writer, which writes its values into
        /// `staging_` first; none for values copied there.
        FieldWriter writer;
    };

    const VulkanContext& context_;
    const ShaderCompiler& compiler_;
    ComputeInterop* compute_ = nullptr;
    PipelineLayout layout_;
    mutable std::unique_ptr<ComputePipeline> pipelines_[2];
    /// The light patterns each pipeline was compiled with.
    mutable std::size_t pipelinePatterns_[2] = {};
    /// The band groups each pipeline was compiled for, and the
    /// frame's.
    mutable std::uint32_t pipelineGroups_[2] = {};
    std::uint32_t bandGroups_ = 16;
    std::map<GridKey, LoadedGrid> grids_;
    bool poolStale_ = true;
    std::size_t poolBytes_ = 0;
    std::unique_ptr<GpuBuffer> pool_;
    std::unique_ptr<GpuBuffer> volumes_;
    std::unique_ptr<GpuBuffer> materials_;
    std::unique_ptr<GpuBuffer> runs_;
    std::unique_ptr<GpuBuffer> lists_;
    std::unique_ptr<GpuBuffer> lightViewWorlds_;
    /// volume.slang's VolumeResources: the buffers above by address.
    std::unique_ptr<GpuBuffer> resources_;
    /// What `Set` gives `Prepare` to write into `resources_`: the
    /// buffers' addresses and the volumes' count.
    std::array<VkDeviceAddress, 5> addresses_{};
    std::uint32_t volumeCount_ = 0;
    PipelineLayout shadowLayout_;
    mutable std::unique_ptr<ComputePipeline> shadowPipeline_;
    mutable std::size_t shadowPipelinePatterns_ = 0;
    /// Dense grids' values on their way to the pool, and the scatters
    /// due (`denseScatterMain`).
    std::unique_ptr<GpuBuffer> staging_;
    bool stagingShared_ = false;
    mutable std::vector<Scatter> scatters_;
    /// The fields' writers by path, with their shapes.
    std::map<std::string,
             std::pair<std::array<std::uint32_t, 3>, FieldWriter>>
        writers_;
    PipelineLayout scatterLayout_;
    mutable std::unique_ptr<ComputePipeline> scatterPipeline_;
    mutable std::size_t scatterPipelinePatterns_ = 0;
    /// The lighting caches (`lightCacheMain`): their cells, all and
    /// the most of one instance, the buffer they fill, and its
    /// pipeline.
    std::optional<Bounds3> bounds_;
    std::vector<Bounds3> boxes_;
    /// The grids' distance grids (`PlanDistanceGrids`), their words,
    /// the builds due (`Prepare`), a build's scratch and its
    /// pipelines.
    struct DistanceBuild {
        std::uint32_t grid = 0;  ///< In the pool, bytes.
        std::uint32_t type = 0;
        std::uint32_t distances = 0;
        glm::ivec3 firstBrick{0};
        glm::ivec3 bricks{0};
    };
    void PlanDistanceGrids();
    std::unique_ptr<GpuBuffer> distances_;
    std::uint64_t distanceWords_ = 0;
    mutable std::vector<DistanceBuild> distanceBuilds_;
    std::unique_ptr<GpuBuffer> distanceScratch_;
    PipelineLayout distanceLayout_;
    mutable std::unique_ptr<ComputePipeline> occupancyPipeline_;
    mutable std::unique_ptr<ComputePipeline> distancePipeline_;
    mutable std::size_t distancePipelinePatterns_ = 0;
    std::uint64_t cacheCells_ = 0;
    std::uint64_t cacheLargest_ = 0;
    mutable std::unique_ptr<GpuBuffer> cache_;
    PipelineLayout cacheLayout_;
    mutable std::unique_ptr<ComputePipeline> cachePipeline_;
    mutable std::size_t cachePipelinePatterns_ = 0;
    mutable std::unique_ptr<GpuImage> shadowMap_;
    mutable std::uint32_t shadowViews_ = 0;
    /// Whether the caches and shadow maps are built again
    /// (`volumeChangesMain`): what they last read, the two indirect
    /// dispatches it writes, the parts the host changed since (a bit
    /// each, caches 1 and maps 2: the grids, the settings, the
    /// buffers) and the counts the last comparison laid out.
    PipelineLayout changeLayout_;
    mutable std::unique_ptr<ComputePipeline> changePipeline_;
    mutable std::size_t changePipelinePatterns_ = 0;
    mutable std::unique_ptr<GpuBuffer> previous_;
    std::unique_ptr<GpuBuffer> rebuild_;
    /// The last frames' verdicts whether what the volumes show
    /// changed at a stroke and whether their grids were written again
    /// (`pass.rebuild[6]` and `[7]`), two words each by frame, which
    /// the host reads a few frames on: where they all would march in
    /// full, the march is one dispatch rather than amortised's two.
    std::unique_ptr<GpuBuffer> changes_;
    /// Starts discrete (volume.slang's `kHostDiscrete`).
    mutable std::uint32_t hostChanged_ = 3 | 16;
    /// The frames in a row, this one's included, that wrote grids
    /// again.
    mutable std::uint32_t gridsWritten_ = 0;
    mutable std::array<std::uint32_t, 4> compared_{};
    std::uint32_t materialCount_ = 0;
    /// Temporal amortisation's histories (volume.slang's
    /// `historyAt`), by a dispatch's first view: this frame's and the
    /// last's, which is current, the images' size, views and bands
    /// they hold, the frame last written, and whether the last
    /// frame's holds for this one.
    struct History {
        std::unique_ptr<GpuBuffer> buffers[2];
        std::uint32_t now = 0;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::uint32_t views = 0;
        std::uint32_t bands = 0;
        std::uint64_t frame = 0;
        bool held = false;
    };
    mutable std::map<std::uint32_t, History> histories_;
    std::uint64_t frame_ = 0;
    std::uint32_t bandCount_ = 0;
    /// The frame's origin, and it less the last's as floats
    /// (`VolumeResources::originShift`).
    glm::dvec3 origin_{0.0};
    glm::vec3 originShift_{0.0f};
    std::uint32_t views_ = 0;
    std::uint32_t seed_ = 0;
    VolumeSettings settings_;
    bool active_ = false;
};

}  // namespace ernie
