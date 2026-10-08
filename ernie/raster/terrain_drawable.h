#pragma once

#include "ernie/device/buffer.h"
#include "ernie/device/compute_interop.h"
#include "ernie/device/frame_uploads.h"
#include "ernie/device/image.h"
#include "ernie/device/pipeline.h"
#include "ernie/geometry/frustum.h"
#include "ernie/geometry/lod_quadtree.h"
#include "ernie/raster/drawable.h"
#include "ernie/raster/renderer.h"
#include "ernie/util/raster_view.h"

#include <glm/glm.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ernie {

/// How terrain fragments are coloured.
enum class TerrainColorMode : std::uint32_t {
    Hillshade = 0,       ///< Material albedo lit by the material's reflectance model.
    Colormap = 1,        ///< Height colormap without shading.
    ShadedColormap = 2,  ///< Height colormap modulated by the lit surface.
};

/// Continuous level-of-detail options; see `LodQuadtree`.
struct LodSettings {
    bool enabled = true;       ///< Off draws every raster cell at full resolution.
    float detail = 4.0f;       ///< First LOD range in level-0 node widths; raised to the crack-free
                               ///< minimum (`LodQuadtree::MinFirstRange`) when lower.
    float morphStart = 0.67f;  ///< Fraction of each level's distance band before morphing starts,
                               ///< clamped to [0.05, 0.95].
    bool freeze = false;       ///< Keep selecting from the eye and frustum of the frame this was
                               ///< turned on, to inspect the selection from elsewhere.
    bool showLevels = false;   ///< Tint patches by LOD level.
};

/// Terrain options for the next frames.
struct TerrainParams {
    float exaggeration = 1.0f;  ///< Vertical exaggeration of heights.
    TerrainColorMode colorMode = TerrainColorMode::ShadedColormap;
    LodSettings lod;
    std::uint32_t material = RasterRenderer::kDefaultMaterial;  ///< Id in the renderer's table.
    std::uint32_t primId = 1;  ///< Written to the PrimId render var; 0 means background.
};

/// A heightfield written on the compute device (CUDA or HIP), which stays there: a contiguous
/// float32 (rows, cols) tensor and how it lies, as `RasterView`.
struct DeviceHeightfield {
    Tensor heights;
    double originX = 0.0;  ///< World x of the north-west corner of the first cell.
    double originY = 0.0;  ///< World y of that corner.
    double cellSize = 1.0;
    std::optional<float> noData;
    /// The cells that changed since the last update as (first column, first row, columns, rows),
    /// so only their gradients and LOD bounds are rebuilt; none for every cell.
    std::optional<std::array<std::uint32_t, 4>> dirty;
    std::uintptr_t stream = 1;  ///< The stream the heights are written on.
};

/// A north-up heightfield raster drawn as a lit, shadowed triangle mesh: a `RasterDrawable`.
///
/// Heights live in an `R32_SFLOAT` image that the vertex shader reads; the mesh has no vertex or
/// index buffers. Each frame `LodQuadtree` selects square patches at continuous levels of detail
/// (CDLOD), and each patch is one instance of a shared grid whose vertices are derived from the
/// vertex and instance indices and morphed between levels. Normals come from a precomputed
/// central-difference gradient texture sampled per fragment, so shading is independent of the mesh
/// level. The world frame is the raster's: x east, y north, z up.
///
/// The colour pass writes the renderer's G-buffer (`terrain_color.slang`), which the renderer's
/// spectral shading pass lights with the reflectance model and spectrum of the terrain's material:
/// the material's albedo map scales its reflectance, its detail albedo texture modulates it and
/// its detail normal texture perturbs the shading normal.
///
/// The main view culls patches against the camera frustum and each shadow cascade against its
/// light frustum, but all of them choose levels and morph from the camera's eye, so shadow casters
/// match the visible surface.
///
/// Reference: F. Strugar, "Continuous Distance-Dependent Level of Detail for Rendering
/// Heightmaps", Journal of Graphics, GPU, and Game Tools 14(4):57-74, 2009.
/// https://doi.org/10.1080/2151237X.2009.10129287
class TerrainDrawable : public RasterDrawable {
  public:
    /// Grid quads along a level-0 LOD node; patches are half this.
    static constexpr std::uint32_t kLodLeafCells = 64;

    /// A terrain drawn by `renderer`, which must outlive it.
    /// @throws std::runtime_error if the shadow pipeline cannot be created.
    explicit TerrainDrawable(RasterRenderer& renderer);
    ~TerrainDrawable() override;
    TerrainDrawable(const TerrainDrawable&) = delete;
    TerrainDrawable& operator=(const TerrainDrawable&) = delete;

    /// Uploads `heights` as the heightfield to draw; the data is copied, so the view need not
    /// outlive the call.
    /// @throws std::invalid_argument if the view is smaller than 2 x 2, malformed (see
    /// `RasterView`), or larger than the device's maximum texture size.
    void SetHeightfield(const RasterView& heights);

    /// Draws heights written on the compute device of `compute` (`SetComputeInterop`), without
    /// a host copy. They are copied device to device into the drawable's own buffer at once (on
    /// the field's stream), and the next frame rebuilds the gradients and LOD bounds of the dirty
    /// cells on the GPU. A first field, or one of another size, waits for the GPU to bound it;
    /// later ones never wait, and the LOD tree takes their bounds a frame or two later, so for
    /// that long it selects and culls with the last bounds.
    /// @throws std::invalid_argument without compute interop, for a tensor that is not a
    /// contiguous float32 (rows, cols) tensor on its device, smaller than 2 x 2 or larger than the
    /// device's textures, or a dirty rectangle outside it.
    void SetHeightfield(const DeviceHeightfield& field);

    /// Maps the heights on the compute device for writing in place: a float32 (rows, cols)
    /// tensor over the drawable's own copy of them, holding the heights as they are. Writes go on
    /// `stream`, which first waits for the frames already submitted; handing the tensor back with
    /// the device `SetHeightfield` (and its dirty rectangle) then draws them without copying.
    /// @throws std::invalid_argument without heights on the compute device.
    Tensor MapHeights(std::uintptr_t stream = 1);

    /// Enables heightfields on the compute device (`SetHeightfield`); `compute` must outlive the
    /// drawable.
    void SetComputeInterop(ComputeInterop* compute) {
        compute_ = compute;
    }

    /// Records the GPU update of heights written on the device since the last frame, before
    /// any pass draws them.
    void RecordCull(VkCommandBuffer commands) const override;

    /// Drops the heightfield, so nothing is drawn. Waits for the device to go idle.
    void ClearHeightfield();

    bool HasHeightfield() const {
        return heights_ != nullptr;
    }

    /// The heights' gradient image (R16G16 float, a mip chain) shading samples, for tests; null
    /// without a heightfield.
    const GpuImage* Gradients() const {
        return gradients_.get();
    }

    std::size_t Width() const {
        return width_;
    }

    std::size_t Height() const {
        return height_;
    }

    /// Options for the next frames.
    void SetParams(const TerrainParams& params) {
        params_ = params;
    }

    const TerrainParams& Params() const {
        return params_;
    }

    /// Triangles drawn in every view by the last frame.
    std::uint64_t TriangleCount() const {
        return triangles_;
    }

    /// LOD patches drawn in view 0 by the last frame.
    std::size_t PatchCount() const {
        return mainRanges_.empty() ? 0 : mainRanges_.front().count;
    }

    /// First LOD range used by the last frame, in world units; 0 with LOD off.
    float LodFirstRange() const {
        return firstRange_;
    }

    const LodQuadtree& Lod() const {
        return tree_;
    }

    /// GPU memory held for the heightfield and the drawable's buffers, in bytes.
    std::uint64_t GpuBytes() const;

    /// Bounds of the drawn mesh at the given vertical exaggeration, relative to the origin
    /// (`SetOrigin`).
    std::pair<glm::vec3, glm::vec3> Bounds(float exaggeration) const;

    void SetOrigin(const glm::dvec3& origin) override {
        origin_ = origin;
    }

    /// Bounds at the current exaggeration; none without a heightfield.
    std::optional<Bounds3> Bounds() const override;

    /// Writes the terrain uniforms of every view and selects the LOD patches for every view (each
    /// from its own eye and frustum) and every cascade (from view 0's eye).
    /// @throws std::invalid_argument if the material's colour pipeline fails to compile.
    void Prepare(const FrameParams& params, std::span<const ShadowCascade> cascades) override;

    void RecordShadow(VkCommandBuffer commands, const RasterFrame& frame, std::size_t cascade,
                      const glm::mat4& lightViewProjection,
                      const ShadowCasters& casters) const override;
    void RecordColor(VkCommandBuffer commands, const RasterFrame& frame, std::size_t view,
                     DrawPhase phase = DrawPhase::All) const override;

  private:
    struct DrawRange {
        std::uint32_t first = 0;
        std::uint32_t count = 0;
    };

    /// The G-buffer pipeline for a frame camera's views, or a pushbroom's.
    void CreateGBufferPipeline(bool pushbroom);
    void ReservePatches(std::size_t count);
    /// Makes the images and buffers of device heights for a `width` x `height` raster of cells
    /// `cellSize` wide and resets the tree and patches; they are written later.
    void AllocateHeightfield(std::uint32_t width, std::uint32_t height, double cellSize);
    /// Records the update of the device heights' cells in `rect` (first column, first row, end
    /// column, end row): the heights image, gradients, their mips and LOD leaf ranges, and the
    /// leaves' copy into `readback` when given.
    void RecordHeightUpdate(VkCommandBuffer commands, const std::array<std::uint32_t, 4>& rect,
                            const GpuBuffer* readback) const;
    /// Takes the LOD leaf ranges the GPU computed into `readback` into the tree and the height
    /// range.
    void TakeLeaves(const GpuBuffer& readback);
    std::vector<DescriptorResource> Resources(const RasterFrame& frame, std::size_t view) const;
    void DrawPatches(VkCommandBuffer commands, DrawRange range, const glm::mat4& light) const;

    /// What maps a point relative to the origin into the tree's coordinates (the terrain's
    /// north-west corner at x = y = 0, heights as they are): add it.
    glm::vec3 LocalFrom() const {
        return glm::vec3(origin_ - glm::dvec3(originX_, originY_, 0.0));
    }

    RasterRenderer& renderer_;
    PipelineLayout layout_;
    /// The G-buffer pipelines for frame cameras and, once one is seen, pushbrooms.
    std::array<std::unique_ptr<GraphicsPipeline>, 2> gbufferPipelines_;
    bool pushbroom_ = false;  ///< Whether the last `Prepare`'s camera is a pushbroom.
    std::unique_ptr<GraphicsPipeline> shadowPipeline_;
    TerrainParams params_;

    GpuBuffer uniforms_;
    std::unique_ptr<GpuImage> heights_;
    std::unique_ptr<GpuImage> gradients_;
    std::unique_ptr<GpuBuffer> patchBuffer_;

    LodQuadtree tree_;
    std::vector<LodPatch> patches_;
    std::vector<DrawRange> mainRanges_;
    std::vector<DrawRange> cascadeRanges_;
    std::uint64_t triangles_ = 0;
    float firstRange_ = 0.0f;
    bool frozen_ = false;
    std::vector<glm::vec3> lodEyes_;  ///< In the tree's coordinates (`LocalFrom`).
    std::vector<Frustum> lodFrusta_;  ///< Likewise.
    glm::dvec3 origin_{0.0};          ///< The frame's origin (`SetOrigin`).
    bool prepared_ = false;

    std::size_t width_ = 0;
    std::size_t height_ = 0;
    double originX_ = 0.0;
    double originY_ = 0.0;
    double pixelSize_ = 1.0;
    float minHeight_ = 0.0f;
    float maxHeight_ = 0.0f;

    // Heightfields on the compute device.
    ComputeInterop* compute_ = nullptr;
    DeviceHeightfield device_;               ///< The last field, its tensor dropped.
    std::shared_ptr<GpuBuffer> deviceHeights_;  ///< The heights, copied from the device.
    Tensor mappedHeights_;  ///< The view of `deviceHeights_` `MapHeights` gives out.
    std::unique_ptr<GpuBuffer> leaves_;      ///< Per LOD leaf, (min, max).
    PipelineLayout updateLayout_;
    std::unique_ptr<ComputePipeline> gradientsPipeline_;
    std::unique_ptr<ComputePipeline> gradientMipPipeline_;
    std::unique_ptr<ComputePipeline> leafRangePipeline_;
    /// The cells changed since the last recorded update, (first column, first row, end column, end
    /// row); empty when none.
    mutable std::optional<std::array<std::uint32_t, 4>> pendingRect_;
    /// Per frame slot, the leaf ranges its frame copies back, and whether it did.
    std::array<std::unique_ptr<GpuBuffer>, FrameUploads::kSlots> readbacks_;
    mutable std::array<bool, FrameUploads::kSlots> readbackPending_{};
};

}  // namespace ernie
