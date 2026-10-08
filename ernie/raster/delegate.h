#pragma once

#include "ernie/device/compute_interop.h"
#include "ernie/device/offscreen_target.h"
#include "ernie/geometry/primitives.h"
#include "ernie/raster/ground_drawable.h"
#include "ernie/raster/mesh_drawable.h"
#include "ernie/raster/renderer.h"
#include "ernie/raster/terrain_drawable.h"
#include "ernie/scene/scene.h"
#include "ernie/session/delegate.h"
#include "ernie/session/outputs.h"
#include "ernie/session/resolve.h"
#include "ernie/util/image.h"
#include "ernie/util/tensor.h"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace ernie {

/// Names of the rasterizer's `RenderStats::counters`.
namespace raster_stats {
inline constexpr std::string_view kTerrainPatches = "terrain patches";       ///< Drawn last frame.
inline constexpr std::string_view kTerrainLodLevels = "terrain LOD levels";  ///< Of the quadtree.
/// Mesh draw ranges the first view drew last frame, after frustum culling.
inline constexpr std::string_view kMeshDraws = "mesh draws";
/// Mesh draw calls the first view recorded last frame: one per mesh drawn directly, one per
/// multi-draw of GPU-culled instances.
inline constexpr std::string_view kMeshDrawCalls = "mesh draw calls";
}  // namespace raster_stats

/// The ground plane's options from the GroundPlane prim at `ground`: `ernie:height`, and its
/// checkerboard from `ernie:checker`, `ernie:checker:size` and `ernie:checker:color0` and `color1`.
/// The material and prim id are left at their defaults.
/// @throws std::invalid_argument for a checker size that is not positive and finite.
GroundParams GroundParamsOf(const Scene& scene, const Path& ground);

/// The terrain's options: vertical exaggeration from the terrain's `ernie:exaggeration`, and
/// colour mode and LOD options from the render settings. The material is left at the default.
TerrainParams TerrainParamsOf(const Scene& scene, const RenderPrims& prims);

/// Ernie's rasterizer as a render delegate for a `Scene`: syncs Material prims into the
/// renderer's material table, the first visible Terrain into a `TerrainDrawable`, and Mesh and
/// PointInstancer prims into a `MeshDrawable`, and renders them with the scene's camera, light and
/// render settings.
///
/// `Sync` re-uploads GPU data only when the attributes behind it changed
/// (`Scene::AttributeVersion`): a deforming mesh rewrites only its points, and moving instances
/// rewrites only the instance arrays. Arrays on a GPU compute device (mesh `points`, instance
/// `positions`, `orientations` and `scales`, contiguous float32) are copied device to device
/// through `ComputeInterop`, connected on first use; author `extent` on such prims to spare a copy
/// to the host for bounds. Meshes that are prototypes of a PointInstancer are drawn
/// only through it; prototypes must be Mesh prims. A mesh without a bound material uses the
/// default material tinted by its `primvars:displayColor` (0.5 grey without one).
class RasterDelegate : public RenderDelegate {
  public:
    /// @throws std::runtime_error if the renderer's pipelines cannot be created.
    /// `presentFormat` is the colour format `Record` draws LdrColor into.
    explicit RasterDelegate(const VulkanContext& context,
                            VkFormat presentFormat = VK_FORMAT_R8G8B8A8_UNORM,
                            BrdfRegistry brdfs = BrdfRegistry::WithBuiltins());
    ~RasterDelegate() override;

    const RendererInfo& Info() const override;
    const BrdfRegistry& Brdfs() const override;
    void AddSensorFilter(SensorFilterModel model) override;
    const SensorFilterRegistry& SensorFilters() const override {
        return renderer_.SensorFilters();
    }

    /// Uploads the materials, terrain, meshes and instances that changed since the last call, or
    /// all of them when `scene` is not the scene of the last call.
    void Sync(const Scene& scene) override;

    /// As `RenderDelegate::Update`; waits for earlier `StepAsync` work, then needs no submitted
    /// frame to still be reading the renderer.
    void Update(const Scene& scene, const Path& settings, VkExtent2D extent) override;

    std::vector<TimelinePoint> SubmitWaits() const override;

    /// Records the frame prepared by the last `Update` into render targets of the delegate's own
    /// and draws its LdrColor into `target.color`. Recreating those targets for a new size or band
    /// count waits for the device.
    void Record(VkCommandBuffer commands, const RenderingTarget& target,
                std::string_view var) override;

    /// As `RenderDelegate::StepAsync`. Products render one after another; GPU work is ordered
    /// after compute copies into the renderer's buffers, and signals the compute side when done.
    StepOperation StepAsync(const Scene& scene, std::span<const Path> products = {},
                            const Path& settings = {}) override;

    ComputeInterop* Compute(std::string* why = nullptr) override;

    Path PrimPath(std::uint32_t id) const override;
    Path Pick(std::uint32_t x, std::uint32_t y) override;

    void SetViewBand(int band) override {
        renderer_.SetPresentBand(band);
    }

    std::optional<double> MeteredLuminance() const override {
        return renderer_.MeteredLuminance();
    }

    /// GPU memory of the renderer, drawables and the view's targets; triangles drawn by the terrain
    /// and meshes; and the `raster_stats` counters.
    RenderStats Stats() const override;

    /// The union of the terrain's and the meshes' bounds.
    std::optional<Bounds3> SceneBounds() const override;

    /// The prims the last `Update` drew.
    const RenderPrims& Prims() const {
        return prims_;
    }

    RasterRenderer& Renderer() {
        return renderer_;
    }

    const RasterRenderer& Renderer() const {
        return renderer_;
    }

    const TerrainDrawable& Terrain() const {
        return terrain_;
    }

    const GroundDrawable& Ground() const {
        return ground_;
    }

    const MeshDrawable& Meshes() const {
        return meshes_;
    }

    /// The renderer's material id for the Material prim at `path`, if it is synced.
    std::optional<std::uint32_t> MaterialId(const Path& path) const;

  private:
    /// Scene, path and attribute versions behind an upload.
    using Key = std::pair<Path, std::array<std::uint64_t, 12>>;

    struct SyncedMaterial {
        std::uint32_t id = 0;
        std::optional<Key> albedoMap;
        std::optional<Key> detailAlbedo;
        std::optional<Key> detailNormal;
    };

    struct SyncedMesh {
        std::uint32_t handle = 0;
        Key key;
        std::uint64_t points = 0;
        bool hasColors = false;
        glm::vec3 tint{0.5f};
    };

    struct SyncedInstancer {
        std::uint32_t handle = 0;
        Key key;
    };

    Key KeyOf(const Scene& scene, const Path& path,
              std::initializer_list<std::string_view> attributes) const;
    void SyncMaterials(const Scene& scene);
    void SyncMaterial(const Scene& scene, const Path& path, SyncedMaterial& material);
    void SyncHeightfield(const Scene& scene, const Path& terrain);
    /// Uploads the map of the frame's dome and the stars of its StarField when they change.
    void SyncSky(const Scene& scene, const RenderPrims& prims);
    void SyncMeshes(const Scene& scene);
    void SyncInstancers(const Scene& scene);
    std::uint32_t MaterialOf(const Scene& scene, const Path& prim) const;
    std::vector<MeshItem> Items(const Scene& scene) const;

    ComputeInterop& RequireCompute();
    void UpdatePrims(const Scene& scene, const RenderPrims& prims, VkExtent2D extent, FrameId frame,
                     std::span<const SensorFilterStep> filters = {},
                     std::vector<glm::dmat4> poses = {}, std::vector<std::int32_t> worlds = {},
                     std::vector<std::string_view> vars = {},
                     std::optional<glm::dmat4> posesParent = std::nullopt,
                     const std::optional<std::string>& overlays = std::nullopt);
    /// Takes the next `FrameUploads` slot for a frame, once the frame that last used it has
    /// finished; returns it.
    std::uint32_t BeginFrame();
    std::shared_ptr<ProductFrame> AcquireFrame(const Path& product, VkExtent2D extent, int bands,
                                               std::uint32_t views, bool batched,
                                               std::span<const std::string_view> vars);
    std::uint32_t PrimIdOf(const Path& path) const;
    void SetMeshPoints(const Scene& scene, const Path& path, std::uint32_t handle);

    const VulkanContext& context_;
    std::unique_ptr<ComputeInterop> compute_;
    bool computeTried_ = false;
    std::string computeWhy_;
    bool meshesOnGpu_ = false;
    RasterRenderer renderer_;
    TerrainDrawable terrain_;
    MeshDrawable meshes_;
    GroundDrawable ground_;
    std::unique_ptr<RenderTargets> viewTargets_;
    mutable std::map<Path, std::uint32_t> primIds_;
    mutable std::vector<Path> primPaths_;
    std::map<Path, std::vector<std::shared_ptr<ProductFrame>>> frames_;
    std::vector<std::shared_ptr<GpuBuffer>> linearPool_;
    /// Per `FrameUploads` slot, the submission that last used it; the slot to use next.
    std::array<std::uint64_t, FrameUploads::kSlots> slotSubmissions_{};
    std::uint32_t nextSlot_ = 0;
    /// The render origin (`FrameParams::origin`), and how far the camera may move from it before
    /// it follows: float precision there is 1024 * 2^-24, about 0.06 mm.
    glm::dvec3 origin_{0.0};
    static constexpr double kRebaseDistance = 1024.0;
    ChangeCursor cursor_;
    std::uint64_t sceneId_ = 0;
    RenderPrims prims_;
    std::optional<Key> heightfield_;
    std::optional<Key> domeMap_;  ///< The dome map last uploaded, and its dome.
    std::optional<Key> stars_;    ///< The StarField last uploaded.
    std::map<Path, SyncedMaterial> materials_;
    std::map<Path, SyncedMesh> syncedMeshes_;
    std::map<Path, SyncedInstancer> instancers_;
};

}  // namespace ernie
