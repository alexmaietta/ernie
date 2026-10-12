// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/compute_interop.h"
#include "ernie/device/offscreen_target.h"
#include "ernie/geometry/primitives.h"
#include "ernie/raster/ground_drawable.h"
#include "ernie/raster/mesh_drawable.h"
#include "ernie/raster/planet_drawable.h"
#include "ernie/raster/points_drawable.h"
#include "ernie/raster/renderer.h"
#include "ernie/raster/terrain_drawable.h"
#include "ernie/scene/scene.h"
#include "ernie/session/delegate.h"
#include "ernie/session/light_links.h"
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
#include <span>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace ernie {

/// Names of the rasterizer's `RenderStats::counters`.
namespace raster_stats {
inline constexpr std::string_view kTerrainPatches =
    "terrain patches";  ///< Drawn last frame.
inline constexpr std::string_view kTerrainLodLevels =
    "terrain LOD levels";  ///< Of the quadtree.
/// Mesh draw ranges the first view drew last frame, after frustum
/// culling.
inline constexpr std::string_view kMeshDraws = "mesh draws";
/// Mesh draw calls the first view recorded last frame: one per mesh
/// drawn directly, one per multi-draw of GPU-culled instances.
inline constexpr std::string_view kMeshDrawCalls = "mesh draw calls";
}  // namespace raster_stats

/// The ground plane's options from the GroundPlane prim at `ground`:
/// `ernie:height`, and its checkerboard from `ernie:checker`,
/// `ernie:checker:size` and `ernie:checker:color0` and `color1`. The
/// material and prim id are left at their defaults.
/// @throws std::invalid_argument for a checker size that is not
/// positive and finite.
GroundParams GroundParamsOf(const Scene& scene, const Path& ground);

/// The terrain's options: vertical exaggeration from the terrain's
/// `ernie:exaggeration`, and colour mode and LOD options from the
/// render settings. The material is left at the default.
TerrainParams TerrainParamsOf(const Scene& scene,
                              const RenderPrims& prims);

/// Ernie's rasterizer as a render delegate for a `Scene`: syncs
/// Material prims into the renderer's material table, the first
/// visible Terrain into a `TerrainDrawable`, and Mesh and
/// PointInstancer prims into a `MeshDrawable`, and renders them with
/// the scene's camera, light and render settings.
///
/// `Sync` re-uploads GPU data only when the attributes behind it
/// changed (`Scene::AttributeVersion`): a deforming mesh rewrites
/// only its points, and moving instances rewrites only the instance
/// arrays. Arrays on a GPU compute device (mesh `points`, instance
/// `positions`, `orientations`, `scales` and `primvars:displayColor`,
/// contiguous float32; mesh triangles as `faceVertexIndices`, int32
/// or uint32, with `faceVertexCounts` all 3 or unset) are copied
/// device to device through
/// `ComputeInterop`, connected on first use; author `extent` on such
/// prims to spare a copy to the host for bounds. Meshes that are
/// prototypes of a PointInstancer are drawn only through it;
/// prototypes must be Mesh prims. A mesh without a bound material
/// uses the default material tinted by its `primvars:displayColor`
/// (0.5 grey without one), or by its instancer's per instance.
class RasterDelegate : public RenderDelegate {
  public:
    /// @throws std::runtime_error if the renderer's pipelines cannot
    /// be created. `presentFormat` is the colour format `Record`
    /// draws LdrColor into.
    explicit RasterDelegate(
        const VulkanContext& context,
        VkFormat presentFormat = VK_FORMAT_R8G8B8A8_UNORM,
        BrdfRegistry brdfs = BrdfRegistry::WithBuiltins());
    ~RasterDelegate() override;

    const RendererInfo& Info() const override;
    const BrdfRegistry& Brdfs() const override;
    void AddSensorFilter(SensorFilterModel model) override;
    void AddLightPattern(LightPatternModel model) override;
    bool HasLightPattern(std::string_view id) const override {
        return renderer_.LightPatterns().Find(id) != nullptr;
    }
    const SensorFilterRegistry& SensorFilters() const override {
        return renderer_.SensorFilters();
    }

    /// Uploads the materials, terrain, meshes and instances that
    /// changed since the last call, or all of them when `scene` is
    /// not the scene of the last call.
    void Sync(const Scene& scene) override;

    /// As `RenderDelegate::Update`; waits for earlier `StepAsync`
    /// work, then needs no submitted frame to still be reading the
    /// renderer. The view runs the sensor filter chain of the
    /// settings' first product, so it shows what that product
    /// renders, AOVs included.
    void Update(const Scene& scene, const Path& settings,
                VkExtent2D extent) override;

    /// As `RenderDelegate::MapArray`: a Mesh's `points`, a
    /// PointInstancer's `positions`, `orientations`, `scales` or
    /// `primvars:displayColor`, a
    /// Points prim's `points`, a Terrain's `heights`, or a Camera's
    /// `ernie:poses` once written on the GPU (as float32, quaternions
    /// w last); or a transform op of a prim posed on the GPU alone in
    /// its group (as the PrimSet's).
    Tensor MapArray(Scene& scene, const Path& path,
                    std::string_view attribute,
                    std::uintptr_t stream = 1) override;

    /// As `RenderDelegate::MapArray` of a PrimSet: transform op
    /// `attribute` (`xformOp:*`) of prims posed on the GPU as one
    /// group, rows of one array written to exactly `prims` in row
    /// order. Orientations map as x, y, z, w unless composed on the
    /// GPU (then as written).
    Tensor MapArray(Scene& scene, const PrimSet& prims,
                    std::string_view attribute,
                    std::uintptr_t stream = 1) override;

    /// `Info().viewVars`, then the AOVs of the view's chain as
    /// "aov:<name>".
    std::vector<std::string> ViewVars() const override;

    std::vector<TimelinePoint> SubmitWaits() const override;

    /// Records the frame prepared by the last `Update` into render
    /// targets of the delegate's own and draws its var `var` into
    /// `target.color`. Recreating those targets for a new size, band
    /// count or AOVs waits for the device.
    void Record(VkCommandBuffer commands,
                const RenderingTarget& target,
                std::string_view var) override;

    /// As `RenderDelegate::StepAsync`. Products render one after
    /// another; GPU work is ordered after compute copies into the
    /// renderer's buffers, and signals the compute side when done.
    StepOperation StepAsync(const Scene& scene,
                            std::span<const Path> products = {},
                            const Path& settings = {},
                            double deltaTime = 0.0) override;

    ComputeInterop* Compute(std::string* why = nullptr) override;

    Path PrimPath(std::uint32_t id) const override;
    Path Pick(std::uint32_t x, std::uint32_t y) override;

    void SetViewBand(int band) override {
        renderer_.SetPresentBand(band);
    }

    void SetViewVar(std::string_view var) override {
        viewMotion_ = var == render_vars::kMotionVector;
    }

    void SetHighlight(std::span<const Path> paths,
                      glm::vec4 colour) override {
        highlight_.assign(paths.begin(), paths.end());
        highlightColour_ = colour;
    }

    std::optional<double> MeteredLuminance() const override {
        return renderer_.MeteredLuminance();
    }

    /// GPU memory of the renderer, drawables and the view's targets;
    /// triangles drawn by the terrain and meshes; and the
    /// `raster_stats` counters.
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

    const PlanetDrawable& Planet() const {
        return planet_;
    }

    const GroundDrawable& Ground() const {
        return ground_;
    }

    const PointsDrawable& Points() const {
        return points_;
    }

    const MeshDrawable& Meshes() const {
        return meshes_;
    }

    /// The renderer's material id for the Material prim at `path`, if
    /// it is synced.
    std::optional<std::uint32_t> MaterialId(const Path& path) const;

  private:
    /// Scene, path and attribute versions behind an upload.
    using Key = std::pair<Path, std::array<std::uint64_t, 12>>;

    struct SyncedMaterial {
        std::uint32_t id = 0;
        std::optional<Key> albedoMap;
        std::optional<Key> detailAlbedo;
        std::optional<Key> detailNormal;
        /// What the albedo texture, normal map, roughness, metallic
        /// and opacity textures were loaded from (`TextureKey`).
        std::array<std::string, 5> textures;
    };

    struct SyncedMesh {
        std::uint32_t handle = 0;
        Key key;
        /// Versions of its points, and of its normals, colours and
        /// triangle indices when written on the GPU.
        std::array<std::uint64_t, 4> versions{};
        bool hasColors = false;
        glm::vec3 tint{0.5f};
    };

    struct SyncedInstancer {
        std::uint32_t handle = 0;
        Key key;
        /// Under a GPU-posed prim: its instance and the transforms
        /// from it down to the instancer, which its instances are
        /// composed under (instancer 0 for none).
        std::uint32_t anchorInstancer = 0;
        std::uint32_t anchorRow = 0;
        glm::dmat4 anchorBelow{1.0};
    };

    struct SyncedPoints {
        std::uint32_t handle = 0;
        Key key;
    };

    /// Prims posed on the GPU (`GpuXformOf`), grouped by the arrays
    /// they share: an instancer of the meshes' drawable over those
    /// arrays, one instance per prim, and the attribute versions
    /// behind its last upload. Kept by the group's first prim and its
    /// number of rows, so arrays written anew each step for the same
    /// prims keep the instancer (and its motion).
    using GpuXformKey = std::pair<Path, std::int64_t>;
    struct GpuXformGroup {
        std::uint32_t handle = 0;
        std::uint64_t version = 0;
        std::uint64_t scene =
            0;  ///< `Scene::Id` of the scene `version` counts in.
        Tensor zeros;  ///< Device positions for a group whose prims
                       ///< have no translate.
        std::size_t members = 0;
        std::int64_t rows = 0;          ///< Rows of its arrays.
        std::optional<Bounds3> extent;  ///< The union of its prims'
                                        ///< extents, if all have one.
        std::vector<std::string>
            ops;  ///< Its prims' ops, in order (`xformOp:*`).
        bool wFirst = false;  ///< Its orientations put w first.
        /// Composed on the GPU (`GpuXform::composed`, or posed under
        /// other GPU-posed prims): per row, the instance it is posed
        /// under and the CPU transforms between.
        bool composed = false;
        std::vector<InstanceParent> parents;
    };
    /// A prim posed on the GPU: its group (in `gpuXformGroups_`), its
    /// instancer and its row there, and the prim whose world
    /// transform its draws are under (none for the world itself).
    struct GpuPosed {
        GpuXformGroup* group = nullptr;
        std::uint32_t instancer = 0;
        std::uint32_t row = 0;
        bool reset = false;
        std::optional<Path> frame;
    };
    /// Where a prim under a GPU-posed prim is posed (`GpuAnchorOf`):
    /// that prim (`at`) and its instance, the world transform of its
    /// frame, and the transforms from it down to the prim.
    struct GpuAnchor {
        Path at;
        std::uint32_t instancer = 0;
        std::uint32_t row = 0;
        glm::dmat4 frame{1.0};
        glm::dmat4 below{1.0};
    };

    /// The nearest prim from `path` up posed on the GPU whose
    /// transform reaches it (none past a prim resetting the transform
    /// stack), as `GpuAnchor`.
    std::optional<GpuAnchor> GpuAnchorOf(const Scene& scene,
                                         const Path& path) const;
    /// `HostArrayReader` of `scene`: GPU arrays the CPU plans with
    /// (worlds), copied to the host once per write (waiting for their
    /// streams) and kept until written again.
    HostArrayReader HostArrays(const Scene& scene) const;

    Key
    KeyOf(const Scene& scene, const Path& path,
          std::initializer_list<std::string_view> attributes) const;
    void SyncMaterials(const Scene& scene);
    void SyncMaterial(const Scene& scene, const Path& path,
                      SyncedMaterial& material);
    void SyncHeightfield(const Scene& scene, const Path& terrain);
    /// Uploads the map of the frame's dome and the stars of its
    /// StarField when they change.
    void SyncSky(const Scene& scene, const RenderPrims& prims);
    /// Lights `frame` with the sun found in its dome's map
    /// (`domeSun_`) in the full dome lighting mode: as the frame's
    /// sun (with its cascades) when it has none, else as a shadowed
    /// distant light; the dome then lights with its map without the
    /// sun.
    void ApplyDomeSun(FrameParams& frame) const;
    /// In the directions mode, lights `frame` with its dome's
    /// shadowed directions (`DomeDirectionsOf`) as distant lights of
    /// the dome's diffuse light, as many as light slots remain, so
    /// the dome lights without its own diffuse light.
    void ApplyDomeDirections(FrameParams& frame) const;
    /// Exposes `frame` automatically when its camera's exposure mode
    /// is auto: meters it under `stateKey`, and adds the stops that
    /// bring the key's latest metered luminance to middle grey,
    /// approached over the camera's adaptation frames.
    void ApplyAutoExposure(const Scene& scene, const Path& stateKey,
                           FrameParams& frame);
    /// Whether a product of `specs` exposes automatically with no
    /// metered frame yet, so `StepAsync` meters one first.
    bool
    NeedsMetering(const Scene& scene, const RenderPrims& base,
                  std::span<const RenderProductSpec> specs) const;
    void SyncMeshes(const Scene& scene);
    void SyncInstancers(const Scene& scene);
    /// Hands the meshes' drawable the arrays of prims posed on the
    /// GPU that changed, one instancer per group of prims sharing
    /// arrays; the meshes under those prims draw as their instances
    /// (`Items`).
    void SyncGpuXforms(const Scene& scene);
    /// The fast path of a simulation step: when `changes` rewrite
    /// only the arrays of whole groups of prims already posed on the
    /// GPU (the same prims, rows and shapes), hands the drawable the
    /// new arrays and returns true; false when a full `Sync` is
    /// needed.
    bool RefreshGpuXforms(const Scene& scene,
                          std::span<const Change> changes);
    /// The views of `cameras` as views posed under GPU-posed prims
    /// (`AnchoredView`), one per camera; none when no camera is under
    /// a GPU-posed prim (their transforms are on the CPU).
    /// @throws std::invalid_argument for such cameras with
    /// `ernie:poses` of their own.
    std::vector<AnchoredView>
    AnchoredViewsOf(const Scene& scene,
                    std::span<const Path> cameras) const;
    /// Hands the drawable a group's arrays from one of its prims'
    /// pose.
    void UploadGpuXformGroup(GpuXformGroup& group,
                             const GpuXform& pose, std::int64_t rows);
    /// Uploads the Points prims that changed: their arrays from the
    /// host or kept on the GPU.
    void SyncPoints(const Scene& scene);
    /// Light linking's groups of the synced geometry (`links_`).
    void SyncLinks(const Scene& scene);
    std::vector<PointsItem> PointsItems(const Scene& scene) const;
    std::uint32_t MaterialOf(const Scene& scene,
                             const Path& prim) const;
    std::vector<MeshItem> Items(const Scene& scene) const;

    ComputeInterop& RequireCompute();
    /// Reads the frame's lasers' pattern image arrays on the GPU into
    /// the renderer's (`RasterRenderer::PatternArrayWrites`): those
    /// written there in place (`MapArray`) only waited for, others
    /// copied there.
    void ReadPatternArrays();
    /// Ends a step's motion once all of it is prepared
    /// (`MeshDrawable::SettleMotion`).
    void SettleMotion();
    /// `stateKey` names whose frames the temporal filters' state
    /// follows: the product, or the settings for the interactive
    /// view.
    void UpdatePrims(
        const Scene& scene, const RenderPrims& prims,
        VkExtent2D extent, FrameId frame, const Path& stateKey,
        std::span<const SensorFilterStep> filters = {},
        std::vector<glm::dmat4> poses = {},
        std::vector<std::int32_t> worlds = {},
        std::vector<std::string_view> vars = {},
        std::optional<glm::dmat4> posesParent = std::nullopt,
        const std::optional<std::string>& overlays = std::nullopt,
        std::vector<FrameParams::SensorChain> chains = {},
        MinimalMode minimal = MinimalMode::Off,
        std::optional<glm::vec3> minimalColor = std::nullopt);
    /// Hands the renderer the state images of `key` for the frame
    /// `Update` just prepared, made (and history dropped) when its
    /// size, views or chain change, and swaps them for the next.
    void PrepareSensorState(const Path& key, VkExtent2D extent,
                            std::uint32_t views);
    /// The state of sensor pass `pass` (one chain of the frame).
    void PrepareSensorState(const Path& key, std::size_t pass,
                            VkExtent2D extent, std::uint32_t views);
    /// Takes the next `FrameUploads` slot for a frame, once the frame
    /// that last used it has finished; returns it.
    std::uint32_t BeginFrame();
    /// `faces` raster layers per view: a camera rendered on cube
    /// faces' (`NeedsCubeFaces`).
    std::shared_ptr<ProductFrame>
    AcquireFrame(const Path& product, VkExtent2D extent, int bands,
                 std::uint32_t views, bool batched,
                 std::span<const std::string_view> vars,
                 std::span<const std::string> aovs = {},
                 std::uint32_t faces = 1, bool minimal = false);
    std::uint32_t PrimIdOf(const Path& path) const;
    void SetMeshPoints(const Scene& scene, const Path& path,
                       std::uint32_t handle);
    /// Hands the mesh its normals and colours written on the GPU.
    void SetMeshAttributes(const Scene& scene, const Path& path,
                           std::uint32_t handle);
    /// Hands the mesh its triangle indices written on the GPU.
    void SetMeshIndices(const Scene& scene, const Path& path,
                        std::uint32_t handle);

    const VulkanContext& context_;
    std::unique_ptr<ComputeInterop> compute_;
    bool computeTried_ = false;
    std::string computeWhy_;
    bool meshesOnGpu_ = false;
    bool terrainOnGpu_ =
        false;  ///< Whether the terrain has the compute interop.
    RasterRenderer renderer_;
    TerrainDrawable terrain_;
    PlanetDrawable planet_;
    MeshDrawable meshes_;
    GroundDrawable ground_;
    PointsDrawable points_;
    std::unique_ptr<RenderTargets> viewTargets_;
    /// The AOVs the view's chain writes, in slot order, as of the
    /// last `Update`.
    std::vector<std::string> viewAovs_;
    // Whether the view holds the opt-in MotionVector, which it
    // renders only while shown.
    bool viewMotion_ = false;
    /// Per product (or the settings for the view), its views' camera
    /// poses at its last frame, for motion blur.
    std::map<Path, std::vector<glm::dmat4>> lastPoses_;
    /// And its lights' poses at its last frame, by light.
    std::map<Path, std::map<Path, LightParams::Pose>> lastLightPoses_;
    /// Per product (or the settings for the view) exposed
    /// automatically, the stops it adds to its camera's exposure.
    std::map<Path, double> autoStops_;
    /// Inside the frame `StepAsync` meters before exposing.
    bool metering_ = false;
    /// The span of the sensor clock the frames being prepared cover
    /// (`FrameParams::time`).
    TimeWindow window_;
    /// The prims `SetHighlight` outlines, resolved to prim ids (with
    /// their descendants') per frame.
    std::vector<Path> highlight_;
    glm::vec4 highlightColour_{1.0f, 0.6f, 0.1f, 1.0f};
    mutable std::map<Path, std::uint32_t> primIds_;
    mutable std::vector<Path> primPaths_;
    std::map<Path, std::vector<std::shared_ptr<ProductFrame>>>
        frames_;
    std::vector<std::shared_ptr<GpuBuffer>> linearPool_;
    /// Per `FrameUploads` slot, the submission that last used it; the
    /// slot to use next.
    std::array<std::uint64_t, FrameUploads::kSlots>
        slotSubmissions_{};
    std::uint32_t nextSlot_ = 0;
    /// The render origin (`FrameParams::origin`), and how far the
    /// camera may move from it before it follows: float precision
    /// there is 1024 * 2^-24, about 0.06 mm.
    glm::dvec3 origin_{0.0};
    static constexpr double kRebaseDistance = 1024.0;
    ChangeCursor cursor_;
    std::uint64_t sceneId_ = 0;
    RenderPrims prims_;
    std::optional<Key> heightfield_;
    std::optional<Key>
        domeMap_;  ///< The dome map last uploaded, and its dome.
    /// Whether the last upload looked for a sun in the map (the full
    /// dome lighting mode, with the dome's shadows on), the sun it
    /// found, and the map without it (`ExtractDomeSun`).
    bool domeSunWanted_ = false;
    std::optional<DomeSun> domeSun_;
    Tensor domeLightingMap_;
    /// The map that lights the scene (the dome's, without its sun
    /// when lit on its own), and its shadowed directions for the
    /// directions mode, made for `domeDirectionCount_` of them when a
    /// frame first asks.
    std::optional<Image> domeLightingImage_;
    mutable std::vector<DomeDirection> domeDirections_;
    mutable std::size_t domeDirectionCount_ = 0;
    /// The temporal filters' state per product: two images, the last
    /// frame's and the next's.
    struct SensorState {
        std::array<std::unique_ptr<GpuImage>, 2> images;
        std::uint32_t next =
            0;                 ///< Which image the next frame writes.
        bool history = false;  ///< Whether the other holds a frame.
        std::string chain;     ///< The plan key it was made for.
    };
    std::map<std::pair<Path, std::size_t>, SensorState>
        sensorStates_;          ///< By product, pass.
    std::optional<Key> stars_;  ///< The StarField last uploaded.
    std::map<Path, SyncedMaterial> materials_;
    std::map<Path, SyncedMesh> syncedMeshes_;
    std::map<Path, SyncedInstancer> instancers_;
    std::map<Path, SyncedPoints> syncedPoints_;
    /// The drawn geometry's light link groups and the lights' masks
    /// over them, as of the last `Sync`.
    LinkGroups links_;
    std::map<GpuXformKey, GpuXformGroup> gpuXformGroups_;
    /// Whether the GPU poses of the product being updated are
    /// positions and quaternions (`PosesAreQuaternions`), for
    /// `UpdatePrims`.
    bool gpuPosesQuaternions_ = false;
    /// Whether the product being updated has views posed under
    /// GPU-posed prims (`AnchoredViewsOf`), whose poses the drawable
    /// writes relative to the frame's origin.
    bool anchoredViews_ = false;
    std::unordered_map<Path, GpuPosed> gpuPosed_;
    /// The scene's version at the last full `Sync`: prims below
    /// GPU-posed ones whose own transform ops are older changed only
    /// with their ancestors (`RefreshGpuXforms`).
    std::uint64_t syncedVersion_ = 0;
    /// `HostArrays`' copies by prim and attribute: the scene and
    /// attribute version they hold.
    struct HostArray {
        std::uint64_t scene = 0;
        std::uint64_t version = 0;
        Tensor host;
    };
    mutable std::map<std::pair<Path, std::string>, HostArray>
        hostArrays_;
};

}  // namespace ernie
