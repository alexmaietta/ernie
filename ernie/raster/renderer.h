// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/buffer.h"
#include "ernie/device/frame_uploads.h"
#include "ernie/device/image.h"
#include "ernie/device/pipeline.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/raster/atmosphere_luts.h"
#include "ernie/raster/drawable.h"
#include "ernie/raster/material.h"
#include "ernie/raster/minimal_targets.h"
#include "ernie/raster/overlay_pass.h"
#include "ernie/raster/render_targets.h"
#include "ernie/raster/sensor_plan.h"
#include "ernie/scene/brdf.h"
#include "ernie/scene/light_pattern.h"

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ernie {

/// The shadow views of `light` over a scene within `sceneMin` to
/// `sceneMax`, for square maps of `resolution` texels:
/// - a distant light: one orthographic view along the light enclosing
///   the scene's bounding sphere; `texelWorldSize` is the world size
///   of a texel;
/// - a sphere light whose cone is at most
///   `Light::kMaxSpotConeDegrees`: one perspective view along its
///   axis covering the cone;
/// - any other sphere light: six 90-degree perspective views, the
///   faces of a cube in the order +x, -x, +y, -y, +z, -z. Perspective
///   views reach from the light (its radius, at least 1/10000 of the
///   far plane) to the scene's farthest corner, and their
///   `texelWorldSize` is the size of a texel one unit away. None for
///   a light without shadows. `position` is the light's position in
///   the coordinates of the bounds, relative to the frame's origin
///   (`FrameParams::Relative`); distant lights ignore it.
std::vector<ShadowCascade> FitLightShadowViews(
    const LightParams& light, const glm::vec3& position,
    const glm::vec3& sceneMin, const glm::vec3& sceneMax,
    std::uint32_t resolution);

/// Ernie's rasterizer: owns what the drawables of a frame share, and
/// draws them, spectrally.
///
/// Shared are the frame uniforms (camera, sun, ambient), the lights,
/// the shadow maps, a table of materials, the reflectance models and
/// the shader compiler. Drawables (`TerrainDrawable`, `MeshDrawable`,
/// ...) are created against a renderer and passed to `Update` and
/// `Record` each frame: the renderer fits the shadow cascades around
/// all their bounds, so every drawable casts shadows on every other.
///
/// A frame is deferred and spectral. The drawables rasterize a
/// G-buffer (shading normals, shadow visibility, albedo scale and
/// modulation, material and prim ids, depth); a compute pass per
/// reflectance model in use turns it into spectral radiance in the
/// camera's bands, lit by the sun's spectral irradiance and the other
/// lights (`Light`); then the camera's sensor filters
/// (`SensorFilter`) turn that into the render vars, and a present
/// pass can draw LdrColor into a colour attachment.
///
/// A frame can render a batch of views of one camera at different
/// poses (`FrameParams::poses`), such as one per environment in
/// reinforcement learning, into layered targets in one command
/// buffer: every view has uniforms of its own, and drawables prepare
/// and draw each view (`RasterDrawable::RecordColor`). A batch's sun
/// shadows use one map fitted to the whole scene rather than cascades
/// fitted to each view: a base layer of what every world shares,
/// copied into one layer per world of the batch's views
/// (`FrameParams::worlds`) on which that world's casters draw, so
/// worlds overlapping in space shadow only themselves. Other lights'
/// maps hold what every world shares.
///
/// The sun's shadows use cascaded shadow maps (`FitCascades`)
/// filtered with rotated Poisson-disk PCF, with slope-scaled depth
/// bias and normal offset against acne, and penumbrae that widen with
/// the distance to the occluder for the sun's angular size
/// (percentage-closer soft shadows). Other lights' shadow maps
/// (`FitLightShadowViews`) are sampled with the same filter in the
/// shading pass.
///
/// References:
/// - L. Williams, "Casting Curved Shadows on Curved Surfaces",
///   SIGGRAPH '78, 270-274, 1978.
///   https://doi.org/10.1145/965139.807402
/// - W. T. Reeves, D. H. Salesin and R. L. Cook, "Rendering
///   Antialiased Shadows with Depth Maps", SIGGRAPH '87, 283-291,
///   1987. https://doi.org/10.1145/37402.37435
/// - D. Holbert, "Saying 'Goodbye' to Shadow Acne", GDC 2011 poster
///   (normal offset shadows).
/// - J. Jimenez, "Next Generation Post Processing in Call of Duty:
///   Advanced Warfare", SIGGRAPH 2014 Advances in Real-Time Rendering
///   course (interleaved gradient noise).
/// - R. Fernando, "Percentage-Closer Soft Shadows", SIGGRAPH 2005
///   Sketches, 2005. https://doi.org/10.1145/1187112.1187153
/// - T. Saito and T. Takahashi, "Comprehensible Rendering of 3-D
///   Shapes", SIGGRAPH '90, 197-206, 1990.
///   https://doi.org/10.1145/97880.97901 (G-buffers and deferred
///   shading).
/// - W. Jakob and J. Hanika, "A Low-Dimensional Function Space for
///   Efficient Spectral Upsampling", Computer Graphics Forum
///   38(2):147-155, 2019 (RGB modulation to spectra).
class RasterRenderer {
  public:
    static constexpr int kMaxCascades = 4;

    /// Lights besides the sun per frame, and shadow map views among
    /// them; lights past the views draw without shadows.
    static constexpr std::size_t kMaxLights = 16;
    static constexpr std::size_t kMaxLightShadowViews = 24;

    /// Material 0, which always exists and drawables use by default:
    /// Lambert with albedo 1.
    static constexpr std::uint32_t kDefaultMaterial = 0;

    /// Shades with the models in `brdfs`, and presents LdrColor into
    /// colour attachments of `presentFormat`.
    /// @throws std::runtime_error if the shared resources cannot be
    /// created.
    explicit RasterRenderer(
        const VulkanContext& context,
        VkFormat presentFormat = VK_FORMAT_R8G8B8A8_UNORM,
        BrdfRegistry brdfs = BrdfRegistry::WithBuiltins());
    ~RasterRenderer();
    RasterRenderer(const RasterRenderer&) = delete;
    RasterRenderer& operator=(const RasterRenderer&) = delete;

    const VulkanContext& Context() const {
        return context_;
    }

    VkFormat PresentFormat() const {
        return presentFormat_;
    }

    const ShaderCompiler& Compiler() const {
        return compiler_;
    }

    /// The atmosphere of the last `Update` and its tables.
    const AtmosphereLuts& AtmosphereTables() const {
        return *atmosphere_;
    }

    const BrdfRegistry& Brdfs() const {
        return brdfs_;
    }

    /// Registers `model` after checking that its Slang module
    /// compiles against `IBrdf`.
    /// @throws std::invalid_argument if the registry rejects the
    /// model (`BrdfRegistry::Add`) or its module fails to compile;
    /// the message includes the compiler's diagnostics.
    void AddBrdf(BrdfModel model);

    /// Registers custom sensor filter `model` after checking that its
    /// Slang module compiles in a sensor kernel, so render products'
    /// chains can use it by id.
    /// @throws std::invalid_argument if the registry rejects the
    /// model (`SensorFilterRegistry:: Add`) or its module fails to
    /// compile; the message includes the compiler's diagnostics.
    void AddSensorFilter(SensorFilterModel model);

    /// Registers light pattern `model` after checking that its Slang
    /// module compiles against `ILightPattern`, so LaserLights can
    /// name it in `ernie:laser:pattern`; the shading passes recompile
    /// with it.
    /// @throws std::invalid_argument if the registry rejects the
    /// pattern (`LightPatternRegistry::Add`) or its module fails to
    /// compile; the message includes the compiler's diagnostics.
    void AddLightPattern(LightPatternModel model);

    const LightPatternRegistry& LightPatterns() const {
        return lightPatterns_;
    }

    const SensorFilterRegistry& SensorFilters() const {
        return sensorFilters_;
    }

    /// Adds a material with `settings` and returns its id.
    /// @throws std::invalid_argument as `SetMaterial`.
    std::uint32_t AddMaterial(const MaterialSettings& settings = {});

    /// Sets a material's settings, resolving its BRDF parameters.
    /// @throws std::invalid_argument for an unknown material or BRDF.
    void SetMaterial(std::uint32_t id,
                     const MaterialSettings& settings);

    /// The material with `id`, for its albedo map and detail
    /// textures.
    /// @throws std::invalid_argument for an unknown material.
    RasterMaterial& Material(std::uint32_t id);
    const RasterMaterial& Material(std::uint32_t id) const;

    bool HasMaterial(std::uint32_t id) const {
        return materials_.contains(id);
    }

    /// Removes a material; drawables using it fall back to the
    /// default material.
    /// @throws std::invalid_argument for the default material or an
    /// unknown one.
    void RemoveMaterial(std::uint32_t id);

    /// Counts additions and removals of materials, so drawables that
    /// resolve material ids (`HasMaterial`) know when to again.
    std::uint64_t MaterialSetVersion() const {
        return materialSetVersion_;
    }

    /// The material drawables should use for `id`: that material, or
    /// the default one if it is gone.
    const RasterMaterial& MaterialOrDefault(std::uint32_t id) const;

    /// Sets the dome's latitude-longitude RGB map (`DomeMapOf`) drawn
    /// behind the scene where the dome is visible
    /// (`DomeLight::visible`), with its mips; none draws its radiance
    /// alone.
    /// @throws std::invalid_argument for an empty map or one larger
    /// than the device allows.
    void SetDomeMap(const Image* map);

    /// Whether the frame `Update` last prepared shadows its dome (the
    /// full dome lighting mode): drawables then provide what occludes
    /// it, such as the terrain's horizon map.
    bool DomeShadowed() const {
        return domeShadowed_;
    }

    /// The view straight down over a terrain's grid of what stands on
    /// it, which the terrain's horizon map reads
    /// (terrain_horizon.slang's `horizonCastersMain`): a light shadow
    /// map layer, drawn with the light views while the dome is
    /// shadowed and a terrain is set (`SetTerrainExtent`),
    /// orthographic over the grid with depth 0 at `top` (relative to
    /// the frame's origin) and 1 `span` below it. The terrain itself
    /// does not draw into it.
    struct CastersView {
        bool valid = false;
        std::uint32_t layer = 0;  ///< In `LightShadowMaps`.
        std::size_t shadowView =
            0;  ///< Its index among the drawables' shadow views.
        std::uint32_t resolution = 0;  ///< Texels a side.
        float top = 0.0f;
        float span = 0.0f;
    };
    const CastersView& Casters() const {
        return casters_;
    }

    /// The terrain's grid for the casters' view, absolute world x of
    /// its west edge, y of its north edge, width and height; none
    /// without a terrain.
    void SetTerrainExtent(const std::optional<glm::dvec4>& extent) {
        terrainExtent_ = extent;
        castersChanged_ = true;
    }

    /// Notes that what casts shadows may have moved or changed (the
    /// delegate calls it when the scene's geometry, transforms or
    /// materials change), so the terrain's horizon over it is
    /// recomputed; `CastersChanged` reads it until the frame's
    /// shadows are recorded.
    void MarkCastersChanged() {
        castersChanged_ = true;
    }
    bool CastersChanged() const {
        return castersChanged_;
    }

    /// The light shadow maps, a D32 2D array (one layer per light
    /// view and the casters' view).
    const GpuImage& LightShadowMaps() const {
        return *lightShadowMaps_;
    }

    /// A terrain's horizon map (terrain_horizon.slang) for the dome
    /// occlusion of its pixels: the image (two layers of four
    /// azimuths), where its texels lie relative to the frame's origin
    /// (north-west corner x and y, texel size, vertical
    /// exaggeration), and its width, height, the terrain's prim id
    /// and 1 to use it.
    struct TerrainHorizon {
        const GpuImage* image = nullptr;
        std::array<float, 4> grid{};
        std::array<std::uint32_t, 4> info{};
        /// The terrain's heights (R32F), for how far a point stands
        /// above the ground: their cell size, the frame origin's
        /// height (subtracted from the terrain's), the no-data value
        /// and 1 when there is one; their width and height.
        const GpuImage* heights = nullptr;
        std::array<float, 4> heightsGrid{};
        std::array<std::uint32_t, 4> heightsSize{};
        /// The heights above the ground the map's levels see from, in
        /// height units.
        std::array<float, 4> levels{};
    };

    /// Sets the terrain horizon this frame's dome occlusion reads; a
    /// default one for none.
    void SetTerrainHorizon(const TerrainHorizon& horizon) {
        terrainHorizon_ = horizon;
    }

    /// As `SetDomeMap(map)`, with `lighting` (when not nullptr)
    /// lighting the scene in its place: the map without a sun lit as
    /// a distant light of its own (`ExtractDomeSun`), while the sky
    /// still shows the sun.
    /// @throws std::invalid_argument as `SetDomeMap`.
    void SetDomeMap(const Image* map, const Image* lighting);

    /// Sets the stars (`StarsOf`) drawn behind the scene while frames
    /// have `FrameParams::stars`, binned into cells of a degree for
    /// each pixel to gather the stars around it.
    void SetStars(std::span<const Star> stars);

    /// Outlines the prims with ids `primIds` (PrimId values) in
    /// `Record`'s presented image, as Omniverse Kit outlines its
    /// selection: `width` pixels each side of their silhouettes, in
    /// display colour `colour` (sRGB-encoded rgb, alpha its opacity),
    /// from frames whose targets hold PrimId. No ids, no outline.
    /// Call between frames, before `Update`.
    void SetHighlight(std::span<const std::uint32_t> primIds,
                      glm::vec4 colour, std::uint32_t width = 2);

    /// Writes the frame, shading and sensor uniforms and the material
    /// table, fits the shadow cascades around the drawables' bounds
    /// and prepares every drawable. Call once per frame after
    /// `BeginFrame` and before `Record`; earlier frames may still be
    /// running, since host data reaches the buffers through `Uploads`
    /// when the frame records.
    /// @throws std::invalid_argument for an invalid camera
    /// (`CameraParams::Check`) or filter chain (`CheckFilterChain`),
    /// a sun or light spectrum with the wrong number of bands, more
    /// than `kMaxLights` lights, or a model whose shading pass fails
    /// to compile. `overlays` are drawn over LdrColor
    /// (`OverlayPass`); the caller selects them by
    /// `FrameParams::overlays`.
    void Update(const FrameParams& params,
                std::span<RasterDrawable* const> drawables,
                std::span<const Overlay* const> overlays = {});

    /// Records a frame of `drawables` into `targets` (`RecordShadows`
    /// then `RecordViews` from view 0): the shadow passes, the
    /// G-buffer, spectral shading and the sensor filters of the last
    /// `Update`. With `present`, then draws view 0's render var `var`
    /// into `present.color` (in the colour attachment layout, of
    /// `PresentFormat`) as a display image (`CanPresent`), and the
    /// overlays over it, whichever var it shows. Before any `Update`
    /// only clears `present`.
    /// @throws std::invalid_argument if `targets` do not have the
    /// camera's extent and bands, or `var` cannot be presented.
    void Record(VkCommandBuffer commands, RenderTargets& targets,
                std::span<const RasterDrawable* const> drawables,
                const RenderingTarget* present = nullptr,
                std::string_view var = render_vars::kLdrColor) const;

    /// Records the drawables' culling (`RasterDrawable::RecordCull`)
    /// and the shadow passes of the last `Update`, which every view
    /// of the frame shares.
    void RecordShadows(
        VkCommandBuffer commands,
        std::span<const RasterDrawable* const> drawables) const;

    /// Records views `firstView` to `firstView + targets.Views()` of
    /// the last `Update` into `targets`, after `RecordShadows`: each
    /// view's G-buffer pass, then spectral shading and the sensor
    /// filters, each pass over all the views before the next, so the
    /// passes of a batch of views need no more barriers than one
    /// view's. A frame of more views than one set of targets holds
    /// (`RenderTargets::MaxViews`) records in several calls. The
    /// `Update`'s overlays are drawn over each view's LdrColor last,
    /// unless `overlays` is false.
    ///
    /// With motion blur (`CameraParams::Subframes` above 1) each
    /// sub-frame draws and shades the views at its time in turn, the
    /// camera between its last frame's pose and this one's and meshes
    /// between their last frame's transforms and these; their
    /// radiance sums into what each row's exposure sees
    /// (motion.slang, a rolling shutter shifting the rows'
    /// exposures), which the sensor filters. The geometry vars and
    /// overlays are the last sub-frame's. Per-instance positions and
    /// deforming points do not blur, and shadows are the frame's.
    /// @throws std::invalid_argument if `targets` do not have the
    /// camera's extent and bands, or hold views past the frame's.
    void RecordViews(VkCommandBuffer commands, RenderTargets& targets,
                     std::span<const RasterDrawable* const> drawables,
                     std::uint32_t firstView,
                     bool overlays = true) const;

    /// Records every view of a frame with minimal shading
    /// (`FrameParams::minimal`) after `RecordShadows`: a chunk of
    /// views at a time (`MinimalTargets::Chunk`), one layered pass in
    /// which the drawables draw each view with their minimal
    /// pipelines, then the resolve pass (minimal_resolve.slang)
    /// writing the held vars' linear buffers.
    /// @throws std::invalid_argument if the last `Update` was not
    /// minimal, or `targets` do not have the camera's extent or the
    /// frame's views.
    void RecordMinimal(
        VkCommandBuffer commands, MinimalTargets& targets,
        std::span<const RasterDrawable* const> drawables) const;

    /// Byte offsets of a view's camera fields in a block of per-view
    /// data, for `PatchViews`; `kNone` leaves a field alone. Matrices
    /// are glm's column-major floats.
    struct ViewFields {
        static constexpr std::uint32_t kNone = 0xFFFFFFFFu;
        std::uint32_t viewProjection = kNone;
        std::uint32_t inverseViewProjection = kNone;
        std::uint32_t eye = kNone;  ///< Three floats.
        std::uint32_t forward =
            kNone;  ///< Three floats, four with `forwardDepth`.
        bool forwardDepth =
            false;  ///< The fourth is the eye's depth along forward.
        std::uint32_t planes =
            kNone;  ///< Six normalised frustum planes (`Frustum`).
        std::uint32_t pose = kNone;  ///< The camera-to-world matrix.
        /// Whether the blocks are one per output view, posed and
        /// projected as the camera (the sensor's), rather than one
        /// per raster view: a cube camera's per view and face, each
        /// turned onto its face and projected as the faces are.
        bool outputViews = false;
    };

    /// Records the patching of every view's `fields` in `blocks` (a
    /// storage buffer of a block per view, `stride` bytes apart from
    /// byte `first`) from the poses the last `Update` took from the
    /// GPU (`FrameParams::posesOnGpu`, in `PoseBuffer`); the caller
    /// follows it with a barrier. Does nothing for other frames.
    void PatchViews(VkCommandBuffer commands, const GpuBuffer& blocks,
                    VkDeviceSize first, VkDeviceSize stride,
                    const ViewFields& fields) const;

    /// Starts preparing a frame in `FrameUploads` slot `slot`, whose
    /// previous frame must have finished on the device; the frame's
    /// host data then reaches the device with its commands
    /// (`Uploads`), so earlier frames may still be running while it
    /// is prepared.
    void BeginFrame(std::uint32_t slot);

    /// Host data for the renderer's and the drawables' buffers,
    /// copied when the frame records.
    FrameUploads& Uploads() const {
        return uploads_;
    }

    /// The buffer the views' poses of a frame with
    /// `FrameParams::posesOnGpu` are copied into before it records
    /// (`ComputeInterop::CopyToBuffer`), row-major 4x4 float32 per
    /// view; valid after that frame's `Update` (or
    /// `EnsurePoseBuffer`).
    GpuBuffer& PoseBuffer() {
        return *poseBuffer_;
    }

    /// `PoseBuffer` to hold up to `views` views, which a compute API
    /// may view in place (`ComputeInterop::View`); a larger one waits
    /// for the device to idle. A view of the previous buffer keeps
    /// that buffer alive, so its values can still be copied.
    const std::shared_ptr<GpuBuffer>&
    EnsurePoseBuffer(std::uint32_t views);

    /// The poses of the last `Update`'s lights posed on the GPU
    /// (`Light::posedOnGpu`), row-major 4x4 float32 per light in
    /// their order, relative to the frame's origin: written in the
    /// prepass (`MeshDrawable::SetAnchoredLights`), read by the light
    /// pass.
    const GpuBuffer& LightPoseBuffer() const {
        return *lightPoses_;
    }

    /// The light shadow views' world-to-clip matrices
    /// (`LightShadowViews`, glm columns): the CPU's, and for views of
    /// lights posed on the GPU, the light pass's
    /// (`ShadowCascade::gpuFitted`).
    const GpuBuffer& LightViewMatrices() const {
        return lightViewMatrices_;
    }

    /// The camera-to-world poses the last frame with poses on the GPU
    /// in this `BeginFrame`'s slot rendered (a finished frame), if it
    /// had `views` views; otherwise empty. Estimates of the next
    /// frame's poses.
    std::vector<glm::dmat4> LastGpuPoses(std::uint32_t views) const;

    /// The number of views of the last `Update`
    /// (`FrameParams::Views`).
    std::uint32_t Views() const {
        return outputViews_;
    }

    /// The sub-frames each view of the last `Update` renders
    /// (`CameraParams::Subframes`).
    std::uint32_t Subframes() const {
        return subframes_;
    }

    /// The raster layers each view of the last `Update` draws: its
    /// camera's cube faces (`NeedsCubeFaces`), else 1. Render targets
    /// must hold as many (`RenderTargets::Faces`).
    std::uint32_t Faces() const {
        return faces_;
    }

    /// Whether `Record` can present render var `name`: every one.
    /// LdrColor shows as is; HdrColor clipped at 1; SpectralRadiance
    /// as grey, one band or their mean (`SetPresentBand`); distances
    /// from white at the camera's near plane to black at its far
    /// plane on a log scale; normals as rgb; Electrons over the full
    /// well and RawDN over the ADC range; ids as hashed colours,
    /// black for the background.
    static bool CanPresent(std::string_view name);

    /// The scene's average luminance in cd/m^2 as the latest finished
    /// frame that metered it measured (`FrameParams::meter`): the
    /// geometric mean of its HdrColor without the darkest tenth and
    /// the brightest twentieth of its lit pixels. None until a
    /// metered frame has finished, or if it saw no light.
    std::optional<double> MeteredLuminance() const {
        return meteredLuminance_;
    }

    /// As `MeteredLuminance`, of the latest finished frame metered
    /// under `key` (`SetMeterKey`): one camera's, among several.
    std::optional<double>
    MeteredLuminance(std::string_view key) const {
        const auto found = meteredByKey_.find(key);
        return found == meteredByKey_.end()
                   ? std::nullopt
                   : std::optional<double>(found->second);
    }

    /// The key the next `Update`'s frame meters under, if it meters.
    void SetMeterKey(std::string key) {
        meterKey_ = std::move(key);
    }

    /// Reads every finished frame's meter at once, rather than as its
    /// slot comes round again: for a frame metered to expose the next
    /// (`RasterDelegate::StepAsync`). Nothing may still be running.
    void ReadFinishedMeters();

    /// Which band `Record` shows when presenting SpectralRadiance: 0
    /// to the band count - 1, or -1 (or any other) for the mean over
    /// the bands. It shows as grey at the photographic image's
    /// brightness: a flat spectrum as bright as there.
    void SetPresentBand(int band) {
        presentBand_ = band;
    }

    /// The sensor passes of the last `Update`: one per chain
    /// (`FrameParams::chains`), in order.
    std::size_t SensorPasses() const {
        return sensorPasses_.size();
    }

    /// State slots per pixel the temporal filters of sensor pass
    /// `pass` keep.
    std::uint32_t SensorStateSlots(std::size_t pass = 0) const {
        return sensorPasses_.at(pass).plan.states;
    }

    /// The last `Update`'s sensor plan of pass `pass`.
    const SensorPlan& Plan(std::size_t pass = 0) const {
        return sensorPasses_.at(pass).plan;
    }

    /// The temporal filters' state of sensor pass `pass` for the next
    /// `Record`: images of `SensorStateSlots` layers per view,
    /// `previous` holding the last frame's (when `history`) and
    /// `next` taking this one's; null for none. Call after `Update`;
    /// the images must outlive the frame's recording.
    void SetSensorState(std::size_t pass, GpuImage* previous,
                        GpuImage* next, bool history) {
        SensorPass& sensor = sensorPasses_.at(pass);
        sensor.statePrevious = previous;
        sensor.stateNext = next;
        sensor.history = history && previous != nullptr;
    }

    /// The spectral bands and filter chain of the last `Update`.
    const SpectralBands& Bands() const {
        return bands_;
    }

    const std::vector<SensorFilter>& Filters() const {
        return filters_;
    }

    /// The image extent of the last `Update`.
    VkExtent2D Extent() const {
        return extent_;
    }

    /// Frame resources for drawables drawing view `view`: its
    /// uniforms, the shadows and samplers (bindings 0 to 4).
    RasterFrame Frame(std::uint32_t view = 0) const;

    /// Cascades fitted by the last `Update`; empty with shadows off.
    const std::vector<ShadowCascade>& Cascades() const {
        return cascades_;
    }

    /// Shadow views of the other lights fitted by the last `Update`,
    /// in light order.
    const std::vector<ShadowCascade>& LightShadowViews() const {
        return lightViews_;
    }

    /// GPU memory held for the frame, shadow maps and materials, in
    /// bytes.
    std::uint64_t GpuBytes() const;

  private:
    void CreateShadowMaps(std::uint32_t resolution,
                          std::uint32_t layers);
    /// Plans a batch's sun shadow maps: the passes, each view's layer
    /// and own fitted map.
    void PlanBatchShadows(const FrameParams& params,
                          const Bounds3& scene,
                          std::span<RasterDrawable* const> drawables);
    void CreateLightShadowMaps(std::uint32_t resolution,
                               std::uint32_t layers);
    void WriteLights(const FrameParams& params,
                     const std::optional<Bounds3>& bounds);
    const ComputePipeline& ShadingPipeline(std::size_t model,
                                           bool pushbroom = false);
    /// Sensor.slang's entry point `entry`, which reads only its lens,
    /// for lens kernel source `lens` (`SensorPlan::LensSource`) in
    /// `layout`, compiled when first wanted.
    const ComputePipeline& LensPipeline(const std::string& lens,
                                        const std::string& entry,
                                        const PipelineLayout& layout);
    /// Compiles the sky pass for a frame camera, or a pushbroom,
    /// unless it has.
    void SkyPipeline(bool pushbroom);
    /// Compiles `plan`'s fused sensor kernels, one per segment,
    /// unless already compiled.
    void SensorPipelines(const SensorPlan& plan);
    void WriteMaterialTable();
    void WriteShading(const FrameParams& params);
    /// The sensor's uniforms for `params`' camera, its views a
    /// sub-frame's each; `raster` the views the raster passes draw (a
    /// cube camera's faces).
    void WriteSensor(const FrameParams& params,
                     const FrameParams& raster);
    void EnsureViewBuffers(std::uint32_t views);
    /// Writes views.slang's camera (and `raster`'s faces' projection)
    /// and sizes the pose buffers for a frame posed on the GPU.
    void WritePoseCamera(const FrameParams& params,
                         const FrameParams& raster);
    /// Records the light pass (light_views.slang) for the lights
    /// posed on the GPU, after the prepass wrote their poses: their
    /// light table entries and shadow views, then every drawable's
    /// `RecordShadowViews`, each followed by a barrier. Nothing
    /// without such lights.
    void RecordLights(
        VkCommandBuffer commands,
        std::span<const RasterDrawable* const> drawables) const;
    /// Records `PatchViews` of the renderer's own per-view uniforms
    /// and every drawable's.
    void RecordViewPoses(
        VkCommandBuffer commands,
        std::span<const RasterDrawable* const> drawables) const;
    /// Builds `targets`' depth pyramid from their depth, every view
    /// at once, leaving it in the general layout.
    void BuildHzb(VkCommandBuffer commands,
                  RenderTargets& targets) const;
    /// Fills mips 1 onward of `image` (a channel image) from mip 0,
    /// each the mean of the one before.
    void BuildPyramid(VkCommandBuffer commands,
                      GpuImage& image) const;
    /// Records the sun's and the other lights' shadow maps of the
    /// frame's casters, with `frame`'s uniforms: a sub-frame's, whose
    /// time (`Uniforms::motion`) places moving casters.
    void
    RecordShadowMaps(VkCommandBuffer commands,
                     std::span<const RasterDrawable* const> drawables,
                     const RasterFrame& frame) const;
    /// Spectral shading and the sky of `targets`' views into their
    /// raster radiance, with the uniforms of view `firstView` on (a
    /// sub-frame's).
    void Shade(VkCommandBuffer commands, RenderTargets& targets,
               std::uint32_t firstView) const;
    /// Adds sub-frame `subframe`'s raster radiance into `targets`'
    /// sum (motion.slang), each row weighted for its share of its
    /// exposure; the last leaves the sum in the raster radiance.
    void Accumulate(VkCommandBuffer commands, RenderTargets& targets,
                    std::uint32_t subframe) const;
    /// Writes each internal view's MotionUniforms for a frame holding
    /// the MotionVector var: the last frame's view-projection from
    /// `frame`'s `PreviousPose` (a cube camera's faces'), and makes
    /// the camera pass.
    void WriteMotion(const FrameParams& params,
                     const FrameParams& frame);
    /// Records the MotionVector var's passes for output views
    /// `firstView` on, after their last sub-frame: the camera's
    /// motion for every pixel (motion_vectors.slang), then each
    /// drawable's moving draws over it
    /// (`RasterDrawable::RecordMotion`), into `targets.Raster`.
    void
    RecordMotion(VkCommandBuffer commands, RenderTargets& targets,
                 std::span<const RasterDrawable* const> drawables,
                 std::uint32_t firstView) const;
    /// The sensor filters of output views `firstView` on.
    void Filter(VkCommandBuffer commands, RenderTargets& targets,
                std::uint32_t firstView) const;
    /// The internal view (uniforms, culling) of output view `view` at
    /// its last sub-frame, whose G-buffer the sensor and overlays
    /// read; its first face's for a cube camera.
    std::uint32_t LastSubframeView(std::uint32_t view) const {
        return ((subframes_ - 1) * outputViews_ + view) * faces_;
    }
    void Present(VkCommandBuffer commands, RenderTargets& targets,
                 const RenderingTarget& present,
                 std::string_view var) const;
    /// Records the measuring of view 0's visible depth range into
    /// this slot's buffer.
    void RecordDepthRange(VkCommandBuffer commands,
                          RenderTargets& targets) const;
    /// Records the counting of view 0's log-luminance histogram into
    /// this slot's buffer.
    void RecordMeter(VkCommandBuffer commands,
                     RenderTargets& targets) const;
    /// Reads slot `slot`'s histogram, if its frame metered, into
    /// `meteredLuminance_` and its key's `meteredByKey_`.
    void ReadMeter(std::uint32_t slot);
    /// The view depths from which the sun's cascades start and to
    /// which they reach for a `ShadowFit::Visible` frame: those a
    /// recent frame of the same stream saw, if its view was close to
    /// this one; otherwise none.
    std::optional<std::pair<float, float>>
    VisibleDepths(const FrameParams& params);
    /// Cascades of a light towards `towards` (a unit vector) over
    /// `cascadeFit_`'s view, `resolution` texels across each: the
    /// sun's, and other distant lights' in their own maps.
    std::vector<ShadowCascade>
    FitViewCascades(const glm::vec3& towards,
                    std::uint32_t resolution) const;
    /// Records the overlays over views `firstView` on of `targets`'
    /// LdrColor.
    void RecordOverlays(VkCommandBuffer commands,
                        RenderTargets& targets,
                        std::uint32_t firstView) const;
    /// Records the map of a cube camera's faces' sensor rows
    /// (`faceRows_`), for its rolling shutter.
    void RecordFaceRows(VkCommandBuffer commands,
                        std::uint32_t firstView) const;
    /// `RecordOverlays` for a camera rendered on cube faces: drawn on
    /// its faces, then resampled through its lens onto `color`; or,
    /// with `store`, into `color` as their colour over black and what
    /// shows through (alpha), for view 0 only, for
    /// `PresentFaceOverlays`.
    void RecordFaceOverlays(VkCommandBuffer commands,
                            RenderTargets& targets,
                            std::uint32_t firstView, GpuImage& color,
                            bool store = false) const;
    /// A cube camera's overlays over the presented var `present`
    /// shows when it is not LdrColor: resampled through the lens
    /// (`lensOverlays_`), then blended over it by the present pass.
    void PresentFaceOverlays(VkCommandBuffer commands,
                             RenderTargets& targets,
                             const RenderingTarget& present) const;
    /// The overlay pass's target for view `view`, reading `targets`'
    /// depth layer `layer`.
    OverlayPass::Target OverlayTarget(RenderTargets& targets,
                                      std::uint32_t view,
                                      std::uint32_t layer) const;

    const VulkanContext& context_;
    VkFormat presentFormat_;
    ShaderCompiler compiler_;
    BrdfRegistry brdfs_;
    LightPatternRegistry lightPatterns_ =
        LightPatternRegistry::WithBuiltins();
    /// Recorded by the const `Record*` calls, so mutable.
    mutable FrameUploads uploads_;
    std::map<std::uint32_t, std::unique_ptr<RasterMaterial>>
        materials_;
    std::uint32_t nextMaterial_ = 1;
    std::uint64_t materialSetVersion_ = 0;
    glm::dvec3 materialsOrigin_{0.0};  ///< The origin the materials'
                                       ///< uniforms are relative to.

    GpuBuffer uniforms_;
    GpuBuffer shadowUniforms_;
    Sampler clampSampler_;
    Sampler shadowSampler_;
    std::unique_ptr<GpuImage> shadowMaps_;
    std::uint32_t shadowResolution_ = 0;
    std::vector<ShadowCascade> cascades_;
    /// What a single view's cascades fit to this frame: its view, a
    /// cube camera's face extent (0 for others), the casters' box
    /// down to any floor they shadow, the split count and weight, and
    /// the depths it saw (a visible fit's). None in a batch, without
    /// bounds, or without a distant light's shadows.
    struct CascadeFit {
        CameraView view;
        float around = 0.0f;
        glm::vec3 casterLow{0.0f};
        glm::vec3 casterHigh{0.0f};
        int count = 1;
        float lambda = 0.5f;
        std::optional<std::pair<float, float>> seen;
    };
    std::optional<CascadeFit> cascadeFit_;
    std::vector<ShadowCascade> lightViews_;
    std::vector<ShadowCascade> shadowViews_;
    std::unique_ptr<GpuImage> lightShadowMaps_;
    CastersView casters_;
    std::optional<glm::dvec4> terrainExtent_;
    mutable bool castersChanged_ = true;
    std::uint32_t lightResolution_ = 0;
    std::vector<glm::vec2> poisson_;
    bool shadowsEnabled_ = false;
    bool updated_ = false;
    /// Internal views: a view per sub-frame, output view and face
    /// (`FrameParams::Subframes`, then `FrameParams::Faces`),
    /// sub-frame `k`'s output view `v`'s face `f` at `(k *
    /// outputViews_ + v) * faces_ + f`, which uniforms, culling and
    /// drawables index.
    std::uint32_t views_ = 1;
    std::uint32_t outputViews_ =
        1;  ///< The frame's own views (`FrameParams::Views`).
    std::uint32_t subframes_ =
        1;  ///< Sub-frames per view: above 1 for motion blur.
    std::uint32_t faces_ =
        1;  ///< Raster layers per view: a cube camera's faces.
    /// Whether the last `Update`'s frame holds the MotionVector var,
    /// which runs the motion passes.
    bool motionVectors_ = false;
    /// Per internal view, frame.slang's MotionUniforms (256 bytes
    /// each), written only for frames holding the MotionVector var;
    /// and the camera pass (motion_vectors.slang), made then.
    std::unique_ptr<GpuBuffer> motionUniforms_;
    std::unique_ptr<PipelineLayout> cameraMotionLayout_;
    std::unique_ptr<ComputePipeline> cameraMotionPipeline_;
    /// The shutter for motion.slang: the first sub-frame's time and
    /// the spacing, the shutter's opening and closing and the rolling
    /// readout, in frames.
    std::array<float, 5> shutter_{};
    /// Whether the last `Update`'s camera rolls its shutter over cube
    /// faces, whose texels then take their sensor rows' times
    /// (`faceRows_`).
    bool RollsFaces() const {
        return faces_ > 1 && subframes_ > 1 && shutter_[4] > 0.0f;
    }
    /// Whether the last `Update`'s views take their poses from
    /// `poseBuffer_`, and the views of `poseHistory_` (the poses it
    /// last rendered) then.
    bool posesOnGpu_ = false;
    bool posesQuaternions_ = false;  ///< Whether `poseBuffer_` holds
                                     ///< positions and quaternions.
    std::array<std::uint32_t, FrameUploads::kSlots>
        poseHistoryViews_{};
    std::array<glm::dvec3, FrameUploads::kSlots>
        poseHistoryOrigins_{};  ///< Their frames' origins.
    /// One pass rendering the sun's casters `casters` for cascade
    /// `cascade` into shadow map layer `layer`, starting from a copy
    /// of layer 0 with `fromBase`.
    struct ShadowPass {
        std::uint32_t layer = 0;
        std::uint32_t cascade = 0;
        ShadowCasters casters;
        bool fromBase = false;
    };
    std::vector<ShadowPass> shadowPasses_;
    std::vector<std::uint32_t>
        viewShadowLayers_;  ///< Per view, its sun map's first layer.
    std::vector<std::int32_t>
        viewSunCascades_;  ///< Per view, its own fitted map, or -1.
    std::uint32_t shadowLayers_ =
        1;  ///< Layers the frame's sun maps take.
    ShadowCasters
        lightCasters_;  ///< Casters of the other lights' maps.
    std::vector<std::string_view>
        requiredVars_;  ///< The vars the last `Update`'s frame needs.

    PipelineLayout shadingLayout_;
    /// Per model and whether for a pushbroom's views.
    std::map<std::pair<std::size_t, bool>,
             std::unique_ptr<ComputePipeline>>
        shadingPipelines_;
    PipelineLayout filterLayout_;
    /// The fused sensor kernel of each plan used so far, by
    /// `SensorPlan::Key`.
    std::map<std::string, std::unique_ptr<ComputePipeline>>
        sensorPipelines_;
    SensorFilterRegistry sensorFilters_ =
        SensorFilterRegistry::WithStandard();
    PipelineLayout presentLayout_;
    PipelineLayout hzbLayout_;
    std::unique_ptr<ComputePipeline> hzbPipeline_;
    /// Pyramids of the channel images for spatial filters
    /// (pyramid.slang, Hi-Z's layout).
    std::unique_ptr<ComputePipeline> pyramidPipeline_;
    PipelineLayout motionLayout_;
    std::unique_ptr<ComputePipeline>
        motionPipeline_;  ///< Sub-frame accumulation (motion.slang).
    PipelineLayout viewsLayout_;
    std::unique_ptr<ComputePipeline> viewsPipeline_;
    GpuBuffer poseCamera_;  ///< views.slang's camera: parent,
                            ///< projection, its inverse.
    std::shared_ptr<GpuBuffer> poseBuffer_;
    std::array<std::unique_ptr<GpuBuffer>, FrameUploads::kSlots>
        poseHistory_;
    std::unique_ptr<GraphicsPipeline> presentFloat_;
    std::unique_ptr<GraphicsPipeline> presentUint_;
    std::unique_ptr<GraphicsPipeline> presentSpectral_;
    std::unique_ptr<GraphicsPipeline> presentOutline_;
    std::unique_ptr<GraphicsPipeline> presentOverlay_;
    std::unique_ptr<GpuBuffer>
        highlight_;  ///< A bit per highlighted prim id.
    /// One chain's sensor kernels: its plan, its custom filters'
    /// parameters and the temporal filters' state images
    /// (`SetSensorState`, or null).
    struct SensorPass {
        SensorPlan plan;
        /// The plan's parameters, then a lens of rays' table
        /// (`tableOffset` on).
        std::unique_ptr<GpuBuffer> params;
        std::uint32_t tableOffset = 0;
        GpuImage* statePrevious = nullptr;
        GpuImage* stateNext = nullptr;
        bool history = false;
    };
    /// A lens of rays' table, as sensor.slang's
    /// `SensorConstants::table` takes it: (0, step, columns, rows);
    /// all zero without one.
    std::array<std::uint32_t, 4> lensTable_{};
    std::vector<SensorPass> sensorPasses_;
    std::unique_ptr<GpuImage>
        statePlaceholder_;  ///< Bound without state.
    std::vector<std::uint32_t> highlightBits_;
    std::uint32_t highlightWords_ = 0;  ///< 0 draws no outline.
    std::uint32_t highlightWidth_ = 2;
    glm::vec4 highlightColour_{1.0f, 0.6f, 0.1f, 1.0f};
    int presentBand_ = -1;  ///< The SpectralRadiance band `Present`
                            ///< shows; -1 their mean.
    float spectralGain_ = 1.0f;  ///< SpectralRadiance's display gain,
                                 ///< from the exposure.
    /// Present ranges of the last `Update`: near and far plane, full
    /// well, ADC maximum.
    std::array<float, 4> presentRange_{};
    GpuImage sigmoidTable_;
    Sampler tableSampler_;
    GpuBuffer shadingUniforms_;
    GpuBuffer sensorUniforms_;
    GpuBuffer lightTable_;
    /// Per light slot, its IES profile's candela
    /// (`LightParams::profile`), where `LightGpu`'s shadow.w points.
    GpuBuffer lightProfiles_;
    /// Per light slot, the integral of its laser's pattern
    /// (`LightPatternModel`), which shading divides it by so the
    /// laser emits its power: 1 for a normalized pattern, else
    /// computed each frame from `patternJobs_` by `patternPipeline_`.
    GpuBuffer patternNorms_;
    GpuBuffer patternJobs_;
    std::uint32_t patternJobCount_ = 0;
    PipelineLayout patternLayout_;
    std::unique_ptr<ComputePipeline> patternPipeline_;
    /// The lasers' pattern data and images (shading.slang's
    /// lightPatternData), grown as they need, and what it holds:
    /// per laser its data and image, so unchanged ones upload once.
    std::unique_ptr<GpuBuffer> patternData_;
    std::vector<std::pair<std::vector<float>,
                          std::shared_ptr<const LightPatternImage>>>
        patternDataHeld_;
    /// Per shaded view and light slot, `LightParams::Beam::kMaxLit`
    /// (sequence frame, weight) pairs: what a laser shows over the
    /// view's span of the sensor clock (shading.slang's
    /// lightTiming), grown as the views need.
    std::unique_ptr<GpuBuffer> lightTiming_;
    /// Records the integration of the frame's light patterns.
    void RecordPatternNorms(VkCommandBuffer commands) const;
    /// Writes `lightTiming_` for `params`' lasers over `views` shaded
    /// views: a single frame's over the camera's exposure ending at
    /// its shutter's close, a sub-frame's over its share of the
    /// shutter around its time.
    void WriteLightTiming(const FrameParams& frame,
                          const FrameParams& params,
                          std::uint32_t views);
    GpuBuffer lightViewMatrices_;
    /// Per light shadow view, a Shadows block (frame.slang) whose
    /// first matrix its shadow pass draws with
    /// (`RasterFrame::shadowFromFrame`), `kLightShadowBlockStride`
    /// apart.
    GpuBuffer lightShadowBlocks_;
    /// Lights posed on the GPU: their poses (`LightPoseBuffer`), the
    /// light pass's records (two float4 each, light_views.slang) and
    /// count, and the pass.
    std::unique_ptr<GpuBuffer> lightPoses_;
    std::unique_ptr<GpuBuffer> lightRecords_;
    std::uint32_t gpuLights_ = 0;
    glm::vec4 lightBoundsLow_{0.0f};
    glm::vec4 lightBoundsHigh_{0.0f};
    PipelineLayout lightsLayout_;
    std::unique_ptr<ComputePipeline> lightsPipeline_;
    std::unique_ptr<GpuBuffer> materialTable_;
    /// What shows behind the scene: the dome's map (a white texel
    /// without one), the stars sorted by grid cell (`kStarColumns` by
    /// `kStarRows`), each cell's first star and count, and the last
    /// `Update`'s kSky* bits (shading.slang); the sky pass runs when
    /// any are set.
    static constexpr std::uint32_t kStarColumns = 360;
    static constexpr std::uint32_t kStarRows = 180;
    std::unique_ptr<GpuImage> domeMap_;
    /// The map that lights the scene when it differs from `domeMap_`
    /// (shading.slang's binding 16).
    std::unique_ptr<GpuImage> domeLighting_;
    /// `map` (a white texel for nullptr) as a float image with its
    /// mips.
    std::unique_ptr<GpuImage> UploadDomeMap(const Image* map) const;

    /// Dome occlusion (the full dome lighting mode): whether this
    /// frame computes it, the passes (shading.slang's `occlusionMain`
    /// and `occlusionFilterMain`, made the first time a frame does),
    /// and their per-view images, raw then filtered, made at the
    /// frame's size when first needed; a placeholder texel binds in
    /// their place otherwise.
    bool domeShadowed_ = false;
    TerrainHorizon terrainHorizon_;
    std::unique_ptr<GpuImage>
        noHorizon_;  ///< Binds in place of a terrain's horizon map.
    std::unique_ptr<GpuImage>
        noHeights_;  ///< Binds in place of a terrain's heights.
    std::unique_ptr<AtmosphereLuts>
        atmosphere_;  ///< The planet's atmosphere and its tables.
    std::unique_ptr<ComputePipeline> occlusionPipeline_;
    std::unique_ptr<ComputePipeline> occlusionFilterPipeline_;
    mutable std::array<std::unique_ptr<GpuImage>, 2> occlusion_;
    std::unique_ptr<GpuImage> noOcclusion_;
    bool hasDomeMap_ = false;
    std::unique_ptr<GpuBuffer> stars_;
    std::unique_ptr<GpuBuffer> starCells_;
    std::size_t starCount_ = 0;
    std::uint32_t sky_ = 0;
    /// For frame cameras and, once one is seen, pushbrooms.
    std::array<std::unique_ptr<ComputePipeline>, 2> skyPipelines_;
    std::vector<std::size_t> modelsInUse_;
    SpectralBands bands_;
    std::vector<SensorFilter> filters_;
    VkExtent2D extent_{0, 0};
    std::unique_ptr<OverlayPass> overlays_;
    /// A cube camera's overlays: each face's drawn over black and
    /// over white (a layer per view and face), then resampled through
    /// the lens onto LdrColor (sensor.slang's `overlayLensMain`, a
    /// pipeline per lens kernel); made when first drawn.
    mutable std::array<std::unique_ptr<GpuImage>, 2> faceOverlays_;
    /// The same resampled for a presented var other than LdrColor:
    /// colour over black, and what shows through as alpha
    /// (`PresentFaceOverlays`).
    mutable std::unique_ptr<GpuImage> lensOverlays_;
    std::unique_ptr<PipelineLayout> overlayLensLayout_;
    /// A cube camera with a rolling shutter: its faces' sensor rows
    /// (sensor.slang's `faceRowsMain`), which motion.slang times
    /// their texels by, made when first wanted; a placeholder for
    /// other cameras.
    mutable std::unique_ptr<GpuImage> faceRows_;
    std::unique_ptr<GpuImage> noFaceRows_;
    std::unique_ptr<PipelineLayout> faceRowsLayout_;
    /// `LensPipeline`'s, by entry point and lens kernel source.
    std::map<std::string, std::unique_ptr<ComputePipeline>,
             std::less<>>
        lensPipelines_;
    /// The lens kernel source (`SensorPlan::LensSource`) of the last
    /// `Update`'s cube camera, for its overlays and faces' rows;
    /// empty for others.
    std::string faceLens_;

    /// Sample distribution shadow maps (`ShadowFit::Visible`): each
    /// `FrameUploads` slot's frame measures its view's visible depth
    /// range into its buffer (`RecordDepthRange`), which a later
    /// frame in the slot reads once that frame has finished.
    struct DepthSample {
        bool measured = false;
        std::uint32_t stream =
            0;  ///< `FrameParams::stream`: whose view it saw.
        glm::dvec3 eye{0.0};
        glm::vec3 forward{0.0f};
        float nearest =
            0.0f;  ///< The visible view depths, once read.
        float farthest = 0.0f;
    };
    PipelineLayout depthRangeLayout_;
    std::unique_ptr<ComputePipeline> depthRangePipeline_;
    std::array<std::unique_ptr<GpuBuffer>, FrameUploads::kSlots>
        depthRanges_;
    std::array<DepthSample, FrameUploads::kSlots> depthSlots_;
    std::map<std::uint32_t, DepthSample>
        seenDepths_;             ///< The latest read, by stream.
    bool measureDepth_ = false;  ///< Whether the last `Update`'s
                                 ///< frame measures its depths.

    /// Exposure metering (`FrameParams::meter`): each slot's frame
    /// counts its view's log-luminance histogram into its buffer
    /// (`RecordMeter`), read by a later frame in the slot.
    PipelineLayout meterLayout_;
    std::unique_ptr<ComputePipeline> meterPipeline_;
    /// The last `Update`'s minimal shading (`RecordMinimal`), its
    /// resolve pass (made the first time a frame asks) and the buffer
    /// bound for vars a frame does not hold.
    MinimalMode minimal_ = MinimalMode::Off;
    glm::vec3 minimalClear_{
        0.0f};  ///< The background's linear colour
                ///< (`FrameParams::clearColor`).
    PipelineLayout resolveLayout_;
    std::unique_ptr<ComputePipeline> resolvePipeline_;
    GpuBuffer unheldVar_;
    std::array<std::unique_ptr<GpuBuffer>, FrameUploads::kSlots>
        meterBins_;
    std::array<bool, FrameUploads::kSlots> metered_{};
    std::optional<double> meteredLuminance_;
    /// The key each slot's frame metered under, the next frame's, and
    /// the latest luminance of each.
    std::array<std::string, FrameUploads::kSlots> meterKeys_;
    std::string meterKey_;
    std::map<std::string, double, std::less<>> meteredByKey_;
    bool meter_ =
        false;  ///< Whether the last `Update`'s frame meters.
    std::vector<std::int32_t>
        viewWorlds_;  ///< Per view of the last `Update`, its world.
    std::array<float, 2>
        clipRange_{};  ///< The last `Update`'s near and far planes.
    bool orthographic_ =
        false;  ///< Whether the last `Update`'s camera is.
    bool pushbroom_ =
        false;  ///< Whether the last `Update`'s camera is one.
};

}  // namespace ernie
