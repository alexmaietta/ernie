#pragma once

#include "ernie/device/buffer.h"
#include "ernie/device/image.h"
#include "ernie/device/pipeline.h"
#include "ernie/device/shader_compiler.h"
#include "ernie/raster/drawable.h"
#include "ernie/raster/material.h"
#include "ernie/raster/render_targets.h"
#include "ernie/scene/brdf.h"

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace ernie {

/// The shadow views of `light` over a scene within `sceneMin` to `sceneMax`, for square maps of
/// `resolution` texels:
/// - a distant light: one orthographic view along the light enclosing the scene's bounding sphere;
///   `texelWorldSize` is the world size of a texel;
/// - a sphere light whose cone is at most `Light::kMaxSpotConeDegrees`: one perspective view along
///   its axis covering the cone;
/// - any other sphere light: six 90-degree perspective views, the faces of a cube in the order
///   +x, -x, +y, -y, +z, -z.
/// Perspective views reach from the light (its radius, at least 1/10000 of the far plane) to the
/// scene's farthest corner, and their `texelWorldSize` is the size of a texel one unit away. None
/// for a light without shadows.
std::vector<ShadowCascade> FitLightShadowViews(const Light& light, const glm::vec3& sceneMin,
                                               const glm::vec3& sceneMax, std::uint32_t resolution);

/// Ernie's rasterizer: owns what the drawables of a frame share, and draws them, spectrally.
///
/// Shared are the frame uniforms (camera, sun, ambient), the lights, the shadow maps, a table of
/// materials, the reflectance models and the shader compiler. Drawables (`TerrainDrawable`,
/// `MeshDrawable`,
/// ...) are created against a renderer and passed to `Update` and `Record` each frame: the
/// renderer fits the shadow cascades around all their bounds, so every drawable casts shadows on
/// every other.
///
/// A frame is deferred and spectral. The drawables rasterize a G-buffer (shading normals, shadow
/// visibility, albedo scale and modulation, material and prim ids, depth); a compute pass per
/// reflectance model in use turns it into spectral radiance in the camera's bands, lit by the
/// sun's spectral irradiance and the other lights (`Light`); then the camera's sensor filters
/// (`SensorFilter`) turn that into the render vars, and a present pass can draw LdrColor into a
/// colour attachment.
///
/// A frame can render a batch of views of one camera at different poses (`FrameParams::poses`),
/// such as one per environment in reinforcement learning, into layered targets in one command
/// buffer: every view has uniforms of its own, and drawables prepare and draw each view
/// (`RasterDrawable::RecordColor`). A batch's sun shadows use one map fitted to the whole scene
/// rather than cascades fitted to each view: a base layer of what every world shares, copied into
/// one layer per world of the batch's views (`FrameParams::worlds`) on which that world's casters
/// draw, so worlds overlapping in space shadow only themselves. Other lights' maps hold what
/// every world shares.
///
/// The sun's shadows use cascaded shadow maps (`FitCascades`) filtered with rotated Poisson-disk
/// PCF, with slope-scaled depth bias and normal offset against acne, and penumbrae that widen with
/// the distance to the occluder for the sun's angular size (percentage-closer soft shadows). Other
/// lights' shadow maps (`FitLightShadowViews`) are sampled with the same filter in the shading
/// pass.
///
/// References:
/// - L. Williams, "Casting Curved Shadows on Curved Surfaces", SIGGRAPH '78, 270-274, 1978.
///   https://doi.org/10.1145/965139.807402
/// - W. T. Reeves, D. H. Salesin and R. L. Cook, "Rendering Antialiased Shadows with Depth Maps",
///   SIGGRAPH '87, 283-291, 1987. https://doi.org/10.1145/37402.37435
/// - D. Holbert, "Saying 'Goodbye' to Shadow Acne", GDC 2011 poster (normal offset shadows).
/// - J. Jimenez, "Next Generation Post Processing in Call of Duty: Advanced Warfare", SIGGRAPH 2014
///   Advances in Real-Time Rendering course (interleaved gradient noise).
/// - R. Fernando, "Percentage-Closer Soft Shadows", SIGGRAPH 2005 Sketches, 2005.
///   https://doi.org/10.1145/1187112.1187153
/// - T. Saito and T. Takahashi, "Comprehensible Rendering of 3-D Shapes", SIGGRAPH '90, 197-206,
///   1990. https://doi.org/10.1145/97880.97901 (G-buffers and deferred shading).
/// - W. Jakob and J. Hanika, "A Low-Dimensional Function Space for Efficient Spectral
///   Upsampling", Computer Graphics Forum 38(2):147-155, 2019 (RGB modulation to spectra).
class RasterRenderer {
  public:
    static constexpr int kMaxCascades = 4;

    /// Lights besides the sun per frame, and shadow map views among them; lights past the views
    /// draw without shadows.
    static constexpr std::size_t kMaxLights = 16;
    static constexpr std::size_t kMaxLightShadowViews = 24;

    /// Material 0, which always exists and drawables use by default: Lambert with albedo 1.
    static constexpr std::uint32_t kDefaultMaterial = 0;

    /// Shades with the models in `brdfs`, and presents LdrColor into colour attachments of
    /// `presentFormat`.
    /// @throws std::runtime_error if the shared resources cannot be created.
    explicit RasterRenderer(const VulkanContext& context,
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

    const BrdfRegistry& Brdfs() const {
        return brdfs_;
    }

    /// Registers `model` after checking that its Slang module compiles against `IBrdf`.
    /// @throws std::invalid_argument if the registry rejects the model (`BrdfRegistry::Add`) or
    /// its module fails to compile; the message includes the compiler's diagnostics.
    void AddBrdf(BrdfModel model);

    /// Adds a material with `settings` and returns its id.
    /// @throws std::invalid_argument as `SetMaterial`.
    std::uint32_t AddMaterial(const MaterialSettings& settings = {});

    /// Sets a material's settings, resolving its BRDF parameters.
    /// @throws std::invalid_argument for an unknown material or BRDF.
    void SetMaterial(std::uint32_t id, const MaterialSettings& settings);

    /// The material with `id`, for its albedo map and detail textures.
    /// @throws std::invalid_argument for an unknown material.
    RasterMaterial& Material(std::uint32_t id);
    const RasterMaterial& Material(std::uint32_t id) const;

    bool HasMaterial(std::uint32_t id) const {
        return materials_.contains(id);
    }

    /// Removes a material; drawables using it fall back to the default material.
    /// @throws std::invalid_argument for the default material or an unknown one.
    void RemoveMaterial(std::uint32_t id);

    /// The material drawables should use for `id`: that material, or the default one if it is gone.
    const RasterMaterial& MaterialOrDefault(std::uint32_t id) const;

    /// Writes the frame, shading and sensor uniforms and the material table, fits the shadow
    /// cascades around the drawables' bounds and prepares every drawable. Call once per frame
    /// before `Record`, only while no submitted frame is still reading the renderer's or the
    /// drawables' buffers.
    /// @throws std::invalid_argument for an invalid camera (`Camera::Check`) or filter
    /// chain (`CheckFilterChain`), a sun or light spectrum with the wrong number of bands, more
    /// than `kMaxLights` lights, or a model whose shading pass fails to compile.
    void Update(const FrameParams& params, std::span<RasterDrawable* const> drawables);

    /// Records a frame of `drawables` into `targets` (`RecordShadows` then `RecordViews` from view
    /// 0): the shadow passes, the G-buffer, spectral shading and the sensor filters of the last
    /// `Update`. With `present`, then draws view 0's render var `var`
    /// into `present.color` (in the colour attachment layout, of `PresentFormat`) as a display
    /// image (`CanPresent`). Before any `Update` only clears `present`.
    /// @throws std::invalid_argument if `targets` do not have the camera's extent and bands, or
    /// `var` cannot be presented.
    void Record(VkCommandBuffer commands, RenderTargets& targets,
                std::span<const RasterDrawable* const> drawables,
                const RenderingTarget* present = nullptr,
                std::string_view var = render_vars::kLdrColor) const;

    /// Records the drawables' culling (`RasterDrawable::RecordCull`) and the shadow passes of the
    /// last `Update`, which every view of the frame shares.
    void RecordShadows(VkCommandBuffer commands,
                       std::span<const RasterDrawable* const> drawables) const;

    /// Records views `firstView` to `firstView + targets.Views()` of the last `Update` into
    /// `targets`, after `RecordShadows`: each view's G-buffer pass, then spectral shading and the
    /// sensor filters, each pass over all the views before the next, so the passes of a batch of
    /// views need no more barriers than one view's. A frame of more views than one set of targets
    /// holds (`RenderTargets::MaxViews`) records in several calls.
    /// @throws std::invalid_argument if `targets` do not have the camera's extent and bands, or
    /// hold views past the frame's.
    void RecordViews(VkCommandBuffer commands, RenderTargets& targets,
                     std::span<const RasterDrawable* const> drawables,
                     std::uint32_t firstView) const;

    /// Byte offsets of a view's camera fields in a block of per-view data, for `PatchViews`;
    /// `kNone` leaves a field alone. Matrices are glm's column-major floats.
    struct ViewFields {
        static constexpr std::uint32_t kNone = 0xFFFFFFFFu;
        std::uint32_t viewProjection = kNone;
        std::uint32_t inverseViewProjection = kNone;
        std::uint32_t eye = kNone;      ///< Three floats.
        std::uint32_t forward = kNone;  ///< Three floats, four with `forwardDepth`.
        bool forwardDepth = false;      ///< The fourth is the eye's depth along forward.
        std::uint32_t planes = kNone;   ///< Six normalised frustum planes (`Frustum`).
        std::uint32_t pose = kNone;     ///< The camera-to-world matrix.
    };

    /// Records the patching of every view's `fields` in `blocks` (a storage buffer of a block per
    /// view, `stride` bytes apart from byte `first`) from the poses the last `Update` took from
    /// the GPU (`FrameParams::posesOnGpu`, in `PoseBuffer`); the caller follows it with a
    /// barrier. Does nothing for other frames.
    void PatchViews(VkCommandBuffer commands, const GpuBuffer& blocks, VkDeviceSize first,
                    VkDeviceSize stride, const ViewFields& fields) const;

    /// The buffer the views' poses of a frame with `FrameParams::posesOnGpu` are copied into
    /// before it records (`CudaInterop::CopyToBuffer`), row-major 4x4 float32 per view; valid
    /// after that frame's `Update`.
    GpuBuffer& PoseBuffer() {
        return *poseBuffer_;
    }

    /// The camera-to-world poses the last frame with poses on the GPU rendered, once it has
    /// finished, if it had `views` views; otherwise empty. Estimates of the next frame's poses.
    std::vector<glm::dmat4> LastGpuPoses(std::uint32_t views) const;

    /// The number of views of the last `Update` (`FrameParams::Views`).
    std::uint32_t Views() const {
        return views_;
    }

    /// Whether `Record` can present render var `name`: every one but SpectralRadiance. LdrColor
    /// shows as is; HdrColor clipped at 1; distances from white at the camera's near plane to
    /// black at its far plane on a log scale; normals as rgb; Electrons over the full well and
    /// RawDN over the ADC range; ids as hashed colours, black for the background.
    static bool CanPresent(std::string_view name);

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

    /// Frame resources for drawables drawing view `view`: its uniforms, the shadows and samplers
    /// (bindings 0 to 4).
    RasterFrame Frame(std::uint32_t view = 0) const;

    /// Cascades fitted by the last `Update`; empty with shadows off.
    const std::vector<ShadowCascade>& Cascades() const {
        return cascades_;
    }

    /// Shadow views of the other lights fitted by the last `Update`, in light order.
    const std::vector<ShadowCascade>& LightShadowViews() const {
        return lightViews_;
    }

    /// GPU memory held for the frame, shadow maps and materials, in bytes.
    std::uint64_t GpuBytes() const;

  private:
    void CreateShadowMaps(std::uint32_t resolution, std::uint32_t layers);
    /// Plans a batch's sun shadow maps: the passes, each view's layer and own fitted map.
    void PlanBatchShadows(const FrameParams& params, const Bounds3& scene,
                          std::span<RasterDrawable* const> drawables);
    void CreateLightShadowMaps(std::uint32_t resolution, std::uint32_t layers);
    void WriteLights(const FrameParams& params, const std::optional<Bounds3>& bounds);
    const ComputePipeline& ShadingPipeline(std::size_t model);
    void WriteMaterialTable();
    void WriteShading(const FrameParams& params);
    void WriteSensor(const FrameParams& params);
    void EnsureViewBuffers(std::uint32_t views);
    /// Writes views.slang's camera and sizes the pose buffers for a frame posed on the GPU.
    void WritePoseCamera(const FrameParams& params);
    /// Records `PatchViews` of the renderer's own per-view uniforms and every drawable's.
    void RecordViewPoses(VkCommandBuffer commands,
                         std::span<const RasterDrawable* const> drawables) const;
    /// Builds `targets`' depth pyramid from their depth, every view at once, leaving it in the
    /// general layout.
    void BuildHzb(VkCommandBuffer commands, RenderTargets& targets) const;
    void ShadeAndFilter(VkCommandBuffer commands, RenderTargets& targets,
                        std::uint32_t firstView) const;
    void Present(VkCommandBuffer commands, RenderTargets& targets, const RenderingTarget& present,
                 std::string_view var) const;

    const VulkanContext& context_;
    VkFormat presentFormat_;
    ShaderCompiler compiler_;
    BrdfRegistry brdfs_;
    std::map<std::uint32_t, std::unique_ptr<RasterMaterial>> materials_;
    std::uint32_t nextMaterial_ = 1;

    GpuBuffer uniforms_;
    GpuBuffer shadowUniforms_;
    Sampler clampSampler_;
    Sampler shadowSampler_;
    std::unique_ptr<GpuImage> shadowMaps_;
    std::uint32_t shadowResolution_ = 0;
    std::vector<ShadowCascade> cascades_;
    std::vector<ShadowCascade> lightViews_;
    std::vector<ShadowCascade> shadowViews_;
    std::unique_ptr<GpuImage> lightShadowMaps_;
    std::uint32_t lightResolution_ = 0;
    std::vector<glm::vec2> poisson_;
    bool shadowsEnabled_ = false;
    bool updated_ = false;
    std::uint32_t views_ = 1;
    /// Whether the last `Update`'s views take their poses from `poseBuffer_`, and the views of
    /// `poseHistory_` (the poses it last rendered) then.
    bool posesOnGpu_ = false;
    std::uint32_t poseHistoryViews_ = 0;
    /// One pass rendering the sun's casters `casters` for cascade `cascade` into shadow map
    /// layer `layer`, starting from a copy of layer 0 with `fromBase`.
    struct ShadowPass {
        std::uint32_t layer = 0;
        std::uint32_t cascade = 0;
        ShadowCasters casters;
        bool fromBase = false;
    };
    std::vector<ShadowPass> shadowPasses_;
    std::vector<std::uint32_t> viewShadowLayers_;  ///< Per view, its sun map's first layer.
    std::vector<std::int32_t> viewSunCascades_;    ///< Per view, its own fitted map, or -1.
    std::uint32_t shadowLayers_ = 1;               ///< Layers the frame's sun maps take.
    ShadowCasters lightCasters_;                   ///< Casters of the other lights' maps.
    std::vector<std::string_view> requiredVars_;   ///< The vars the last `Update`'s frame needs.

    PipelineLayout shadingLayout_;
    std::map<std::size_t, std::unique_ptr<ComputePipeline>> shadingPipelines_;
    PipelineLayout filterLayout_;
    std::array<std::unique_ptr<ComputePipeline>, kSensorFilters.size()> filterPipelines_;
    PipelineLayout presentLayout_;
    PipelineLayout hzbLayout_;
    std::unique_ptr<ComputePipeline> hzbPipeline_;
    PipelineLayout viewsLayout_;
    std::unique_ptr<ComputePipeline> viewsPipeline_;
    GpuBuffer poseCamera_;  ///< views.slang's camera: parent, projection, its inverse.
    std::unique_ptr<GpuBuffer> poseBuffer_;
    std::unique_ptr<GpuBuffer> poseHistory_;
    std::unique_ptr<GraphicsPipeline> presentFloat_;
    std::unique_ptr<GraphicsPipeline> presentUint_;
    /// Present ranges of the last `Update`: near and far plane, full well, ADC maximum.
    std::array<float, 4> presentRange_{};
    GpuImage sigmoidTable_;
    Sampler tableSampler_;
    GpuBuffer shadingUniforms_;
    GpuBuffer sensorUniforms_;
    GpuBuffer lightTable_;
    GpuBuffer lightViewMatrices_;
    std::unique_ptr<GpuBuffer> materialTable_;
    std::vector<std::size_t> modelsInUse_;
    SpectralBands bands_;
    std::vector<SensorFilter> filters_;
    VkExtent2D extent_{0, 0};
};

}  // namespace ernie
