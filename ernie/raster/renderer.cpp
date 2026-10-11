// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/renderer.h"

#include "ernie/device/gpu_profile.h"
#include "ernie/kernel/embedded_shaders.h"
#include "ernie/raster/poisson_disk.h"
#include "ernie/scene/wide_view.h"
#include "ernie/session/sampling.h"

#include <glm/gtc/packing.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace ernie {

namespace {

struct Uniforms {
    float viewProjection[16];
    float sun[4];
    float forward[4];
    float eye[4];
    std::uint32_t flags[4];
    float sunViewProjection[16];  ///< A batch view's own fitted sun
                                  ///< map, with `sunShadow[2]` 1.
    float sunShadow[4];  ///< Its texel size and depth span in world
                         ///< units, and the flag.
    float motion[4];  ///< The sub-frame's share from the last frame's
                      ///< transforms.
    float strip[4];   ///< A pushbroom's intrinsics
                      ///< (`PushbroomStrip::Intrinsics`).
    float
        minimal[4];  ///< Minimal shading's constant colour and mode.
};
static_assert(sizeof(Uniforms) == 256,
              "Uniforms must match frame.slang");

struct ShadowUniforms {
    float lightViewProjection[4][16];
    float splits[4];
    float texelWorld[4];
    float depthWorld[4];
    float params[4];
    float sun[4];
    std::uint32_t debug[4];
    float poisson[32];
};
static_assert(sizeof(ShadowUniforms) == 480,
              "ShadowUniforms must match frame.slang");

struct ShadingUniforms {
    float invViewProj[16];
    float eye[4];
    float sun[4];
    float range[4];
    std::uint32_t size[4];
    float irradiance[SpectralBands::kMaxBands];
    float lightShadow[4];
    float poisson[32];
    float domeSh[9][4];
    float dome[SpectralBands::kMaxBands];
    std::uint32_t view[4];
    float toDome[3][4];       ///< Rows of the rotation from world
                              ///< directions to the dome's frame.
    float toCatalogue[3][4];  ///< And to the star catalogue's.
    std::uint32_t sky[4];     ///< kSky* bits, the dome map's width,
                              ///< height and mip levels.
    float starGrid[4];        ///< The star grid's columns and rows.
    float strip[4];           ///< A pushbroom's intrinsics
                              ///< (`PushbroomStrip::Intrinsics`).
    /// Dome lighting: occluded (1), diffuse (1, 0 in the limited
    /// mode), the occlusion's world radius and its largest radius in
    /// pixels.
    float domeLighting[4];
    /// The atmosphere (km): the planet's centre relative to the
    /// frame's origin and the bottom radius; the top radius, 1 with
    /// an atmosphere, the sine of the sun's angular radius.
    float atmosphere[2][4];
    float padding[36];  ///< To the views' stride, so shading reads
                        ///< them as one array.
};
static_assert(sizeof(ShadingUniforms) == 1280,
              "ShadingUniforms must match shading.slang");

/// shading.slang's kSky* bits: what its sky pass draws behind the
/// scene, and whether a dome lights it (for shading only; `sky_`
/// holds the others).
constexpr std::uint32_t kSkyDome = 1;
constexpr std::uint32_t kSkyMap = 2;
constexpr std::uint32_t kSkyStars = 4;
constexpr std::uint32_t kSkyLighting = 8;
constexpr std::uint32_t kSkyAtmosphere = 16;

struct LightGpu {
    float position[4];
    float axis[4];
    float cone[4];
    float shadow[4];
    float extentX[4];
    float extentY[4];
    float spectrum[SpectralBands::kMaxBands];
    float tinted[SpectralBands::kMaxBands];
    float beam[4];
    float beamExtra[4];
    /// x: its pattern's index in the registry, y: the laser's
    /// wavelength (nm), z: the frame's time (s); `patternExtent` xy
    /// its tangent-plane half extents (0 unbounded); `patternLayout`
    /// its data and image in `patternData_` (shading.slang's
    /// LightPatternParams); its parameters.
    float pattern[4];
    float patternExtent[4];
    std::uint32_t patternLayout[4];
    float patternParams[16];
};
static_assert(sizeof(LightGpu) == 752,
              "LightGpu must match shading.slang");

/// A light pattern's integration (`patternNormMain`): its index in
/// the registry and light slot, its half extents, wavelength and
/// time, its data and image, and its parameters.
struct PatternJob {
    std::uint32_t info[4];
    float values[4];
    std::uint32_t layout[4];
    float params[16];
};

constexpr DescriptorBinding kPatternBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {29, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
};

/// Integrates each job's pattern over its extent on a grid of 256 x
/// 256 cells, a workgroup per job, into `norms` at its slot.
constexpr const char* kPatternNormSlang = R"(
import light_pattern;
import light_patterns;

struct PatternJob {
    uint4 info;
    float4 values;
    uint4 layout;
    float4 params[4];
}
[[vk::binding(0, 0)]] StructuredBuffer<PatternJob> jobs;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> norms;

static const uint kCells = 256;
groupshared float partial[kCells];

[shader("compute")]
[numthreads(256, 1, 1)]
void patternNormMain(uint3 group : SV_GroupID,
                     uint3 thread : SV_GroupThreadID) {
    let job = jobs[group.x];
    let extent = job.values.xy;
    let cell = 2.0 * extent / float(kCells);
    LightPatternParams p;
    p.values = job.params;
    p.layout = job.layout;
    LightPatternInput input;
    input.footprint = cell;
    input.beam = float2(0.0);
    input.wavelength = job.values.z;
    input.time = job.values.w;
    input.t.y = -extent.y + (float(thread.x) + 0.5) * cell.y;
    var sum = 0.0;
    for (uint i = 0; i < kCells; ++i) {
        input.t.x = -extent.x + (float(i) + 0.5) * cell.x;
        sum += max(lightPattern(job.info.x, input, p), 0.0);
    }
    partial[thread.x] = sum * cell.x * cell.y;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = kCells / 2; stride > 0; stride /= 2) {
        if (thread.x < stride) {
            partial[thread.x] += partial[thread.x + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (thread.x == 0) {
        norms[job.info.y] = partial[0];
    }
}
)";

/// One light's IES profile on the GPU (shading.slang's kProfileRows x
/// kProfileColumns floats).
constexpr VkDeviceSize kProfileBytes = LightParams::kProfileRows *
                                       LightParams::kProfileColumns *
                                       sizeof(float);

struct MaterialGpu {
    std::uint32_t info[4];
    float params[BrdfRegistry::kMaxParams];
    float reflectance[SpectralBands::kMaxBands];
    /// Spectral radiance the surface emits per band (W m^-2 sr^-1
    /// nm^-1).
    float emission[SpectralBands::kMaxBands];
};
static_assert(sizeof(MaterialGpu) == 592,
              "MaterialGpu must match shading.slang");

/// Reflectance luminance below which a material counts as black; its
/// spectrum's shape is flat.
constexpr float kBlackLuminance = 1e-6f;

struct SensorUniforms {
    float invViewProj[16];
    float eye[4];
    float forward[4];
    float intrinsics[4];
    std::uint32_t size[4];
    float distortion[4];
    float lens[4];
    float sensor[4];
    float adc[4];
    std::uint32_t noise[4];
    float clear[4];
    float colour[SpectralBands::kMaxBands][4];
    float qe[SpectralBands::kMaxBands][4];
    float range[4];   ///< First band's lower wavelength, band width
                      ///< (nm), the lens f-number.
    float optics[4];  ///< Diffraction f-number, pixel pitch (um),
                      ///< focal length (mm), focus.
    float photons[SpectralBands::kMaxBands];  ///< Per band: photons
                                              ///< per unit radiance
                                              ///< (no QE).
    /// A cube camera's faces (`FrameParams::Faces`): their shared
    /// projection's inverse, from clip space to the face's camera
    /// frame.
    float faceInvProj[16];
    /// Faces (1 without), the faces' focal length in raster pixels,
    /// their principal point.
    float faces[4];
    /// The unit direction towards the sun in the view's camera frame;
    /// 1 when the sun lights the frame.
    float sun[4];
    float facesPadding[40];
    /// Per band, two to a float4: the lens's image scale (lateral
    /// colour) and focal length shift in millimetres (axial colour);
    /// 1 and 0 without chromatic aberration.
    float chromatic[SpectralBands::kMaxBands / 2][4];
    /// Per band: the radiance whose image puts the sun's irradiance
    /// at normal incidence on the focal plane, E_sun 4 N^2 / (pi T),
    /// for stray light; 0 without the sun.
    float sunRadiance[SpectralBands::kMaxBands];
};
static_assert(sizeof(SensorUniforms) == 3584,
              "SensorUniforms must match sensor.slang");

/// Per-view uniforms sit in one buffer at multiples of this, the
/// largest `minUniformBufferOffsetAlignment` Vulkan allows.
constexpr VkDeviceSize kUniformAlignment = 256;

constexpr VkDeviceSize Aligned(VkDeviceSize bytes) {
    return (bytes + kUniformAlignment - 1) / kUniformAlignment *
           kUniformAlignment;
}

constexpr VkDeviceSize kUniformsStride = Aligned(sizeof(Uniforms));
constexpr VkDeviceSize kShadingStride =
    Aligned(sizeof(ShadingUniforms));
constexpr VkDeviceSize kSensorStride =
    Aligned(sizeof(SensorUniforms));
/// Light shadow views' Shadows blocks (`lightShadowBlocks_`), bound
/// one at a time.
constexpr VkDeviceSize kLightShadowBlockStride =
    Aligned(sizeof(ShadowUniforms));
static_assert(
    kShadingStride == sizeof(ShadingUniforms) &&
        kSensorStride == sizeof(SensorUniforms),
    "Shading and the filters index the views' uniforms as arrays");

/// Samples per axis of the RGB-to-spectrum coefficient table.
constexpr std::uint32_t kTableResolution = 17;

constexpr VkShaderStageFlags kCompute = VK_SHADER_STAGE_COMPUTE_BIT;

constexpr DescriptorBinding kShadingBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {6, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {7, VK_DESCRIPTOR_TYPE_SAMPLER, kCompute},
    {8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {10, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {11, VK_DESCRIPTOR_TYPE_SAMPLER, kCompute},
    {12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {13, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {14, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {16, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {17, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {18, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {19, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {20, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {21, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {22, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {23, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {24, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {25, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {26, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {27, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {28, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {29, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
};

/// shading.slang's ShadingConstants: the dispatch's model and first
/// view, and the terrain horizon the dome occlusion reads
/// (`RasterRenderer::TerrainHorizon`).
struct ShadingConstants {
    std::uint32_t model[4];
    float horizonGrid[4];
    std::uint32_t horizonInfo[4];
    float heightsGrid[4];
    std::uint32_t heightsSize[4];
    float horizonLevels[4];
};
static_assert(sizeof(ShadingConstants) == 96,
              "ShadingConstants must match shading.slang");

constexpr DescriptorBinding kFilterBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {10, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {11, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {12, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {13, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {16, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    // Sampled linearly for the channel images' pyramids.
    {17, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kCompute},
    {18, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {19, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    // The AOV slots.
    {20, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {21, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {22, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {23, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {24, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {25, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {26, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {27, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    // The temporal filters' state, previous and next.
    {28, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {29, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    // The MotionVector var.
    {30, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    // Per raster pixel, its motion (xy) and motion blur's coverage
    // (z): `RenderTargets::Raster`. 32 descriptors in all, as many as
    // devices push at least.
    {31, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
};

/// A plan that runs custom filter `model`, with the built-in filters
/// its domain needs, and holds a var downstream of it: for checking
/// the filter compiles.
SensorPlan ProbePlan(const SensorFilterModel& model,
                     const SensorFilterRegistry& registry) {
    std::vector<SensorFilter> before;
    std::string_view held = render_vars::kHdrColor;
    if (model.kind == FilterKind::Integrator) {
        // In place of qe or cie: straight from the lens, holding what
        // it makes.
        std::vector<SensorFilterStep> chain{{model.id, {}}};
        const std::string_view vars[] = {
            model.domain == FilterDomain::Electrons
                ? render_vars::kElectrons
                : render_vars::kHdrColor};
        return PlanSensor(chain, vars, registry);
    }
    switch (model.domain) {
    case FilterDomain::Spectrum:
        break;
    case FilterDomain::Electrons:
        before = {SensorFilter::Qe};
        held = render_vars::kElectrons;
        break;
    case FilterDomain::Digital:
        before = {SensorFilter::Qe, SensorFilter::Adc};
        held = render_vars::kRawDn;
        break;
    case FilterDomain::Rgb:
        before = {SensorFilter::Cie};
        break;
    case FilterDomain::Display:
        before = {SensorFilter::Cie, SensorFilter::Exposure};
        held = render_vars::kLdrColor;
        break;
    }
    std::vector<SensorFilterStep> chain = StepsOf(before);
    chain.push_back({model.id, {}});
    if (model.domain == FilterDomain::Spectrum) {
        chain.push_back({std::string(NameOf(SensorFilter::Cie)), {}});
    }
    const std::string_view vars[] = {held};
    return PlanSensor(chain, vars, registry);
}

/// The modules of the shading and sky passes with `brdf` as the
/// active model and `patterns`' light patterns, for a frame camera's
/// views or a pushbroom's.
std::vector<ShaderModuleSource>
ShadingModules(const BrdfModel& brdf,
               const LightPatternRegistry& patterns, bool pushbroom) {
    std::vector<ShaderModuleSource> modules =
        LightPatternModules(patterns);
    modules.insert(modules.end(),
                   {{"brdf", std::string(BrdfSlang())},
                    {BrdfModuleName(brdf), brdf.slang},
                    ProjectionModule(pushbroom),
                    {"atmosphere", shaders::kAtmosphereSlang},
                    {"shading", "import " + BrdfModuleName(brdf) +
                                    ";\ntypealias ActiveBrdf = " +
                                    brdf.typeName + ";\n" +
                                    shaders::kShadingSlang}});
    return modules;
}

/// The modules of `plan`'s kernel: the sensor, its custom filters'
/// and its entry point.
std::vector<ShaderModuleSource> SensorModules(const SensorPlan& plan,
                                              std::uint32_t segment) {
    std::vector<ShaderModuleSource> modules{
        ProjectionModule(plan.pushbroom),
        {"lens", shaders::kLensSlang},
        {"cube_faces", shaders::kCubeFacesSlang},
        {"lensProjection", plan.LensSource()},
        {"sensor", shaders::kSensorSlang}};
    for (auto& [name, source] : plan.Modules()) {
        modules.push_back({name, source});
    }
    modules.push_back({"sensorPlan", plan.Source(segment)});
    return modules;
}

/// The key of `plan`'s kernel `segment` among the compiled sensor
/// kernels.
std::string SensorKernelKey(const SensorPlan& plan,
                            std::uint32_t segment) {
    return plan.Key() + "/" + std::to_string(segment);
}

/// hzb.slang: the source level and the destination level.
constexpr DescriptorBinding kHzbBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
};

/// motion.slang: the sub-frame's radiance, the sum, its ids and the
/// coverage. motion_vectors.slang: the depth, the raster image
/// written and the views' MotionUniforms. sensor.slang's
/// `overlayLensMain`: the sensor's uniforms, LdrColor and parameters
/// (a lens of rays' table), and a cube camera's faces' overlays over
/// black and over white.
constexpr DescriptorBinding kOverlayLensBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {32, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {33, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
};

constexpr DescriptorBinding kCameraMotionBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
};

/// frame.slang's MotionUniforms.
struct MotionUniforms {
    float previousViewProj[16];
    float inverseViewProj[16];
    float size[4];
    float padding[28];
};
static_assert(sizeof(MotionUniforms) == 256,
              "MotionUniforms must match frame.slang");

constexpr DescriptorBinding kMotionBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
};

/// sensor.slang's `faceRowsClearMain` and `faceRowsMain`: the
/// sensor's uniforms and parameters (a lens of rays' table), and a
/// cube camera's faces' sensor rows.
constexpr DescriptorBinding kFaceRowsBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {34, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
};

/// views.slang: the camera, the poses and the per-view blocks it
/// patches.
constexpr DescriptorBinding kViewsBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
};

/// light_views.slang's `lightsMain`: records, poses, the light table,
/// the shadow views' matrices and their Shadows blocks.
constexpr DescriptorBinding kLightsBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
};

/// minimal_resolve.slang: the attachments, the views' uniforms, and
/// the vars' linear buffers.
constexpr DescriptorBinding kResolveBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
};

/// depth_range.slang: the views' depth and the range it reduces to.
constexpr DescriptorBinding kDepthRangeBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
};

/// meter.slang: HdrColor and its histogram.
constexpr DescriptorBinding kMeterBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
};

/// meter.slang's histogram: its bins, the log2 luminance its first
/// starts at, and its bins per stop, spanning 2^-12 to 2^20 cd/m^2
/// (starlight to well above the sunlit sky).
constexpr std::uint32_t kMeterBins = 64;
constexpr float kMeterLow = -12.0f;
constexpr float kMeterBinsPerStop = 2.0f;

/// How far a view may move from the one whose depths a cascade fit
/// reuses (`ShadowFit::Visible`): its eye by this fraction of the
/// nearest depth seen, its direction by this cosine.
constexpr double kVisibleFitEyeShift = 0.1;
constexpr float kVisibleFitTurn = 0.995f;

/// views.slang's camera: parent-to-world, projection and its inverse.
constexpr VkDeviceSize kPoseCameraBytes = 5 * sizeof(glm::mat4);

/// Per-view uniforms the CPU writes and views.slang may patch.
constexpr VkBufferUsageFlags kViewUniformUsage =
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

constexpr DescriptorBinding kPresentBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
     VK_SHADER_STAGE_FRAGMENT_BIT},
    {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
     VK_SHADER_STAGE_FRAGMENT_BIT},
    {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
     VK_SHADER_STAGE_FRAGMENT_BIT},
    {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_FRAGMENT_BIT},
};

/// present.slang's kShowSpectral: SpectralRadiance's bands, through
/// presentSpectral.
constexpr std::uint32_t kShowSpectral = 7;

/// How present.slang shows a render var (its kShow* constants), and
/// whether it is an integer image.
struct PresentMode {
    std::string_view var;
    std::uint32_t show;
    bool integer;
};

constexpr std::array<PresentMode, 11> kPresentModes{{
    {render_vars::kSpectralRadiance, kShowSpectral, false},
    {render_vars::kLdrColor, 0, false},
    {render_vars::kHdrColor, 1, false},
    {render_vars::kDistanceToImagePlane, 2, false},
    {render_vars::kDistanceToCamera, 2, false},
    {render_vars::kNormal, 3, false},
    {render_vars::kElectrons, 4, false},
    {render_vars::kPrimId, 5, true},
    {render_vars::kInstanceId, 5, true},
    {render_vars::kRawDn, 6, true},
    {render_vars::kMotionVector, 8, false},
}};

/// AOVs, as linear rgb over the electrons' white, the scale filters'
/// charge outputs share.
constexpr PresentMode kAovPresentMode{"", 4, false};

const PresentMode* PresentModeOf(std::string_view var) {
    for (const PresentMode& mode : kPresentModes) {
        if (mode.var == var) {
            return &mode;
        }
    }
    if (var.starts_with(kAovPrefix) ||
        std::find(kAovSlotNames.begin(), kAovSlotNames.end(), var) !=
            kAovSlotNames.end()) {
        return &kAovPresentMode;
    }
    return nullptr;
}

/// Jakob-Hanika coefficients over a `kTableResolution`^3 grid of
/// linear RGB, red along x, green along y, blue across layers, as
/// RGBA32F texels; computed once per process.
const std::vector<float>& SigmoidTable() {
    static const std::vector<float> table = [] {
        const std::uint32_t n = kTableResolution;
        std::vector<float> texels(static_cast<std::size_t>(n) * n *
                                  n * 4);
        for (std::uint32_t b = 0; b < n; ++b) {
            for (std::uint32_t g = 0; g < n; ++g) {
                for (std::uint32_t r = 0; r < n; ++r) {
                    const glm::dvec3 rgb(r / double(n - 1),
                                         g / double(n - 1),
                                         b / double(n - 1));
                    const glm::dvec3 c =
                        RgbToSigmoid(rgb).coefficients;
                    float* texel =
                        &texels[((static_cast<std::size_t>(b) * n +
                                  g) *
                                     n +
                                 r) *
                                4];
                    texel[0] = static_cast<float>(c.x);
                    texel[1] = static_cast<float>(c.y);
                    texel[2] = static_cast<float>(c.z);
                    texel[3] = 0.0f;
                }
            }
        }
        return texels;
    }();
    return table;
}

/// A barrier making compute and graphics writes visible to later
/// compute, fragment and transfer reads.
void ShaderBarrier(VkCommandBuffer commands) {
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.dstAccessMask =
        VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commands, &dependency);
}

/// The integral of the CIE y function over 360-830 nm, in nm, by
/// midpoints of 1 nm steps as light spectra are scaled
/// (`RelativeSpectrum`).
double LuminanceIntegral() {
    static const double integral = [] {
        double sum = 0.0;
        for (double w = 360.5; w < 830.0; w += 1.0) {
            sum += CieXyz(w).y;
        }
        return sum;
    }();
    return integral;
}

void CopyMatrix(float (&target)[16], const glm::mat4& matrix) {
    std::memcpy(target, glm::value_ptr(matrix), sizeof(target));
}

std::uint32_t Groups(std::uint32_t size) {
    return (size + 7) / 8;
}

/// A unit vector perpendicular enough to `direction` to serve as a
/// look-at up vector.
glm::vec3 UpFor(const glm::vec3& direction) {
    return std::abs(direction.z) > 0.99f
               ? glm::vec3(0.0f, 1.0f, 0.0f)
               : glm::vec3(0.0f, 0.0f, 1.0f);
}

/// World depth spanned by the [0, 1] depth of an orthographic
/// view-projection.
float OrthographicDepth(const glm::mat4& viewProjection) {
    const float gradient = glm::length(
        glm::vec3(viewProjection[0][2], viewProjection[1][2],
                  viewProjection[2][2]));
    return gradient > 0.0f ? 1.0f / gradient : 0.0f;
}

/// A compute shader that evaluates `model` once, to check that its
/// module implements `IBrdf`.
std::vector<ShaderModuleSource>
BrdfCheckModules(const BrdfModel& model) {
    return {
        {"brdf", std::string(BrdfSlang())},
        {BrdfModuleName(model), model.slang},
        {"brdf_check",
         "import brdf;\nimport " + BrdfModuleName(model) +
             ";\nRWStructuredBuffer<float3> result;\n"
             "[shader(\"compute\")]\n[numthreads(1, 1, 1)]\n"
             "void main() {\n    " +
             model.typeName +
             " model;\n    BrdfInput input;\n    BrdfParams params;\n"
             "    result[0] = model.radianceFactor(input, "
             "params);\n}\n"}};
}

}  // namespace

namespace {

/// Whether frame `params` shadows its dome or its atmosphere's sky
/// light: the full or directions dome lighting mode with a shadowing
/// dome, or an atmosphere, whose sky light the same occlusion darkens
/// under overhangs and in the terrain's valleys; screen-space (so not
/// for pushbroom strips).
bool DomeShadowedOf(const FrameParams& params) {
    const bool dome = params.dome && params.dome->shadow &&
                      (params.domeMode == DomeLightMode::Full ||
                       params.domeMode == DomeLightMode::Directions);
    return (dome || params.atmosphere) && !params.camera.pushbroom;
}

}  // namespace

std::vector<ShadowCascade> FitLightShadowViews(
    const LightParams& light, const glm::vec3& position,
    const glm::vec3& sceneMin, const glm::vec3& sceneMax,
    std::uint32_t resolution) {
    if (!light.shadows) {
        return {};
    }
    const float texels = static_cast<float>(std::max(resolution, 1u));
    const glm::vec3 centre = 0.5f * (sceneMin + sceneMax);
    if (light.type == LightType::Distant) {
        const float radius =
            std::max(0.5f * glm::length(sceneMax - sceneMin), 1e-3f);
        const glm::vec3 towards = glm::normalize(light.direction);
        ShadowCascade view;
        view.lightViewProjection =
            glm::orthoRH_ZO(-radius, radius, -radius, radius, radius,
                            3.0f * radius) *
            glm::lookAtRH(centre + 2.0f * radius * towards, centre,
                          UpFor(towards));
        view.texelWorldSize = 2.0f * radius / texels;
        return {view};
    }
    float farPlane = 1e-3f;
    for (int i = 0; i < 8; ++i) {
        const glm::vec3 corner((i & 1) ? sceneMax.x : sceneMin.x,
                               (i & 2) ? sceneMax.y : sceneMin.y,
                               (i & 4) ? sceneMax.z : sceneMin.z);
        farPlane = std::max(farPlane,
                            1.01f * glm::length(corner - position));
    }
    // A laser's spot: orthographic along the beam (a patterned laser
    // spreads from its aperture, so a spot light's perspective view
    // over the pattern below).
    if (light.type == LightType::Laser && light.beam.IsSpot()) {
        // Orthographic along the beam, as wide as the beam's widest
        // (at the aperture or the scene's far side) out to 2.5 1/e^2
        // radii, where it carries e^-12.5 of its peak.
        const LightParams::Beam& beam = light.beam;
        float widest = 0.0f;
        for (const float z :
             {-beam.waistDistance, farPlane - beam.waistDistance}) {
            const glm::vec2 t = z / glm::max(beam.rayleigh, 1e-12f);
            const glm::vec2 w = beam.waist * glm::sqrt(1.0f + t * t);
            widest = std::max(widest, std::max(w.x, w.y));
        }
        const float half = std::max(2.5f * widest, 1e-6f);
        const glm::vec3 along = glm::normalize(light.direction);
        ShadowCascade view;
        view.lightViewProjection =
            glm::orthoRH_ZO(-half, half, -half, half, 0.0f,
                            farPlane) *
            glm::lookAtRH(position, position + along, UpFor(along));
        view.texelWorldSize = 2.0f * half / texels;
        return {view};
    }
    const float nearPlane = std::max(light.radius, 1e-4f * farPlane);
    const auto perspective = [&](const glm::vec3& axis, float fov) {
        ShadowCascade view;
        view.lightViewProjection =
            glm::perspectiveRH_ZO(fov, 1.0f, nearPlane, farPlane) *
            glm::lookAtRH(position, position + axis, UpFor(axis));
        view.texelWorldSize = 2.0f * std::tan(0.5f * fov) / texels;
        return view;
    };
    if (light.coneAngle <= LightParams::kMaxSpotConeDegrees) {
        return {perspective(
            glm::normalize(light.direction),
            2.0f * glm::radians(std::max(light.coneAngle, 0.5f)))};
    }
    std::vector<ShadowCascade> faces;
    for (const glm::vec3 axis :
         {glm::vec3(1, 0, 0), glm::vec3(-1, 0, 0), glm::vec3(0, 1, 0),
          glm::vec3(0, -1, 0), glm::vec3(0, 0, 1),
          glm::vec3(0, 0, -1)}) {
        faces.push_back(perspective(axis, glm::radians(90.0f)));
    }
    return faces;
}

RasterRenderer::RasterRenderer(const VulkanContext& context,
                               VkFormat presentFormat,
                               BrdfRegistry brdfs)
    : context_(context), presentFormat_(presentFormat),
      brdfs_(std::move(brdfs)), uploads_(context),
      uniforms_(context, kUniformsStride, kViewUniformUsage,
                BufferMemory::Device),
      shadowUniforms_(context, sizeof(ShadowUniforms),
                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      BufferMemory::Device),
      clampSampler_(context, SamplerDesc{}),
      shadowSampler_(
          context, SamplerDesc{VK_FILTER_LINEAR,
                               VK_SAMPLER_MIPMAP_MODE_NEAREST,
                               VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                               1.0f, true}),
      poisson_(PoissonDisk(16)),
      shadingLayout_(context, kShadingBindings,
                     sizeof(ShadingConstants)),
      filterLayout_(context, kFilterBindings, 32),
      presentLayout_(context, kPresentBindings, 32),
      hzbLayout_(context, kHzbBindings, 16),
      motionLayout_(context, kMotionBindings, 48),
      viewsLayout_(context, kViewsBindings, 48),
      poseCamera_(context, kPoseCameraBytes,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  BufferMemory::Device),
      sigmoidTable_(context,
                    ImageDesc{kTableResolution, kTableResolution,
                              kTableResolution, 1,
                              VK_FORMAT_R32G32B32A32_SFLOAT,
                              VK_IMAGE_USAGE_SAMPLED_BIT}),
      tableSampler_(context,
                    SamplerDesc{VK_FILTER_LINEAR,
                                VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                1.0f, false}),
      shadingUniforms_(context, kShadingStride, kViewUniformUsage,
                       BufferMemory::Device),
      sensorUniforms_(context, kSensorStride, kViewUniformUsage,
                      BufferMemory::Device),
      lightTable_(context, kMaxLights * sizeof(LightGpu),
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  BufferMemory::Device),
      lightProfiles_(context, kMaxLights * kProfileBytes,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     BufferMemory::Device),
      patternNorms_(context, kMaxLights * sizeof(float),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    BufferMemory::Device),
      patternJobs_(context, kMaxLights * sizeof(PatternJob),
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   BufferMemory::Device),
      patternLayout_(context, kPatternBindings, 0),
      patternData_(std::make_unique<GpuBuffer>(
          context, sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          BufferMemory::Device)),
      lightViewMatrices_(
          context, kMaxLightShadowViews * sizeof(glm::mat4),
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device),
      lightShadowBlocks_(
          context, kMaxLightShadowViews * kLightShadowBlockStride,
          kViewUniformUsage, BufferMemory::Device),
      lightsLayout_(context, kLightsBindings, 48),
      depthRangeLayout_(context, kDepthRangeBindings, 32),
      meterLayout_(context, kMeterBindings, 32),
      resolveLayout_(context, kResolveBindings, 32),
      unheldVar_(context, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 BufferMemory::Device) {
    if (brdfs_.Models().empty()) {
        throw std::invalid_argument(
            "RasterRenderer needs at least one BRDF");
    }
    MaterialSettings defaults;
    if (brdfs_.Find(defaults.brdf) == nullptr) {
        defaults.brdf = brdfs_.Models().front().id;
    }
    materials_.emplace(
        kDefaultMaterial,
        std::make_unique<RasterMaterial>(context_, uploads_));
    SetMaterial(kDefaultMaterial, defaults);
    CreateShadowMaps(1, kMaxCascades);
    CreateLightShadowMaps(1, 1);

    const std::vector<float>& table = SigmoidTable();
    const std::size_t layerBytes =
        static_cast<std::size_t>(kTableResolution) *
        kTableResolution * 4 * sizeof(float);
    GpuBuffer staging(context_, table.size() * sizeof(float), 0,
                      BufferMemory::Upload);
    staging.Write(std::as_bytes(std::span(table)));
    context_.Submit([&](VkCommandBuffer commands) {
        sigmoidTable_.Transition(
            commands, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        std::vector<VkBufferImageCopy> regions;
        for (std::uint32_t layer = 0; layer < kTableResolution;
             ++layer) {
            VkBufferImageCopy region{};
            region.bufferOffset = layer * layerBytes;
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                       layer, 1};
            region.imageExtent = {kTableResolution, kTableResolution,
                                  1};
            regions.push_back(region);
        }
        vkCmdCopyBufferToImage(
            commands, staging.Handle(), sigmoidTable_.Handle(),
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            static_cast<std::uint32_t>(regions.size()),
            regions.data());
        sigmoidTable_.Transition(
            commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });

    SetDomeMap(nullptr);
    SetStars({});
    SkyPipeline(false);
    noOcclusion_ = std::make_unique<GpuImage>(
        context_, ImageDesc{1, 1, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT,
                            VK_IMAGE_USAGE_STORAGE_BIT, false, true});
    noFaceRows_ = std::make_unique<GpuImage>(
        context_, ImageDesc{1, 1, 1, 1, VK_FORMAT_R32_UINT,
                            VK_IMAGE_USAGE_STORAGE_BIT, false, true});
    context_.Submit([this](VkCommandBuffer commands) {
        noOcclusion_->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
        noFaceRows_->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    });
    // Two layers of horizons at the nadir: nothing occludes.
    noHorizon_ = std::make_unique<GpuImage>(
        context_, ImageDesc{1, 1, 2, 1, VK_FORMAT_R16G16B16A16_SFLOAT,
                            VK_IMAGE_USAGE_SAMPLED_BIT, false, true});
    const std::array<std::uint16_t, 4> nadir{
        glm::packHalf1x16(-100.0f), glm::packHalf1x16(-100.0f),
        glm::packHalf1x16(-100.0f), glm::packHalf1x16(-100.0f)};
    const std::span<const std::byte> layers[] = {
        std::as_bytes(std::span(nadir)),
        std::as_bytes(std::span(nadir))};
    noHorizon_->Upload(layers);
    // A height in place of a terrain's, read only with its horizon
    // map.
    noHeights_ = std::make_unique<GpuImage>(
        context_,
        ImageDesc{1, 1, 1, 1, VK_FORMAT_R32_SFLOAT,
                  VK_IMAGE_USAGE_SAMPLED_BIT, false, false});
    const float flat = 0.0f;
    const std::span<const std::byte> height[] = {
        std::as_bytes(std::span(&flat, 1))};
    noHeights_->Upload(height);
    atmosphere_ =
        std::make_unique<AtmosphereLuts>(context_, compiler_);

    const std::vector<ShaderModuleSource> hzbModules{
        {"hzb", shaders::kHzbSlang}};
    const std::vector<ShaderModuleSource> pyramidModules{
        {"pyramid", shaders::kPyramidSlang}};
    pyramidPipeline_ = std::make_unique<ComputePipeline>(
        context_, hzbLayout_,
        compiler_.CompileSpirv(pyramidModules, "pyramidMain"));
    const std::vector<ShaderModuleSource> motionModules{
        {"motion", shaders::kMotionSlang}};
    motionPipeline_ = std::make_unique<ComputePipeline>(
        context_, motionLayout_,
        compiler_.CompileSpirv(motionModules, "motionMain"));
    hzbPipeline_ = std::make_unique<ComputePipeline>(
        context_, hzbLayout_,
        compiler_.CompileSpirv(hzbModules, "hzbMain"));
    const std::vector<ShaderModuleSource> viewsModules{
        {"cube_faces", shaders::kCubeFacesSlang},
        {"views", shaders::kViewsSlang}};
    viewsPipeline_ = std::make_unique<ComputePipeline>(
        context_, viewsLayout_,
        compiler_.CompileSpirv(viewsModules, "viewsMain"));
    const std::vector<ShaderModuleSource> lightModules{
        {"light_views", shaders::kLightViewsSlang}};
    lightsPipeline_ = std::make_unique<ComputePipeline>(
        context_, lightsLayout_,
        compiler_.CompileSpirv(lightModules, "lightsMain"));
    const std::vector<ShaderModuleSource> presentModules{
        {"present", shaders::kPresentSlang}};
    const std::vector<std::uint32_t> vertex =
        compiler_.CompileSpirv(presentModules, "presentVertex");
    const auto presentPipeline = [&](const char* entry,
                                     bool blend = false) {
        const std::vector<std::uint32_t> fragment =
            compiler_.CompileSpirv(presentModules, entry);
        GraphicsPipelineDesc present;
        present.vertexSpirv = vertex;
        present.fragmentSpirv = fragment;
        present.colorFormat = presentFormat_;
        present.depthFormat = VK_FORMAT_UNDEFINED;
        present.alphaBlend = blend;
        return std::make_unique<GraphicsPipeline>(
            context_, presentLayout_, present);
    };
    presentFloat_ = presentPipeline("presentFloat");
    presentUint_ = presentPipeline("presentUint");
    presentSpectral_ = presentPipeline("presentSpectral");
    presentOutline_ = presentPipeline("presentOutline", true);
    highlight_ = std::make_unique<GpuBuffer>(
        context_, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        BufferMemory::Device);
    statePlaceholder_ = std::make_unique<GpuImage>(
        context_, ImageDesc{1, 1, 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
                            VK_IMAGE_USAGE_STORAGE_BIT |
                                VK_IMAGE_USAGE_SAMPLED_BIT,
                            false, true});
    overlays_ = std::make_unique<OverlayPass>(
        context_, compiler_, uploads_, presentFormat_);
    const std::vector<ShaderModuleSource> depthRangeModules{
        {"depth_range", shaders::kDepthRangeSlang}};
    depthRangePipeline_ = std::make_unique<ComputePipeline>(
        context_, depthRangeLayout_,
        compiler_.CompileSpirv(depthRangeModules, "depthRangeMain"));
    for (std::unique_ptr<GpuBuffer>& range : depthRanges_) {
        range = std::make_unique<GpuBuffer>(
            context_, 2 * sizeof(std::uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            BufferMemory::Readback);
    }
    const std::vector<ShaderModuleSource> meterModules{
        {"meter", shaders::kMeterSlang}};
    meterPipeline_ = std::make_unique<ComputePipeline>(
        context_, meterLayout_,
        compiler_.CompileSpirv(meterModules, "meterMain"));
    for (std::unique_ptr<GpuBuffer>& bins : meterBins_) {
        bins = std::make_unique<GpuBuffer>(
            context_, kMeterBins * sizeof(std::uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            BufferMemory::Readback);
    }
}

RasterRenderer::~RasterRenderer() {
    vkDeviceWaitIdle(context_.Device());
}

void RasterRenderer::AddSensorFilter(SensorFilterModel model) {
    SensorFilterRegistry candidate = sensorFilters_;
    candidate.Add(model);
    try {
        const SensorPlan probe = ProbePlan(model, candidate);
        for (std::uint32_t segment = 0; segment < probe.segments;
             ++segment) {
            compiler_.CompileSpirv(SensorModules(probe, segment),
                                   "sensorMain");
        }
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(
            "Sensor filter " + model.id +
            " failed to compile: " + error.what());
    }
    sensorFilters_ = std::move(candidate);
}

void RasterRenderer::AddLightPattern(LightPatternModel model) {
    LightPatternRegistry candidate = lightPatterns_;
    candidate.Add(model);
    std::vector<ShaderModuleSource> modules =
        LightPatternModules(candidate);
    modules.push_back({"pattern_norms", kPatternNormSlang});
    std::vector<std::uint32_t> spirv;
    try {
        spirv = compiler_.CompileSpirv(modules, "patternNormMain");
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(
            "Light pattern " + model.id +
            " failed to compile: " + error.what());
    }
    // The passes that draw patterns compile again with it.
    vkDeviceWaitIdle(context_.Device());
    lightPatterns_ = std::move(candidate);
    patternPipeline_ = std::make_unique<ComputePipeline>(
        context_, patternLayout_, spirv);
    shadingPipelines_.clear();
    skyPipelines_ = {};
    occlusionPipeline_.reset();
    occlusionFilterPipeline_.reset();
}

void RasterRenderer::RecordPatternNorms(
    VkCommandBuffer commands) const {
    if (patternJobCount_ == 0) {
        return;
    }
    ERNIE_GPU_ZONE(context_, commands, "Light patterns");
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      patternPipeline_->Handle());
    const DescriptorResource resources[] = {
        DescriptorResource::Of(patternJobs_),
        DescriptorResource::Of(patternNorms_),
        DescriptorResource::Of(*patternData_)};
    patternLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                        resources);
    vkCmdDispatch(commands, patternJobCount_, 1, 1);
}

void RasterRenderer::AddBrdf(BrdfModel model) {
    BrdfRegistry candidate = brdfs_;
    candidate.Add(model);
    try {
        compiler_.CompileSpirv(BrdfCheckModules(model), "main");
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(
            "BRDF " + model.id +
            " failed to compile: " + error.what());
    }
    brdfs_ = std::move(candidate);
}

std::uint32_t
RasterRenderer::AddMaterial(const MaterialSettings& settings) {
    if (brdfs_.Find(settings.brdf) == nullptr) {
        throw std::invalid_argument("Unknown BRDF: " + settings.brdf);
    }
    const std::uint32_t id = nextMaterial_++;
    materials_.emplace(
        id, std::make_unique<RasterMaterial>(context_, uploads_));
    materials_.at(id)->SetOrigin(materialsOrigin_);
    ++materialSetVersion_;
    SetMaterial(id, settings);
    return id;
}

void RasterRenderer::SetMaterial(std::uint32_t id,
                                 const MaterialSettings& settings) {
    const BrdfModel* model = brdfs_.Find(settings.brdf);
    if (model == nullptr) {
        throw std::invalid_argument("Unknown BRDF: " + settings.brdf);
    }
    Material(id).Update(
        settings, ResolveBrdfParams(*model, settings.brdfParams));
}

RasterMaterial& RasterRenderer::Material(std::uint32_t id) {
    const auto it = materials_.find(id);
    if (it == materials_.end()) {
        throw std::invalid_argument("Unknown material " +
                                    std::to_string(id));
    }
    return *it->second;
}

const RasterMaterial&
RasterRenderer::Material(std::uint32_t id) const {
    const auto it = materials_.find(id);
    if (it == materials_.end()) {
        throw std::invalid_argument("Unknown material " +
                                    std::to_string(id));
    }
    return *it->second;
}

void RasterRenderer::RemoveMaterial(std::uint32_t id) {
    if (id == kDefaultMaterial) {
        throw std::invalid_argument(
            "The default material cannot be removed");
    }
    const auto it = materials_.find(id);
    if (it == materials_.end()) {
        throw std::invalid_argument("Unknown material " +
                                    std::to_string(id));
    }
    ++materialSetVersion_;
    vkDeviceWaitIdle(context_.Device());
    materials_.erase(it);
}

const RasterMaterial&
RasterRenderer::MaterialOrDefault(std::uint32_t id) const {
    const auto it = materials_.find(id);
    return it != materials_.end() ? *it->second
                                  : *materials_.at(kDefaultMaterial);
}

void RasterRenderer::SetDomeMap(const Image* map) {
    SetDomeMap(map, nullptr);
}

void RasterRenderer::SetDomeMap(const Image* map,
                                const Image* lighting) {
    auto image = UploadDomeMap(map);
    auto lit =
        lighting != nullptr ? UploadDomeMap(lighting) : nullptr;
    vkDeviceWaitIdle(context_.Device());
    domeMap_ = std::move(image);
    domeLighting_ = std::move(lit);
    hasDomeMap_ = map != nullptr;
}

std::unique_ptr<GpuImage>
RasterRenderer::UploadDomeMap(const Image* map) const {
    if (map != nullptr) {
        if (map->width == 0 || map->height == 0 ||
            map->pixels.size() !=
                static_cast<std::size_t>(map->width) * map->height) {
            throw std::invalid_argument("The dome's map is empty");
        }
        const std::uint32_t maxSize =
            context_.Limits().maxImageDimension2D;
        if (map->width > maxSize || map->height > maxSize) {
            throw std::invalid_argument(
                "The dome's map exceeds the device's maximum texture "
                "size");
        }
    }
    const Image white(1, 1, glm::vec4(1.0f));
    const std::vector<Image> mips =
        MipChain(map != nullptr ? *map : white, MipFilter::Average);
    // Float texels, filtered by the shader itself: linear filtering
    // of float32 is optional.
    auto image = std::make_unique<GpuImage>(
        context_,
        ImageDesc{mips.front().width, mips.front().height, 1,
                  static_cast<std::uint32_t>(mips.size()),
                  VK_FORMAT_R32G32B32A32_SFLOAT,
                  VK_IMAGE_USAGE_SAMPLED_BIT});
    std::vector<std::span<const std::byte>> spans;
    for (const Image& mip : mips) {
        spans.push_back(std::as_bytes(std::span(mip.pixels)));
    }
    image->Upload(spans);
    return image;
}

void RasterRenderer::SetStars(std::span<const Star> stars) {
    // Each star's grid cell, by declination row from the south pole
    // and right ascension column.
    const auto cellOf = [](const Star& star) {
        constexpr double kTwoPi = 2.0 * M_PI;
        const double ascension = std::fmod(
            std::fmod(static_cast<double>(star.rightAscension),
                      kTwoPi) +
                kTwoPi,
            kTwoPi);
        const auto column = std::min<std::uint32_t>(
            static_cast<std::uint32_t>(ascension / kTwoPi *
                                       kStarColumns),
            kStarColumns - 1);
        const auto row = std::min<std::uint32_t>(
            static_cast<std::uint32_t>(
                std::max(static_cast<double>(star.declination) +
                             M_PI / 2.0,
                         0.0) /
                M_PI * kStarRows),
            kStarRows - 1);
        return row * kStarColumns + column;
    };
    std::vector<std::uint32_t> order(stars.size());
    for (std::uint32_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::vector<std::uint32_t> cells(stars.size());
    for (std::size_t i = 0; i < stars.size(); ++i) {
        cells[i] = cellOf(stars[i]);
    }
    std::stable_sort(order.begin(), order.end(),
                     [&cells](std::uint32_t a, std::uint32_t b) {
                         return cells[a] < cells[b];
                     });
    std::vector<glm::vec4> packed;
    packed.reserve(std::max<std::size_t>(stars.size(), 1));
    std::vector<glm::uvec2> ranges(
        static_cast<std::size_t>(kStarColumns) * kStarRows,
        glm::uvec2(0));
    for (const std::uint32_t i : order) {
        glm::uvec2& range = ranges[cells[i]];
        if (range.y == 0) {
            range.x = static_cast<std::uint32_t>(packed.size());
        }
        range.y += 1;
        const Star& star = stars[i];
        packed.emplace_back(star.rightAscension, star.declination,
                            star.temperature, star.irradiance555);
    }
    if (packed.empty()) {
        packed.emplace_back(0.0f);  // buffers cannot be empty
    }
    auto starBuffer = std::make_unique<GpuBuffer>(
        context_, packed.size() * sizeof(glm::vec4),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
    auto cellBuffer = std::make_unique<GpuBuffer>(
        context_, ranges.size() * sizeof(glm::uvec2),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
    starBuffer->Write(std::as_bytes(std::span(packed)));
    cellBuffer->Write(std::as_bytes(std::span(ranges)));
    vkDeviceWaitIdle(context_.Device());
    stars_ = std::move(starBuffer);
    starCells_ = std::move(cellBuffer);
    starCount_ = stars.size();
}

void RasterRenderer::CreateShadowMaps(std::uint32_t resolution,
                                      std::uint32_t layers) {
    vkDeviceWaitIdle(context_.Device());
    shadowMaps_ = std::make_unique<GpuImage>(
        context_,
        ImageDesc{resolution, resolution,
                  std::max<std::uint32_t>(layers, kMaxCascades), 1,
                  VK_FORMAT_D32_SFLOAT,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT});
    context_.Submit([this](VkCommandBuffer commands) {
        shadowMaps_->Transition(
            commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    shadowResolution_ = resolution;
}

void RasterRenderer::CreateLightShadowMaps(std::uint32_t resolution,
                                           std::uint32_t layers) {
    vkDeviceWaitIdle(context_.Device());
    lightShadowMaps_ = std::make_unique<GpuImage>(
        context_,
        ImageDesc{resolution, resolution, layers, 1,
                  VK_FORMAT_D32_SFLOAT,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT,
                  false, true});
    context_.Submit([this](VkCommandBuffer commands) {
        lightShadowMaps_->Transition(
            commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    lightResolution_ = resolution;
}

void RasterRenderer::WriteLights(
    const FrameParams& params, const std::optional<Bounds3>& bounds) {
    ERNIE_ZONE("Lights");
    const std::uint32_t resolution =
        std::max(params.shadows.lightResolution, 1u);
    lightViews_.clear();
    std::vector<LightGpu> table(params.lights.size());
    std::vector<glm::vec4>
        lightRecords;  ///< The light pass's, for lights posed on the
                       ///< GPU.
    std::vector<PatternJob> patternJobs;
    // Per light slot, its pattern's integral where known (1 for a
    // normalized pattern); the pattern pass writes the others.
    std::vector<float> patternNorms(params.lights.size(), 1.0f);
    // The lasers' pattern data and images, as uploaded to
    // `patternData_`.
    std::vector<float> patternData;
    decltype(patternDataHeld_) patternDataHeld;
    for (std::size_t i = 0; i < params.lights.size(); ++i) {
        const LightParams& given = params.lights[i];
        // A laser's pattern, by name; a custom one sizes its own
        // extent, which its shadow view covers.
        const LightPatternModel* model = nullptr;
        LightParams patterned;
        if (given.type == LightType::Laser) {
            model = lightPatterns_.Find(given.beam.pattern);
            if (model == nullptr) {
                throw std::invalid_argument(
                    "Unknown light pattern '" + given.beam.pattern +
                    "': register it with AddLightPattern");
            }
            if (given.beam.extent == glm::vec2(0.0f) &&
                !model->normalized) {
                patterned = given;
                patterned.beam.extent =
                    glm::vec2(static_cast<float>(model->extent));
                patterned.coneAngle = static_cast<float>(
                    std::min(glm::degrees(std::atan(model->extent *
                                                    std::sqrt(2.0))) +
                                 1.0,
                             180.0));
            }
        }
        const LightParams& light =
            patterned.type == LightType::Laser ? patterned : given;
        LightGpu& entry = table[i];
        const glm::vec3 axis = glm::normalize(light.direction);
        const bool distant = light.type == LightType::Distant;
        const glm::vec3 where =
            distant ? axis : params.Relative(light.position);
        entry.position[0] = where.x;
        entry.position[1] = where.y;
        entry.position[2] = where.z;
        entry.position[3] = static_cast<float>(light.type);
        // A dome's direction lights with the model's diffuse response
        // alone (shading.slang).
        entry.shadow[1] = light.domeDiffuse ? 1.0f : 0.0f;
        entry.axis[0] = axis.x;
        entry.axis[1] = axis.y;
        entry.axis[2] = axis.z;
        entry.axis[3] = light.radius;
        const glm::vec3 x = glm::normalize(light.tangent);
        const glm::vec3 y = glm::cross(x, axis);
        const glm::vec2 half = light.type == LightType::Disk
                                   ? glm::vec2(light.radius)
                                   : 0.5f * light.size;
        for (int k = 0; k < 3; ++k) {
            entry.extentX[k] = x[k] * half.x;
            entry.extentY[k] = y[k] * half.y;
        }
        // The light's size for its shadows' penumbrae
        // (shading.slang): a local light's radius, a rect's or
        // cylinder's the mean of its half extents; a distant light's
        // tangent of its angular radius.
        float shadowRadius = 0.0f;
        switch (light.type) {
        case LightType::Sphere:
        case LightType::Disk:
            shadowRadius = light.radius;
            break;
        case LightType::Rect:
            shadowRadius = 0.5f * (half.x + half.y);
            break;
        case LightType::Cylinder:
            shadowRadius = 0.5f * (half.x + light.radius);
            break;
        default:
            break;
        }
        entry.extentX[3] = shadowRadius;
        entry.extentY[3] =
            distant ? std::tan(0.5f * glm::radians(light.angle))
                    : 0.0f;
        const bool cone = !distant && light.coneAngle < 180.0f;
        const float softness =
            std::clamp(light.coneSoftness, 0.0f, 1.0f);
        entry.cone[0] =
            cone ? std::cos(glm::radians(light.coneAngle)) : -2.0f;
        entry.cone[1] =
            cone ? std::cos(glm::radians(light.coneAngle *
                                         (1.0f - softness)))
                 : -1.0f;
        const bool laser = light.type == LightType::Laser;
        // A laser's spot posed on the GPU has no shadows yet: the
        // light pass fits perspective views only.
        if (bounds &&
            !(laser && light.posedOnGpu && light.beam.IsSpot())) {
            std::vector<ShadowCascade> views = FitLightShadowViews(
                light, params.Relative(light.position), bounds->first,
                bounds->second, resolution);
            if (!views.empty() && lightViews_.size() + views.size() <=
                                      kMaxLightShadowViews) {
                entry.cone[2] =
                    static_cast<float>(lightViews_.size());
                entry.cone[3] = static_cast<float>(views.size());
                entry.shadow[0] = views.front().texelWorldSize;
                if (light.posedOnGpu) {
                    for (std::size_t v = 0; v < views.size(); ++v) {
                        views[v].gpuFitted = true;
                        views[v].gpuMatrix =
                            static_cast<std::uint32_t>(
                                lightViews_.size() + v);
                    }
                }
                lightViews_.insert(lightViews_.end(), views.begin(),
                                   views.end());
            }
        }
        // A focused light's emission is spectrum x (1 - tint) x
        // |cos|^focus + spectrum x tint.
        entry.shadow[2] =
            distant ? 0.0f : std::max(light.focus, 0.0f);
        // Its IES profile, in its slot of the profile buffer: the
        // slot plus 1, 0 without.
        if (!distant &&
            light.profile.size() == kProfileBytes / sizeof(float)) {
            uploads_.Write(lightProfiles_,
                           std::as_bytes(std::span(light.profile)),
                           i * kProfileBytes);
            entry.shadow[3] = static_cast<float>(i + 1);
        }
        for (std::size_t b = 0; b < light.spectrum.size(); ++b) {
            const float tint =
                b < light.focusTint.size() && entry.shadow[2] > 0.0f
                    ? std::clamp(light.focusTint[b], 0.0f, 1.0f)
                    : 0.0f;
            entry.spectrum[b] = light.spectrum[b] * (1.0f - tint);
            entry.tinted[b] = light.spectrum[b] * tint;
        }
        // A laser's beam: its waist radii and Rayleigh ranges, and
        // the waist's distance along it.
        if (laser) {
            entry.beam[0] = light.beam.waist.x;
            entry.beam[1] = light.beam.waist.y;
            entry.beam[2] = light.beam.rayleigh.x;
            entry.beam[3] = light.beam.rayleigh.y;
            entry.beamExtra[0] = light.beam.waistDistance;
            // Its pattern: whether it is the beam itself (0, a spot)
            // or spreads from the aperture (1), its efficiency and
            // zero order; the pattern's index, the laser's wavelength
            // and the frame's time (none yet), its extent, its data
            // and image, and its parameters (defaults for those not
            // given).
            const LightParams::Beam& beam = light.beam;
            entry.beamExtra[1] = beam.IsSpot() ? 0.0f : 1.0f;
            entry.beamExtra[2] = beam.efficiency;
            entry.beamExtra[3] = beam.zeroOrder;
            const auto index = static_cast<std::uint32_t>(
                lightPatterns_.IndexOf(model->id));
            entry.pattern[0] = static_cast<float>(index);
            entry.pattern[1] = beam.wavelength;
            entry.pattern[2] = 0.0f;
            entry.patternExtent[0] = beam.extent.x;
            entry.patternExtent[1] = beam.extent.y;
            entry.patternLayout[0] =
                static_cast<std::uint32_t>(patternData.size());
            entry.patternLayout[1] =
                static_cast<std::uint32_t>(beam.patternData.size());
            patternData.insert(patternData.end(),
                               beam.patternData.begin(),
                               beam.patternData.end());
            if (const LightPatternImage* image =
                    beam.patternImage.get()) {
                entry.patternLayout[2] =
                    static_cast<std::uint32_t>(patternData.size());
                entry.patternLayout[3] =
                    image->width | (image->height << 16);
                patternData.insert(patternData.end(),
                                   image->texels.begin(),
                                   image->texels.end());
            }
            patternDataHeld.emplace_back(beam.patternData,
                                         beam.patternImage);
            std::vector<float> values = model->Defaults();
            for (std::size_t p = 0;
                 p < values.size() && p < beam.patternParams.size();
                 ++p) {
                values[p] = beam.patternParams[p];
            }
            std::copy(values.begin(), values.end(),
                      entry.patternParams);
            // Its integral to compute (`patternNormMain`), unless
            // it is 1.
            if (!model->normalized) {
                PatternJob job{};
                job.info[0] = index;
                job.info[1] = static_cast<std::uint32_t>(i);
                job.values[0] = beam.extent.x;
                job.values[1] = beam.extent.y;
                job.values[2] = entry.pattern[1];
                job.values[3] = entry.pattern[2];
                std::copy(entry.patternLayout,
                          entry.patternLayout + 4, job.layout);
                std::copy(values.begin(), values.end(), job.params);
                patternJobs.push_back(job);
            }
        }
        if (light.posedOnGpu) {
            // The light pass poses it and fits its views as
            // `FitLightShadowViews` would.
            const auto kind =
                distant ? 0u
                : light.coneAngle <= LightParams::kMaxSpotConeDegrees
                    ? 1u
                    : 2u;
            const std::uint32_t record[4] = {
                static_cast<std::uint32_t>(i),
                static_cast<std::uint32_t>(entry.cone[2]),
                static_cast<std::uint32_t>(entry.cone[3]), kind};
            glm::vec4 bits;
            std::memcpy(&bits, record, sizeof(bits));
            lightRecords.push_back(bits);
            lightRecords.emplace_back(
                2.0f * glm::radians(std::max(light.coneAngle, 0.5f)),
                light.radius, 0.0f, 0.0f);
        }
    }
    if (!patternNorms.empty()) {
        uploads_.Write(patternNorms_,
                       std::as_bytes(std::span(patternNorms)));
    }
    // The pattern data and images, when they change.
    if (patternDataHeld != patternDataHeld_ && !patternData.empty()) {
        const VkDeviceSize bytes = patternData.size() * sizeof(float);
        if (patternData_->Size() < bytes) {
            vkDeviceWaitIdle(context_.Device());
            patternData_ = std::make_unique<GpuBuffer>(
                context_, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                BufferMemory::Device);
        }
        uploads_.Write(*patternData_,
                       std::as_bytes(std::span(patternData)));
    }
    patternDataHeld_ = std::move(patternDataHeld);
    patternJobCount_ = static_cast<std::uint32_t>(patternJobs.size());
    if (!patternJobs.empty()) {
        uploads_.Write(patternJobs_,
                       std::as_bytes(std::span(patternJobs)));
        if (patternPipeline_ == nullptr) {
            std::vector<ShaderModuleSource> modules =
                LightPatternModules(lightPatterns_);
            modules.push_back({"pattern_norms", kPatternNormSlang});
            patternPipeline_ = std::make_unique<ComputePipeline>(
                context_, patternLayout_,
                compiler_.CompileSpirv(modules, "patternNormMain"));
        }
    }
    if (!table.empty()) {
        uploads_.Write(lightTable_, std::as_bytes(std::span(table)));
    }
    // The casters' view over the terrain's grid, for its horizon map,
    // while the dome is shadowed: orthographic straight down from
    // above the scene, its x and y the world's.
    const CastersView before = casters_;
    casters_ = {};
    if (terrainExtent_ && bounds && DomeShadowedOf(params) &&
        lightViews_.size() < kMaxLightShadowViews) {
        const glm::vec3 corner = params.Relative(
            glm::dvec3(terrainExtent_->x, terrainExtent_->y, 0.0));
        const auto width = static_cast<float>(terrainExtent_->z);
        const auto height = static_cast<float>(terrainExtent_->w);
        const float top = bounds->second.z + 1.0f;
        const float span = top - bounds->first.z + 1.0f;
        ShadowCascade view;
        view.lightViewProjection =
            glm::orthoRH_ZO(corner.x, corner.x + width,
                            corner.y - height, corner.y, 0.0f, span) *
            glm::lookAtRH(glm::vec3(0.0f, 0.0f, top),
                          glm::vec3(0.0f, 0.0f, top - 1.0f),
                          glm::vec3(0.0f, 1.0f, 0.0f));
        view.texelWorldSize =
            std::max(width, height) / static_cast<float>(resolution);
        casters_.valid = true;
        casters_.layer =
            static_cast<std::uint32_t>(lightViews_.size());
        casters_.shadowView = cascades_.size() + lightViews_.size();
        casters_.resolution = resolution;
        casters_.top = top;
        casters_.span = span;
        lightViews_.push_back(view);
    }
    if (casters_.valid != before.valid ||
        casters_.layer != before.layer ||
        casters_.resolution != before.resolution ||
        casters_.top != before.top || casters_.span != before.span) {
        castersChanged_ = true;
    }
    std::vector<glm::mat4> matrices;
    std::vector<std::byte> blocks(lightViews_.size() *
                                  kLightShadowBlockStride);
    for (std::size_t v = 0; v < lightViews_.size(); ++v) {
        matrices.push_back(lightViews_[v].lightViewProjection);
        std::memcpy(
            blocks.data() + v * kLightShadowBlockStride,
            glm::value_ptr(lightViews_[v].lightViewProjection),
            sizeof(glm::mat4));
    }
    if (!matrices.empty()) {
        uploads_.Write(lightViewMatrices_,
                       std::as_bytes(std::span(matrices)));
        uploads_.Write(lightShadowBlocks_, blocks);
        const auto layers =
            static_cast<std::uint32_t>(lightViews_.size());
        if (resolution != lightResolution_ ||
            layers > lightShadowMaps_->Desc().layers) {
            CreateLightShadowMaps(resolution, layers);
        }
    }
    gpuLights_ = static_cast<std::uint32_t>(lightRecords.size() / 2);
    if (gpuLights_ > 0) {
        // Grown only once no frame still reads them.
        const VkDeviceSize recordBytes =
            lightRecords.size() * sizeof(glm::vec4);
        const VkDeviceSize poseBytes =
            VkDeviceSize{gpuLights_} * sizeof(glm::mat4);
        if (lightRecords_ == nullptr ||
            lightRecords_->Size() < recordBytes ||
            lightPoses_ == nullptr ||
            lightPoses_->Size() < poseBytes) {
            vkDeviceWaitIdle(context_.Device());
            lightRecords_ = std::make_unique<GpuBuffer>(
                context_, recordBytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                BufferMemory::Device);
            lightPoses_ = std::make_unique<GpuBuffer>(
                context_, poseBytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                BufferMemory::Device);
        }
        uploads_.Write(*lightRecords_,
                       std::as_bytes(std::span(lightRecords)));
        lightBoundsLow_ =
            bounds ? glm::vec4(bounds->first, 0.0f) : glm::vec4(0.0f);
        lightBoundsHigh_ = bounds ? glm::vec4(bounds->second, 0.0f)
                                  : glm::vec4(0.0f);
    }
}

void RasterRenderer::RecordLights(
    VkCommandBuffer commands,
    std::span<const RasterDrawable* const> drawables) const {
    if (gpuLights_ == 0) {
        return;
    }
    ERNIE_GPU_ZONE(context_, commands, "Lights posed on the GPU");
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      lightsPipeline_->Handle());
    const DescriptorResource resources[] = {
        DescriptorResource::Of(*lightRecords_),
        DescriptorResource::Of(*lightPoses_),
        DescriptorResource::Of(lightTable_),
        DescriptorResource::Of(lightViewMatrices_),
        DescriptorResource::Of(lightShadowBlocks_)};
    lightsLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                       resources);
    struct {
        std::uint32_t layout[4];
        glm::vec4 low;
        glm::vec4 high;
    } constants{{gpuLights_,
                 static_cast<std::uint32_t>(sizeof(LightGpu) /
                                            sizeof(glm::vec4)),
                 static_cast<std::uint32_t>(kLightShadowBlockStride /
                                            sizeof(glm::vec4)),
                 0u},
                lightBoundsLow_,
                lightBoundsHigh_};
    lightsLayout_.PushConstants(commands, &constants,
                                sizeof(constants));
    vkCmdDispatch(commands, (gpuLights_ + 63) / 64, 1, 1);
    ShaderBarrier(commands);
    // The views' cull views follow their fitted matrices.
    for (const RasterDrawable* drawable : drawables) {
        drawable->RecordShadowViews(commands);
    }
    ShaderBarrier(commands);
}

void RasterRenderer::PlanBatchShadows(
    const FrameParams& params, const Bounds3& scene,
    std::span<RasterDrawable* const> drawables) {
    ERNIE_ZONE("Plan batch shadows");
    const ShadowSettings& shadows = params.shadows;
    LightParams sun;
    sun.type = LightType::Distant;
    sun.direction = params.sunDirection;
    // A fit of the sun's map around `low` to `high`, one cascade
    // covering every view depth. It reaches receivers below the
    // casters, such as a ground plane, without a floor
    // (`ShadowFloor`): they lie in the casters' shadow along the
    // light, and the one cascade has no far split.
    const auto fit = [&](const glm::vec3& low,
                         const glm::vec3& high) {
        std::vector<ShadowCascade> views = FitLightShadowViews(
            sun, glm::vec3(0.0f), low, high, shadowResolution_);
        ShadowCascade cascade = views.front();
        cascade.splitFar = std::numeric_limits<float>::max();
        return cascade;
    };
    cascades_ = {fit(scene.first, scene.second)};
    std::vector<std::int32_t> worlds;
    for (std::uint32_t v = 0; v < views_; ++v) {
        worlds.push_back(params.ViewWorld(v));
    }
    std::sort(worlds.begin(), worlds.end());
    worlds.erase(std::unique(worlds.begin(), worlds.end()),
                 worlds.end());
    // One map for all: every view of every world, the worlds' shared
    // casters only, or every world's casters.
    if (worlds == std::vector<std::int32_t>{-1} ||
        shadows.worlds != WorldShadows::Separate) {
        const std::int32_t casting =
            shadows.worlds == WorldShadows::Shared
                ? ShadowCasters::kNoWorld
                : ShadowCasters::kAllWorlds;
        shadowPasses_.push_back(
            {0, 0, ShadowCasters{true, casting}, false});
        return;
    }
    // A map per world: layer 0 holds the shared casters. Fitted to
    // the scene, each world's layer is a copy of it with the world's
    // casters drawn on top; fitted to the world, it draws the shared
    // casters again with its own light view. Views of every world
    // (-1) get every caster.
    shadowPasses_.push_back(
        {0, 0, ShadowCasters{true, ShadowCasters::kNoWorld}, false});
    const std::uint32_t maxLayers =
        context_.Limits().maxImageArrayLayers;
    std::map<std::int32_t, std::uint32_t> layerOf;
    std::map<std::int32_t, std::uint32_t> cascadeOf;
    // Every world's bounds at once, over the drawables' geometry.
    std::map<std::int32_t, Bounds3> worldBounds;
    if (shadows.worldFit == WorldShadowFit::World) {
        for (const RasterDrawable* drawable : drawables) {
            for (const auto& [world, box] :
                 drawable->WorldsBounds()) {
                const auto [it, added] =
                    worldBounds.try_emplace(world, box);
                if (!added) {
                    it->second = Bounds3(
                        glm::min(it->second.first, box.first),
                        glm::max(it->second.second, box.second));
                }
            }
        }
    }
    for (const std::int32_t world : worlds) {
        if (shadowPasses_.size() == maxLayers) {
            break;  // Further worlds get the shared casters' shadows
                    // only.
        }
        const auto layer =
            static_cast<std::uint32_t>(shadowPasses_.size());
        layerOf[world] = layer;
        std::optional<Bounds3> own;
        if (shadows.worldFit == WorldShadowFit::World &&
            world != -1) {
            if (const auto found = worldBounds.find(world);
                found != worldBounds.end()) {
                own = found->second;
            }
        }
        if (!own) {
            shadowPasses_.push_back(
                {layer, 0, ShadowCasters{false, world}, true});
            continue;
        }
        const glm::vec3 margin(shadows.worldMargin);
        const auto cascade =
            static_cast<std::uint32_t>(cascades_.size());
        cascades_.push_back(
            fit(own->first - margin, own->second + margin));
        cascades_.back().world = world;
        cascadeOf[world] = cascade;
        shadowPasses_.push_back(
            {layer, cascade, ShadowCasters{true, world}, false});
    }
    for (std::uint32_t v = 0; v < views_; ++v) {
        const auto layer = layerOf.find(params.ViewWorld(v));
        viewShadowLayers_[v] =
            layer != layerOf.end() ? layer->second : 0;
        const auto cascade = cascadeOf.find(params.ViewWorld(v));
        viewSunCascades_[v] =
            cascade != cascadeOf.end()
                ? static_cast<std::int32_t>(cascade->second)
                : -1;
    }
    shadowLayers_ = static_cast<std::uint32_t>(shadowPasses_.size());
}

void RasterRenderer::Update(
    const FrameParams& frame,
    std::span<RasterDrawable* const> drawables,
    std::span<const Overlay* const> overlays) {
    ERNIE_ZONE("Renderer::Update");
    ERNIE_PLOT("Views", frame.Views());
    // With motion blur, a view per sub-frame and view
    // (`FrameParams::Subframes`): the drawables cull and draw each,
    // and the sensor reads their sum.
    frame.camera.Check();
    subframes_ = static_cast<std::uint32_t>(frame.camera.Subframes());
    if (subframes_ > 1 && frame.posesOnGpu) {
        throw std::invalid_argument(
            "Motion blur needs the views' poses on the host");
    }
    if (frame.camera.pushbroom && frame.posesOnGpu) {
        throw std::invalid_argument(
            "A pushbroom camera needs the views' poses on the host");
    }
    std::optional<FrameParams> expanded;
    if (subframes_ > 1) {
        expanded = frame.Subframes();
    }
    const FrameParams& params = expanded ? *expanded : frame;
    // A wide fisheye draws each view on cube faces
    // (`FrameParams::Faces`): the raster passes see `raster`, its
    // views the faces' and its camera theirs; the sensor keeps
    // `params`' camera, following its lens's rays onto them.
    faces_ = static_cast<std::uint32_t>(CubeFacesOf(frame.camera));
    std::optional<FrameParams> facesOf;
    if (faces_ > 1) {
        facesOf = params.Faces(static_cast<int>(faces_));
    }
    const FrameParams& raster = facesOf ? *facesOf : params;
    outputViews_ = static_cast<std::uint32_t>(frame.Views());
    const CameraParams& shutter = frame.camera;
    const double first = shutter.SubframeTime(0);
    shutter_ = {
        static_cast<float>(first),
        static_cast<float>(
            subframes_ > 1 ? shutter.SubframeTime(1) - first : 1.0),
        static_cast<float>(shutter.shutterOpen),
        static_cast<float>(shutter.shutterClose),
        static_cast<float>(shutter.rollingShutter)};
    // The cascades follow the frame's own view.
    const CameraView view = frame.View();
    if (params.lights.size() > kMaxLights) {
        throw std::invalid_argument("A frame has at most " +
                                    std::to_string(kMaxLights) +
                                    " lights besides the sun");
    }
    for (const LightParams& light : params.lights) {
        if (light.spectrum.size() !=
            static_cast<std::size_t>(params.camera.bands.count)) {
            throw std::invalid_argument(
                "Every light's spectrum needs one value per camera "
                "band");
        }
        if (glm::length(light.direction) == 0.0f) {
            throw std::invalid_argument(
                "A light's direction must not be zero");
        }
        const bool area = light.type == LightType::Rect ||
                          light.type == LightType::Disk ||
                          light.type == LightType::Cylinder;
        if (area && std::abs(glm::dot(
                        glm::normalize(light.tangent),
                        glm::normalize(light.direction))) > 1e-3f) {
            throw std::invalid_argument(
                "A rect or disk light's tangent must be "
                "perpendicular to "
                "its direction");
        }
    }
    if (params.dome &&
        params.dome->radiance.size() !=
            static_cast<std::size_t>(params.camera.bands.count)) {
        throw std::invalid_argument(
            "The dome's radiance needs one value per camera band");
    }
    views_ = static_cast<std::uint32_t>(raster.Views());
    // Minimal shading draws one layer per view (`MinimalTargets`).
    minimal_ = params.minimal;
    minimalClear_ = params.clearColor;
    if (minimal_ != MinimalMode::Off && resolvePipeline_ == nullptr) {
        const std::vector<ShaderModuleSource> resolve{
            {"minimal_resolve", shaders::kMinimalResolveSlang}};
        resolvePipeline_ = std::make_unique<ComputePipeline>(
            context_, resolveLayout_,
            compiler_.CompileSpirv(resolve, "resolveMain"));
    }
    const std::uint32_t viewsPerTargets =
        minimal_ != MinimalMode::Off
            ? RenderTargets::MaxViews(context_, 1, 1)
            : RenderTargets::MaxViews(
                  context_, params.camera.bands.count, faces_);
    if (minimal_ != MinimalMode::Off) {
        if (subframes_ > 1 || faces_ > 1 || params.camera.pushbroom ||
            !params.camera.lens.IsPinhole()) {
            throw std::invalid_argument(
                "Minimal shading draws pinhole frame cameras without "
                "motion blur");
        }
        for (const RasterDrawable* drawable : drawables) {
            if (!drawable->DrawsMinimal() && drawable->Bounds()) {
                throw std::invalid_argument(
                    "Minimal shading draws meshes and the ground "
                    "plane, "
                    "not yet terrain, planets or point clouds");
            }
        }
    }
    EnsureViewBuffers(views_);
    posesOnGpu_ = params.posesOnGpu;
    posesQuaternions_ = params.posesOnGpu && params.posesQuaternions;
    if (posesOnGpu_) {
        WritePoseCamera(params, raster);
    }
    // Views posed on the GPU are planned as a batch, whose shadows do
    // not follow the views. A cube camera's faces are one view's: its
    // cascades surround its eye, holding every face.
    const bool batch = outputViews_ > 1 || posesOnGpu_;
    const float around =
        faces_ > 1 ? static_cast<float>(CubeFaceExtent(std::min(
                         params.camera.width, params.camera.height)))
                   : 0.0f;
    const SensorModel& sensor = params.camera.sensor;
    presentRange_ = {
        static_cast<float>(params.camera.nearPlane),
        static_cast<float>(params.camera.farPlane),
        static_cast<float>(sensor.fullWellElectrons),
        static_cast<float>(std::ldexp(1.0, sensor.adcBits) - 1.0)};

    // Everything below is relative to the frame's origin, the
    // drawables' bounds included.
    for (RasterDrawable* drawable : drawables) {
        drawable->SetOrigin(params.origin);
    }
    if (params.origin != materialsOrigin_) {
        materialsOrigin_ = params.origin;
        for (auto& [id, material] : materials_) {
            material->SetOrigin(params.origin);
        }
    }
    std::optional<Bounds3> bounds;
    for (const RasterDrawable* drawable : drawables) {
        if (const std::optional<Bounds3> box = drawable->Bounds()) {
            bounds =
                bounds
                    ? Bounds3(glm::min(bounds->first, box->first),
                              glm::max(bounds->second, box->second))
                    : *box;
        }
    }
    const ShadowSettings& shadows = params.shadows;
    shadowsEnabled_ = params.sun && shadows.enabled &&
                      bounds.has_value() &&
                      params.sunDirection.z > 0.0f;
    cascades_.clear();
    shadowPasses_.clear();
    viewShadowLayers_.assign(views_, 0);
    viewSunCascades_.assign(views_, -1);
    shadowLayers_ = 1;
    // Other lights' maps hold what a single view sees, or in a batch
    // what every world shares (every world's casters when merged).
    lightCasters_ =
        !batch ? ShadowCasters{true, params.ViewWorld(0)}
               : ShadowCasters{true,
                               shadows.worlds == WorldShadows::Merged
                                   ? ShadowCasters::kAllWorlds
                                   : ShadowCasters::kNoWorld};
    if (shadowsEnabled_ && batch) {
        // A batch's per-world maps use their own, smaller resolution.
        const std::uint32_t worldResolution =
            shadows.worldResolution > 0
                ? shadows.worldResolution
                : ShadowSettings::AutoWorldResolution(
                      params.camera.width, params.camera.height);
        const std::uint32_t resolution =
            shadows.worlds == WorldShadows::Separate
                ? worldResolution
                : shadows.resolution;
        if (resolution != shadowResolution_) {
            CreateShadowMaps(resolution, kMaxCascades);
        }
        PlanBatchShadows(raster, *bounds, drawables);
        if (shadowLayers_ > shadowMaps_->Desc().layers) {
            CreateShadowMaps(resolution, shadowLayers_);
        }
    }
    if (shadowsEnabled_ && !batch) {
        if (shadows.resolution != shadowResolution_) {
            CreateShadowMaps(shadows.resolution, kMaxCascades);
        }
        // Where the shadows land counts too: on a ground plane, which
        // has no bounds, they reach past the casters, and down to it
        // when the casters float above it.
        glm::vec3 casters = bounds->first;
        for (const RasterDrawable* drawable : drawables) {
            if (const std::optional<float> floor =
                    drawable->ShadowFloor()) {
                casters.z = std::min(casters.z, *floor);
            }
        }
        const auto [low, high] =
            WithShadows(casters, bounds->second, params.sunDirection);
        // How far the casters reach along the view; a cube camera's
        // faces look every way.
        float farthest = 0.0f;
        for (int i = 0; i < 8; ++i) {
            const glm::vec3 corner((i & 1) ? high.x : low.x,
                                   (i & 2) ? high.y : low.y,
                                   (i & 4) ? high.z : low.z);
            farthest = std::max(
                farthest,
                around > 0.0f
                    ? glm::length(corner - view.eye)
                    : glm::dot(corner - view.eye, view.forward));
        }
        // A margin keeps surfaces at exactly the farthest depth, such
        // as flat ground under a nadir camera, inside the last
        // cascade.
        float shadowDistance =
            std::min(view.farPlane, 1.01f * farthest);
        float splitNear = view.nearPlane;
        // Sample distribution shadow maps: the cascades span only the
        // depths the view sees, within those the casters can shadow.
        if (shadows.fit == ShadowFit::Visible) {
            if (const std::optional<std::pair<float, float>> seen =
                    VisibleDepths(params)) {
                shadowDistance =
                    std::max(std::min(shadowDistance, seen->second),
                             view.nearPlane * 1.01f);
                splitNear = std::clamp(seen->first, view.nearPlane,
                                       0.95f * shadowDistance);
            }
        }
        cascades_ = FitCascades(
            view, shadowDistance, params.sunDirection, low, high,
            std::clamp(shadows.cascades, 1, kMaxCascades),
            shadows.splitLambda, shadowResolution_, splitNear,
            around);
        // A single view's cascades hold what it sees: the shared
        // casters and its world's.
        for (std::uint32_t c = 0; c < cascades_.size(); ++c) {
            shadowPasses_.push_back(
                {c, c, ShadowCasters{true, params.ViewWorld(0)},
                 false});
        }
        shadowLayers_ = static_cast<std::uint32_t>(cascades_.size());
    }
    // A metering frame counts its luminance for later frames; this
    // slot's last one has finished.
    ReadMeter();
    meter_ = params.meter && !batch;
    metered_[uploads_.Slot()] = meter_;
    // A visible fit's frame measures what it sees, for later frames'
    // cascades.
    measureDepth_ = shadowsEnabled_ && !batch &&
                    params.shadows.fit == ShadowFit::Visible;
    if (measureDepth_) {
        DepthSample& sample = depthSlots_[uploads_.Slot()];
        sample.measured = true;
        sample.stream = params.stream;
        sample.eye = glm::dvec3(params.camera.transform[3]);
        sample.forward = view.forward;
    }

    {
        ERNIE_ZONE("Frame uniforms");
        // Every view's in one upload: the array frame.slang reads
        // (`kUniformsStride` apart).
        static_assert(kUniformsStride == sizeof(Uniforms),
                      "Views' uniforms are an array");
        std::vector<Uniforms> viewUniforms(views_);
        for (std::uint32_t v = 0; v < views_; ++v) {
            const CameraView viewOf = raster.View(v);
            Uniforms& uniforms = viewUniforms[v];
            // A pushbroom projects through its strip in place of a
            // view-projection (`projectWorld`).
            if (params.camera.pushbroom) {
                const PushbroomStrip strip = params.Strip(v);
                CopyMatrix(uniforms.viewProjection, strip.Packed());
                const glm::vec4 intrinsics = strip.Intrinsics();
                std::memcpy(uniforms.strip,
                            glm::value_ptr(intrinsics),
                            sizeof(uniforms.strip));
            } else {
                CopyMatrix(uniforms.viewProjection,
                           viewOf.ViewProjection());
            }
            uniforms.sun[0] = params.sunDirection.x;
            uniforms.sun[1] = params.sunDirection.y;
            uniforms.sun[2] = params.sunDirection.z;
            uniforms.sun[3] = params.ambient;
            uniforms.forward[0] = viewOf.forward.x;
            uniforms.forward[1] = viewOf.forward.y;
            uniforms.forward[2] = viewOf.forward.z;
            uniforms.forward[3] =
                glm::dot(viewOf.eye, viewOf.forward);
            uniforms.eye[0] = viewOf.eye.x;
            uniforms.eye[1] = viewOf.eye.y;
            uniforms.eye[2] = viewOf.eye.z;
            uniforms.eye[3] = params.exposure;
            uniforms.flags[0] =
                static_cast<std::uint32_t>(params.wireframe);
            uniforms.flags[1] = viewShadowLayers_[v];
            // Its layer of the render targets it draws into: a frame
            // of more views renders them in chunks of
            // `RenderTargets::MaxViews`, and each sub-frame into the
            // same targets; a cube camera's view its faces'
            // consecutive layers.
            const std::uint32_t outputView =
                (v / faces_) % outputViews_;
            uniforms.flags[2] =
                (outputView % viewsPerTargets) * faces_ + v % faces_;
            uniforms.flags[3] = params.camera.orthographic ? 1u : 0u;
            uniforms.motion[0] =
                subframes_ > 1
                    ? static_cast<float>(
                          1.0 +
                          frame.camera.SubframeTime(static_cast<int>(
                              v / (outputViews_ * faces_))))
                    : 1.0f;
            if (viewSunCascades_[v] >= 0) {
                // A view of a world with its own fitted map samples
                // it in place of cascade 0.
                const ShadowCascade& own =
                    cascades_[static_cast<std::size_t>(
                        viewSunCascades_[v])];
                std::memcpy(uniforms.sunViewProjection,
                            glm::value_ptr(own.lightViewProjection),
                            sizeof(uniforms.sunViewProjection));
                uniforms.sunShadow[0] = own.texelWorldSize;
                uniforms.sunShadow[1] =
                    OrthographicDepth(own.lightViewProjection);
                uniforms.sunShadow[2] = 1.0f;
            }
            // Minimal shading's colour and mode (surface.slang's
            // kMinimal*).
            uniforms.minimal[0] = params.minimalColor.r;
            uniforms.minimal[1] = params.minimalColor.g;
            uniforms.minimal[2] = params.minimalColor.b;
            const std::uint32_t mode =
                params.minimal == MinimalMode::Constant   ? 1u
                : params.minimal == MinimalMode::Textured ? 2u
                                                          : 0u;
            uniforms.minimal[3] =
                static_cast<float>(mode | (params.sun ? 0u : 4u));
        }
        uploads_.Write(uniforms_,
                       std::as_bytes(std::span(viewUniforms)));
    }

    ShadowUniforms shadowUniforms{};
    // A batch's views all sample cascade 0 (or their own fitted map).
    const std::size_t sampled =
        batch ? std::min<std::size_t>(cascades_.size(), 1)
              : cascades_.size();
    for (std::size_t c = 0; c < sampled; ++c) {
        std::memcpy(shadowUniforms.lightViewProjection[c],
                    glm::value_ptr(cascades_[c].lightViewProjection),
                    64);
        shadowUniforms.splits[c] = cascades_[c].splitFar;
        shadowUniforms.texelWorld[c] = cascades_[c].texelWorldSize;
        shadowUniforms.depthWorld[c] =
            OrthographicDepth(cascades_[c].lightViewProjection);
    }
    shadowUniforms.sun[0] = std::tan(
        0.5f * glm::radians(std::max(params.sunAngle, 0.0f)));
    shadowUniforms.params[0] = shadows.softness;
    shadowUniforms.params[1] = shadows.normalOffset;
    shadowUniforms.params[2] = static_cast<float>(sampled);
    shadowUniforms.params[3] = shadowsEnabled_ ? 1.0f : 0.0f;
    shadowUniforms.debug[0] = shadows.showCascades ? 1u : 0u;
    for (std::size_t i = 0; i < poisson_.size(); ++i) {
        shadowUniforms.poisson[2 * i] = poisson_[i].x;
        shadowUniforms.poisson[2 * i + 1] = poisson_[i].y;
    }
    uploads_.Write(shadowUniforms_,
                   std::as_bytes(std::span(&shadowUniforms, 1)));

    params.camera.Check();
    if (params.camera.sensor.channels.size() > 4) {
        throw std::invalid_argument(
            "A sensor has at most four channels");
    }
    if (!params.sunIrradiance.empty() &&
        params.sunIrradiance.size() !=
            static_cast<std::size_t>(params.camera.bands.count)) {
        throw std::invalid_argument(
            "The sun's irradiance needs one value per camera band");
    }
    // Only the filters producing the wanted vars run, fused into one
    // kernel that writes only them, per chain: the frame's own, or
    // each of its chains for the vars that chain writes.
    std::vector<FrameParams::SensorChain> chains = params.chains;
    if (chains.empty()) {
        chains.push_back({params.filters, params.vars, 0});
    }
    requiredVars_.clear();
    std::vector<SensorPass> passes(chains.size());
    for (std::size_t c = 0; c < chains.size(); ++c) {
        const FrameParams::SensorChain& chain = chains[c];
        const std::vector<std::string_view> held =
            RequiredVars(chain.vars);
        for (const std::string_view var : held) {
            if (std::find(requiredVars_.begin(), requiredVars_.end(),
                          var) != requiredVars_.end()) {
                throw std::invalid_argument(
                    "Render var '" + std::string(var) +
                    "' comes from two sensor filter chains");
            }
            requiredVars_.push_back(var);
        }
        SensorPass& pass = passes[c];
        pass.plan = PlanSensor(chain.filters.empty()
                                   ? StepsOf(DefaultFilterChain())
                                   : chain.filters,
                               held, sensorFilters_, chain.aovOffset);
        pass.plan.lens =
            LensKernelOf(params.camera.lens, params.camera.pushbroom);
        pass.plan.pushbroom = params.camera.pushbroom;
        pass.plan.cube = faces_ > 1;
        // A lens of rays reads its table after the filters'
        // parameters (the kernel's bindings are all taken).
        const std::shared_ptr<const LensRayTable>& table =
            params.camera.lens.rays;
        pass.plan.table = table != nullptr;
        pass.plan.chromatic =
            !params.camera.lens.chromaticScale.Empty();
        std::vector<float> uploaded = pass.plan.params;
        pass.tableOffset =
            static_cast<std::uint32_t>(uploaded.size());
        if (table != nullptr) {
            for (const glm::vec3& ray : table->rays) {
                uploaded.insert(uploaded.end(),
                                {ray.x, ray.y, ray.z});
            }
        }
        // Each pass's parameters in a buffer of its own; kept from
        // the last frame's pass at the same place in the order when
        // large enough, which only ever waits when one grows.
        const VkDeviceSize bytes =
            std::max<VkDeviceSize>(uploaded.size(), 1) *
            sizeof(float);
        if (c < sensorPasses_.size() &&
            sensorPasses_[c].params != nullptr &&
            sensorPasses_[c].params->Size() >= bytes) {
            pass.params = std::move(sensorPasses_[c].params);
        } else {
            vkDeviceWaitIdle(context_.Device());
            pass.params = std::make_unique<GpuBuffer>(
                context_, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                BufferMemory::Device);
        }
        if (!uploaded.empty()) {
            uploads_.Write(*pass.params,
                           std::as_bytes(std::span(uploaded)));
        }
        SensorPipelines(pass.plan);
    }
    sensorPasses_ = std::move(passes);
    lensTable_ = {};
    if (const auto& table = params.camera.lens.rays) {
        lensTable_ = {0u, table->step, table->columns, table->rows};
    }
    filters_ = sensorPasses_.front().plan.filters;
    bands_ = params.camera.bands;
    extent_ = {params.camera.width, params.camera.height};
    overlays_->Prepare(overlays, params.origin,
                       params.camera.pushbroom);
    // A cube camera's overlays resample through its lens as the
    // sensor does, and with a rolling shutter its faces' texels take
    // the times of the sensor rows the lens maps them to.
    faceLens_.clear();
    if (faces_ > 1) {
        faceLens_ = sensorPasses_.front().plan.LensSource();
        if (!overlays_->Empty()) {
            if (overlayLensLayout_ == nullptr) {
                overlayLensLayout_ = std::make_unique<PipelineLayout>(
                    context_, kOverlayLensBindings, 32);
            }
            LensPipeline(faceLens_, "overlayLensMain",
                         *overlayLensLayout_);
        }
        if (RollsFaces()) {
            if (faceRowsLayout_ == nullptr) {
                faceRowsLayout_ = std::make_unique<PipelineLayout>(
                    context_, kFaceRowsBindings, 32);
            }
            LensPipeline(faceLens_, "faceRowsClearMain",
                         *faceRowsLayout_);
            LensPipeline(faceLens_, "faceRowsMain", *faceRowsLayout_);
        }
    }
    viewWorlds_.assign(views_, -1);
    for (std::size_t v = 0;
         v < std::min<std::size_t>(raster.worlds.size(), views_);
         ++v) {
        viewWorlds_[v] = raster.worlds[v];
    }
    clipRange_ = {static_cast<float>(params.camera.nearPlane),
                  static_cast<float>(params.camera.farPlane)};
    orthographic_ = params.camera.orthographic;
    pushbroom_ = params.camera.pushbroom;
    WriteMaterialTable();
    for (const std::size_t model : modelsInUse_) {
        ShadingPipeline(model, pushbroom_);
    }
    SkyPipeline(pushbroom_);
    sky_ = 0;
    if (params.dome && params.dome->visible) {
        sky_ |= kSkyDome | (hasDomeMap_ ? kSkyMap : 0u);
    }
    if (params.atmosphere) {
        sky_ |= kSkyAtmosphere;
    }
    if (params.stars && starCount_ > 0) {
        sky_ |= kSkyStars;
    }
    atmosphere_->Set(params.atmosphere, bands_);
    // The lights first: shading reads the size of the light shadow
    // maps they make.
    WriteLights(params, bounds);
    WriteShading(raster);
    // A cube camera's motion is its faces', which the sensor maps
    // through the lens.
    motionVectors_ =
        std::find(requiredVars_.begin(), requiredVars_.end(),
                  render_vars::kMotionVector) != requiredVars_.end();
    if (motionVectors_) {
        WriteMotion(raster,
                    faces_ > 1 ? frame.Faces(static_cast<int>(faces_))
                               : frame);
    }
    WriteSensor(params, raster);

    shadowViews_ = cascades_;
    shadowViews_.insert(shadowViews_.end(), lightViews_.begin(),
                        lightViews_.end());
    for (RasterDrawable* drawable : drawables) {
        drawable->Prepare(raster, shadowViews_);
    }
    updated_ = true;
}

const ComputePipeline&
RasterRenderer::LensPipeline(const std::string& lens,
                             const std::string& entry,
                             const PipelineLayout& layout) {
    const std::string key = entry + "\n" + lens;
    if (const auto it = lensPipelines_.find(key);
        it != lensPipelines_.end()) {
        return *it->second;
    }
    // The sensor's module without a plan: the entry point needs only
    // the lens.
    const std::vector<ShaderModuleSource> modules{
        ProjectionModule(false),
        {"lens", shaders::kLensSlang},
        {"cube_faces", shaders::kCubeFacesSlang},
        {"lensProjection", lens},
        {"sensor", shaders::kSensorSlang}};
    auto pipeline = std::make_unique<ComputePipeline>(
        context_, layout, compiler_.CompileSpirv(modules, entry));
    return *lensPipelines_.emplace(key, std::move(pipeline))
                .first->second;
}

const ComputePipeline&
RasterRenderer::ShadingPipeline(std::size_t model, bool pushbroom) {
    if (const auto it = shadingPipelines_.find({model, pushbroom});
        it != shadingPipelines_.end()) {
        return *it->second;
    }
    std::vector<std::uint32_t> spirv;
    const BrdfModel& brdf = brdfs_.Models()[model];
    try {
        spirv = compiler_.CompileSpirv(
            ShadingModules(brdf, lightPatterns_, pushbroom),
            "shadeMain");
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(
            "BRDF " + brdf.id +
            " failed to compile for shading: " + error.what());
    }
    auto pipeline = std::make_unique<ComputePipeline>(
        context_, shadingLayout_, spirv);
    return *shadingPipelines_
                .emplace(std::pair(model, pushbroom),
                         std::move(pipeline))
                .first->second;
}

void RasterRenderer::SkyPipeline(bool pushbroom) {
    std::unique_ptr<ComputePipeline>& sky =
        skyPipelines_[pushbroom ? 1 : 0];
    if (sky == nullptr) {
        // The sky pass shares shading's module, composed with the
        // first model, which it ignores.
        sky = std::make_unique<ComputePipeline>(
            context_, shadingLayout_,
            compiler_.CompileSpirv(
                ShadingModules(brdfs_.Models().front(),
                               lightPatterns_, pushbroom),
                "skyMain"));
    }
}

void RasterRenderer::WriteMaterialTable() {
    ERNIE_ZONE("Material table");
    const std::uint32_t count = materials_.rbegin()->first + 1;
    std::vector<MaterialGpu> table(count, MaterialGpu{});
    std::set<std::size_t> models;
    const std::span<const BrdfModel> registered = brdfs_.Models();
    for (const auto& [id, material] : materials_) {
        MaterialGpu& entry = table[id];
        const auto model =
            std::find_if(registered.begin(), registered.end(),
                         [&](const BrdfModel& m) {
                             return m.id == material->Settings().brdf;
                         });
        const auto index =
            static_cast<std::size_t>(model - registered.begin());
        entry.info[0] = static_cast<std::uint32_t>(index);
        // The parameter the G-buffer's per-pixel value replaces, plus
        // 1 (0 for none).
        const MaterialSettings& settings = material->Settings();
        if (settings.metallicTexture && settings.metallicParam >= 0) {
            entry.info[1] =
                static_cast<std::uint32_t>(settings.metallicParam) +
                1;
        }
        models.insert(index);
        const std::vector<float>& params = material->Params();
        std::copy_n(params.begin(),
                    std::min<std::size_t>(params.size(),
                                          BrdfRegistry::kMaxParams),
                    entry.params);
        std::vector<float> shape =
            material->Reflectance().Bin(bands_);
        const float luminance = material->Luminance();
        for (float& value : shape) {
            value = luminance > kBlackLuminance ? value / luminance
                                                : 1.0f;
        }
        std::copy(shape.begin(), shape.end(), entry.reflectance);
        // Emission: its colour upsampled as a reflectance is
        // (`RgbToSpectrum`, on its largest component's scale) under
        // the equal-energy illuminant of 1 cd/m^2 per unit.
        const glm::vec3 emission = settings.emission;
        const float peak =
            std::max({emission.r, emission.g, emission.b});
        if (peak > 0.0f) {
            entry.info[2] = 1;
            const std::vector<float> emitted =
                RgbToSpectrum(glm::dvec3(emission / peak))
                    .Bin(bands_);
            const double perNit =
                1.0 / (kLuminousEfficacy * LuminanceIntegral());
            for (std::size_t b = 0; b < emitted.size(); ++b) {
                entry.emission[b] =
                    static_cast<float>(emitted[b] * peak * perNit);
            }
        }
    }
    modelsInUse_.assign(models.begin(), models.end());
    const VkDeviceSize bytes = table.size() * sizeof(MaterialGpu);
    if (materialTable_ == nullptr || materialTable_->Size() < bytes) {
        vkDeviceWaitIdle(context_.Device());
        materialTable_ = std::make_unique<GpuBuffer>(
            context_, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            BufferMemory::Device);
    }
    uploads_.Write(*materialTable_, std::as_bytes(std::span(table)));
}

void RasterRenderer::WriteShading(const FrameParams& params) {
    ERNIE_ZONE("Shading uniforms");
    ShadingUniforms shading{};
    shading.sun[0] = params.sunDirection.x;
    shading.sun[1] = params.sunDirection.y;
    shading.sun[2] = params.sunDirection.z;
    shading.sun[3] = params.ambient;
    shading.range[0] = static_cast<float>(bands_.minNm);
    shading.range[1] = static_cast<float>(bands_.Width());
    shading.range[2] = static_cast<float>(bands_.count);
    shading.range[3] = params.camera.orthographic ? 1.0f : 0.0f;
    shading.size[0] = extent_.width;
    shading.size[1] = extent_.height;
    shading.size[2] = kTableResolution;
    shading.size[3] =
        static_cast<std::uint32_t>(params.lights.size());
    for (int i = 0; i < bands_.count && params.sun; ++i) {
        shading.irradiance[i] =
            params.sunIrradiance.empty()
                ? static_cast<float>(M_PI / (kLuminousEfficacy *
                                             LuminanceIntegral()))
                : params.sunIrradiance[static_cast<std::size_t>(i)];
    }
    if (params.dome) {
        for (std::size_t k = 0; k < params.dome->sh.size(); ++k) {
            shading.domeSh[k][0] = params.dome->sh[k].r;
            shading.domeSh[k][1] = params.dome->sh[k].g;
            shading.domeSh[k][2] = params.dome->sh[k].b;
        }
        std::copy(params.dome->radiance.begin(),
                  params.dome->radiance.end(), shading.dome);
    }
    // Rows of the rotations, so the shader dots directions with them.
    const auto rows = [](const glm::mat3& rotation,
                         float (&out)[3][4]) {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                out[r][c] = rotation[c][r];
            }
        }
    };
    rows(params.dome ? params.dome->toDome : glm::mat3(1.0f),
         shading.toDome);
    rows(params.stars ? params.stars->toCatalogue : glm::mat3(1.0f),
         shading.toCatalogue);
    const ImageDesc& map = domeMap_->Desc();
    // What the sky pass draws, and for shading whether a dome lights
    // the scene at all.
    shading.sky[0] = sky_ | (params.dome ? kSkyLighting : 0u);
    // The dome lighting mode: shadowed (screen-space, so not for
    // pushbroom strips) and diffuse.
    const bool domeShadowed = DomeShadowedOf(params);
    shading.domeLighting[0] = domeShadowed ? 1.0f : 0.0f;
    domeShadowed_ = domeShadowed;
    if (domeShadowed && occlusionPipeline_ == nullptr) {
        // The occlusion passes share shading's module, composed with
        // the first model, which they ignore.
        const auto modules = ShadingModules(brdfs_.Models().front(),
                                            lightPatterns_, false);
        occlusionPipeline_ = std::make_unique<ComputePipeline>(
            context_, shadingLayout_,
            compiler_.CompileSpirv(modules, "occlusionMain"));
        occlusionFilterPipeline_ = std::make_unique<ComputePipeline>(
            context_, shadingLayout_,
            compiler_.CompileSpirv(modules, "occlusionFilterMain"));
    }
    // No diffuse dome light in the limited mode, nor when shadowed
    // directions carry it.
    shading.domeLighting[1] =
        params.domeMode == DomeLightMode::Limited ||
                params.domeDirectionsLit
            ? 0.0f
            : 1.0f;
    shading.domeLighting[2] = params.domeShadowRadius;
    shading.domeLighting[3] = 32.0f;
    // The atmosphere in kilometres: the planet's centre relative to
    // the frame's origin (from double, as it lies thousands of
    // kilometres away) and its radii, and the sine of the sun's
    // angular radius, over which it sets behind the planet.
    if (atmosphere_->Active()) {
        const BandedAtmosphere& banded = atmosphere_->Banded();
        const glm::dvec3 centre =
            (atmosphere_->Params().planetCentre - params.origin) *
            1e-3;
        shading.atmosphere[0][0] = static_cast<float>(centre.x);
        shading.atmosphere[0][1] = static_cast<float>(centre.y);
        shading.atmosphere[0][2] = static_cast<float>(centre.z);
        shading.atmosphere[0][3] =
            static_cast<float>(banded.bottomRadius * 1e-3);
        shading.atmosphere[1][0] = static_cast<float>(
            (banded.bottomRadius + banded.topAltitude) * 1e-3);
        shading.atmosphere[1][1] = 1.0f;
        shading.atmosphere[1][2] = std::max(
            std::sin(glm::radians(0.5f * params.sunAngle)), 1e-5f);
    }
    shading.sky[1] = map.width;
    shading.sky[2] = map.height;
    shading.sky[3] = map.mips;
    shading.starGrid[0] = static_cast<float>(kStarColumns);
    shading.starGrid[1] = static_cast<float>(kStarRows);
    shading.lightShadow[0] = params.shadows.softness;
    shading.lightShadow[1] = params.shadows.normalOffset;
    shading.lightShadow[2] = static_cast<float>(lightResolution_);
    for (std::size_t i = 0; i < poisson_.size(); ++i) {
        shading.poisson[2 * i] = poisson_[i].x;
        shading.poisson[2 * i + 1] = poisson_[i].y;
    }
    // Every view's in one upload, `kShadingStride` apart.
    std::vector<ShadingUniforms> viewShading;
    viewShading.reserve(views_);
    for (std::uint32_t v = 0; v < views_; ++v) {
        const CameraView view = params.View(v);
        if (params.camera.pushbroom) {
            // Its pixels' rays come from its strip
            // (projection_pushbroom.slang).
            const PushbroomStrip strip = params.Strip(v);
            CopyMatrix(shading.invViewProj, strip.Packed());
            const glm::vec4 intrinsics = strip.Intrinsics();
            std::memcpy(shading.strip, glm::value_ptr(intrinsics),
                        sizeof(shading.strip));
        } else {
            CopyMatrix(shading.invViewProj,
                       glm::inverse(view.ViewProjection()));
        }
        shading.eye[0] = view.eye.x;
        shading.eye[1] = view.eye.y;
        shading.eye[2] = view.eye.z;
        // The lights the view sees: the shared ones and its world's,
        // or all for every world.
        const std::int32_t world = params.ViewWorld(v);
        shading.view[0] = 0;
        for (std::size_t k = 0; k < params.lights.size(); ++k) {
            const std::int32_t of = params.lights[k].world;
            if (of == -1 || world == -1 || of == world) {
                shading.view[0] |= 1u << k;
            }
        }
        viewShading.push_back(shading);
    }
    uploads_.Write(shadingUniforms_,
                   std::as_bytes(std::span(viewShading)));
}

void RasterRenderer::WriteSensor(const FrameParams& params,
                                 const FrameParams& raster) {
    ERNIE_ZONE("Sensor uniforms");
    const CameraParams& camera = params.camera;
    const glm::mat4& projection = params.View().projection;
    const double rasterFocal = static_cast<double>(projection[1][1]) *
                               static_cast<double>(extent_.height) /
                               2.0;
    SensorUniforms sensor{};
    sensor.intrinsics[0] =
        static_cast<float>(rasterFocal * params.overscan);
    sensor.intrinsics[1] = static_cast<float>(
        extent_.width / 2.0 *
        (1.0 - static_cast<double>(projection[2][0])));
    sensor.intrinsics[2] = static_cast<float>(
        extent_.height / 2.0 *
        (1.0 + static_cast<double>(projection[2][1])));
    sensor.intrinsics[3] = static_cast<float>(params.overscan);
    if (camera.orthographic || camera.pushbroom) {
        // An orthographic principal point is a shift of the
        // projection, not a skew: pixels per scene unit stand in for
        // the focal length, which the pinhole lens maps through
        // unchanged. A pushbroom's view is its strip's envelope: its
        // own focal length instead.
        const glm::dvec2 centre = camera.PrincipalPoint();
        sensor.intrinsics[1] = static_cast<float>(centre.x);
        sensor.intrinsics[2] = static_cast<float>(centre.y);
        if (camera.pushbroom) {
            sensor.intrinsics[0] =
                static_cast<float>(camera.FocalLengthPixels());
        }
    }
    sensor.size[0] = extent_.width;
    sensor.size[1] = extent_.height;
    sensor.size[2] = static_cast<std::uint32_t>(bands_.count);
    sensor.size[3] =
        static_cast<std::uint32_t>(camera.sensor.channels.size());
    const LensModel& lens = camera.lens;
    // The kernel's projection (`LensKernelOf`) reads Brown-Conrady's
    // terms or the fisheye's.
    const bool fisheye = lens.projection == LensProjection::Fisheye;
    sensor.distortion[0] =
        static_cast<float>(fisheye ? lens.fisheye[0] : lens.k1);
    sensor.distortion[1] =
        static_cast<float>(fisheye ? lens.fisheye[1] : lens.k2);
    sensor.distortion[2] =
        static_cast<float>(fisheye ? lens.fisheye[2] : lens.k3);
    sensor.distortion[3] =
        static_cast<float>(fisheye ? lens.fisheye[3] : lens.p1);
    sensor.lens[0] = static_cast<float>(lens.p2);
    // Telecentric (orthographic) optics have no fall-off off axis.
    sensor.lens[1] =
        lens.vignetting && !camera.orthographic ? 1.0f : 0.0f;
    sensor.lens[2] =
        static_cast<float>(camera.LinearExposureScale() *
                           static_cast<double>(params.exposure));
    // A flat spectrum of spectral radiance L has luminance 683 L
    // times the y function's integral; shown with this gain it is as
    // bright as in the photographic image.
    spectralGain_ =
        static_cast<float>(static_cast<double>(sensor.lens[2]) *
                           kLuminousEfficacy * LuminanceIntegral());
    sensor.lens[3] = static_cast<float>(camera.exposureTime);
    const SensorModel& model = camera.sensor;
    sensor.sensor[0] = static_cast<float>(model.fullWellElectrons);
    sensor.sensor[1] = static_cast<float>(model.readNoiseElectrons);
    sensor.sensor[2] = static_cast<float>(model.darkCurrentElectrons);
    sensor.sensor[3] = static_cast<float>(model.prnu);
    sensor.adc[0] = static_cast<float>(model.dsnuElectrons);
    sensor.adc[1] =
        static_cast<float>(model.ElectronsPerDn(camera.iso));
    sensor.adc[2] = static_cast<float>(model.blackLevelDn);
    sensor.adc[3] =
        static_cast<float>(std::exp2(model.adcBits) - 1.0);
    sensor.noise[0] = model.noise ? 1u : 0u;
    sensor.noise[1] = model.seed;
    sensor.noise[2] =
        FrameSeed(model.seed, params.stream, params.frame);
    sensor.clear[0] = params.clearColor.r;
    sensor.clear[1] = params.clearColor.g;
    sensor.clear[2] = params.clearColor.b;
    // The background holds the sky's radiance; or with motion blur,
    // every pixel the sub-frames'.
    sensor.clear[3] = sky_ != 0 ? 1.0f : subframes_ > 1 ? 2.0f : 0.0f;
    sensor.range[0] = static_cast<float>(bands_.minNm);
    sensor.range[1] = static_cast<float>(bands_.Width());
    // The optics filters read (sensor.slang's PixelContext): the
    // lens's f-number for depth of field (0, a pinhole), the one
    // diffraction takes (the lens's, else the exposure's), the
    // sensor's pixel pitch, the focal length and the focus distance.
    // Depth of field follows a thin lens's at its distance from the
    // eye, which telecentric (orthographic) optics do not have: they
    // stay all in focus.
    sensor.range[2] = static_cast<float>(
        camera.orthographic ? 0.0 : camera.lensFStop);
    sensor.range[3] = camera.orthographic ? 1.0f : 0.0f;
    sensor.optics[0] = static_cast<float>(
        camera.lensFStop > 0.0 ? camera.lensFStop : camera.fStop);
    sensor.optics[1] =
        static_cast<float>(camera.PixelPitchMetres() * 1e6);
    sensor.optics[2] = static_cast<float>(camera.focalLengthMm);
    sensor.optics[3] = static_cast<float>(camera.focusDistance);
    sensor.noise[3] = static_cast<std::uint32_t>(params.frame);
    const std::vector<glm::dvec3> weights = CieWeights(bands_);
    std::vector<glm::dvec3> colours;
    for (int i = 0; i < bands_.count; ++i) {
        colours.push_back(EqualEnergyXyzToLinearSrgb(
            weights[static_cast<std::size_t>(i)] *
            kLuminousEfficacy));
    }
    // A monochrome sensor's photographic outputs are grey: its
    // channel's spectral response, scaled so that a flat spectrum
    // keeps the luminance the CIE observer gives it.
    if (model.channels.size() == 1) {
        const std::vector<float> response =
            model.channels.front().quantumEfficiency.Bin(bands_);
        double luminance = 0.0;
        double total = 0.0;
        for (int i = 0; i < bands_.count; ++i) {
            const glm::dvec3& rgb =
                colours[static_cast<std::size_t>(i)];
            luminance +=
                0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
            total += response[static_cast<std::size_t>(i)];
        }
        const double scale = total > 0.0 ? luminance / total : 0.0;
        for (int i = 0; i < bands_.count; ++i) {
            colours[static_cast<std::size_t>(i)] = glm::dvec3(
                scale * response[static_cast<std::size_t>(i)]);
        }
    }
    for (int i = 0; i < bands_.count; ++i) {
        const glm::dvec3& rgb = colours[static_cast<std::size_t>(i)];
        sensor.colour[i][0] = static_cast<float>(rgb.r);
        sensor.colour[i][1] = static_cast<float>(rgb.g);
        sensor.colour[i][2] = static_cast<float>(rgb.b);
    }
    const std::vector<float> photons = camera.PhotonsPerRadiance();
    std::copy(photons.begin(), photons.end(), sensor.photons);
    for (std::size_t c = 0; c < model.channels.size(); ++c) {
        const std::vector<float> factors =
            camera.ElectronsPerRadiance(c);
        for (int i = 0; i < bands_.count; ++i) {
            sensor.qe[i][c] = factors[static_cast<std::size_t>(i)];
        }
    }
    // A cube camera's faces: where their image planes land in the
    // raster, and their depth back to their camera frames (every face
    // shares the projection). Chromatic aberration per band, its mean
    // over the band: the image's scale and the focal length's shift.
    for (int i = 0; i < bands_.count; ++i) {
        const double low = bands_.Lower(i);
        const double high = low + bands_.Width();
        const double scale =
            lens.chromaticScale.Empty()
                ? 1.0
                : lens.chromaticScale.Average(low, high);
        const double shift =
            lens.chromaticFocusMm.Empty()
                ? 0.0
                : lens.chromaticFocusMm.Average(low, high);
        sensor.chromatic[i / 2][(i % 2) * 2] =
            static_cast<float>(scale);
        sensor.chromatic[i / 2][(i % 2) * 2 + 1] =
            static_cast<float>(shift);
    }
    // The sun for stray light: its irradiance as the radiance whose
    // image puts it on the focal plane, through the camera equation
    // (`CameraParams::IrradiancePerRadiance` on axis).
    const double focal = camera.IrradiancePerRadiance({0.0, 0.0});
    for (int i = 0; i < bands_.count && params.sun && focal > 0.0 &&
                    std::isfinite(focal);
         ++i) {
        const double irradiance =
            params.sunIrradiance.empty()
                ? M_PI / (kLuminousEfficacy * LuminanceIntegral())
                : static_cast<double>(
                      params.sunIrradiance[static_cast<std::size_t>(
                          i)]);
        sensor.sunRadiance[i] =
            static_cast<float>(irradiance / focal);
    }
    sensor.sun[3] = params.sun ? 1.0f : 0.0f;
    sensor.faces[0] = static_cast<float>(faces_);
    if (faces_ > 1) {
        const glm::dvec3 face = FrameParams::FaceIntrinsics(
            extent_.width, extent_.height);
        sensor.faces[1] = static_cast<float>(face.x);
        sensor.faces[2] = static_cast<float>(face.y);
        sensor.faces[3] = static_cast<float>(face.z);
        CopyMatrix(sensor.faceInvProj,
                   glm::inverse(raster.View(0).projection));
    }
    // The sensor filters the output views, reading the G-buffer of
    // their last sub-frame (`params` holds a view per sub-frame and
    // output view).
    for (std::uint32_t v = 0; v < outputViews_; ++v) {
        const std::uint32_t last =
            (subframes_ - 1) * outputViews_ + v;
        const CameraView view = params.View(last);
        if (params.camera.pushbroom) {
            // Its strip; its distortion across the line and the line
            // (lens.slang's Pushbroom).
            const PushbroomStrip strip = params.Strip(last);
            CopyMatrix(sensor.invViewProj, strip.Packed());
            sensor.distortion[0] = static_cast<float>(lens.k1);
            sensor.distortion[1] = static_cast<float>(lens.k2);
            sensor.distortion[2] = static_cast<float>(lens.k3);
            sensor.distortion[3] = strip.line;
        } else {
            CopyMatrix(sensor.invViewProj,
                       glm::inverse(view.ViewProjection()));
        }
        sensor.eye[0] = view.eye.x;
        sensor.eye[1] = view.eye.y;
        sensor.eye[2] = view.eye.z;
        sensor.forward[0] = view.forward.x;
        sensor.forward[1] = view.forward.y;
        sensor.forward[2] = view.forward.z;
        const glm::vec3 sun =
            glm::mat3(view.view) * params.sunDirection;
        const glm::vec3 towards =
            glm::length(sun) > 0.0f ? glm::normalize(sun) : sun;
        sensor.sun[0] = towards.x;
        sensor.sun[1] = towards.y;
        sensor.sun[2] = towards.z;
        // Every view of a batch has noise of its own; view 0's is the
        // single view's.
        sensor.noise[2] =
            FrameSeed(model.seed, params.stream ^ (v * 0x9E3779B9u),
                      params.frame);
        uploads_.Write(sensorUniforms_,
                       std::as_bytes(std::span(&sensor, 1)),
                       v * kSensorStride);
    }
}

void RasterRenderer::WriteMotion(const FrameParams& params,
                                 const FrameParams& frame) {
    ERNIE_ZONE("Motion uniforms");
    const VkDeviceSize bytes =
        VkDeviceSize{views_} * sizeof(MotionUniforms);
    if (motionUniforms_ == nullptr ||
        motionUniforms_->Size() < bytes) {
        vkDeviceWaitIdle(context_.Device());
        motionUniforms_ = std::make_unique<GpuBuffer>(
            context_, bytes, kViewUniformUsage, BufferMemory::Device);
    }
    if (cameraMotionPipeline_ == nullptr) {
        cameraMotionLayout_ = std::make_unique<PipelineLayout>(
            context_, kCameraMotionBindings, 16);
        const std::vector<ShaderModuleSource> modules{
            {"motion_vectors", shaders::kMotionVectorsSlang}};
        cameraMotionPipeline_ = std::make_unique<ComputePipeline>(
            context_, *cameraMotionLayout_,
            compiler_.CompileSpirv(modules, "cameraMotionMain"));
    }
    std::vector<MotionUniforms> uniforms(views_);
    for (std::uint32_t v = 0; v < views_; ++v) {
        MotionUniforms& view = uniforms[v];
        // The view's camera at the last frame, relative to this
        // frame's origin (a cube camera's face's, `frame` being its
        // faces'); a pushbroom's motion is zero.
        const CameraView before = frame.camera.RasterView(
            frame.PreviousPose(v % (outputViews_ * faces_)),
            frame.overscan);
        CopyMatrix(view.previousViewProj, before.ViewProjection());
        CopyMatrix(view.inverseViewProj,
                   glm::inverse(params.View(v).ViewProjection()));
        view.size[0] = static_cast<float>(extent_.width);
        view.size[1] = static_cast<float>(extent_.height);
        view.size[2] = pushbroom_ ? 0.0f : 1.0f;
    }
    uploads_.Write(*motionUniforms_,
                   std::as_bytes(std::span(uniforms)));
}

void RasterRenderer::RecordMotion(
    VkCommandBuffer commands, RenderTargets& targets,
    std::span<const RasterDrawable* const> drawables,
    std::uint32_t firstView) const {
    ERNIE_GPU_ZONE(context_, commands, "Motion vectors");
    GpuImage& raster = targets.Raster(true);
    raster.Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    // Every pixel's motion from the camera's, reading the last
    // sub-frame's depth.
    const DescriptorResource camera[] = {
        DescriptorResource::Of(
            targets.Depth().View(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        DescriptorResource::Of(raster.View(),
                               VK_IMAGE_LAYOUT_GENERAL),
        DescriptorResource::Of(*motionUniforms_)};
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      cameraMotionPipeline_->Handle());
    cameraMotionLayout_->Push(commands,
                              VK_PIPELINE_BIND_POINT_COMPUTE, camera);
    // Every raster layer: a cube camera's views' faces too.
    const std::uint32_t first = LastSubframeView(firstView);
    const std::uint32_t layers = targets.RasterLayers();
    const std::uint32_t constants[4] = {first, 0, 0, 0};
    cameraMotionLayout_->PushConstants(commands, constants,
                                       sizeof(constants));
    vkCmdDispatch(commands, Groups(extent_.width),
                  Groups(extent_.height), layers);
    ShaderBarrier(commands);

    // Then what moved draws its own over it, where its surfaces are
    // the G-buffer's.
    raster.Transition(commands,
                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    targets.Depth().Transition(
        commands, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
    RenderingTarget target;
    target.colors = {raster.View()};
    target.loadColor = true;
    target.depth = targets.Depth().View();
    target.loadDepth = true;
    target.extent = extent_;
    target.layers = layers;
    BeginRendering(commands, target);
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const std::uint32_t view = first + layer;
        const DescriptorResource motion = DescriptorResource::Of(
            *motionUniforms_,
            VkDeviceSize{view} * sizeof(MotionUniforms),
            sizeof(MotionUniforms));
        for (const RasterDrawable* drawable : drawables) {
            drawable->RecordMotion(commands, Frame(view), view,
                                   motion);
        }
    }
    vkCmdEndRendering(commands);
    raster.Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    targets.Depth().Transition(
        commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void RasterRenderer::EnsureViewBuffers(std::uint32_t views) {
    if (uniforms_.Size() >= views * kUniformsStride) {
        return;
    }
    vkDeviceWaitIdle(context_.Device());
    uniforms_ = GpuBuffer(context_, views * kUniformsStride,
                          kViewUniformUsage, BufferMemory::Device);
    shadingUniforms_ =
        GpuBuffer(context_, views * kShadingStride, kViewUniformUsage,
                  BufferMemory::Device);
    sensorUniforms_ =
        GpuBuffer(context_, views * kSensorStride, kViewUniformUsage,
                  BufferMemory::Device);
}

RasterFrame RasterRenderer::Frame(std::uint32_t view) const {
    return {DescriptorResource::Of(uniforms_),
            view,
            DescriptorResource::Of(shadowUniforms_),
            DescriptorResource::Of(shadowMaps_->View()),
            DescriptorResource::Of(shadowSampler_),
            DescriptorResource::Of(clampSampler_)};
}

bool RasterRenderer::CanPresent(std::string_view name) {
    return PresentModeOf(name) != nullptr;
}

void RasterRenderer::Record(
    VkCommandBuffer commands, RenderTargets& targets,
    std::span<const RasterDrawable* const> drawables,
    const RenderingTarget* present, std::string_view var) const {
    if (present != nullptr && !CanPresent(var)) {
        throw std::invalid_argument("Render var '" +
                                    std::string(var) +
                                    "' cannot be presented");
    }
    if (!updated_) {
        if (present != nullptr) {
            BeginRendering(commands, *present);
            vkCmdEndRendering(commands);
        }
        return;
    }
    if (present != nullptr && targets.Var(var) == nullptr) {
        throw std::invalid_argument(
            "The frame holds no render var '" + std::string(var) +
            "'");
    }
    RecordShadows(commands, drawables);
    // With a present target the overlays go over what it shows,
    // whichever var, not LdrColor; a cube camera's, which follow its
    // lens, onto LdrColor when it shows that (and else not).
    const bool lensOverlays = faces_ > 1 && present != nullptr &&
                              var == render_vars::kLdrColor;
    RecordViews(commands, targets, drawables, 0,
                present == nullptr || lensOverlays);
    if (present != nullptr) {
        Present(commands, targets, *present, var);
        if (!overlays_->Empty() && faces_ == 1) {
            ERNIE_GPU_ZONE(context_, commands, "Overlays");
            targets.Depth().Transition(
                commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            OverlayPass::Target target =
                OverlayTarget(targets, LastSubframeView(0), 0);
            target.color = present->color;
            target.format = presentFormat_;
            target.extent = present->extent;
            overlays_->Record(commands, target);
        }
    }
}

void RasterRenderer::RecordShadows(
    VkCommandBuffer commands,
    std::span<const RasterDrawable* const> drawables) const {
    if (!updated_) {
        return;
    }
    ERNIE_ZONE("RecordShadows");
    {
        ERNIE_GPU_ZONE(context_, commands, "Uploads");
        // The frame's host data first, after every earlier frame's
        // use of the buffers it fills.
        uploads_.Record(commands);
    }
    // The atmosphere's tables, when it or the bands changed, before
    // any view shades with them.
    atmosphere_->Record(commands);
    {
        ERNIE_GPU_ZONE(context_, commands, "Prepass");
        // Poses composed on the GPU, and views posed under them,
        // before anything reads them.
        for (const RasterDrawable* drawable : drawables) {
            drawable->RecordPrepass(commands);
        }
        // The custom light patterns' integrals, which shading reads.
        RecordPatternNorms(commands);
        ShaderBarrier(commands);
    }
    RecordLights(commands, drawables);
    if (posesOnGpu_) {
        ERNIE_GPU_ZONE(context_, commands, "View poses");
        RecordViewPoses(commands, drawables);
    }
    {
        ERNIE_GPU_ZONE(context_, commands, "Cull");
        // Culling and other compute the draws read, once for every
        // view and shadow view.
        for (const RasterDrawable* drawable : drawables) {
            drawable->RecordCull(commands);
        }
        ShaderBarrier(commands);
    }
    // The first sub-frame's (the frame's own without motion blur);
    // `RecordViews` the others'.
    RecordShadowMaps(commands, drawables, Frame(0));
    {
        ERNIE_GPU_ZONE(context_, commands, "After shadows");
        // What reads the shadow maps before the views draw, such as
        // the terrain's horizon over what stands on it.
        for (const RasterDrawable* drawable : drawables) {
            drawable->RecordAfterShadows(commands);
        }
        ShaderBarrier(commands);
        castersChanged_ = false;
    }
}

void RasterRenderer::RecordShadowMaps(
    VkCommandBuffer commands,
    std::span<const RasterDrawable* const> drawables,
    const RasterFrame& frame) const {
    if (shadowsEnabled_) {
        ERNIE_GPU_ZONE(context_, commands, "Sun shadows");
        shadowMaps_->Transition(
            commands, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
        // One layered pass for the passes of a kind, each drawing
        // into its layer (`RasterFrame::shadowLayer`), rather than a
        // pass per layer: a batch has a layer per world.
        const auto record = [&](bool fromBase) {
            RenderingTarget shadow;
            shadow.depth = shadowMaps_->View();
            shadow.extent = {shadowResolution_, shadowResolution_};
            shadow.layers = shadowMaps_->Desc().layers;
            shadow.loadDepth = fromBase;
            BeginRendering(commands, shadow);
            for (const ShadowPass& pass : shadowPasses_) {
                if (pass.fromBase != fromBase) {
                    continue;
                }
                RasterFrame layered = frame;
                layered.shadowLayer = pass.layer;
                for (const RasterDrawable* drawable : drawables) {
                    drawable->RecordShadow(
                        commands, layered, pass.cascade,
                        cascades_[pass.cascade].lightViewProjection,
                        pass.casters);
                }
            }
            vkCmdEndRendering(commands);
        };
        record(false);
        // Each world's layer starts as a copy of the shared casters'
        // layer 0.
        std::vector<VkImageCopy> copies;
        for (const ShadowPass& pass : shadowPasses_) {
            if (pass.fromBase) {
                VkImageCopy copy{};
                copy.srcSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0,
                                       0, 1};
                copy.dstSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0,
                                       pass.layer, 1};
                copy.extent = {shadowResolution_, shadowResolution_,
                               1};
                copies.push_back(copy);
            }
        }
        if (!copies.empty()) {
            shadowMaps_->Transition(commands,
                                    VK_IMAGE_LAYOUT_GENERAL);
            vkCmdCopyImage(commands, shadowMaps_->Handle(),
                           VK_IMAGE_LAYOUT_GENERAL,
                           shadowMaps_->Handle(),
                           VK_IMAGE_LAYOUT_GENERAL,
                           static_cast<std::uint32_t>(copies.size()),
                           copies.data());
            shadowMaps_->Transition(
                commands, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
            record(true);
        }
        shadowMaps_->Transition(
            commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    if (!lightViews_.empty()) {
        ERNIE_GPU_ZONE(context_, commands, "Light shadows");
        lightShadowMaps_->Transition(
            commands, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
        for (std::size_t v = 0; v < lightViews_.size(); ++v) {
            RenderingTarget shadow;
            shadow.depth = lightShadowMaps_->LayerView(
                static_cast<std::uint32_t>(v));
            shadow.extent = {lightResolution_, lightResolution_};
            BeginRendering(commands, shadow);
            // Each view draws with its own Shadows block's matrix,
            // which the light pass fits for lights posed on the GPU.
            RasterFrame own = frame;
            own.shadows = DescriptorResource::Of(
                lightShadowBlocks_, v * kLightShadowBlockStride,
                sizeof(ShadowUniforms));
            own.shadowFromFrame = true;
            for (const RasterDrawable* drawable : drawables) {
                drawable->RecordShadow(
                    commands, own, cascades_.size() + v,
                    lightViews_[v].lightViewProjection,
                    lightCasters_);
            }
            vkCmdEndRendering(commands);
        }
        lightShadowMaps_->Transition(
            commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
}

void RasterRenderer::RecordViews(
    VkCommandBuffer commands, RenderTargets& targets,
    std::span<const RasterDrawable* const> drawables,
    std::uint32_t firstView, bool overlays) const {
    if (!updated_) {
        return;
    }
    if (targets.Extent().width != extent_.width ||
        targets.Extent().height != extent_.height ||
        targets.Bands() != bands_.count) {
        throw std::invalid_argument(
            "Render targets need the camera's extent and band count");
    }
    if (targets.Faces() != faces_) {
        throw std::invalid_argument(
            "Render targets need " + std::to_string(faces_) +
            " raster layers per view for the camera (cube faces)");
    }
    if (firstView + targets.Views() > outputViews_) {
        throw std::invalid_argument(
            "Render targets hold views past the frame's " +
            std::to_string(outputViews_));
    }
    for (const std::string_view name : requiredVars_) {
        if (targets.Var(name) == nullptr) {
            throw std::invalid_argument(
                "Render targets lack render var " +
                std::string(name) + ", which the frame needs");
        }
    }
    // With occlusion culling, each view draws its occluders, then a
    // depth pyramid of them is built for all the views, the
    // occlusion-culled instances are tested against it, and each view
    // draws those it does not hide on top.
    const bool occlusion =
        std::any_of(drawables.begin(), drawables.end(),
                    [](const RasterDrawable* drawable) {
                        return drawable->CullsOcclusion();
                    });
    ERNIE_ZONE("RecordViews");
    // Each sub-frame in turn draws and shades its views into the one
    // set of targets; with motion blur their radiance sums into what
    // each row's exposure sees, which the sensor filters. A cube
    // camera's views draw their faces, consecutive internal views and
    // layers.
    const std::uint32_t layers = targets.RasterLayers();
    if (RollsFaces()) {
        RecordFaceRows(commands, firstView);
    }
    for (std::uint32_t subframe = 0; subframe < subframes_;
         ++subframe) {
        const std::uint32_t first =
            (subframe * outputViews_ + firstView) * faces_;
        // Each sub-frame's shadows from its casters where they are at
        // its time (the first's come from `RecordShadows`, unless an
        // earlier chunk of views left the last one's); the cascades
        // stay fitted to the frame's own view.
        if (subframes_ > 1 && (subframe > 0 || firstView > 0)) {
            RecordShadowMaps(commands, drawables,
                             Frame(subframe * outputViews_ * faces_));
        }
        // One layered pass for all the views: each view's draws go to
        // its layer (`Uniforms::flags`).
        const auto draw = [&](DrawPhase phase, bool load) {
            targets.PrepareGBuffer(commands);
            BeginRendering(commands, targets.GBufferTarget(load));
            // Each drawable draws every view in turn, keeping what
            // they bind alike bound.
            std::vector<RasterFrame> frames;
            frames.reserve(layers);
            for (std::uint32_t v = 0; v < layers; ++v) {
                frames.push_back(Frame(first + v));
            }
            for (const RasterDrawable* drawable : drawables) {
                drawable->RecordColorViews(commands, frames, first,
                                           phase);
            }
            vkCmdEndRendering(commands);
        };
        if (!occlusion) {
            ERNIE_GPU_ZONE(context_, commands, "G-buffer");
            draw(DrawPhase::All, false);
        } else {
            {
                ERNIE_GPU_ZONE(context_, commands,
                               "G-buffer occluders");
                draw(DrawPhase::Occluders, false);
            }
            {
                ERNIE_GPU_ZONE(context_, commands,
                               "Hi-Z and occlusion cull");
                BuildHzb(commands, targets);
                for (const RasterDrawable* drawable : drawables) {
                    drawable->RecordOcclusionCull(
                        commands, targets.Hzb(), first, layers);
                }
                ShaderBarrier(commands);
            }
            ERNIE_GPU_ZONE(context_, commands, "G-buffer occludees");
            draw(DrawPhase::Occludees, true);
        }
        Shade(commands, targets, first);
        if (subframes_ > 1) {
            Accumulate(commands, targets, subframe);
        }
    }
    // The MotionVector var's passes, over the last sub-frame's
    // G-buffer, only when it is held.
    if (motionVectors_) {
        RecordMotion(commands, targets, drawables, firstView);
    }
    Filter(commands, targets, firstView);
    if (measureDepth_ && firstView == 0) {
        RecordDepthRange(commands, targets);
    }
    if (meter_ && firstView == 0) {
        RecordMeter(commands, targets);
    }
    if (overlays) {
        RecordOverlays(commands, targets, firstView);
    }
}

void RasterRenderer::RecordMinimal(
    VkCommandBuffer commands, MinimalTargets& targets,
    std::span<const RasterDrawable* const> drawables) const {
    if (!updated_) {
        return;
    }
    if (minimal_ == MinimalMode::Off) {
        throw std::invalid_argument("The last update was not of a "
                                    "frame with minimal shading");
    }
    if (targets.Extent().width != extent_.width ||
        targets.Extent().height != extent_.height ||
        targets.Views() != outputViews_) {
        throw std::invalid_argument(
            "Minimal targets need the camera's extent and the "
            "frame's views");
    }
    ERNIE_ZONE("RecordMinimal");
    const float clear[3] = {minimalClear_.r, minimalClear_.g,
                            minimalClear_.b};
    const auto sampled = [](const GpuImage& image) {
        return DescriptorResource::Of(
            image.View(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    };
    for (std::uint32_t first = 0; first < targets.Views();
         first += targets.Chunk()) {
        const std::uint32_t views =
            std::min(targets.Chunk(), targets.Views() - first);
        {
            ERNIE_GPU_ZONE(context_, commands, "Minimal draw");
            targets.PrepareDraw(commands);
            BeginRendering(commands, targets.Target(views, clear));
            std::vector<RasterFrame> frames;
            frames.reserve(views);
            for (std::uint32_t v = 0; v < views; ++v) {
                frames.push_back(Frame(first + v));
            }
            for (const RasterDrawable* drawable : drawables) {
                if (drawable->DrawsMinimal()) {
                    drawable->RecordColorViews(commands, frames,
                                               first, DrawPhase::All);
                }
            }
            vkCmdEndRendering(commands);
        }
        ERNIE_GPU_ZONE(context_, commands, "Minimal resolve");
        targets.PrepareResolve(commands);
        std::uint32_t holds = 0;
        std::vector<DescriptorResource> resources = {
            sampled(targets.Color()), sampled(targets.Ids()),
            sampled(targets.Depth()),
            DescriptorResource::Of(uniforms_),
            DescriptorResource::Of(shadingUniforms_)};
        const std::string_view vars[] = {
            render_vars::kLdrColor,
            render_vars::kDistanceToImagePlane,
            render_vars::kDistanceToCamera,
            render_vars::kNormal,
            render_vars::kPrimId,
            render_vars::kInstanceId};
        for (std::size_t i = 0; i < std::size(vars); ++i) {
            const FrameImages::LinearVar linear =
                targets.Linear(vars[i]);
            holds |= linear.buffer != nullptr ? 1u << i : 0u;
            resources.push_back(DescriptorResource::Of(
                linear.buffer != nullptr ? *linear.buffer
                                         : unheldVar_));
        }
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                          resolvePipeline_->Handle());
        resolveLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                            resources);
        const std::uint32_t constants[8] = {
            extent_.width,
            extent_.height,
            views,
            first,
            holds,
            static_cast<std::uint32_t>(kUniformsStride),
            static_cast<std::uint32_t>(kShadingStride),
            0u};
        resolveLayout_.PushConstants(commands, constants,
                                     sizeof(constants));
        vkCmdDispatch(commands, Groups(extent_.width),
                      Groups(extent_.height), views);
        // The next chunk draws into the same attachments once this
        // one is read.
        ShaderBarrier(commands);
    }
}

void RasterRenderer::RecordDepthRange(VkCommandBuffer commands,
                                      RenderTargets& targets) const {
    ERNIE_GPU_ZONE(context_, commands, "Depth range");
    const GpuBuffer& range = *depthRanges_[uploads_.Slot()];
    // Nearest starts at the largest float, farthest at 0.
    vkCmdFillBuffer(commands, range.Handle(), 0,
                    sizeof(std::uint32_t), 0x7F7FFFFFu);
    vkCmdFillBuffer(commands, range.Handle(), sizeof(std::uint32_t),
                    sizeof(std::uint32_t), 0u);
    targets.Depth().Transition(
        commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    ShaderBarrier(commands);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      depthRangePipeline_->Handle());
    const DescriptorResource resources[] = {
        DescriptorResource::Of(targets.Depth().View()),
        DescriptorResource::Of(range)};
    depthRangeLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                           resources);
    const struct {
        std::uint32_t size[4];
        float clip[4];
    } constants{
        {extent_.width, extent_.height, 0, orthographic_ ? 1u : 0u},
        {clipRange_[0], clipRange_[1], 0.0f, 0.0f}};
    depthRangeLayout_.PushConstants(commands, &constants,
                                    sizeof(constants));
    // A cube camera's view 0 is its faces' first layers.
    vkCmdDispatch(commands, (extent_.width + 15) / 16,
                  (extent_.height + 15) / 16, faces_);
    ShaderBarrier(commands);
}

void RasterRenderer::RecordMeter(VkCommandBuffer commands,
                                 RenderTargets& targets) const {
    GpuImage* hdr = targets.Var(render_vars::kHdrColor);
    if (hdr == nullptr) {
        return;
    }
    ERNIE_GPU_ZONE(context_, commands, "Exposure meter");
    const GpuBuffer& bins = *meterBins_[uploads_.Slot()];
    vkCmdFillBuffer(commands, bins.Handle(), 0, VK_WHOLE_SIZE, 0u);
    hdr->Transition(commands,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    ShaderBarrier(commands);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      meterPipeline_->Handle());
    const DescriptorResource resources[] = {
        DescriptorResource::Of(hdr->View()),
        DescriptorResource::Of(bins)};
    meterLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      resources);
    const struct {
        std::uint32_t size[4];
        float range[4];
    } constants{{extent_.width, extent_.height, 0, 0},
                {kMeterLow, kMeterBinsPerStop, 0.0f, 0.0f}};
    meterLayout_.PushConstants(commands, &constants,
                               sizeof(constants));
    vkCmdDispatch(commands, (extent_.width + 15) / 16,
                  (extent_.height + 15) / 16, 1);
    hdr->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
}

void RasterRenderer::ReadMeter() {
    const std::uint32_t slot = uploads_.Slot();
    if (!metered_[slot]) {
        return;
    }
    metered_[slot] = false;
    std::array<std::uint32_t, kMeterBins> bins{};
    meterBins_[slot]->Read(std::as_writable_bytes(std::span(bins)));
    std::uint64_t total = 0;
    for (const std::uint32_t count : bins) {
        total += count;
    }
    if (total == 0) {
        meteredLuminance_.reset();
        return;
    }
    // The mean log luminance of the pixels between the darkest tenth
    // and the brightest twentieth, so a few very dark or bright ones
    // do not swing it.
    const double low = 0.10 * static_cast<double>(total);
    const double high = 0.95 * static_cast<double>(total);
    double seen = 0.0;
    double weighted = 0.0;
    double counted = 0.0;
    for (std::uint32_t bin = 0; bin < kMeterBins; ++bin) {
        const double count = bins[bin];
        const double from = std::max(seen, low);
        const double to = std::min(seen + count, high);
        if (to > from) {
            const double centre =
                kMeterLow + (bin + 0.5) / kMeterBinsPerStop;
            weighted += centre * (to - from);
            counted += to - from;
        }
        seen += count;
    }
    meteredLuminance_ =
        counted > 0.0 ? std::optional(std::exp2(weighted / counted))
                      : std::nullopt;
}

std::optional<std::pair<float, float>>
RasterRenderer::VisibleDepths(const FrameParams& params) {
    // This slot's last frame has finished (`BeginFrame`): keep what
    // it saw.
    DepthSample& slot = depthSlots_[uploads_.Slot()];
    if (slot.measured) {
        std::array<std::uint32_t, 2> bits{};
        depthRanges_[uploads_.Slot()]->Read(
            std::as_writable_bytes(std::span(bits)));
        slot.nearest = std::bit_cast<float>(bits[0]);
        slot.farthest = std::bit_cast<float>(bits[1]);
        slot.measured = false;
        if (slot.farthest > 0.0f && slot.nearest <= slot.farthest) {
            seenDepths_[slot.stream] = slot;
        }
    }
    const auto seen = seenDepths_.find(params.stream);
    if (seen == seenDepths_.end()) {
        return std::nullopt;
    }
    // Only for a view close to the one that saw them, and with a
    // margin for what it moved.
    const DepthSample& sample = seen->second;
    const glm::dvec3 eye(params.camera.transform[3]);
    const glm::vec3 forward = params.View().forward;
    if (glm::length(eye - sample.eye) >
            kVisibleFitEyeShift * sample.nearest ||
        glm::dot(forward, sample.forward) < kVisibleFitTurn) {
        return std::nullopt;
    }
    return std::pair(0.9f * sample.nearest, 1.1f * sample.farthest);
}

OverlayPass::Target
RasterRenderer::OverlayTarget(RenderTargets& targets,
                              std::uint32_t view,
                              std::uint32_t layer) const {
    OverlayPass::Target target;
    target.extent = extent_;
    // The overlays' own pass binds the one view's uniforms
    // (overlay.slang).
    target.uniforms = DescriptorResource::Of(
        uniforms_, view * kUniformsStride, sizeof(Uniforms));
    target.depth = targets.Depth().View();
    target.depthLayer = layer;
    target.world = view < viewWorlds_.size() ? viewWorlds_[view] : -1;
    target.nearPlane = clipRange_[0];
    target.farPlane = clipRange_[1];
    target.orthographic = orthographic_;
    target.pushbroom = pushbroom_;
    return target;
}

void RasterRenderer::RecordOverlays(VkCommandBuffer commands,
                                    RenderTargets& targets,
                                    std::uint32_t firstView) const {
    GpuImage* color = targets.Var(render_vars::kLdrColor);
    if (overlays_->Empty() || color == nullptr) {
        return;
    }
    ERNIE_GPU_ZONE(context_, commands, "Overlays");
    targets.Depth().Transition(
        commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (faces_ > 1) {
        RecordFaceOverlays(commands, targets, firstView, *color);
        return;
    }
    color->Transition(commands,
                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    for (std::uint32_t v = 0; v < targets.Views(); ++v) {
        OverlayPass::Target target = OverlayTarget(
            targets, LastSubframeView(firstView + v), v);
        target.color = color->LayerView(v);
        target.format = VK_FORMAT_R8G8B8A8_UNORM;
        overlays_->Record(commands, target);
    }
    color->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
}

void RasterRenderer::RecordFaceRows(VkCommandBuffer commands,
                                    std::uint32_t firstView) const {
    ERNIE_GPU_ZONE(context_, commands, "Cube face rows");
    if (faceRows_ == nullptr ||
        faceRows_->Desc().width != extent_.width ||
        faceRows_->Desc().height != extent_.height ||
        faceRows_->Desc().layers != faces_) {
        vkDeviceWaitIdle(context_.Device());
        faceRows_ = std::make_unique<GpuImage>(
            context_,
            ImageDesc{extent_.width, extent_.height, faces_, 1,
                      VK_FORMAT_R32_UINT, VK_IMAGE_USAGE_STORAGE_BIT,
                      false, true});
    }
    faceRows_->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    const DescriptorResource resources[] = {
        DescriptorResource::Of(sensorUniforms_),
        DescriptorResource::Of(*sensorPasses_.front().params),
        DescriptorResource::Of(faceRows_->View(),
                               VK_IMAGE_LAYOUT_GENERAL)};
    const std::uint32_t constants[8] = {
        firstView,
        0u,
        0u,
        0u,
        sensorPasses_.front().tableOffset,
        lensTable_[1],
        lensTable_[2],
        lensTable_[3]};
    // Cleared, then each sensor pixel's row scattered to the face
    // texels its tap reads.
    for (const char* entry : {"faceRowsClearMain", "faceRowsMain"}) {
        const ComputePipeline& pipeline =
            *lensPipelines_.at(entry + ("\n" + faceLens_));
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipeline.Handle());
        faceRowsLayout_->Push(
            commands, VK_PIPELINE_BIND_POINT_COMPUTE, resources);
        faceRowsLayout_->PushConstants(commands, constants,
                                       sizeof(constants));
        vkCmdDispatch(
            commands, Groups(extent_.width), Groups(extent_.height),
            std::string_view(entry) == "faceRowsMain" ? 1u : faces_);
        ShaderBarrier(commands);
    }
}

void RasterRenderer::RecordFaceOverlays(VkCommandBuffer commands,
                                        RenderTargets& targets,
                                        std::uint32_t firstView,
                                        GpuImage& color) const {
    // Each face draws its overlays as a frame camera's view, over
    // black and over white.
    const std::uint32_t layers = targets.RasterLayers();
    for (std::unique_ptr<GpuImage>& image : faceOverlays_) {
        if (image == nullptr ||
            image->Desc().width != extent_.width ||
            image->Desc().height != extent_.height ||
            image->Desc().layers != layers) {
            vkDeviceWaitIdle(context_.Device());
            image = std::make_unique<GpuImage>(
                context_,
                ImageDesc{extent_.width, extent_.height, layers, 1,
                          VK_FORMAT_R8G8B8A8_UNORM,
                          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                              VK_IMAGE_USAGE_SAMPLED_BIT,
                          false, true});
        }
        image->Transition(commands,
                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }
    RenderingTarget clear;
    clear.colors = {faceOverlays_[0]->View(),
                    faceOverlays_[1]->View()};
    clear.clearColors = {VkClearColorValue{{0.0f, 0.0f, 0.0f, 0.0f}},
                         VkClearColorValue{{1.0f, 1.0f, 1.0f, 1.0f}}};
    clear.extent = extent_;
    clear.layers = layers;
    BeginRendering(commands, clear);
    vkCmdEndRendering(commands);
    const std::uint32_t first = LastSubframeView(firstView);
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        for (const std::unique_ptr<GpuImage>& image : faceOverlays_) {
            OverlayPass::Target target =
                OverlayTarget(targets, first + layer, layer);
            target.color = image->LayerView(layer);
            target.format = VK_FORMAT_R8G8B8A8_UNORM;
            overlays_->Record(commands, target);
        }
    }
    // Then resampled through the lens onto the sensor's image.
    for (const std::unique_ptr<GpuImage>& image : faceOverlays_) {
        image->Transition(commands,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    color.Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    const ComputePipeline& pipeline =
        *lensPipelines_.at("overlayLensMain\n" + faceLens_);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipeline.Handle());
    const DescriptorResource resources[] = {
        DescriptorResource::Of(sensorUniforms_),
        DescriptorResource::Of(color.View(), VK_IMAGE_LAYOUT_GENERAL),
        DescriptorResource::Of(*sensorPasses_.front().params),
        DescriptorResource::Of(
            faceOverlays_[0]->View(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        DescriptorResource::Of(
            faceOverlays_[1]->View(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)};
    overlayLensLayout_->Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                             resources);
    const std::uint32_t constants[8] = {
        firstView,
        0u,
        0u,
        0u,
        sensorPasses_.front().tableOffset,
        lensTable_[1],
        lensTable_[2],
        lensTable_[3]};
    overlayLensLayout_->PushConstants(commands, constants,
                                      sizeof(constants));
    vkCmdDispatch(commands, Groups(extent_.width),
                  Groups(extent_.height), targets.Views());
    ShaderBarrier(commands);
}

void RasterRenderer::BeginFrame(std::uint32_t slot) {
    uploads_.Begin(slot);
}

void RasterRenderer::WritePoseCamera(const FrameParams& params,
                                     const FrameParams& raster) {
    const glm::mat4 projection = params.View().projection;
    const glm::mat4 faceProjection = raster.View().projection;
    // Relative to the frame's origin, so the views' poses are too.
    const glm::mat4 camera[] = {params.Relative(params.posesParent),
                                projection, glm::inverse(projection),
                                faceProjection,
                                glm::inverse(faceProjection)};
    uploads_.Write(poseCamera_, std::as_bytes(std::span(camera)));
    // A pose per output view, a cube camera's faces turning them on
    // the GPU.
    EnsurePoseBuffer(outputViews_);
    poseHistoryViews_[uploads_.Slot()] = outputViews_;
    poseHistoryOrigins_[uploads_.Slot()] = params.origin;
}

const std::shared_ptr<GpuBuffer>&
RasterRenderer::EnsurePoseBuffer(std::uint32_t views) {
    const VkDeviceSize bytes =
        std::max(views, 1u) * sizeof(glm::mat4);
    if (poseBuffer_ == nullptr || poseBuffer_->Size() < bytes) {
        vkDeviceWaitIdle(context_.Device());
        poseBuffer_ = std::make_shared<GpuBuffer>(
            context_, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            BufferMemory::Shared);
        poseHistoryViews_.fill(0);
        for (std::unique_ptr<GpuBuffer>& history : poseHistory_) {
            history = std::make_unique<GpuBuffer>(
                context_, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                BufferMemory::Readback);
        }
    }
    return poseBuffer_;
}

std::vector<glm::dmat4>
RasterRenderer::LastGpuPoses(std::uint32_t views) const {
    std::vector<glm::dmat4> poses;
    const std::uint32_t slot = uploads_.Slot();
    if (poseHistory_[slot] == nullptr ||
        views != poseHistoryViews_[slot] || views == 0) {
        return poses;
    }
    std::vector<glm::mat4> read(views);
    poseHistory_[slot]->Read(std::as_writable_bytes(std::span(read)));
    // Back in the world: the poses were relative to that frame's
    // origin.
    for (const glm::mat4& pose : read) {
        glm::dmat4 world(pose);
        world[3] += glm::dvec4(poseHistoryOrigins_[slot], 0.0);
        poses.push_back(world);
    }
    return poses;
}

void RasterRenderer::PatchViews(VkCommandBuffer commands,
                                const GpuBuffer& blocks,
                                VkDeviceSize first,
                                VkDeviceSize stride,
                                const ViewFields& fields) const {
    if (!posesOnGpu_) {
        return;
    }
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      viewsPipeline_->Handle());
    const DescriptorResource resources[] = {
        DescriptorResource::Of(poseCamera_),
        DescriptorResource::Of(*poseBuffer_),
        DescriptorResource::Of(blocks)};
    viewsLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      resources);
    // A block per output view, or per raster view: a cube camera's
    // views and faces.
    const std::uint32_t count =
        fields.outputViews ? outputViews_ : views_;
    const std::uint32_t faces = fields.outputViews ? 1u : faces_;
    const std::uint32_t constants[12] = {
        count,
        static_cast<std::uint32_t>(stride),
        static_cast<std::uint32_t>(first),
        (fields.forwardDepth ? 1u : 0u) |
            (posesQuaternions_ ? 2u : 0u),
        fields.viewProjection,
        fields.inverseViewProjection,
        fields.eye,
        fields.forward,
        fields.planes,
        fields.pose,
        faces,
        0u};
    viewsLayout_.PushConstants(commands, constants,
                               sizeof(constants));
    vkCmdDispatch(commands, (count + 63) / 64, 1, 1);
}

void RasterRenderer::RecordViewPoses(
    VkCommandBuffer commands,
    std::span<const RasterDrawable* const> drawables) const {
    ERNIE_ZONE("Record view poses");
    ViewFields frame;
    frame.viewProjection = offsetof(Uniforms, viewProjection);
    frame.eye = offsetof(Uniforms, eye);
    frame.forward = offsetof(Uniforms, forward);
    frame.forwardDepth = true;
    PatchViews(commands, uniforms_, 0, kUniformsStride, frame);
    ViewFields shading;
    shading.inverseViewProjection =
        offsetof(ShadingUniforms, invViewProj);
    shading.eye = offsetof(ShadingUniforms, eye);
    PatchViews(commands, shadingUniforms_, 0, kShadingStride,
               shading);
    ViewFields sensor;
    sensor.inverseViewProjection =
        offsetof(SensorUniforms, invViewProj);
    sensor.eye = offsetof(SensorUniforms, eye);
    sensor.forward = offsetof(SensorUniforms, forward);
    sensor.outputViews = true;
    PatchViews(commands, sensorUniforms_, 0, kSensorStride, sensor);
    // The poses rendered, for the CPU's estimates of the next
    // frame's.
    ViewFields history;
    history.pose = 0;
    history.outputViews = true;
    PatchViews(commands, *poseHistory_[uploads_.Slot()], 0,
               sizeof(glm::mat4), history);
    for (const RasterDrawable* drawable : drawables) {
        drawable->RecordViewPoses(commands);
    }
    ShaderBarrier(commands);
}

void RasterRenderer::BuildHzb(VkCommandBuffer commands,
                              RenderTargets& targets) const {
    ERNIE_ZONE("Record Hi-Z");
    GpuImage& depth = targets.Depth();
    GpuImage& hzb = targets.Hzb();
    depth.Transition(commands,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    hzb.Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      hzbPipeline_->Handle());
    std::uint32_t sourceWidth = depth.Desc().width;
    std::uint32_t sourceHeight = depth.Desc().height;
    for (std::uint32_t mip = 0; mip < hzb.Desc().mips; ++mip) {
        const std::uint32_t width =
            std::max(hzb.Desc().width >> mip, 1u);
        const std::uint32_t height =
            std::max(hzb.Desc().height >> mip, 1u);
        const DescriptorResource resources[] = {
            mip == 0
                ? DescriptorResource::Of(depth.View())
                : DescriptorResource::Of(hzb.MipView(mip - 1),
                                         VK_IMAGE_LAYOUT_GENERAL),
            DescriptorResource::Of(hzb.MipView(mip),
                                   VK_IMAGE_LAYOUT_GENERAL)};
        hzbLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                        resources);
        const std::uint32_t size[4] = {sourceWidth, sourceHeight,
                                       width, height};
        hzbLayout_.PushConstants(commands, size, sizeof(size));
        vkCmdDispatch(commands, Groups(width), Groups(height),
                      targets.RasterLayers());
        ShaderBarrier(commands);
        sourceWidth = width;
        sourceHeight = height;
    }
}

void RasterRenderer::BuildPyramid(VkCommandBuffer commands,
                                  GpuImage& image) const {
    ERNIE_GPU_ZONE(context_, commands, "Sensor pyramid");
    image.Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pyramidPipeline_->Handle());
    for (std::uint32_t mip = 1; mip < image.Desc().mips; ++mip) {
        const std::uint32_t sourceWidth =
            std::max(image.Desc().width >> (mip - 1), 1u);
        const std::uint32_t sourceHeight =
            std::max(image.Desc().height >> (mip - 1), 1u);
        const std::uint32_t width =
            std::max(image.Desc().width >> mip, 1u);
        const std::uint32_t height =
            std::max(image.Desc().height >> mip, 1u);
        const DescriptorResource resources[] = {
            DescriptorResource::Of(image.MipView(mip - 1),
                                   VK_IMAGE_LAYOUT_GENERAL),
            DescriptorResource::Of(image.MipView(mip),
                                   VK_IMAGE_LAYOUT_GENERAL)};
        hzbLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                        resources);
        const std::uint32_t size[4] = {sourceWidth, sourceHeight,
                                       width, height};
        hzbLayout_.PushConstants(commands, size, sizeof(size));
        vkCmdDispatch(commands, Groups(width), Groups(height),
                      image.Desc().layers);
        ShaderBarrier(commands);
    }
}

void RasterRenderer::Shade(VkCommandBuffer commands,
                           RenderTargets& targets,
                           std::uint32_t firstView) const {
    ERNIE_ZONE("Shade");
    // Shading writes the pixels it shades; the lens reads background
    // ones as no radiance, so the raster radiance needs no clearing.
    targets.PrepareCompute(commands);

    // Every pass covers all the views in one dispatch, view `z` in
    // layer `z` of each image (a group of a layer per band of the
    // spectral ones) with uniforms `firstView + z`; a cube camera's
    // views are its faces, internal views and raster layers alike.
    const auto sampled = [](const GpuImage& image) {
        return DescriptorResource::Of(
            image.View(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    };
    const auto general = [](const GpuImage& image) {
        return DescriptorResource::Of(image.View(),
                                      VK_IMAGE_LAYOUT_GENERAL);
    };
    const std::uint32_t groupsX = Groups(extent_.width);
    const std::uint32_t groupsY = Groups(extent_.height);
    const std::uint32_t views = targets.RasterLayers();
    {
        ERNIE_GPU_ZONE(context_, commands, "Shading");
        if (domeShadowed_) {
            for (std::unique_ptr<GpuImage>& image : occlusion_) {
                if (image == nullptr ||
                    image->Desc().width != extent_.width ||
                    image->Desc().height != extent_.height ||
                    image->Desc().layers != views) {
                    if (image != nullptr) {
                        vkDeviceWaitIdle(context_.Device());
                    }
                    image = std::make_unique<GpuImage>(
                        context_,
                        ImageDesc{
                            extent_.width, extent_.height, views, 1,
                            VK_FORMAT_R16G16B16A16_SFLOAT,
                            VK_IMAGE_USAGE_STORAGE_BIT, false, true});
                }
                image->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
            }
        }
        const DescriptorResource shading[] = {
            DescriptorResource::Of(shadingUniforms_),
            sampled(targets.GBuffer(0)),
            sampled(targets.GBuffer(1)),
            sampled(targets.GBuffer(2)),
            sampled(targets.Depth()),
            DescriptorResource::Of(*materialTable_),
            DescriptorResource::Of(sigmoidTable_.View()),
            DescriptorResource::Of(tableSampler_),
            general(targets.RasterRadiance()),
            DescriptorResource::Of(lightTable_),
            DescriptorResource::Of(lightShadowMaps_->View()),
            DescriptorResource::Of(shadowSampler_),
            DescriptorResource::Of(lightViewMatrices_),
            sampled(*domeMap_),
            DescriptorResource::Of(*stars_),
            DescriptorResource::Of(*starCells_),
            sampled(domeLighting_ != nullptr ? *domeLighting_
                                             : *domeMap_),
            general(occlusion_[0] != nullptr ? *occlusion_[0]
                                             : *noOcclusion_),
            general(occlusion_[1] != nullptr ? *occlusion_[1]
                                             : *noOcclusion_),
            sampled(terrainHorizon_.image != nullptr
                        ? *terrainHorizon_.image
                        : *noHorizon_),
            DescriptorResource::Of(lightProfiles_),
            sampled(terrainHorizon_.heights != nullptr
                        ? *terrainHorizon_.heights
                        : *noHeights_),
            sampled(atmosphere_->Transmittance()),
            sampled(atmosphere_->MultipleScattering()),
            DescriptorResource::Of(atmosphere_->Medium()),
            DescriptorResource::Of(atmosphere_->Densities()),
            DescriptorResource::Of(atmosphere_->Phases()),
            DescriptorResource::Of(patternNorms_),
            sampled(atmosphere_->SkyIrradiance()),
            DescriptorResource::Of(*patternData_)};
        // The dispatch's model and first view, and the terrain
        // horizon (none without its image).
        const auto constantsFor = [&](std::uint32_t model) {
            ShadingConstants constants{};
            constants.model[0] = model;
            constants.model[1] = firstView;
            if (terrainHorizon_.image != nullptr) {
                std::copy(terrainHorizon_.grid.begin(),
                          terrainHorizon_.grid.end(),
                          constants.horizonGrid);
                std::copy(terrainHorizon_.info.begin(),
                          terrainHorizon_.info.end(),
                          constants.horizonInfo);
                std::copy(terrainHorizon_.levels.begin(),
                          terrainHorizon_.levels.end(),
                          constants.horizonLevels);
                if (terrainHorizon_.heights != nullptr) {
                    std::copy(terrainHorizon_.heightsGrid.begin(),
                              terrainHorizon_.heightsGrid.end(),
                              constants.heightsGrid);
                    std::copy(terrainHorizon_.heightsSize.begin(),
                              terrainHorizon_.heightsSize.end(),
                              constants.heightsSize);
                }
            }
            return constants;
        };
        // Dome occlusion for the full dome lighting mode: per pixel,
        // then filtered, before shading reads it.
        if (domeShadowed_ && occlusionPipeline_ != nullptr) {
            ERNIE_GPU_ZONE(context_, commands, "Dome occlusion");
            const ShadingConstants constants = constantsFor(0);
            for (const ComputePipeline* pipeline :
                 {occlusionPipeline_.get(),
                  occlusionFilterPipeline_.get()}) {
                vkCmdBindPipeline(commands,
                                  VK_PIPELINE_BIND_POINT_COMPUTE,
                                  pipeline->Handle());
                shadingLayout_.Push(commands,
                                    VK_PIPELINE_BIND_POINT_COMPUTE,
                                    shading);
                shadingLayout_.PushConstants(commands, &constants,
                                             sizeof(constants));
                vkCmdDispatch(commands, groupsX, groupsY, views);
                ShaderBarrier(commands);
            }
        }
        for (const std::size_t model : modelsInUse_) {
            vkCmdBindPipeline(
                commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                shadingPipelines_.at({model, pushbroom_})->Handle());
            shadingLayout_.Push(
                commands, VK_PIPELINE_BIND_POINT_COMPUTE, shading);
            const ShadingConstants constants =
                constantsFor(static_cast<std::uint32_t>(model));
            shadingLayout_.PushConstants(commands, &constants,
                                         sizeof(constants));
            vkCmdDispatch(commands, groupsX, groupsY, views);
        }
        // The pixels no geometry covers, when the dome, stars or the
        // atmosphere show there; else the lens reads them as no
        // radiance.
        if (sky_ != 0) {
            ERNIE_GPU_ZONE(context_, commands, "Sky");
            vkCmdBindPipeline(
                commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                skyPipelines_[pushbroom_ ? 1 : 0]->Handle());
            shadingLayout_.Push(
                commands, VK_PIPELINE_BIND_POINT_COMPUTE, shading);
            const ShadingConstants constants = constantsFor(0);
            shadingLayout_.PushConstants(commands, &constants,
                                         sizeof(constants));
            vkCmdDispatch(commands, groupsX, groupsY, views);
        }
        ShaderBarrier(commands);
    }
}

void RasterRenderer::Accumulate(VkCommandBuffer commands,
                                RenderTargets& targets,
                                std::uint32_t subframe) const {
    ERNIE_GPU_ZONE(context_, commands, "Motion blur");
    GpuImage& sum = targets.RadianceSum();
    GpuImage& coverage = targets.Raster(true);
    sum.Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    coverage.Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    const DescriptorResource resources[] = {
        DescriptorResource::Of(targets.RasterRadiance().View(),
                               VK_IMAGE_LAYOUT_GENERAL),
        DescriptorResource::Of(sum.View(), VK_IMAGE_LAYOUT_GENERAL),
        DescriptorResource::Of(
            targets.GBuffer(2).View(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        DescriptorResource::Of(coverage.View(),
                               VK_IMAGE_LAYOUT_GENERAL),
        DescriptorResource::Of(RollsFaces() ? faceRows_->View()
                                            : noFaceRows_->View(),
                               VK_IMAGE_LAYOUT_GENERAL)};
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      motionPipeline_->Handle());
    motionLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                       resources);
    struct {
        std::uint32_t frame[4];
        float times[4];
        float rolling[4];
    } constants{{subframe, subframes_,
                 static_cast<std::uint32_t>(bands_.count),
                 sky_ != 0 ? 1u : 0u},
                {shutter_[0], shutter_[1], shutter_[2], shutter_[3]},
                {shutter_[4],
                 RollsFaces() ? static_cast<float>(faces_) : 0.0f,
                 0.0f, 0.0f}};
    motionLayout_.PushConstants(commands, &constants,
                                sizeof(constants));
    vkCmdDispatch(commands, Groups(extent_.width),
                  Groups(extent_.height),
                  targets.RasterLayers() *
                      static_cast<std::uint32_t>(bands_.count));
    ShaderBarrier(commands);
}

void RasterRenderer::Filter(VkCommandBuffer commands,
                            RenderTargets& targets,
                            std::uint32_t firstView) const {
    const auto sampled = [](const GpuImage& image) {
        return DescriptorResource::Of(
            image.View(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    };
    const auto general = [](const GpuImage& image) {
        return DescriptorResource::Of(image.View(),
                                      VK_IMAGE_LAYOUT_GENERAL);
    };
    const std::uint32_t groupsX = Groups(extent_.width);
    const std::uint32_t groupsY = Groups(extent_.height);
    const std::uint32_t views = targets.Views();
    GpuImage& raster =
        targets.Raster(subframes_ > 1 || motionVectors_);
    raster.Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    ERNIE_GPU_ZONE(context_, commands, "Sensor filters");
    std::vector<DescriptorResource> filters{
        DescriptorResource::Of(sensorUniforms_),
        general(targets.RasterRadiance()),
        sampled(targets.GBuffer(0)), sampled(targets.GBuffer(2)),
        sampled(targets.Depth())};
    // Bindings 5 to 14; MotionVector, bound last, at 30.
    for (const std::string_view name : RenderTargets::VarNames()) {
        if (name != render_vars::kMotionVector) {
            filters.push_back(general(targets.Bound(name)));
        }
    }
    // Each chain's kernels in turn over the one raster image, each
    // writing its own vars.
    for (const SensorPass& pass : sensorPasses_) {
        const SensorPlan& plan = pass.plan;
        std::vector<DescriptorResource> bound = filters;
        bound.push_back(DescriptorResource::Of(*pass.params));
        // Each segment reads the images the one before stored and
        // stores those the next reads, the two sets alternating
        // (placeholders for a plan that passes none on).
        const auto image = [&](bool spectrum,
                               std::uint32_t set) -> GpuImage& {
            GpuImage& sensorImage = targets.SensorImage(
                spectrum, set,
                spectrum ? plan.spectrumImages : plan.channelImages,
                plan.pyramids != 0);
            sensorImage.Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
            return sensorImage;
        };
        // The temporal filters' state, from the last frame and for
        // the next, or a placeholder.
        const bool state =
            plan.states > 0 && pass.stateNext != nullptr;
        GpuImage& previous = state && pass.statePrevious != nullptr
                                 ? *pass.statePrevious
                                 : *statePlaceholder_;
        GpuImage& next = state ? *pass.stateNext : *statePlaceholder_;
        previous.Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
        next.Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
        // The AOV slots the frame holds, which filters' `writeOutput`
        // writes; whether the state holds history, and its slots per
        // view. And a lens of rays' table, at its place in this
        // pass's parameters.
        const std::uint32_t constants[8] = {
            firstView,
            plan.aovs,
            state && pass.history ? 1u : 0u,
            state ? plan.states : 0u,
            pass.tableOffset,
            lensTable_[1],
            lensTable_[2],
            lensTable_[3]};
        for (std::uint32_t segment = 0; segment < plan.segments;
             ++segment) {
            std::vector<DescriptorResource> resources = bound;
            resources.push_back(general(image(true, segment + 1)));
            // Read through a linear sampler too, for pyramids.
            resources.push_back(DescriptorResource::Of(
                image(false, segment + 1).View(),
                VK_IMAGE_LAYOUT_GENERAL, clampSampler_));
            resources.push_back(general(image(true, segment)));
            resources.push_back(general(image(false, segment)));
            for (std::size_t slot = 0; slot < kMaxAovs; ++slot) {
                resources.push_back(general(targets.BoundAov(slot)));
            }
            resources.push_back(general(previous));
            resources.push_back(general(next));
            resources.push_back(
                general(targets.Bound(render_vars::kMotionVector)));
            resources.push_back(general(raster));
            vkCmdBindPipeline(
                commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                sensorPipelines_.at(SensorKernelKey(plan, segment))
                    ->Handle());
            filterLayout_.Push(
                commands, VK_PIPELINE_BIND_POINT_COMPUTE, resources);
            filterLayout_.PushConstants(commands, constants,
                                        sizeof(constants));
            vkCmdDispatch(commands, groupsX, groupsY, views);
            ShaderBarrier(commands);
            // The next segment reads what this one stored (set
            // `segment`); its spatial filter's pyramid of it, when it
            // reads one.
            if (((plan.pyramids >> (segment + 1)) & 1u) != 0) {
                BuildPyramid(commands, image(false, segment));
            }
        }
    }
}

void RasterRenderer::SensorPipelines(const SensorPlan& plan) {
    for (std::uint32_t segment = 0; segment < plan.segments;
         ++segment) {
        const std::string key = SensorKernelKey(plan, segment);
        if (!sensorPipelines_.contains(key)) {
            ERNIE_ZONE("Compile sensor kernel");
            sensorPipelines_.emplace(
                key,
                std::make_unique<ComputePipeline>(
                    context_, filterLayout_,
                    compiler_.CompileSpirv(
                        SensorModules(plan, segment), "sensorMain")));
        }
    }
}

void RasterRenderer::Present(VkCommandBuffer commands,
                             RenderTargets& targets,
                             const RenderingTarget& present,
                             std::string_view var) const {
    const PresentMode& mode = *PresentModeOf(var);
    GpuImage& image = *targets.Var(var);
    image.Transition(commands,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    // The outline reads PrimId in the same pass, so it is made
    // readable before the pass starts.
    GpuImage* ids = highlightWords_ > 0
                        ? targets.Var(render_vars::kPrimId)
                        : nullptr;
    if (ids != nullptr && ids != &image) {
        ids->Transition(commands,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    RenderingTarget target = present;
    target.depth = VK_NULL_HANDLE;
    BeginRendering(commands, target);
    const bool spectral = mode.show == kShowSpectral;
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      (spectral       ? presentSpectral_
                       : mode.integer ? presentUint_
                                      : presentFloat_)
                          ->Handle());
    // Every binding takes the var, SpectralRadiance's view 0 bands at
    // 2; the pipeline reads the one of its type.
    const DescriptorResource view = DescriptorResource::Of(
        image.LayerView(0), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    const DescriptorResource bands = DescriptorResource::Of(
        spectral ? image.GroupView(0) : image.View(),
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    const std::array<DescriptorResource, 4> views{
        view, view, bands, DescriptorResource::Of(*highlight_)};
    presentLayout_.Push(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        views);
    const bool srgb = presentFormat_ == VK_FORMAT_R8G8B8A8_SRGB ||
                      presentFormat_ == VK_FORMAT_B8G8R8A8_SRGB;
    struct {
        std::uint32_t flags[4];
        float range[4];
    } constants{{srgb ? 1u : 0u, mode.show, 0, 0},
                {presentRange_[0], presentRange_[1], presentRange_[2],
                 presentRange_[3]}};
    if (var == render_vars::kRawDn) {
        constants.range[2] = presentRange_[3];
    }
    if (spectral) {
        constants.flags[2] =
            presentBand_ >= 0 && presentBand_ < bands_.count
                ? static_cast<std::uint32_t>(presentBand_)
                : 0xFFFFFFFFu;
        constants.flags[3] = static_cast<std::uint32_t>(bands_.count);
        constants.range[3] = spectralGain_;
    }
    presentLayout_.PushConstants(commands, &constants,
                                 sizeof(constants));
    vkCmdDraw(commands, 3, 1, 0, 0);

    // The highlighted prims' outline over whatever var shows, from
    // the PrimId var.
    if (ids != nullptr) {
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          presentOutline_->Handle());
        const DescriptorResource idView = DescriptorResource::Of(
            ids->LayerView(0),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        const std::array<DescriptorResource, 4> outline{
            idView, idView, bands,
            DescriptorResource::Of(*highlight_)};
        presentLayout_.Push(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            outline);
        struct {
            std::uint32_t flags[4];
            float colour[4];
        } outlineConstants{
            {srgb ? 1u : 0u, highlightWidth_, highlightWords_, 0},
            {highlightColour_.r, highlightColour_.g,
             highlightColour_.b, highlightColour_.a}};
        presentLayout_.PushConstants(commands, &outlineConstants,
                                     sizeof(outlineConstants));
        vkCmdDraw(commands, 3, 1, 0, 0);
        vkCmdEndRendering(commands);
        if (ids != &image) {
            ids->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
        }
    } else {
        vkCmdEndRendering(commands);
    }
    image.Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
}

void RasterRenderer::SetHighlight(
    std::span<const std::uint32_t> primIds, glm::vec4 colour,
    std::uint32_t width) {
    highlightColour_ = colour;
    highlightWidth_ = std::max(width, 1u);
    std::vector<std::uint32_t> words;
    for (const std::uint32_t id : primIds) {
        if (id == 0) {
            continue;
        }
        if (words.size() <= id / 32) {
            words.resize(id / 32 + 1, 0);
        }
        words[id / 32] |= 1u << (id % 32);
    }
    if (words == highlightBits_) {
        return;
    }
    highlightBits_ = words;
    highlightWords_ = static_cast<std::uint32_t>(words.size());
    if (words.empty()) {
        return;
    }
    const VkDeviceSize bytes = words.size() * sizeof(std::uint32_t);
    if (highlight_->Size() < bytes) {
        vkDeviceWaitIdle(context_.Device());
        highlight_ = std::make_unique<GpuBuffer>(
            context_, std::max<VkDeviceSize>(bytes, 256),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
    }
    // Through the frame's uploads, ordered with the frames still
    // reading the buffer.
    uploads_.Write(*highlight_,
                   std::as_bytes(std::span(highlightBits_)));
}

std::uint64_t RasterRenderer::GpuBytes() const {
    std::uint64_t bytes =
        uniforms_.Size() + shadowUniforms_.Size() +
        shadowMaps_->Bytes() + lightShadowMaps_->Bytes() +
        lightTable_.Size() + lightProfiles_.Size() +
        lightViewMatrices_.Size() + sigmoidTable_.Bytes() +
        shadingUniforms_.Size() + sensorUniforms_.Size() +
        (materialTable_ != nullptr ? materialTable_->Size() : 0) +
        overlays_->GpuBytes() + domeMap_->Bytes() +
        (domeLighting_ != nullptr ? domeLighting_->Bytes() : 0) +
        stars_->Size() + starCells_->Size();
    for (const auto& [id, material] : materials_) {
        bytes += material->GpuBytes();
    }
    return bytes;
}

}  // namespace ernie
