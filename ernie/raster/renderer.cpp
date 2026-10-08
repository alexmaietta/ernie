#include "ernie/raster/renderer.h"

#include "ernie/device/gpu_profile.h"
#include "ernie/kernel/embedded_shaders.h"
#include "ernie/raster/poisson_disk.h"
#include "ernie/session/sampling.h"

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
    float sunViewProjection[16];  ///< A batch view's own fitted sun map, with `sunShadow[2]` 1.
    float sunShadow[4];           ///< Its texel size and depth span in world units, and the flag.
};
static_assert(sizeof(Uniforms) == 208, "Uniforms must match frame.slang");

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
static_assert(sizeof(ShadowUniforms) == 480, "ShadowUniforms must match frame.slang");

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
    float toDome[3][4];       ///< Rows of the rotation from world directions to the dome's frame.
    float toCatalogue[3][4];  ///< And to the star catalogue's.
    std::uint32_t sky[4];     ///< kSky* bits, the dome map's width, height and mip levels.
    float starGrid[4];        ///< The star grid's columns and rows.
    float padding[52];        ///< To the views' stride, so shading reads them as one array.
};
static_assert(sizeof(ShadingUniforms) == 1280, "ShadingUniforms must match shading.slang");

/// shading.slang's kSky* bits: what its sky pass draws behind the scene.
constexpr std::uint32_t kSkyDome = 1;
constexpr std::uint32_t kSkyMap = 2;
constexpr std::uint32_t kSkyStars = 4;

struct LightGpu {
    float position[4];
    float axis[4];
    float cone[4];
    float shadow[4];
    float extentX[4];
    float extentY[4];
    float spectrum[SpectralBands::kMaxBands];
};
static_assert(sizeof(LightGpu) == 352, "LightGpu must match shading.slang");

struct MaterialGpu {
    std::uint32_t info[4];
    float params[BrdfRegistry::kMaxParams];
    float reflectance[SpectralBands::kMaxBands];
};
static_assert(sizeof(MaterialGpu) == 336, "MaterialGpu must match shading.slang");

/// Reflectance luminance below which a material counts as black; its spectrum's shape is flat.
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
    float range[4];   ///< First band's lower wavelength, band width (nm), the lens f-number.
    float optics[4];  ///< Diffraction f-number, pixel pitch (um), focal length (mm), focus.
    float photons[SpectralBands::kMaxBands];  ///< Per band: photons per unit radiance (no QE).
};
static_assert(sizeof(SensorUniforms) == 2560, "SensorUniforms must match sensor.slang");

/// Per-view uniforms sit in one buffer at multiples of this, the largest
/// `minUniformBufferOffsetAlignment` Vulkan allows.
constexpr VkDeviceSize kUniformAlignment = 256;

constexpr VkDeviceSize Aligned(VkDeviceSize bytes) {
    return (bytes + kUniformAlignment - 1) / kUniformAlignment * kUniformAlignment;
}

constexpr VkDeviceSize kUniformsStride = Aligned(sizeof(Uniforms));
constexpr VkDeviceSize kShadingStride = Aligned(sizeof(ShadingUniforms));
constexpr VkDeviceSize kSensorStride = Aligned(sizeof(SensorUniforms));
static_assert(kShadingStride == sizeof(ShadingUniforms) && kSensorStride == sizeof(SensorUniforms),
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
};

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
    {17, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
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
};

/// A plan that runs custom filter `model`, with the built-in filters its domain needs, and holds a
/// var downstream of it: for checking the filter compiles.
SensorPlan ProbePlan(const SensorFilterModel& model, const SensorFilterRegistry& registry) {
    std::vector<SensorFilter> before;
    std::string_view held = render_vars::kHdrColor;
    if (model.kind == FilterKind::Integrator) {
        // In place of qe or cie: straight from the lens, holding what it makes.
        std::vector<SensorFilterStep> chain{{model.id, {}}};
        const std::string_view vars[] = {model.domain == FilterDomain::Electrons
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

/// The modules of `plan`'s kernel: the sensor, its custom filters' and its entry point.
std::vector<ShaderModuleSource> SensorModules(const SensorPlan& plan, std::uint32_t segment) {
    std::vector<ShaderModuleSource> modules{{"sensor", shaders::kSensorSlang}};
    for (auto& [name, source] : plan.Modules()) {
        modules.push_back({name, source});
    }
    modules.push_back({"sensorPlan", plan.Source(segment)});
    return modules;
}

/// The key of `plan`'s kernel `segment` among the compiled sensor kernels.
std::string SensorKernelKey(const SensorPlan& plan, std::uint32_t segment) {
    return plan.Key() + "/" + std::to_string(segment);
}

/// hzb.slang: the source level and the destination level.
constexpr DescriptorBinding kHzbBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
};

/// views.slang: the camera, the poses and the per-view blocks it patches.
constexpr DescriptorBinding kViewsBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
    {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
};

/// depth_range.slang: the views' depth and the range it reduces to.
constexpr DescriptorBinding kDepthRangeBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
};

/// meter.slang: HdrColor and its histogram.
constexpr DescriptorBinding kMeterBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_COMPUTE_BIT},
    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
};

/// meter.slang's histogram: its bins, the log2 luminance its first starts at, and its bins per
/// stop, spanning 2^-12 to 2^20 cd/m^2 (starlight to well above the sunlit sky).
constexpr std::uint32_t kMeterBins = 64;
constexpr float kMeterLow = -12.0f;
constexpr float kMeterBinsPerStop = 2.0f;

/// How far a view may move from the one whose depths a cascade fit reuses (`ShadowFit::Visible`):
/// its eye by this fraction of the nearest depth seen, its direction by this cosine.
constexpr double kVisibleFitEyeShift = 0.1;
constexpr float kVisibleFitTurn = 0.995f;

/// views.slang's camera: parent-to-world, projection and its inverse.
constexpr VkDeviceSize kPoseCameraBytes = 3 * sizeof(glm::mat4);

/// Per-view uniforms the CPU writes and views.slang may patch.
constexpr VkBufferUsageFlags kViewUniformUsage =
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

constexpr DescriptorBinding kPresentBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT},
    {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT},
    {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT},
    {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT},
};

/// present.slang's kShowSpectral: SpectralRadiance's bands, through presentSpectral.
constexpr std::uint32_t kShowSpectral = 7;

/// How present.slang shows a render var (its kShow* constants), and whether it is an integer
/// image.
struct PresentMode {
    std::string_view var;
    std::uint32_t show;
    bool integer;
};

constexpr std::array<PresentMode, 10> kPresentModes{{
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
}};

const PresentMode* PresentModeOf(std::string_view var) {
    for (const PresentMode& mode : kPresentModes) {
        if (mode.var == var) {
            return &mode;
        }
    }
    return nullptr;
}

/// Jakob-Hanika coefficients over a `kTableResolution`^3 grid of linear RGB, red along x, green
/// along y, blue across layers, as RGBA32F texels; computed once per process.
const std::vector<float>& SigmoidTable() {
    static const std::vector<float> table = [] {
        const std::uint32_t n = kTableResolution;
        std::vector<float> texels(static_cast<std::size_t>(n) * n * n * 4);
        for (std::uint32_t b = 0; b < n; ++b) {
            for (std::uint32_t g = 0; g < n; ++g) {
                for (std::uint32_t r = 0; r < n; ++r) {
                    const glm::dvec3 rgb(r / double(n - 1), g / double(n - 1), b / double(n - 1));
                    const glm::dvec3 c = RgbToSigmoid(rgb).coefficients;
                    float* texel = &texels[((static_cast<std::size_t>(b) * n + g) * n + r) * 4];
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

/// A barrier making compute and graphics writes visible to later compute, fragment and transfer
/// reads.
void ShaderBarrier(VkCommandBuffer commands) {
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commands, &dependency);
}

/// The integral of the CIE y function over 360-830 nm, in nm, by midpoints of 1 nm steps as
/// light spectra are scaled (`RelativeSpectrum`).
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

/// A unit vector perpendicular enough to `direction` to serve as a look-at up vector.
glm::vec3 UpFor(const glm::vec3& direction) {
    return std::abs(direction.z) > 0.99f ? glm::vec3(0.0f, 1.0f, 0.0f)
                                         : glm::vec3(0.0f, 0.0f, 1.0f);
}

/// World depth spanned by the [0, 1] depth of an orthographic view-projection.
float OrthographicDepth(const glm::mat4& viewProjection) {
    const float gradient =
        glm::length(glm::vec3(viewProjection[0][2], viewProjection[1][2], viewProjection[2][2]));
    return gradient > 0.0f ? 1.0f / gradient : 0.0f;
}

/// A compute shader that evaluates `model` once, to check that its module implements `IBrdf`.
std::vector<ShaderModuleSource> BrdfCheckModules(const BrdfModel& model) {
    return {{"brdf", std::string(BrdfSlang())},
            {BrdfModuleName(model), model.slang},
            {"brdf_check", "import brdf;\nimport " + BrdfModuleName(model) +
                               ";\nRWStructuredBuffer<float3> result;\n"
                               "[shader(\"compute\")]\n[numthreads(1, 1, 1)]\n"
                               "void main() {\n    " +
                               model.typeName +
                               " model;\n    BrdfInput input;\n    BrdfParams params;\n"
                               "    result[0] = model.radianceFactor(input, params);\n}\n"}};
}

}  // namespace

std::vector<ShadowCascade> FitLightShadowViews(const Light& light, const glm::vec3& position,
                                               const glm::vec3& sceneMin, const glm::vec3& sceneMax,
                                               std::uint32_t resolution) {
    if (!light.shadows) {
        return {};
    }
    const float texels = static_cast<float>(std::max(resolution, 1u));
    const glm::vec3 centre = 0.5f * (sceneMin + sceneMax);
    if (light.type == LightType::Distant) {
        const float radius = std::max(0.5f * glm::length(sceneMax - sceneMin), 1e-3f);
        const glm::vec3 towards = glm::normalize(light.direction);
        ShadowCascade view;
        view.lightViewProjection =
            glm::orthoRH_ZO(-radius, radius, -radius, radius, radius, 3.0f * radius) *
            glm::lookAtRH(centre + 2.0f * radius * towards, centre, UpFor(towards));
        view.texelWorldSize = 2.0f * radius / texels;
        return {view};
    }
    float farPlane = 1e-3f;
    for (int i = 0; i < 8; ++i) {
        const glm::vec3 corner((i & 1) ? sceneMax.x : sceneMin.x, (i & 2) ? sceneMax.y : sceneMin.y,
                               (i & 4) ? sceneMax.z : sceneMin.z);
        farPlane = std::max(farPlane, 1.01f * glm::length(corner - position));
    }
    const float nearPlane = std::max(light.radius, 1e-4f * farPlane);
    const auto perspective = [&](const glm::vec3& axis, float fov) {
        ShadowCascade view;
        view.lightViewProjection = glm::perspectiveRH_ZO(fov, 1.0f, nearPlane, farPlane) *
                                   glm::lookAtRH(position, position + axis, UpFor(axis));
        view.texelWorldSize = 2.0f * std::tan(0.5f * fov) / texels;
        return view;
    };
    if (light.coneAngle <= Light::kMaxSpotConeDegrees) {
        return {perspective(glm::normalize(light.direction),
                            2.0f * glm::radians(std::max(light.coneAngle, 0.5f)))};
    }
    std::vector<ShadowCascade> faces;
    for (const glm::vec3 axis : {glm::vec3(1, 0, 0), glm::vec3(-1, 0, 0), glm::vec3(0, 1, 0),
                                 glm::vec3(0, -1, 0), glm::vec3(0, 0, 1), glm::vec3(0, 0, -1)}) {
        faces.push_back(perspective(axis, glm::radians(90.0f)));
    }
    return faces;
}

RasterRenderer::RasterRenderer(const VulkanContext& context, VkFormat presentFormat,
                               BrdfRegistry brdfs)
    : context_(context), presentFormat_(presentFormat), brdfs_(std::move(brdfs)), uploads_(context),
      uniforms_(context, kUniformsStride, kViewUniformUsage, BufferMemory::Device),
      shadowUniforms_(context, sizeof(ShadowUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      BufferMemory::Device),
      clampSampler_(context, SamplerDesc{}),
      shadowSampler_(context, SamplerDesc{VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                          VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 1.0f, true}),
      poisson_(PoissonDisk(16)), shadingLayout_(context, kShadingBindings, 16),
      filterLayout_(context, kFilterBindings, 16), presentLayout_(context, kPresentBindings, 32),
      hzbLayout_(context, kHzbBindings, 16), viewsLayout_(context, kViewsBindings, 48),
      poseCamera_(context, kPoseCameraBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  BufferMemory::Device),
      sigmoidTable_(context, ImageDesc{kTableResolution, kTableResolution, kTableResolution, 1,
                                       VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT}),
      tableSampler_(context, SamplerDesc{VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                         VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 1.0f, false}),
      shadingUniforms_(context, kShadingStride, kViewUniformUsage, BufferMemory::Device),
      sensorUniforms_(context, kSensorStride, kViewUniformUsage, BufferMemory::Device),
      filterParams_(context, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device),
      lightTable_(context, kMaxLights * sizeof(LightGpu), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  BufferMemory::Device),
      lightViewMatrices_(context, kMaxLightShadowViews * sizeof(glm::mat4),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device),
      depthRangeLayout_(context, kDepthRangeBindings, 32),
      meterLayout_(context, kMeterBindings, 32) {
    if (brdfs_.Models().empty()) {
        throw std::invalid_argument("RasterRenderer needs at least one BRDF");
    }
    MaterialSettings defaults;
    if (brdfs_.Find(defaults.brdf) == nullptr) {
        defaults.brdf = brdfs_.Models().front().id;
    }
    materials_.emplace(kDefaultMaterial, std::make_unique<RasterMaterial>(context_, uploads_));
    SetMaterial(kDefaultMaterial, defaults);
    CreateShadowMaps(1, kMaxCascades);
    CreateLightShadowMaps(1, 1);

    const std::vector<float>& table = SigmoidTable();
    const std::size_t layerBytes =
        static_cast<std::size_t>(kTableResolution) * kTableResolution * 4 * sizeof(float);
    GpuBuffer staging(context_, table.size() * sizeof(float), 0, BufferMemory::Upload);
    staging.Write(std::as_bytes(std::span(table)));
    context_.Submit([&](VkCommandBuffer commands) {
        sigmoidTable_.Transition(commands, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        std::vector<VkBufferImageCopy> regions;
        for (std::uint32_t layer = 0; layer < kTableResolution; ++layer) {
            VkBufferImageCopy region{};
            region.bufferOffset = layer * layerBytes;
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, layer, 1};
            region.imageExtent = {kTableResolution, kTableResolution, 1};
            regions.push_back(region);
        }
        vkCmdCopyBufferToImage(commands, staging.Handle(), sigmoidTable_.Handle(),
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               static_cast<std::uint32_t>(regions.size()), regions.data());
        sigmoidTable_.Transition(commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });

    SetDomeMap(nullptr);
    SetStars({});
    {
        // The sky pass shares shading's module, composed with the first model, which it ignores.
        const BrdfModel& brdf = brdfs_.Models().front();
        const std::vector<ShaderModuleSource> modules{
            {"brdf", std::string(BrdfSlang())},
            {BrdfModuleName(brdf), brdf.slang},
            {"shading", "import " + BrdfModuleName(brdf) + ";\ntypealias ActiveBrdf = " +
                            brdf.typeName + ";\n" + shaders::kShadingSlang}};
        skyPipeline_ = std::make_unique<ComputePipeline>(
            context_, shadingLayout_, compiler_.CompileSpirv(modules, "skyMain"));
    }

    const std::vector<ShaderModuleSource> hzbModules{{"hzb", shaders::kHzbSlang}};
    hzbPipeline_ = std::make_unique<ComputePipeline>(context_, hzbLayout_,
                                                     compiler_.CompileSpirv(hzbModules, "hzbMain"));
    const std::vector<ShaderModuleSource> viewsModules{{"views", shaders::kViewsSlang}};
    viewsPipeline_ = std::make_unique<ComputePipeline>(
        context_, viewsLayout_, compiler_.CompileSpirv(viewsModules, "viewsMain"));
    const std::vector<ShaderModuleSource> presentModules{{"present", shaders::kPresentSlang}};
    const std::vector<std::uint32_t> vertex =
        compiler_.CompileSpirv(presentModules, "presentVertex");
    const auto presentPipeline = [&](const char* entry, bool blend = false) {
        const std::vector<std::uint32_t> fragment = compiler_.CompileSpirv(presentModules, entry);
        GraphicsPipelineDesc present;
        present.vertexSpirv = vertex;
        present.fragmentSpirv = fragment;
        present.colorFormat = presentFormat_;
        present.depthFormat = VK_FORMAT_UNDEFINED;
        present.alphaBlend = blend;
        return std::make_unique<GraphicsPipeline>(context_, presentLayout_, present);
    };
    presentFloat_ = presentPipeline("presentFloat");
    presentUint_ = presentPipeline("presentUint");
    presentSpectral_ = presentPipeline("presentSpectral");
    presentOutline_ = presentPipeline("presentOutline", true);
    highlight_ = std::make_unique<GpuBuffer>(context_, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                             BufferMemory::Device);
    statePlaceholder_ = std::make_unique<GpuImage>(
        context_, ImageDesc{1, 1, 1, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, false, true});
    overlays_ = std::make_unique<OverlayPass>(context_, compiler_, uploads_, presentFormat_);
    const std::vector<ShaderModuleSource> depthRangeModules{
        {"depth_range", shaders::kDepthRangeSlang}};
    depthRangePipeline_ = std::make_unique<ComputePipeline>(
        context_, depthRangeLayout_, compiler_.CompileSpirv(depthRangeModules, "depthRangeMain"));
    for (std::unique_ptr<GpuBuffer>& range : depthRanges_) {
        range = std::make_unique<GpuBuffer>(
            context_, 2 * sizeof(std::uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            BufferMemory::Readback);
    }
    const std::vector<ShaderModuleSource> meterModules{{"meter", shaders::kMeterSlang}};
    meterPipeline_ = std::make_unique<ComputePipeline>(
        context_, meterLayout_, compiler_.CompileSpirv(meterModules, "meterMain"));
    for (std::unique_ptr<GpuBuffer>& bins : meterBins_) {
        bins = std::make_unique<GpuBuffer>(
            context_, kMeterBins * sizeof(std::uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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
        for (std::uint32_t segment = 0; segment < probe.segments; ++segment) {
            compiler_.CompileSpirv(SensorModules(probe, segment), "sensorMain");
        }
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument("Sensor filter " + model.id +
                                    " failed to compile: " + error.what());
    }
    sensorFilters_ = std::move(candidate);
}

void RasterRenderer::AddBrdf(BrdfModel model) {
    BrdfRegistry candidate = brdfs_;
    candidate.Add(model);
    try {
        compiler_.CompileSpirv(BrdfCheckModules(model), "main");
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument("BRDF " + model.id + " failed to compile: " + error.what());
    }
    brdfs_ = std::move(candidate);
}

std::uint32_t RasterRenderer::AddMaterial(const MaterialSettings& settings) {
    if (brdfs_.Find(settings.brdf) == nullptr) {
        throw std::invalid_argument("Unknown BRDF: " + settings.brdf);
    }
    const std::uint32_t id = nextMaterial_++;
    materials_.emplace(id, std::make_unique<RasterMaterial>(context_, uploads_));
    materials_.at(id)->SetOrigin(materialsOrigin_);
    ++materialSetVersion_;
    SetMaterial(id, settings);
    return id;
}

void RasterRenderer::SetMaterial(std::uint32_t id, const MaterialSettings& settings) {
    const BrdfModel* model = brdfs_.Find(settings.brdf);
    if (model == nullptr) {
        throw std::invalid_argument("Unknown BRDF: " + settings.brdf);
    }
    Material(id).Update(settings, ResolveBrdfParams(*model, settings.brdfParams));
}

RasterMaterial& RasterRenderer::Material(std::uint32_t id) {
    const auto it = materials_.find(id);
    if (it == materials_.end()) {
        throw std::invalid_argument("Unknown material " + std::to_string(id));
    }
    return *it->second;
}

const RasterMaterial& RasterRenderer::Material(std::uint32_t id) const {
    const auto it = materials_.find(id);
    if (it == materials_.end()) {
        throw std::invalid_argument("Unknown material " + std::to_string(id));
    }
    return *it->second;
}

void RasterRenderer::RemoveMaterial(std::uint32_t id) {
    if (id == kDefaultMaterial) {
        throw std::invalid_argument("The default material cannot be removed");
    }
    const auto it = materials_.find(id);
    if (it == materials_.end()) {
        throw std::invalid_argument("Unknown material " + std::to_string(id));
    }
    ++materialSetVersion_;
    vkDeviceWaitIdle(context_.Device());
    materials_.erase(it);
}

const RasterMaterial& RasterRenderer::MaterialOrDefault(std::uint32_t id) const {
    const auto it = materials_.find(id);
    return it != materials_.end() ? *it->second : *materials_.at(kDefaultMaterial);
}

void RasterRenderer::SetDomeMap(const Image* map) {
    if (map != nullptr) {
        if (map->width == 0 || map->height == 0 ||
            map->pixels.size() != static_cast<std::size_t>(map->width) * map->height) {
            throw std::invalid_argument("The dome's map is empty");
        }
        const std::uint32_t maxSize = context_.Limits().maxImageDimension2D;
        if (map->width > maxSize || map->height > maxSize) {
            throw std::invalid_argument("The dome's map exceeds the device's maximum texture size");
        }
    }
    const Image white(1, 1, glm::vec4(1.0f));
    const std::vector<Image> mips = MipChain(map != nullptr ? *map : white, MipFilter::Average);
    // Float texels, filtered by the shader itself: linear filtering of float32 is optional.
    auto image = std::make_unique<GpuImage>(
        context_, ImageDesc{mips.front().width, mips.front().height, 1,
                            static_cast<std::uint32_t>(mips.size()),
                            VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT});
    std::vector<std::span<const std::byte>> spans;
    for (const Image& mip : mips) {
        spans.push_back(std::as_bytes(std::span(mip.pixels)));
    }
    image->Upload(spans);
    vkDeviceWaitIdle(context_.Device());
    domeMap_ = std::move(image);
    hasDomeMap_ = map != nullptr;
}

void RasterRenderer::SetStars(std::span<const Star> stars) {
    // Each star's grid cell, by declination row from the south pole and right ascension column.
    const auto cellOf = [](const Star& star) {
        constexpr double kTwoPi = 2.0 * M_PI;
        const double ascension = std::fmod(
            std::fmod(static_cast<double>(star.rightAscension), kTwoPi) + kTwoPi, kTwoPi);
        const auto column = std::min<std::uint32_t>(
            static_cast<std::uint32_t>(ascension / kTwoPi * kStarColumns), kStarColumns - 1);
        const auto row = std::min<std::uint32_t>(
            static_cast<std::uint32_t>(
                std::max(static_cast<double>(star.declination) + M_PI / 2.0, 0.0) / M_PI *
                kStarRows),
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
                     [&cells](std::uint32_t a, std::uint32_t b) { return cells[a] < cells[b]; });
    std::vector<glm::vec4> packed;
    packed.reserve(std::max<std::size_t>(stars.size(), 1));
    std::vector<glm::uvec2> ranges(static_cast<std::size_t>(kStarColumns) * kStarRows,
                                   glm::uvec2(0));
    for (const std::uint32_t i : order) {
        glm::uvec2& range = ranges[cells[i]];
        if (range.y == 0) {
            range.x = static_cast<std::uint32_t>(packed.size());
        }
        range.y += 1;
        const Star& star = stars[i];
        packed.emplace_back(star.rightAscension, star.declination, star.temperature,
                            star.irradiance555);
    }
    if (packed.empty()) {
        packed.emplace_back(0.0f);  // buffers cannot be empty
    }
    auto starBuffer = std::make_unique<GpuBuffer>(
        context_, packed.size() * sizeof(glm::vec4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        BufferMemory::Device);
    auto cellBuffer = std::make_unique<GpuBuffer>(
        context_, ranges.size() * sizeof(glm::uvec2), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        BufferMemory::Device);
    starBuffer->Write(std::as_bytes(std::span(packed)));
    cellBuffer->Write(std::as_bytes(std::span(ranges)));
    vkDeviceWaitIdle(context_.Device());
    stars_ = std::move(starBuffer);
    starCells_ = std::move(cellBuffer);
    starCount_ = stars.size();
}

void RasterRenderer::CreateShadowMaps(std::uint32_t resolution, std::uint32_t layers) {
    vkDeviceWaitIdle(context_.Device());
    shadowMaps_ = std::make_unique<GpuImage>(
        context_,
        ImageDesc{resolution, resolution, std::max<std::uint32_t>(layers, kMaxCascades), 1,
                  VK_FORMAT_D32_SFLOAT,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT});
    context_.Submit([this](VkCommandBuffer commands) {
        shadowMaps_->Transition(commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    shadowResolution_ = resolution;
}

void RasterRenderer::CreateLightShadowMaps(std::uint32_t resolution, std::uint32_t layers) {
    vkDeviceWaitIdle(context_.Device());
    lightShadowMaps_ = std::make_unique<GpuImage>(
        context_,
        ImageDesc{resolution, resolution, layers, 1, VK_FORMAT_D32_SFLOAT,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, false,
                  true});
    context_.Submit([this](VkCommandBuffer commands) {
        lightShadowMaps_->Transition(commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    lightResolution_ = resolution;
}

void RasterRenderer::WriteLights(const FrameParams& params, const std::optional<Bounds3>& bounds) {
    ERNIE_ZONE("Lights");
    const std::uint32_t resolution = std::max(params.shadows.lightResolution, 1u);
    lightViews_.clear();
    std::vector<LightGpu> table(params.lights.size());
    for (std::size_t i = 0; i < params.lights.size(); ++i) {
        const Light& light = params.lights[i];
        LightGpu& entry = table[i];
        const glm::vec3 axis = glm::normalize(light.direction);
        const bool distant = light.type == LightType::Distant;
        const glm::vec3 where = distant ? axis : params.Relative(light.position);
        entry.position[0] = where.x;
        entry.position[1] = where.y;
        entry.position[2] = where.z;
        entry.position[3] = static_cast<float>(light.type);
        entry.axis[0] = axis.x;
        entry.axis[1] = axis.y;
        entry.axis[2] = axis.z;
        entry.axis[3] = light.radius;
        const glm::vec3 x = glm::normalize(light.tangent);
        const glm::vec3 y = glm::cross(x, axis);
        const glm::vec2 half =
            light.type == LightType::Disk ? glm::vec2(light.radius) : 0.5f * light.size;
        for (int k = 0; k < 3; ++k) {
            entry.extentX[k] = x[k] * half.x;
            entry.extentY[k] = y[k] * half.y;
        }
        const bool cone = !distant && light.coneAngle < 180.0f;
        const float softness = std::clamp(light.coneSoftness, 0.0f, 1.0f);
        entry.cone[0] = cone ? std::cos(glm::radians(light.coneAngle)) : -2.0f;
        entry.cone[1] = cone ? std::cos(glm::radians(light.coneAngle * (1.0f - softness))) : -1.0f;
        if (bounds) {
            const std::vector<ShadowCascade> views = FitLightShadowViews(
                light, params.Relative(light.position), bounds->first, bounds->second, resolution);
            if (!views.empty() && lightViews_.size() + views.size() <= kMaxLightShadowViews) {
                entry.cone[2] = static_cast<float>(lightViews_.size());
                entry.cone[3] = static_cast<float>(views.size());
                entry.shadow[0] = views.front().texelWorldSize;
                lightViews_.insert(lightViews_.end(), views.begin(), views.end());
            }
        }
        std::copy(light.spectrum.begin(), light.spectrum.end(), entry.spectrum);
    }
    if (!table.empty()) {
        uploads_.Write(lightTable_, std::as_bytes(std::span(table)));
    }
    std::vector<glm::mat4> matrices;
    for (const ShadowCascade& view : lightViews_) {
        matrices.push_back(view.lightViewProjection);
    }
    if (!matrices.empty()) {
        uploads_.Write(lightViewMatrices_, std::as_bytes(std::span(matrices)));
        const auto layers = static_cast<std::uint32_t>(lightViews_.size());
        if (resolution != lightResolution_ || layers > lightShadowMaps_->Desc().layers) {
            CreateLightShadowMaps(resolution, layers);
        }
    }
}

void RasterRenderer::PlanBatchShadows(const FrameParams& params, const Bounds3& scene,
                                      std::span<RasterDrawable* const> drawables) {
    ERNIE_ZONE("Plan batch shadows");
    const ShadowSettings& shadows = params.shadows;
    Light sun;
    sun.type = LightType::Distant;
    sun.direction = params.sunDirection;
    // A fit of the sun's map around `low` to `high`, one cascade covering every view depth.
    const auto fit = [&](const glm::vec3& low, const glm::vec3& high) {
        std::vector<ShadowCascade> views =
            FitLightShadowViews(sun, glm::vec3(0.0f), low, high, shadowResolution_);
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
    worlds.erase(std::unique(worlds.begin(), worlds.end()), worlds.end());
    // One map for all: every view of every world, the worlds' shared casters only, or every
    // world's casters.
    if (worlds == std::vector<std::int32_t>{-1} || shadows.worlds != WorldShadows::Separate) {
        const std::int32_t casting = shadows.worlds == WorldShadows::Shared
                                         ? ShadowCasters::kNoWorld
                                         : ShadowCasters::kAllWorlds;
        shadowPasses_.push_back({0, 0, ShadowCasters{true, casting}, false});
        return;
    }
    // A map per world: layer 0 holds the shared casters. Fitted to the scene, each world's layer
    // is a copy of it with the world's casters drawn on top; fitted to the world, it draws the
    // shared casters again with its own light view. Views of every world (-1) get every caster.
    shadowPasses_.push_back({0, 0, ShadowCasters{true, ShadowCasters::kNoWorld}, false});
    const std::uint32_t maxLayers = context_.Limits().maxImageArrayLayers;
    std::map<std::int32_t, std::uint32_t> layerOf;
    std::map<std::int32_t, std::uint32_t> cascadeOf;
    for (const std::int32_t world : worlds) {
        if (shadowPasses_.size() == maxLayers) {
            break;  // Further worlds get the shared casters' shadows only.
        }
        const auto layer = static_cast<std::uint32_t>(shadowPasses_.size());
        layerOf[world] = layer;
        std::optional<Bounds3> own;
        if (shadows.worldFit == WorldShadowFit::World && world != -1) {
            for (const RasterDrawable* drawable : drawables) {
                if (const std::optional<Bounds3> box = drawable->WorldBounds(world)) {
                    own = own ? Bounds3(glm::min(own->first, box->first),
                                        glm::max(own->second, box->second))
                              : *box;
                }
            }
        }
        if (!own) {
            shadowPasses_.push_back({layer, 0, ShadowCasters{false, world}, true});
            continue;
        }
        const glm::vec3 margin(shadows.worldMargin);
        const auto cascade = static_cast<std::uint32_t>(cascades_.size());
        cascades_.push_back(fit(own->first - margin, own->second + margin));
        cascadeOf[world] = cascade;
        shadowPasses_.push_back({layer, cascade, ShadowCasters{true, world}, false});
    }
    for (std::uint32_t v = 0; v < views_; ++v) {
        const auto layer = layerOf.find(params.ViewWorld(v));
        viewShadowLayers_[v] = layer != layerOf.end() ? layer->second : 0;
        const auto cascade = cascadeOf.find(params.ViewWorld(v));
        viewSunCascades_[v] =
            cascade != cascadeOf.end() ? static_cast<std::int32_t>(cascade->second) : -1;
    }
    shadowLayers_ = static_cast<std::uint32_t>(shadowPasses_.size());
}

void RasterRenderer::Update(const FrameParams& params, std::span<RasterDrawable* const> drawables,
                            std::span<const Overlay* const> overlays) {
    ERNIE_ZONE("Renderer::Update");
    ERNIE_PLOT("Views", params.Views());
    const CameraView view = params.View();
    if (params.lights.size() > kMaxLights) {
        throw std::invalid_argument("A frame has at most " + std::to_string(kMaxLights) +
                                    " lights besides the sun");
    }
    for (const Light& light : params.lights) {
        if (light.spectrum.size() != static_cast<std::size_t>(params.camera.bands.count)) {
            throw std::invalid_argument("Every light's spectrum needs one value per camera band");
        }
        if (glm::length(light.direction) == 0.0f) {
            throw std::invalid_argument("A light's direction must not be zero");
        }
        const bool area = light.type == LightType::Rect || light.type == LightType::Disk;
        if (area && std::abs(glm::dot(glm::normalize(light.tangent),
                                      glm::normalize(light.direction))) > 1e-3f) {
            throw std::invalid_argument("A rect or disk light's tangent must be perpendicular to "
                                        "its direction");
        }
    }
    if (params.dome &&
        params.dome->radiance.size() != static_cast<std::size_t>(params.camera.bands.count)) {
        throw std::invalid_argument("The dome's radiance needs one value per camera band");
    }
    views_ = static_cast<std::uint32_t>(params.Views());
    const std::uint32_t viewsPerTargets =
        RenderTargets::MaxViews(context_, params.camera.bands.count);
    EnsureViewBuffers(views_);
    posesOnGpu_ = params.posesOnGpu;
    if (posesOnGpu_) {
        WritePoseCamera(params);
    }
    // Views posed on the GPU are planned as a batch, whose shadows do not follow the views.
    const bool batch = views_ > 1 || posesOnGpu_;
    const SensorModel& sensor = params.camera.sensor;
    presentRange_ = {static_cast<float>(params.camera.nearPlane),
                     static_cast<float>(params.camera.farPlane),
                     static_cast<float>(sensor.fullWellElectrons),
                     static_cast<float>(std::ldexp(1.0, sensor.adcBits) - 1.0)};

    // Everything below is relative to the frame's origin, the drawables' bounds included.
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
            bounds = bounds ? Bounds3(glm::min(bounds->first, box->first),
                                      glm::max(bounds->second, box->second))
                            : *box;
        }
    }
    const ShadowSettings& shadows = params.shadows;
    shadowsEnabled_ =
        params.sun && shadows.enabled && bounds.has_value() && params.sunDirection.z > 0.0f;
    cascades_.clear();
    shadowPasses_.clear();
    viewShadowLayers_.assign(views_, 0);
    viewSunCascades_.assign(views_, -1);
    shadowLayers_ = 1;
    // Other lights' maps hold what a single view sees, or in a batch what every world shares
    // (every world's casters when merged).
    lightCasters_ = !batch ? ShadowCasters{true, params.ViewWorld(0)}
                           : ShadowCasters{true, shadows.worlds == WorldShadows::Merged
                                                     ? ShadowCasters::kAllWorlds
                                                     : ShadowCasters::kNoWorld};
    if (shadowsEnabled_ && batch) {
        // A batch's per-world maps use their own, smaller resolution.
        const std::uint32_t worldResolution =
            shadows.worldResolution > 0
                ? shadows.worldResolution
                : ShadowSettings::AutoWorldResolution(params.camera.width, params.camera.height);
        const std::uint32_t resolution =
            shadows.worlds == WorldShadows::Separate ? worldResolution : shadows.resolution;
        if (resolution != shadowResolution_) {
            CreateShadowMaps(resolution, kMaxCascades);
        }
        PlanBatchShadows(params, *bounds, drawables);
        if (shadowLayers_ > shadowMaps_->Desc().layers) {
            CreateShadowMaps(resolution, shadowLayers_);
        }
    }
    if (shadowsEnabled_ && !batch) {
        if (shadows.resolution != shadowResolution_) {
            CreateShadowMaps(shadows.resolution, kMaxCascades);
        }
        // Where the shadows land counts too: on a ground plane, which has no bounds, they reach
        // past the casters.
        const auto [low, high] = WithShadows(bounds->first, bounds->second, params.sunDirection);
        float farthest = 0.0f;
        for (int i = 0; i < 8; ++i) {
            const glm::vec3 corner((i & 1) ? high.x : low.x, (i & 2) ? high.y : low.y,
                                   (i & 4) ? high.z : low.z);
            farthest = std::max(farthest, glm::dot(corner - view.eye, view.forward));
        }
        // A margin keeps surfaces at exactly the farthest depth, such as flat ground under a
        // nadir camera, inside the last cascade.
        float shadowDistance = std::min(view.farPlane, 1.01f * farthest);
        float splitNear = view.nearPlane;
        // Sample distribution shadow maps: the cascades span only the depths the view sees,
        // within those the casters can shadow.
        if (shadows.fit == ShadowFit::Visible) {
            if (const std::optional<std::pair<float, float>> seen = VisibleDepths(params)) {
                shadowDistance = std::max(std::min(shadowDistance, seen->second),
                                          view.nearPlane * 1.01f);
                splitNear = std::clamp(seen->first, view.nearPlane, 0.95f * shadowDistance);
            }
        }
        cascades_ = FitCascades(view, shadowDistance, params.sunDirection, low, high,
                                std::clamp(shadows.cascades, 1, kMaxCascades), shadows.splitLambda,
                                shadowResolution_, splitNear);
        // A single view's cascades hold what it sees: the shared casters and its world's.
        for (std::uint32_t c = 0; c < cascades_.size(); ++c) {
            shadowPasses_.push_back({c, c, ShadowCasters{true, params.ViewWorld(0)}, false});
        }
        shadowLayers_ = static_cast<std::uint32_t>(cascades_.size());
    }
    // A metering frame counts its luminance for later frames; this slot's last one has finished.
    ReadMeter();
    meter_ = params.meter && !batch;
    metered_[uploads_.Slot()] = meter_;
    // A visible fit's frame measures what it sees, for later frames' cascades.
    measureDepth_ = shadowsEnabled_ && !batch && params.shadows.fit == ShadowFit::Visible;
    if (measureDepth_) {
        DepthSample& sample = depthSlots_[uploads_.Slot()];
        sample.measured = true;
        sample.stream = params.stream;
        sample.eye = glm::dvec3(params.camera.transform[3]);
        sample.forward = view.forward;
    }

    {
        ERNIE_ZONE("Frame uniforms");
        for (std::uint32_t v = 0; v < views_; ++v) {
            const CameraView viewOf = params.View(v);
            Uniforms uniforms{};
            const glm::mat4 viewProjection = viewOf.ViewProjection();
            std::memcpy(uniforms.viewProjection, glm::value_ptr(viewProjection),
                        sizeof(uniforms.viewProjection));
            uniforms.sun[0] = params.sunDirection.x;
            uniforms.sun[1] = params.sunDirection.y;
            uniforms.sun[2] = params.sunDirection.z;
            uniforms.sun[3] = params.ambient;
            uniforms.forward[0] = viewOf.forward.x;
            uniforms.forward[1] = viewOf.forward.y;
            uniforms.forward[2] = viewOf.forward.z;
            uniforms.forward[3] = glm::dot(viewOf.eye, viewOf.forward);
            uniforms.eye[0] = viewOf.eye.x;
            uniforms.eye[1] = viewOf.eye.y;
            uniforms.eye[2] = viewOf.eye.z;
            uniforms.eye[3] = params.exposure;
            uniforms.flags[0] = static_cast<std::uint32_t>(params.wireframe);
            uniforms.flags[1] = viewShadowLayers_[v];
            // Its layer of the render targets it draws into: a frame of more views renders them in
            // chunks of `RenderTargets::MaxViews`.
            uniforms.flags[2] = v % viewsPerTargets;
            if (viewSunCascades_[v] >= 0) {
                // A view of a world with its own fitted map samples it in place of cascade 0.
                const ShadowCascade& own = cascades_[static_cast<std::size_t>(viewSunCascades_[v])];
                std::memcpy(uniforms.sunViewProjection, glm::value_ptr(own.lightViewProjection),
                            sizeof(uniforms.sunViewProjection));
                uniforms.sunShadow[0] = own.texelWorldSize;
                uniforms.sunShadow[1] = OrthographicDepth(own.lightViewProjection);
                uniforms.sunShadow[2] = 1.0f;
            }
            uploads_.Write(uniforms_, std::as_bytes(std::span(&uniforms, 1)), v * kUniformsStride);
        }
    }

    ShadowUniforms shadowUniforms{};
    // A batch's views all sample cascade 0 (or their own fitted map).
    const std::size_t sampled =
        batch ? std::min<std::size_t>(cascades_.size(), 1) : cascades_.size();
    for (std::size_t c = 0; c < sampled; ++c) {
        std::memcpy(shadowUniforms.lightViewProjection[c],
                    glm::value_ptr(cascades_[c].lightViewProjection), 64);
        shadowUniforms.splits[c] = cascades_[c].splitFar;
        shadowUniforms.texelWorld[c] = cascades_[c].texelWorldSize;
        shadowUniforms.depthWorld[c] = OrthographicDepth(cascades_[c].lightViewProjection);
    }
    shadowUniforms.sun[0] = std::tan(0.5f * glm::radians(std::max(params.sunAngle, 0.0f)));
    shadowUniforms.params[0] = shadows.softness;
    shadowUniforms.params[1] = shadows.normalOffset;
    shadowUniforms.params[2] = static_cast<float>(sampled);
    shadowUniforms.params[3] = shadowsEnabled_ ? 1.0f : 0.0f;
    shadowUniforms.debug[0] = shadows.showCascades ? 1u : 0u;
    for (std::size_t i = 0; i < poisson_.size(); ++i) {
        shadowUniforms.poisson[2 * i] = poisson_[i].x;
        shadowUniforms.poisson[2 * i + 1] = poisson_[i].y;
    }
    uploads_.Write(shadowUniforms_, std::as_bytes(std::span(&shadowUniforms, 1)));

    params.camera.Check();
    if (params.camera.sensor.channels.size() > 4) {
        throw std::invalid_argument("A sensor has at most four channels");
    }
    if (!params.sunIrradiance.empty() &&
        params.sunIrradiance.size() != static_cast<std::size_t>(params.camera.bands.count)) {
        throw std::invalid_argument("The sun's irradiance needs one value per camera band");
    }
    // Only the filters producing the wanted vars run, fused into one kernel that writes only them.
    requiredVars_ = RequiredVars(params.vars);
    sensorPlan_ =
        PlanSensor(params.filters.empty() ? StepsOf(DefaultFilterChain()) : params.filters,
                   requiredVars_, sensorFilters_);
    if (!sensorPlan_.params.empty()) {
        const VkDeviceSize bytes = sensorPlan_.params.size() * sizeof(float);
        if (filterParams_.Size() < bytes) {
            vkDeviceWaitIdle(context_.Device());
            filterParams_ = GpuBuffer(context_, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      BufferMemory::Device);
        }
        uploads_.Write(filterParams_, std::as_bytes(std::span(sensorPlan_.params)));
    }
    filters_ = sensorPlan_.filters;
    SensorPipelines(sensorPlan_);
    bands_ = params.camera.bands;
    extent_ = {params.camera.width, params.camera.height};
    overlays_->Prepare(overlays, params.origin);
    viewWorlds_.assign(views_, -1);
    for (std::size_t v = 0; v < std::min<std::size_t>(params.worlds.size(), views_); ++v) {
        viewWorlds_[v] = params.worlds[v];
    }
    clipRange_ = {static_cast<float>(params.camera.nearPlane),
                  static_cast<float>(params.camera.farPlane)};
    WriteMaterialTable();
    for (const std::size_t model : modelsInUse_) {
        ShadingPipeline(model);
    }
    sky_ = 0;
    if (params.dome && params.dome->visible) {
        sky_ |= kSkyDome | (hasDomeMap_ ? kSkyMap : 0u);
    }
    if (params.stars && starCount_ > 0) {
        sky_ |= kSkyStars;
    }
    // The lights first: shading reads the size of the light shadow maps they make.
    WriteLights(params, bounds);
    WriteShading(params);
    WriteSensor(params);

    shadowViews_ = cascades_;
    shadowViews_.insert(shadowViews_.end(), lightViews_.begin(), lightViews_.end());
    for (RasterDrawable* drawable : drawables) {
        drawable->Prepare(params, shadowViews_);
    }
    updated_ = true;
}

const ComputePipeline& RasterRenderer::ShadingPipeline(std::size_t model) {
    if (const auto it = shadingPipelines_.find(model); it != shadingPipelines_.end()) {
        return *it->second;
    }
    const BrdfModel& brdf = brdfs_.Models()[model];
    const std::vector<ShaderModuleSource> modules{
        {"brdf", std::string(BrdfSlang())},
        {BrdfModuleName(brdf), brdf.slang},
        {"shading", "import " + BrdfModuleName(brdf) + ";\ntypealias ActiveBrdf = " +
                        brdf.typeName + ";\n" + shaders::kShadingSlang}};
    std::vector<std::uint32_t> spirv;
    try {
        spirv = compiler_.CompileSpirv(modules, "shadeMain");
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument("BRDF " + brdf.id +
                                    " failed to compile for shading: " + error.what());
    }
    auto pipeline = std::make_unique<ComputePipeline>(context_, shadingLayout_, spirv);
    return *shadingPipelines_.emplace(model, std::move(pipeline)).first->second;
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
                         [&](const BrdfModel& m) { return m.id == material->Settings().brdf; });
        const auto index = static_cast<std::size_t>(model - registered.begin());
        entry.info[0] = static_cast<std::uint32_t>(index);
        models.insert(index);
        const std::vector<float>& params = material->Params();
        std::copy_n(params.begin(), std::min<std::size_t>(params.size(), BrdfRegistry::kMaxParams),
                    entry.params);
        std::vector<float> shape = material->Reflectance().Bin(bands_);
        const float luminance = material->Luminance();
        for (float& value : shape) {
            value = luminance > kBlackLuminance ? value / luminance : 1.0f;
        }
        std::copy(shape.begin(), shape.end(), entry.reflectance);
    }
    modelsInUse_.assign(models.begin(), models.end());
    const VkDeviceSize bytes = table.size() * sizeof(MaterialGpu);
    if (materialTable_ == nullptr || materialTable_->Size() < bytes) {
        vkDeviceWaitIdle(context_.Device());
        materialTable_ = std::make_unique<GpuBuffer>(
            context_, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
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
    shading.size[0] = extent_.width;
    shading.size[1] = extent_.height;
    shading.size[2] = kTableResolution;
    shading.size[3] = static_cast<std::uint32_t>(params.lights.size());
    for (int i = 0; i < bands_.count && params.sun; ++i) {
        shading.irradiance[i] =
            params.sunIrradiance.empty()
                ? static_cast<float>(M_PI / (kLuminousEfficacy * LuminanceIntegral()))
                : params.sunIrradiance[static_cast<std::size_t>(i)];
    }
    if (params.dome) {
        for (std::size_t k = 0; k < params.dome->sh.size(); ++k) {
            shading.domeSh[k][0] = params.dome->sh[k].r;
            shading.domeSh[k][1] = params.dome->sh[k].g;
            shading.domeSh[k][2] = params.dome->sh[k].b;
        }
        std::copy(params.dome->radiance.begin(), params.dome->radiance.end(), shading.dome);
    }
    // Rows of the rotations, so the shader dots directions with them.
    const auto rows = [](const glm::mat3& rotation, float (&out)[3][4]) {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                out[r][c] = rotation[c][r];
            }
        }
    };
    rows(params.dome ? params.dome->toDome : glm::mat3(1.0f), shading.toDome);
    rows(params.stars ? params.stars->toCatalogue : glm::mat3(1.0f), shading.toCatalogue);
    const ImageDesc& map = domeMap_->Desc();
    shading.sky[0] = sky_;
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
    for (std::uint32_t v = 0; v < views_; ++v) {
        const CameraView view = params.View(v);
        CopyMatrix(shading.invViewProj, glm::inverse(view.ViewProjection()));
        shading.eye[0] = view.eye.x;
        shading.eye[1] = view.eye.y;
        shading.eye[2] = view.eye.z;
        // The lights the view sees: the shared ones and its world's, or all for every world.
        const std::int32_t world = params.ViewWorld(v);
        shading.view[0] = 0;
        for (std::size_t k = 0; k < params.lights.size(); ++k) {
            const std::int32_t of = params.lights[k].world;
            if (of == -1 || world == -1 || of == world) {
                shading.view[0] |= 1u << k;
            }
        }
        uploads_.Write(shadingUniforms_, std::as_bytes(std::span(&shading, 1)), v * kShadingStride);
    }
}

void RasterRenderer::WriteSensor(const FrameParams& params) {
    ERNIE_ZONE("Sensor uniforms");
    const Camera& camera = params.camera;
    const glm::mat4& projection = params.View().projection;
    const double rasterFocal =
        static_cast<double>(projection[1][1]) * static_cast<double>(extent_.height) / 2.0;
    SensorUniforms sensor{};
    sensor.intrinsics[0] = static_cast<float>(rasterFocal * params.overscan);
    sensor.intrinsics[1] =
        static_cast<float>(extent_.width / 2.0 * (1.0 - static_cast<double>(projection[2][0])));
    sensor.intrinsics[2] =
        static_cast<float>(extent_.height / 2.0 * (1.0 + static_cast<double>(projection[2][1])));
    sensor.intrinsics[3] = static_cast<float>(params.overscan);
    sensor.size[0] = extent_.width;
    sensor.size[1] = extent_.height;
    sensor.size[2] = static_cast<std::uint32_t>(bands_.count);
    sensor.size[3] = static_cast<std::uint32_t>(camera.sensor.channels.size());
    const LensModel& lens = camera.lens;
    sensor.distortion[0] = static_cast<float>(lens.k1);
    sensor.distortion[1] = static_cast<float>(lens.k2);
    sensor.distortion[2] = static_cast<float>(lens.k3);
    sensor.distortion[3] = static_cast<float>(lens.p1);
    sensor.lens[0] = static_cast<float>(lens.p2);
    sensor.lens[1] = lens.vignetting ? 1.0f : 0.0f;
    sensor.lens[2] =
        static_cast<float>(camera.LinearExposureScale() * static_cast<double>(params.exposure));
    // A flat spectrum of spectral radiance L has luminance 683 L times the y function's integral;
    // shown with this gain it is as bright as in the photographic image.
    spectralGain_ = static_cast<float>(static_cast<double>(sensor.lens[2]) * kLuminousEfficacy *
                                       LuminanceIntegral());
    sensor.lens[3] = static_cast<float>(camera.exposureTime);
    const SensorModel& model = camera.sensor;
    sensor.sensor[0] = static_cast<float>(model.fullWellElectrons);
    sensor.sensor[1] = static_cast<float>(model.readNoiseElectrons);
    sensor.sensor[2] = static_cast<float>(model.darkCurrentElectrons);
    sensor.sensor[3] = static_cast<float>(model.prnu);
    sensor.adc[0] = static_cast<float>(model.dsnuElectrons);
    sensor.adc[1] = static_cast<float>(model.ElectronsPerDn(camera.iso));
    sensor.adc[2] = static_cast<float>(model.blackLevelDn);
    sensor.adc[3] = static_cast<float>(std::exp2(model.adcBits) - 1.0);
    sensor.noise[0] = model.noise ? 1u : 0u;
    sensor.noise[1] = model.seed;
    sensor.noise[2] = FrameSeed(model.seed, params.stream, params.frame);
    sensor.clear[0] = params.clearColor.r;
    sensor.clear[1] = params.clearColor.g;
    sensor.clear[2] = params.clearColor.b;
    sensor.clear[3] = sky_ != 0 ? 1.0f : 0.0f;  // the background holds the sky's radiance
    sensor.range[0] = static_cast<float>(bands_.minNm);
    sensor.range[1] = static_cast<float>(bands_.Width());
    // The optics filters read (sensor.slang's PixelContext): the lens's f-number for depth of
    // field (0, a pinhole), the one diffraction takes (the lens's, else the exposure's), the
    // sensor's pixel pitch, the focal length and the focus distance.
    sensor.range[2] = static_cast<float>(camera.lensFStop);
    sensor.optics[0] =
        static_cast<float>(camera.lensFStop > 0.0 ? camera.lensFStop : camera.fStop);
    sensor.optics[1] = static_cast<float>(camera.PixelPitchMetres() * 1e6);
    sensor.optics[2] = static_cast<float>(camera.focalLengthMm);
    sensor.optics[3] = static_cast<float>(camera.focusDistance);
    sensor.noise[3] = static_cast<std::uint32_t>(params.frame);
    const std::vector<glm::dvec3> weights = CieWeights(bands_);
    std::vector<glm::dvec3> colours;
    for (int i = 0; i < bands_.count; ++i) {
        colours.push_back(
            EqualEnergyXyzToLinearSrgb(weights[static_cast<std::size_t>(i)] * kLuminousEfficacy));
    }
    // A monochrome sensor's photographic outputs are grey: its channel's spectral response,
    // scaled so that a flat spectrum keeps the luminance the CIE observer gives it.
    if (model.channels.size() == 1) {
        const std::vector<float> response = model.channels.front().quantumEfficiency.Bin(bands_);
        double luminance = 0.0;
        double total = 0.0;
        for (int i = 0; i < bands_.count; ++i) {
            const glm::dvec3& rgb = colours[static_cast<std::size_t>(i)];
            luminance += 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
            total += response[static_cast<std::size_t>(i)];
        }
        const double scale = total > 0.0 ? luminance / total : 0.0;
        for (int i = 0; i < bands_.count; ++i) {
            colours[static_cast<std::size_t>(i)] =
                glm::dvec3(scale * response[static_cast<std::size_t>(i)]);
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
        const std::vector<float> factors = camera.ElectronsPerRadiance(c);
        for (int i = 0; i < bands_.count; ++i) {
            sensor.qe[i][c] = factors[static_cast<std::size_t>(i)];
        }
    }
    for (std::uint32_t v = 0; v < views_; ++v) {
        const CameraView view = params.View(v);
        CopyMatrix(sensor.invViewProj, glm::inverse(view.ViewProjection()));
        sensor.eye[0] = view.eye.x;
        sensor.eye[1] = view.eye.y;
        sensor.eye[2] = view.eye.z;
        sensor.forward[0] = view.forward.x;
        sensor.forward[1] = view.forward.y;
        sensor.forward[2] = view.forward.z;
        // Every view of a batch has noise of its own; view 0's is the single view's.
        sensor.noise[2] = FrameSeed(model.seed, params.stream ^ (v * 0x9E3779B9u), params.frame);
        uploads_.Write(sensorUniforms_, std::as_bytes(std::span(&sensor, 1)), v * kSensorStride);
    }
}

void RasterRenderer::EnsureViewBuffers(std::uint32_t views) {
    if (uniforms_.Size() >= views * kUniformsStride) {
        return;
    }
    vkDeviceWaitIdle(context_.Device());
    uniforms_ =
        GpuBuffer(context_, views * kUniformsStride, kViewUniformUsage, BufferMemory::Device);
    shadingUniforms_ =
        GpuBuffer(context_, views * kShadingStride, kViewUniformUsage, BufferMemory::Device);
    sensorUniforms_ =
        GpuBuffer(context_, views * kSensorStride, kViewUniformUsage, BufferMemory::Device);
}

RasterFrame RasterRenderer::Frame(std::uint32_t view) const {
    return {DescriptorResource::Of(uniforms_, view * kUniformsStride, sizeof(Uniforms)),
            DescriptorResource::Of(shadowUniforms_), DescriptorResource::Of(shadowMaps_->View()),
            DescriptorResource::Of(shadowSampler_), DescriptorResource::Of(clampSampler_)};
}

bool RasterRenderer::CanPresent(std::string_view name) {
    return PresentModeOf(name) != nullptr;
}

void RasterRenderer::Record(VkCommandBuffer commands, RenderTargets& targets,
                            std::span<const RasterDrawable* const> drawables,
                            const RenderingTarget* present, std::string_view var) const {
    if (present != nullptr && !CanPresent(var)) {
        throw std::invalid_argument("Render var '" + std::string(var) + "' cannot be presented");
    }
    if (!updated_) {
        if (present != nullptr) {
            BeginRendering(commands, *present);
            vkCmdEndRendering(commands);
        }
        return;
    }
    RecordShadows(commands, drawables);
    // With a present target the overlays go over what it shows, whichever var, not LdrColor.
    RecordViews(commands, targets, drawables, 0, present == nullptr);
    if (present != nullptr) {
        Present(commands, targets, *present, var);
        if (!overlays_->Empty()) {
            ERNIE_GPU_ZONE(context_, commands, "Overlays");
            targets.Depth().Transition(commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            OverlayPass::Target target = OverlayTarget(targets, 0, 0);
            target.color = present->color;
            target.format = presentFormat_;
            target.extent = present->extent;
            overlays_->Record(commands, target);
        }
    }
}

void RasterRenderer::RecordShadows(VkCommandBuffer commands,
                                   std::span<const RasterDrawable* const> drawables) const {
    if (!updated_) {
        return;
    }
    ERNIE_ZONE("RecordShadows");
    {
        ERNIE_GPU_ZONE(context_, commands, "Uploads");
        // The frame's host data first, after every earlier frame's use of the buffers it fills.
        uploads_.Record(commands);
    }
    if (posesOnGpu_) {
        ERNIE_GPU_ZONE(context_, commands, "View poses");
        RecordViewPoses(commands, drawables);
    }
    {
        ERNIE_GPU_ZONE(context_, commands, "Cull");
        // Culling and other compute the draws read, once for every view and shadow view.
        for (const RasterDrawable* drawable : drawables) {
            drawable->RecordCull(commands);
        }
        ShaderBarrier(commands);
    }
    const RasterFrame frame = Frame(0);
    if (shadowsEnabled_) {
        ERNIE_GPU_ZONE(context_, commands, "Sun shadows");
        shadowMaps_->Transition(commands, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
        const auto record = [&](const ShadowPass& pass) {
            RenderingTarget shadow;
            shadow.depth = shadowMaps_->LayerView(pass.layer);
            shadow.extent = {shadowResolution_, shadowResolution_};
            shadow.loadDepth = pass.fromBase;
            BeginRendering(commands, shadow);
            for (const RasterDrawable* drawable : drawables) {
                drawable->RecordShadow(commands, frame, pass.cascade,
                                       cascades_[pass.cascade].lightViewProjection, pass.casters);
            }
            vkCmdEndRendering(commands);
        };
        for (const ShadowPass& pass : shadowPasses_) {
            if (!pass.fromBase) {
                record(pass);
            }
        }
        // Each world's layer starts as a copy of the shared casters' layer 0.
        std::vector<VkImageCopy> copies;
        for (const ShadowPass& pass : shadowPasses_) {
            if (pass.fromBase) {
                VkImageCopy copy{};
                copy.srcSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
                copy.dstSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, pass.layer, 1};
                copy.extent = {shadowResolution_, shadowResolution_, 1};
                copies.push_back(copy);
            }
        }
        if (!copies.empty()) {
            shadowMaps_->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
            vkCmdCopyImage(commands, shadowMaps_->Handle(), VK_IMAGE_LAYOUT_GENERAL,
                           shadowMaps_->Handle(), VK_IMAGE_LAYOUT_GENERAL,
                           static_cast<std::uint32_t>(copies.size()), copies.data());
            shadowMaps_->Transition(commands, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
            for (const ShadowPass& pass : shadowPasses_) {
                if (pass.fromBase) {
                    record(pass);
                }
            }
        }
        shadowMaps_->Transition(commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    if (!lightViews_.empty()) {
        ERNIE_GPU_ZONE(context_, commands, "Light shadows");
        lightShadowMaps_->Transition(commands, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
        for (std::size_t v = 0; v < lightViews_.size(); ++v) {
            RenderingTarget shadow;
            shadow.depth = lightShadowMaps_->LayerView(static_cast<std::uint32_t>(v));
            shadow.extent = {lightResolution_, lightResolution_};
            BeginRendering(commands, shadow);
            for (const RasterDrawable* drawable : drawables) {
                drawable->RecordShadow(commands, frame, cascades_.size() + v,
                                       lightViews_[v].lightViewProjection, lightCasters_);
            }
            vkCmdEndRendering(commands);
        }
        lightShadowMaps_->Transition(commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
}

void RasterRenderer::RecordViews(VkCommandBuffer commands, RenderTargets& targets,
                                 std::span<const RasterDrawable* const> drawables,
                                 std::uint32_t firstView, bool overlays) const {
    if (!updated_) {
        return;
    }
    if (targets.Extent().width != extent_.width || targets.Extent().height != extent_.height ||
        targets.Bands() != bands_.count) {
        throw std::invalid_argument("Render targets need the camera's extent and band count");
    }
    if (firstView + targets.Views() > views_) {
        throw std::invalid_argument("Render targets hold views past the frame's " +
                                    std::to_string(views_));
    }
    for (const std::string_view name : requiredVars_) {
        if (targets.Var(name) == nullptr) {
            throw std::invalid_argument("Render targets lack render var " + std::string(name) +
                                        ", which the frame needs");
        }
    }
    // With occlusion culling, each view draws its occluders, then a depth pyramid of them is
    // built for all the views, the occlusion-culled instances are tested against it, and each view
    // draws those it does not hide on top.
    const bool occlusion =
        std::any_of(drawables.begin(), drawables.end(),
                    [](const RasterDrawable* drawable) { return drawable->CullsOcclusion(); });
    // One layered pass for all the views: each view's draws go to its layer (`Uniforms::flags`).
    const auto draw = [&](DrawPhase phase, bool load) {
        targets.PrepareGBuffer(commands);
        BeginRendering(commands, targets.GBufferTarget(load));
        for (std::uint32_t v = 0; v < targets.Views(); ++v) {
            const RasterFrame frame = Frame(firstView + v);
            for (const RasterDrawable* drawable : drawables) {
                drawable->RecordColor(commands, frame, firstView + v, phase);
            }
        }
        vkCmdEndRendering(commands);
    };
    ERNIE_ZONE("RecordViews");
    if (!occlusion) {
        ERNIE_GPU_ZONE(context_, commands, "G-buffer");
        draw(DrawPhase::All, false);
    } else {
        {
            ERNIE_GPU_ZONE(context_, commands, "G-buffer occluders");
            draw(DrawPhase::Occluders, false);
        }
        {
            ERNIE_GPU_ZONE(context_, commands, "Hi-Z and occlusion cull");
            BuildHzb(commands, targets);
            for (const RasterDrawable* drawable : drawables) {
                drawable->RecordOcclusionCull(commands, targets.Hzb(), firstView, targets.Views());
            }
            ShaderBarrier(commands);
        }
        ERNIE_GPU_ZONE(context_, commands, "G-buffer occludees");
        draw(DrawPhase::Occludees, true);
    }
    ShadeAndFilter(commands, targets, firstView);
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

void RasterRenderer::RecordDepthRange(VkCommandBuffer commands, RenderTargets& targets) const {
    ERNIE_GPU_ZONE(context_, commands, "Depth range");
    const GpuBuffer& range = *depthRanges_[uploads_.Slot()];
    // Nearest starts at the largest float, farthest at 0.
    vkCmdFillBuffer(commands, range.Handle(), 0, sizeof(std::uint32_t), 0x7F7FFFFFu);
    vkCmdFillBuffer(commands, range.Handle(), sizeof(std::uint32_t), sizeof(std::uint32_t), 0u);
    targets.Depth().Transition(commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    ShaderBarrier(commands);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, depthRangePipeline_->Handle());
    const DescriptorResource resources[] = {DescriptorResource::Of(targets.Depth().View()),
                                            DescriptorResource::Of(range)};
    depthRangeLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE, resources);
    const struct {
        std::uint32_t size[4];
        float clip[4];
    } constants{{extent_.width, extent_.height, 0, 0}, {clipRange_[0], clipRange_[1], 0.0f, 0.0f}};
    depthRangeLayout_.PushConstants(commands, &constants, sizeof(constants));
    vkCmdDispatch(commands, (extent_.width + 15) / 16, (extent_.height + 15) / 16, 1);
    ShaderBarrier(commands);
}

void RasterRenderer::RecordMeter(VkCommandBuffer commands, RenderTargets& targets) const {
    GpuImage* hdr = targets.Var(render_vars::kHdrColor);
    if (hdr == nullptr) {
        return;
    }
    ERNIE_GPU_ZONE(context_, commands, "Exposure meter");
    const GpuBuffer& bins = *meterBins_[uploads_.Slot()];
    vkCmdFillBuffer(commands, bins.Handle(), 0, VK_WHOLE_SIZE, 0u);
    hdr->Transition(commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    ShaderBarrier(commands);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, meterPipeline_->Handle());
    const DescriptorResource resources[] = {DescriptorResource::Of(hdr->View()),
                                            DescriptorResource::Of(bins)};
    meterLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE, resources);
    const struct {
        std::uint32_t size[4];
        float range[4];
    } constants{{extent_.width, extent_.height, 0, 0}, {kMeterLow, kMeterBinsPerStop, 0.0f, 0.0f}};
    meterLayout_.PushConstants(commands, &constants, sizeof(constants));
    vkCmdDispatch(commands, (extent_.width + 15) / 16, (extent_.height + 15) / 16, 1);
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
    // The mean log luminance of the pixels between the darkest tenth and the brightest
    // twentieth, so a few very dark or bright ones do not swing it.
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
            const double centre = kMeterLow + (bin + 0.5) / kMeterBinsPerStop;
            weighted += centre * (to - from);
            counted += to - from;
        }
        seen += count;
    }
    meteredLuminance_ = counted > 0.0 ? std::optional(std::exp2(weighted / counted)) : std::nullopt;
}

std::optional<std::pair<float, float>> RasterRenderer::VisibleDepths(const FrameParams& params) {
    // This slot's last frame has finished (`BeginFrame`): keep what it saw.
    DepthSample& slot = depthSlots_[uploads_.Slot()];
    if (slot.measured) {
        std::array<std::uint32_t, 2> bits{};
        depthRanges_[uploads_.Slot()]->Read(std::as_writable_bytes(std::span(bits)));
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
    // Only for a view close to the one that saw them, and with a margin for what it moved.
    const DepthSample& sample = seen->second;
    const glm::dvec3 eye(params.camera.transform[3]);
    const glm::vec3 forward = params.View().forward;
    if (glm::length(eye - sample.eye) > kVisibleFitEyeShift * sample.nearest ||
        glm::dot(forward, sample.forward) < kVisibleFitTurn) {
        return std::nullopt;
    }
    return std::pair(0.9f * sample.nearest, 1.1f * sample.farthest);
}

OverlayPass::Target RasterRenderer::OverlayTarget(RenderTargets& targets, std::uint32_t view,
                                                  std::uint32_t layer) const {
    OverlayPass::Target target;
    target.extent = extent_;
    target.uniforms = Frame(view).uniforms;
    target.depth = targets.Depth().View();
    target.depthLayer = layer;
    target.world = view < viewWorlds_.size() ? viewWorlds_[view] : -1;
    target.nearPlane = clipRange_[0];
    target.farPlane = clipRange_[1];
    return target;
}

void RasterRenderer::RecordOverlays(VkCommandBuffer commands, RenderTargets& targets,
                                    std::uint32_t firstView) const {
    GpuImage* color = targets.Var(render_vars::kLdrColor);
    if (overlays_->Empty() || color == nullptr) {
        return;
    }
    ERNIE_GPU_ZONE(context_, commands, "Overlays");
    targets.Depth().Transition(commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    color->Transition(commands, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    for (std::uint32_t v = 0; v < targets.Views(); ++v) {
        OverlayPass::Target target = OverlayTarget(targets, firstView + v, v);
        target.color = color->LayerView(v);
        target.format = VK_FORMAT_R8G8B8A8_UNORM;
        overlays_->Record(commands, target);
    }
    color->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
}

void RasterRenderer::BeginFrame(std::uint32_t slot) {
    uploads_.Begin(slot);
}

void RasterRenderer::WritePoseCamera(const FrameParams& params) {
    const glm::mat4 projection = params.View().projection;
    // Relative to the frame's origin, so the views' poses are too.
    const glm::mat4 camera[] = {params.Relative(params.posesParent), projection,
                                glm::inverse(projection)};
    uploads_.Write(poseCamera_, std::as_bytes(std::span(camera)));
    const VkDeviceSize bytes = views_ * sizeof(glm::mat4);
    if (poseBuffer_ == nullptr || poseBuffer_->Size() < bytes) {
        vkDeviceWaitIdle(context_.Device());
        poseBuffer_ = std::make_unique<GpuBuffer>(
            context_, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Shared);
        poseHistoryViews_.fill(0);
        for (std::unique_ptr<GpuBuffer>& history : poseHistory_) {
            history = std::make_unique<GpuBuffer>(
                context_, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Readback);
        }
    }
    poseHistoryViews_[uploads_.Slot()] = views_;
    poseHistoryOrigins_[uploads_.Slot()] = params.origin;
}

std::vector<glm::dmat4> RasterRenderer::LastGpuPoses(std::uint32_t views) const {
    std::vector<glm::dmat4> poses;
    const std::uint32_t slot = uploads_.Slot();
    if (poseHistory_[slot] == nullptr || views != poseHistoryViews_[slot] || views == 0) {
        return poses;
    }
    std::vector<glm::mat4> read(views);
    poseHistory_[slot]->Read(std::as_writable_bytes(std::span(read)));
    // Back in the world: the poses were relative to that frame's origin.
    for (const glm::mat4& pose : read) {
        glm::dmat4 world(pose);
        world[3] += glm::dvec4(poseHistoryOrigins_[slot], 0.0);
        poses.push_back(world);
    }
    return poses;
}

void RasterRenderer::PatchViews(VkCommandBuffer commands, const GpuBuffer& blocks,
                                VkDeviceSize first, VkDeviceSize stride,
                                const ViewFields& fields) const {
    if (!posesOnGpu_) {
        return;
    }
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, viewsPipeline_->Handle());
    const DescriptorResource resources[] = {DescriptorResource::Of(poseCamera_),
                                            DescriptorResource::Of(*poseBuffer_),
                                            DescriptorResource::Of(blocks)};
    viewsLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE, resources);
    const std::uint32_t constants[12] = {views_,
                                         static_cast<std::uint32_t>(stride),
                                         static_cast<std::uint32_t>(first),
                                         fields.forwardDepth ? 1u : 0u,
                                         fields.viewProjection,
                                         fields.inverseViewProjection,
                                         fields.eye,
                                         fields.forward,
                                         fields.planes,
                                         fields.pose,
                                         0u,
                                         0u};
    viewsLayout_.PushConstants(commands, constants, sizeof(constants));
    vkCmdDispatch(commands, (views_ + 63) / 64, 1, 1);
}

void RasterRenderer::RecordViewPoses(VkCommandBuffer commands,
                                     std::span<const RasterDrawable* const> drawables) const {
    ERNIE_ZONE("Record view poses");
    ViewFields frame;
    frame.viewProjection = offsetof(Uniforms, viewProjection);
    frame.eye = offsetof(Uniforms, eye);
    frame.forward = offsetof(Uniforms, forward);
    frame.forwardDepth = true;
    PatchViews(commands, uniforms_, 0, kUniformsStride, frame);
    ViewFields shading;
    shading.inverseViewProjection = offsetof(ShadingUniforms, invViewProj);
    shading.eye = offsetof(ShadingUniforms, eye);
    PatchViews(commands, shadingUniforms_, 0, kShadingStride, shading);
    ViewFields sensor;
    sensor.inverseViewProjection = offsetof(SensorUniforms, invViewProj);
    sensor.eye = offsetof(SensorUniforms, eye);
    sensor.forward = offsetof(SensorUniforms, forward);
    PatchViews(commands, sensorUniforms_, 0, kSensorStride, sensor);
    // The poses rendered, for the CPU's estimates of the next frame's.
    ViewFields history;
    history.pose = 0;
    PatchViews(commands, *poseHistory_[uploads_.Slot()], 0, sizeof(glm::mat4), history);
    for (const RasterDrawable* drawable : drawables) {
        drawable->RecordViewPoses(commands);
    }
    ShaderBarrier(commands);
}

void RasterRenderer::BuildHzb(VkCommandBuffer commands, RenderTargets& targets) const {
    ERNIE_ZONE("Record Hi-Z");
    GpuImage& depth = targets.Depth();
    GpuImage& hzb = targets.Hzb();
    depth.Transition(commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    hzb.Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, hzbPipeline_->Handle());
    std::uint32_t sourceWidth = depth.Desc().width;
    std::uint32_t sourceHeight = depth.Desc().height;
    for (std::uint32_t mip = 0; mip < hzb.Desc().mips; ++mip) {
        const std::uint32_t width = std::max(hzb.Desc().width >> mip, 1u);
        const std::uint32_t height = std::max(hzb.Desc().height >> mip, 1u);
        const DescriptorResource resources[] = {
            mip == 0 ? DescriptorResource::Of(depth.View())
                     : DescriptorResource::Of(hzb.MipView(mip - 1), VK_IMAGE_LAYOUT_GENERAL),
            DescriptorResource::Of(hzb.MipView(mip), VK_IMAGE_LAYOUT_GENERAL)};
        hzbLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE, resources);
        const std::uint32_t size[4] = {sourceWidth, sourceHeight, width, height};
        hzbLayout_.PushConstants(commands, size, sizeof(size));
        vkCmdDispatch(commands, Groups(width), Groups(height), targets.Views());
        ShaderBarrier(commands);
        sourceWidth = width;
        sourceHeight = height;
    }
}

void RasterRenderer::ShadeAndFilter(VkCommandBuffer commands, RenderTargets& targets,
                                    std::uint32_t firstView) const {
    ERNIE_ZONE("ShadeAndFilter");
    // Shading writes the pixels it shades; the lens reads background ones as no radiance, so
    // the raster radiance needs no clearing.
    targets.PrepareCompute(commands);

    // Every pass covers all the views in one dispatch, view `z` in layer `z` of each image (a
    // group of a layer per band of the spectral ones) with uniforms `firstView + z`.
    const auto sampled = [](const GpuImage& image) {
        return DescriptorResource::Of(image.View(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    };
    const auto general = [](const GpuImage& image) {
        return DescriptorResource::Of(image.View(), VK_IMAGE_LAYOUT_GENERAL);
    };
    const std::uint32_t groupsX = Groups(extent_.width);
    const std::uint32_t groupsY = Groups(extent_.height);
    const std::uint32_t views = targets.Views();
    {
        ERNIE_GPU_ZONE(context_, commands, "Shading");
        const DescriptorResource shading[] = {DescriptorResource::Of(shadingUniforms_),
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
                                              DescriptorResource::Of(*starCells_)};
        for (const std::size_t model : modelsInUse_) {
            vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                              shadingPipelines_.at(model)->Handle());
            shadingLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE, shading);
            const std::uint32_t constants[4] = {static_cast<std::uint32_t>(model), firstView, 0, 0};
            shadingLayout_.PushConstants(commands, constants, sizeof(constants));
            vkCmdDispatch(commands, groupsX, groupsY, views);
        }
        // The pixels no geometry covers, when the dome or stars show there; else the lens reads
        // them as no radiance.
        if (sky_ != 0) {
            vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, skyPipeline_->Handle());
            shadingLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE, shading);
            const std::uint32_t constants[4] = {0, firstView, 0, 0};
            shadingLayout_.PushConstants(commands, constants, sizeof(constants));
            vkCmdDispatch(commands, groupsX, groupsY, views);
        }
        ShaderBarrier(commands);
    }

    ERNIE_GPU_ZONE(context_, commands, "Sensor filters");
    std::vector<DescriptorResource> filters{
        DescriptorResource::Of(sensorUniforms_), general(targets.RasterRadiance()),
        sampled(targets.GBuffer(0)), sampled(targets.GBuffer(2)), sampled(targets.Depth())};
    for (const std::string_view name : RenderTargets::VarNames()) {
        filters.push_back(general(targets.Bound(name)));
    }
    filters.push_back(DescriptorResource::Of(filterParams_));
    // Each segment reads the images the one before stored and stores those the next reads, the
    // two sets alternating (placeholders for a plan that passes none on).
    const auto image = [&](bool spectrum, std::uint32_t set) -> GpuImage& {
        GpuImage& image = targets.SensorImage(
            spectrum, set, spectrum ? sensorPlan_.spectrumImages : sensorPlan_.channelImages);
        image.Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
        return image;
    };
    // The temporal filters' state, from the last frame and for the next, or a placeholder.
    const bool state = sensorPlan_.states > 0 && stateNext_ != nullptr;
    GpuImage& previous = state && statePrevious_ != nullptr ? *statePrevious_ : *statePlaceholder_;
    GpuImage& next = state ? *stateNext_ : *statePlaceholder_;
    previous.Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    next.Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    // The AOV slots the frame holds, which filters' `writeOutput` writes; whether the state holds
    // history, and its slots per view.
    const std::uint32_t constants[4] = {firstView, sensorPlan_.aovs,
                                        state && stateHistory_ ? 1u : 0u,
                                        state ? sensorPlan_.states : 0u};
    for (std::uint32_t segment = 0; segment < sensorPlan_.segments; ++segment) {
        std::vector<DescriptorResource> resources = filters;
        resources.push_back(general(image(true, segment + 1)));
        resources.push_back(general(image(false, segment + 1)));
        resources.push_back(general(image(true, segment)));
        resources.push_back(general(image(false, segment)));
        for (std::size_t slot = 0; slot < kMaxAovs; ++slot) {
            resources.push_back(general(targets.BoundAov(slot)));
        }
        resources.push_back(general(previous));
        resources.push_back(general(next));
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                          sensorPipelines_.at(SensorKernelKey(sensorPlan_, segment))->Handle());
        filterLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE, resources);
        filterLayout_.PushConstants(commands, constants, sizeof(constants));
        vkCmdDispatch(commands, groupsX, groupsY, views);
        ShaderBarrier(commands);
    }
}

void RasterRenderer::SensorPipelines(const SensorPlan& plan) {
    for (std::uint32_t segment = 0; segment < plan.segments; ++segment) {
        const std::string key = SensorKernelKey(plan, segment);
        if (!sensorPipelines_.contains(key)) {
            ERNIE_ZONE("Compile sensor kernel");
            sensorPipelines_.emplace(
                key, std::make_unique<ComputePipeline>(
                         context_, filterLayout_,
                         compiler_.CompileSpirv(SensorModules(plan, segment), "sensorMain")));
        }
    }
}

void RasterRenderer::Present(VkCommandBuffer commands, RenderTargets& targets,
                             const RenderingTarget& present, std::string_view var) const {
    const PresentMode& mode = *PresentModeOf(var);
    GpuImage& image = *targets.Var(var);
    image.Transition(commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    // The outline reads PrimId in the same pass, so it is made readable before the pass starts.
    GpuImage* ids = highlightWords_ > 0 ? targets.Var(render_vars::kPrimId) : nullptr;
    if (ids != nullptr && ids != &image) {
        ids->Transition(commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    RenderingTarget target = present;
    target.depth = VK_NULL_HANDLE;
    BeginRendering(commands, target);
    const bool spectral = mode.show == kShowSpectral;
    vkCmdBindPipeline(
        commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
        (spectral ? presentSpectral_ : mode.integer ? presentUint_ : presentFloat_)->Handle());
    // Every binding takes the var, SpectralRadiance's view 0 bands at 2; the pipeline reads the
    // one of its type.
    const DescriptorResource view =
        DescriptorResource::Of(image.LayerView(0), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    const DescriptorResource bands = DescriptorResource::Of(
        spectral ? image.GroupView(0) : image.View(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    const std::array<DescriptorResource, 4> views{view, view, bands,
                                                  DescriptorResource::Of(*highlight_)};
    presentLayout_.Push(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, views);
    const bool srgb =
        presentFormat_ == VK_FORMAT_R8G8B8A8_SRGB || presentFormat_ == VK_FORMAT_B8G8R8A8_SRGB;
    struct {
        std::uint32_t flags[4];
        float range[4];
    } constants{{srgb ? 1u : 0u, mode.show, 0, 0},
                {presentRange_[0], presentRange_[1], presentRange_[2], presentRange_[3]}};
    if (var == render_vars::kRawDn) {
        constants.range[2] = presentRange_[3];
    }
    if (spectral) {
        constants.flags[2] = presentBand_ >= 0 && presentBand_ < bands_.count
                                 ? static_cast<std::uint32_t>(presentBand_)
                                 : 0xFFFFFFFFu;
        constants.flags[3] = static_cast<std::uint32_t>(bands_.count);
        constants.range[3] = spectralGain_;
    }
    presentLayout_.PushConstants(commands, &constants, sizeof(constants));
    vkCmdDraw(commands, 3, 1, 0, 0);

    // The highlighted prims' outline over whatever var shows, from the PrimId var.
    if (ids != nullptr) {
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, presentOutline_->Handle());
        const DescriptorResource idView =
            DescriptorResource::Of(ids->LayerView(0), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        const std::array<DescriptorResource, 4> outline{idView, idView, bands,
                                                        DescriptorResource::Of(*highlight_)};
        presentLayout_.Push(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, outline);
        struct {
            std::uint32_t flags[4];
            float colour[4];
        } outlineConstants{{srgb ? 1u : 0u, highlightWidth_, highlightWords_, 0},
                           {highlightColour_.r, highlightColour_.g, highlightColour_.b,
                            highlightColour_.a}};
        presentLayout_.PushConstants(commands, &outlineConstants, sizeof(outlineConstants));
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

void RasterRenderer::SetHighlight(std::span<const std::uint32_t> primIds, glm::vec4 colour,
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
        highlight_ = std::make_unique<GpuBuffer>(context_, std::max<VkDeviceSize>(bytes, 256),
                                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                 BufferMemory::Device);
    }
    // Through the frame's uploads, ordered with the frames still reading the buffer.
    uploads_.Write(*highlight_, std::as_bytes(std::span(highlightBits_)));
}

std::uint64_t RasterRenderer::GpuBytes() const {
    std::uint64_t bytes = uniforms_.Size() + shadowUniforms_.Size() + shadowMaps_->Bytes() +
                          lightShadowMaps_->Bytes() + lightTable_.Size() +
                          lightViewMatrices_.Size() + sigmoidTable_.Bytes() +
                          shadingUniforms_.Size() + sensorUniforms_.Size() +
                          (materialTable_ != nullptr ? materialTable_->Size() : 0) +
                          overlays_->GpuBytes() + domeMap_->Bytes() + stars_->Size() +
                          starCells_->Size();
    for (const auto& [id, material] : materials_) {
        bytes += material->GpuBytes();
    }
    return bytes;
}

}  // namespace ernie
