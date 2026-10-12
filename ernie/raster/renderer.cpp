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
    float forward[4];
    float eye[4];
    std::uint32_t flags[4];
    /// Minimal shading's distant lights: towards each, and its share
    /// of the brightest's light.
    float minimalLights[4][4];
    float motion[4];  ///< The sub-frame's share from the last frame's
                      ///< transforms.
    float strip[4];   ///< A pushbroom's intrinsics
                      ///< (`PushbroomStrip::Intrinsics`).
    float
        minimal[4];  ///< Minimal shading's constant colour and mode.
    /// Minimal shading's shadow of its main light in this view: its
    /// world's map's place in the shared one's clip space (x and y
    /// scale and offset; depth scale and offset), the world's slot,
    /// and 1 when it shadows the view.
    float minimalShadow[2][4];
};
static_assert(sizeof(Uniforms) == 256,
              "Uniforms must match frame.slang");
static_assert(offsetof(Uniforms, forward) == 64 &&
                  offsetof(Uniforms, eye) == 80 &&
                  offsetof(Uniforms, flags) == 96,
              "minimal_resolve.slang reads these offsets");

struct ShadowUniforms {
    float lightViewProjection[4][16];
    float splits[4];
    float texelWorld[4];
    float depthWorld[4];
    float params[4];
    /// Minimal shading's main light's maps: the first's layer, the
    /// share of a layer they take, how many, and 1 with maps.
    float minimal[4];
    std::uint32_t debug[4];
    float poisson[32];
    /// A virtual shadow map's level being drawn: its block's address
    /// (low, high), the level, 1.
    std::uint32_t virtualMap[4];
};
static_assert(sizeof(ShadowUniforms) == 496,
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
    /// The lights the view sees (a bit each), its world's slot among
    /// the per-world maps, where its lights start in `lightTable_`,
    /// and the float bits of the camera's aperture diameter, m.
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
    /// an atmosphere, unused, and the sky's greatest degree of
    /// polarization.
    float atmosphere[2][4];
    /// The scene's ambient light's colour, linear RGB over its
    /// luminance (its luminance is `sun[3]`'s); w unused.
    float ambient[4];
    /// Celestial bodies (km): each centre relative to the frame's
    /// origin and radius, 0 for none; and its normal albedo, linear
    /// RGB. They fill the views' stride, so shading reads them as one
    /// array.
    float bodies[4][4];
    float bodyAlbedo[4][4];
};
static_assert(sizeof(ShadingUniforms) == 1280,
              "ShadingUniforms must match shading.slang");

/// A view's own shading uniforms (shading.slang's ShadingView), after
/// the ShadingUniforms every view shares, whose `invViewProj`, `eye`,
/// `view` and `strip` go unused: what differs between the views of a
/// batch.
struct ShadingView {
    float invViewProj[16];
    float eye[4];
    std::uint32_t view[4];
    float strip[4];
    float padding[4];
};
static_assert(sizeof(ShadingView) == 128,
              "ShadingView must match shading.slang");

/// shading.slang's kSky* bits: what its sky pass draws behind the
/// scene, and whether a dome lights it (for shading only; `sky_`
/// holds the others).
constexpr std::uint32_t kSkyDome = 1;
constexpr std::uint32_t kSkyMap = 2;
constexpr std::uint32_t kSkyStars = 4;
constexpr std::uint32_t kSkyLighting = 8;
constexpr std::uint32_t kSkyAtmosphere = 16;
constexpr std::uint32_t kSkyBodies = 32;

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
    /// wavelength (nm), z: the frame's time (s), w: its sequence's
    /// frames (`lightTiming_` says which show); `patternExtent` xy
    /// its tangent-plane half extents (0 unbounded), z how volumes
    /// light by its beams (1 a spot's, 2 a grid's, integrated across
    /// rays; 0 sampled); `patternLayout`
    /// its data and image in `patternData_` (shading.slang's
    /// LightPatternParams); its parameters.
    float pattern[4];
    float patternExtent[4];
    std::uint32_t patternLayout[4];
    float patternParams[16];
    /// x: texels along each side of its shadow maps, in that corner
    /// of their layers; y: 1 for a batch's distant light, whose maps
    /// are the shared one then one per world; z: 1 for a laser
    /// showing speckle; w: its linewidth, nm.
    float map[4];
    /// Light linking (`LinkGroups`): the groups it lights, then those
    /// casting its shadows; 1 when it casts shadows at all; 1 for a
    /// pulsed light other than a laser, scaled by its timing.
    std::uint32_t links[4];
    /// xyz: its electric field's direction (`polarizationAxis`), w:
    /// the fraction of its light polarized along it.
    float polarization[4];
};
static_assert(sizeof(LightGpu) == 800,
              "LightGpu must match shading.slang");

/// `v` turned by `angle` radians about unit `axis` (Rodrigues'
/// rotation formula).
glm::dvec3 Rotated(const glm::dvec3& v, const glm::dvec3& axis,
                   double angle) {
    const double c = std::cos(angle);
    return v * c + glm::cross(axis, v) * std::sin(angle) +
           axis * glm::dot(axis, v) * (1.0 - c);
}

/// A distant light's direction `s` of the way along the great circle
/// from its last frame's (`LightParams::previous`) to this frame's.
glm::dvec3 TurnedDirection(const LightParams& light, double s) {
    const glm::dvec3 to = glm::normalize(glm::dvec3(light.direction));
    if (!light.previous) {
        return to;
    }
    const glm::dvec3 from =
        glm::normalize(glm::dvec3(light.previous->direction));
    const glm::dvec3 across = glm::cross(from, to);
    const double sine = glm::length(across);
    if (!(sine > 1e-12)) {
        return to;
    }
    return Rotated(from, across / sine,
                   s * std::atan2(sine, glm::dot(from, to)));
}

/// With `subframes` above 1, a copy of `table` (`params`' lights) per
/// sub-frame, `RasterRenderer::kMaxLights` apart: each light that
/// moved since the last frame (`LightParams::previous`) posed where
/// it is at the sub-frame's time, as the views are
/// (`FrameParams::Subframes`); lights posed on the GPU copied as
/// they are, for the light pass to pose each sub-frame's
/// (`RecordShadowMaps`). Empty when none moved and none is posed on
/// the GPU, or past `RasterRenderer::kMaxLightSubframes`. Distant
/// lights turn
/// along the great circle between their directions
/// (`TurnedDirection`); moving lights' shadow views follow them
/// (`RasterRenderer::subframeLightViews_`).
std::vector<LightGpu>
SubframeLights(const FrameParams& params,
               const std::vector<LightGpu>& table,
               std::uint32_t subframes) {
    const std::vector<LightParams>& lights = params.lights;
    const auto moves = [](const LightParams& light) {
        return light.previous.has_value();
    };
    const auto onGpu = [](const LightParams& light) {
        return light.posedOnGpu;
    };
    if (subframes < 2 ||
        subframes > RasterRenderer::kMaxLightSubframes ||
        (std::none_of(lights.begin(), lights.end(), moves) &&
         std::none_of(lights.begin(), lights.end(), onGpu))) {
        return {};
    }
    std::vector<LightGpu> copies(subframes *
                                 RasterRenderer::kMaxLights);
    for (std::uint32_t k = 0; k < subframes; ++k) {
        // The last frame is at time -1, this one at 0.
        const double s =
            1.0 + params.camera.SubframeTime(static_cast<int>(k));
        for (std::size_t i = 0; i < table.size(); ++i) {
            LightGpu entry = table[i];
            const LightParams& light = lights[i];
            if (moves(light) && light.type == LightType::Distant) {
                // Turned along the great circle from its last
                // direction to this one, short of it by the
                // sub-frame's share: its direction, axis and field.
                const glm::dvec3 from = glm::normalize(
                    glm::dvec3(light.previous->direction));
                const glm::dvec3 to =
                    glm::normalize(glm::dvec3(light.direction));
                const glm::dvec3 across = glm::cross(from, to);
                const double sine = glm::length(across);
                if (sine > 1e-12) {
                    const double back =
                        -(1.0 - s) *
                        std::atan2(sine, glm::dot(from, to));
                    for (float* vector : {entry.position, entry.axis,
                                          entry.polarization}) {
                        const glm::dvec3 turned =
                            Rotated(glm::dvec3(vector[0], vector[1],
                                               vector[2]),
                                    across / sine, back);
                        vector[0] = static_cast<float>(turned.x);
                        vector[1] = static_cast<float>(turned.y);
                        vector[2] = static_cast<float>(turned.z);
                    }
                }
            } else if (moves(light)) {
                const glm::dmat4 now =
                    LightParams::Pose{light.position, light.direction,
                                      light.tangent}
                        .Matrix();
                const glm::dmat4 then =
                    InterpolatePose(light.previous->Matrix(), now, s);
                const glm::mat3 turn(then * glm::inverse(now));
                const glm::vec3 where =
                    params.Relative(glm::dvec3(then[3]));
                for (float* vector :
                     {entry.axis, entry.extentX, entry.extentY,
                      entry.polarization}) {
                    const glm::vec3 turned =
                        turn *
                        glm::vec3(vector[0], vector[1], vector[2]);
                    vector[0] = turned.x;
                    vector[1] = turned.y;
                    vector[2] = turned.z;
                }
                entry.position[0] = where.x;
                entry.position[1] = where.y;
                entry.position[2] = where.z;
            }
            copies[k * RasterRenderer::kMaxLights + i] = entry;
        }
    }
    return copies;
}

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

constexpr DescriptorBinding kPatternMipBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_SHADER_STAGE_COMPUTE_BIT},
};

/// One mip level of a pattern image array (`patternMipMain`): the
/// level above's first texel and size, and this level's.
struct PatternMipStep {
    std::uint32_t source[4];  ///< Offset, width, height.
    std::uint32_t target[4];  ///< Offset, width, height.
};

/// Builds mip level `step.target` of a pattern image array from the
/// level above it in place, each texel the mean of the 2 x 2 above
/// it, clamped at odd edges as `MipChain` does.
constexpr const char* kPatternMipSlang = R"(
struct PatternMipStep {
    uint4 source;
    uint4 target;
}
[[vk::binding(0, 0)]] RWStructuredBuffer<float> texels;
[[vk::push_constant]] ConstantBuffer<PatternMipStep> step;

[shader("compute")]
[numthreads(16, 16, 1)]
void patternMipMain(uint3 id : SV_DispatchThreadID) {
    if (id.x >= step.target.y || id.y >= step.target.z) {
        return;
    }
    let width = step.source.y;
    let last = uint2(width, step.source.z) - 1;
    let x0 = min(2 * id.x, last.x);
    let x1 = min(2 * id.x + 1, last.x);
    let y0 = min(2 * id.y, last.y);
    let y1 = min(2 * id.y + 1, last.y);
    let from = step.source.x;
    texels[step.target.x + id.y * step.target.y + id.x] =
        0.25 * (texels[from + y0 * width + x0] + texels[from + y0 * width + x1] +
                texels[from + y1 * width + x0] + texels[from + y1 * width + x1]);
}
)";

/// Texels of a `width` x `height` image and every mip level below it,
/// each `max(1, size >> n)` on a side down to 1 x 1.
std::size_t MipTexels(std::uint32_t width, std::uint32_t height) {
    std::size_t total = 0;
    while (true) {
        total += static_cast<std::size_t>(width) * height;
        if (width == 1 && height == 1) {
            return total;
        }
        width = std::max(width / 2, 1u);
        height = std::max(height / 2, 1u);
    }
}

/// Integrates each job's pattern over its extent on a grid of 256 x
/// 256 cells, averaged over its sequence's frames (`info.z`), a
/// workgroup per job, into `norms` at its slot.
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
    input.frames = max(job.info.z, 1);
    input.t.y = -extent.y + (float(thread.x) + 0.5) * cell.y;
    // Averaged over the sequence's frames.
    var sum = 0.0;
    for (uint frame = 0; frame < input.frames; ++frame) {
        input.frame = frame;
        for (uint i = 0; i < kCells; ++i) {
            input.t.x = -extent.x + (float(i) + 0.5) * cell.x;
            sum += max(lightPattern(job.info.x, input, p), 0.0);
        }
    }
    partial[thread.x] = sum * cell.x * cell.y / float(input.frames);
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
    /// x: its rms surface height, m, for laser speckle; yzw unused.
    float surface[4];
};
static_assert(sizeof(MaterialGpu) == 608,
              "MaterialGpu must match shading.slang");

/// Reflectance luminance below which a material counts as black; its
/// spectrum's shape is flat.
constexpr float kBlackLuminance = 1e-6f;

/// Distant lights whose stray light the sensor adds, such as the sun
/// (sensor.slang's `kStraySources`).
constexpr std::size_t kStraySources = 4;

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
    /// A pushbroom's lens: Brown-Conrady p1 and p2 (its k1 to k3 and
    /// line in `distortion`).
    float pushbroom[4];
    /// Distant lights for stray light: the unit direction towards
    /// each in the view's camera frame; w 1 for a light, 0 past the
    /// last.
    float sources[kStraySources][4];
    /// The camera's polarizer (`Polarizer`): its angle (radians), 1
    /// for a mosaic (its angle per pixel), and its principal
    /// transmittances k1 and k2; k1 = k2 = 1 for none.
    float polarizer[4];
    float facesPadding[20];
    /// Per band, two to a float4: the lens's image scale (lateral
    /// colour) and focal length shift in millimetres (axial colour);
    /// 1 and 0 without chromatic aberration.
    float chromatic[SpectralBands::kMaxBands / 2][4];
    /// Per light of `sources` and band: the radiance whose image puts
    /// its irradiance at normal incidence on the focal plane,
    /// E 4 N^2 / (pi T), for stray light.
    float sourceRadiance[kStraySources][SpectralBands::kMaxBands];
};
static_assert(sizeof(SensorUniforms) == 4352 &&
                  offsetof(SensorUniforms, faceInvProj) == 2560,
              "SensorUniforms must match sensor.slang");

/// A view's own sensor uniforms, after the shared SensorUniforms
/// (whose per-view fields go unused) in the buffer sensor.slang reads
/// (`sensorData`): what differs between the views of a batch.
struct SensorView {
    float invViewProj[16];
    float pose[16];  ///< Camera to world, for stray light's sources.
    float eye[4];
    float forward[4];
    float distortion[4];    ///< A pushbroom's line differs per view.
    std::uint32_t seed[4];  ///< The frame seed; the rest unused.
};
static_assert(sizeof(SensorView) == 192,
              "SensorView must match sensor.slang");

/// Per-view uniforms sit in one buffer at multiples of this, the
/// largest `minUniformBufferOffsetAlignment` Vulkan allows.
constexpr VkDeviceSize kUniformAlignment = 256;

constexpr VkDeviceSize Aligned(VkDeviceSize bytes) {
    return (bytes + kUniformAlignment - 1) / kUniformAlignment *
           kUniformAlignment;
}

constexpr VkDeviceSize kUniformsStride = Aligned(sizeof(Uniforms));
/// The shading uniforms every view shares, then each view's
/// ShadingView.
constexpr VkDeviceSize kShadingShared = sizeof(ShadingUniforms);
constexpr VkDeviceSize kShadingStride = sizeof(ShadingView);
/// The sensor's shared uniforms, then every view's SensorView.
constexpr VkDeviceSize kSensorShared = sizeof(SensorUniforms);
constexpr VkDeviceSize kSensorStride = sizeof(SensorView);
/// Light shadow views' Shadows blocks (`lightShadowBlocks_`), bound
/// one at a time.
constexpr VkDeviceSize kLightShadowBlockStride =
    Aligned(sizeof(ShadowUniforms));
static_assert(kShadingShared % 16 == 0 && kShadingStride % 16 == 0,
              "Shading reads the views' blocks as arrays after the "
              "shared uniforms");

/// Where the volume pass reads a view's block of the shading
/// uniforms.
VolumePass::ViewLayout VolumeLayout() {
    using Offset = std::uint32_t;
    // volume.slang mirrors LightGpu.
    static_assert(offsetof(LightGpu, axis) == 16 &&
                      offsetof(LightGpu, cone) == 32 &&
                      offsetof(LightGpu, shadow) == 48 &&
                      offsetof(LightGpu, extentX) == 64 &&
                      offsetof(LightGpu, extentY) == 80 &&
                      offsetof(LightGpu, spectrum) == 96 &&
                      offsetof(LightGpu, tinted) == 352 &&
                      offsetof(LightGpu, beam) == 608 &&
                      offsetof(LightGpu, beamExtra) == 624 &&
                      offsetof(LightGpu, pattern) == 640 &&
                      offsetof(LightGpu, patternExtent) == 656 &&
                      offsetof(LightGpu, patternLayout) == 672 &&
                      offsetof(LightGpu, patternParams) == 688 &&
                      offsetof(LightGpu, map) == 752 &&
                      offsetof(LightGpu, links) == 768 &&
                      offsetof(LightGpu, polarization) == 784 &&
                      sizeof(LightGpu) == 800,
                  "volume.slang's LightGpu must match");
    // It reads the views' uniforms 16 bytes at a time.
    static_assert(kShadingStride % 16 == 0 &&
                      offsetof(ShadingView, invViewProj) % 16 == 0 &&
                      offsetof(ShadingView, strip) % 16 == 0 &&
                      offsetof(ShadingUniforms, range) % 16 == 0 &&
                      offsetof(ShadingUniforms, size) % 16 == 0 &&
                      offsetof(ShadingView, view) % 16 == 0 &&
                      offsetof(ShadingUniforms, lightShadow) % 16 ==
                          0 &&
                      offsetof(ShadingUniforms, atmosphere) % 16 == 0,
                  "volume.slang reads the views' uniforms as uint4s");
    // A view's own fields in its block, after the shared uniforms;
    // the shared ones from the start.
    return {
        static_cast<Offset>(kShadingStride),
        static_cast<Offset>(kShadingShared +
                            offsetof(ShadingView, invViewProj)),
        static_cast<Offset>(kShadingShared +
                            offsetof(ShadingView, strip)),
        static_cast<Offset>(offsetof(ShadingUniforms, range)),
        static_cast<Offset>(offsetof(ShadingUniforms, size)),
        static_cast<Offset>(kShadingShared +
                            offsetof(ShadingView, view)),
        static_cast<Offset>(offsetof(ShadingUniforms, lightShadow)),
        static_cast<Offset>(offsetof(ShadingUniforms, atmosphere))};
}

/// Samples per axis of the RGB-to-spectrum coefficient table.
constexpr std::uint32_t kTableResolution = 17;

constexpr VkShaderStageFlags kCompute = VK_SHADER_STAGE_COMPUTE_BIT;

// Shading pushes its per-dispatch images and a block of its
// resources (`ShadingResources`); the rest it reads by address and
// from the bindless heap.
constexpr DescriptorBinding kShadingBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kCompute},
    {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {4, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    {8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {17, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {18, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    // light_pattern's pattern data.
    {29, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
};

/// shading.slang's ShadingResources: its buffers' addresses, then its
/// textures' and samplers' indices in the bindless heap.
struct ShadingResources {
    VkDeviceAddress views;
    VkDeviceAddress materials;
    VkDeviceAddress lights;
    VkDeviceAddress lightViews;
    VkDeviceAddress lightProfiles;
    VkDeviceAddress patternNorms;
    VkDeviceAddress lightTiming;
    VkDeviceAddress stars;
    VkDeviceAddress starCells;
    VkDeviceAddress atmosphereMedium;
    VkDeviceAddress atmosphereDensities;
    VkDeviceAddress atmospherePhases;
    std::uint32_t sigmoidTable;
    std::uint32_t tableSampler;
    std::uint32_t lightShadowMaps;
    std::uint32_t lightShadowSampler;
    std::uint32_t domeMap;
    std::uint32_t domeLighting;
    std::uint32_t horizonMap;
    std::uint32_t terrainHeights;
    std::uint32_t atmosphereTransmittance;
    std::uint32_t atmosphereMultiple;
    std::uint32_t atmosphereSkyIrradiance;
    std::uint32_t atmosphereRefraction;
    std::uint32_t volumeShadows;
    std::uint32_t volumeShadowSize;
    /// Each view's own uniforms (`ShadingView`), after the shared
    /// ones at `views`.
    VkDeviceAddress viewBlocks;
    /// The virtual shadow map (virtual_shadow.slang's
    /// `VirtualShadow`) and the light slot it shadows, ~0 for none.
    VkDeviceAddress virtualShadow;
    std::uint32_t virtualLight;
    std::uint32_t virtualDebug;
};
static_assert(sizeof(ShadingResources) == 176 &&
                  offsetof(ShadingResources, sigmoidTable) == 96 &&
                  offsetof(ShadingResources, volumeShadows) == 144,
              "ShadingResources must match shading.slang and "
              "volume.slang");

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

/// sensor.slang's SensorConstants, which the sensor kernels and the
/// lens passes (the cube camera's overlays and face rows) push.
struct SensorConstants {
    std::uint32_t view[4];
    std::uint32_t table[4];
    std::uint32_t stokes[4];
    VkDeviceAddress params;
    std::uint32_t normalSlot;
    std::uint32_t depthSlot;
    /// The frame's linear vars' and AOV slots' addresses
    /// (sensor.slang's `SensorLinear`), those held so a bit each in
    /// `stokes[1]` and `stokes[2]`.
    VkDeviceAddress linear;
};
/// `RenderTargets::LinearVars::written`'s var bits, below its AOV
/// slots'.
constexpr std::uint32_t kLinearVarBits =
    (1u << RenderTargets::LinearVars::kVars) - 1u;
static_assert(sizeof(SensorConstants) == 72,
              "SensorConstants must match sensor.slang");

constexpr DescriptorBinding kFilterBindings[] = {
    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kCompute},
    {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    // DegreeOfLinearPolarization; the G-buffer's normal and depth
    // come from the bindless heap.
    {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
    {3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kCompute},
    // AngleOfLinearPolarization.
    {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
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
    // LinearStokes; the filters' parameters are read through their
    // address.
    {15, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kCompute},
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

/// What `shadeMain` is compiled with, as bits of a key: the air's
/// light (an atmosphere or celestial bodies), polarization and
/// lasers, and in bits 8 up the most lights it shades. A frame shades
/// with the leanest it needs (`ShadingVariantOf`): the full kernel
/// holds every band's air and every light's state at once, 255
/// registers a thread and spilling (`ERNIE_PIPELINE_STATISTICS`),
/// where a scene without air, polarization or lasers needs about 170
/// and shades in two thirds the time.
constexpr std::uint32_t kShadeAir = 1;
constexpr std::uint32_t kShadePolarization = 2;
constexpr std::uint32_t kShadeLasers = 4;
constexpr std::uint32_t kShadeLightsShift = 8;
constexpr std::uint32_t kFullShading =
    kShadeAir | kShadePolarization | kShadeLasers |
    (16u << kShadeLightsShift);

/// `variant`'s constants for shading.slang, ahead of its source.
std::string ShadingVariantSource(std::uint32_t variant) {
    const auto flag = [variant](std::uint32_t bit) {
        return (variant & bit) != 0 ? "true" : "false";
    };
    return std::string("static const bool kShadeAir = ") +
           flag(kShadeAir) +
           ";\nstatic const bool kShadePolarization = " +
           flag(kShadePolarization) +
           ";\nstatic const bool kShadeLasers = " +
           flag(kShadeLasers) +
           ";\nstatic const uint kShadeLights = " +
           std::to_string(variant >> kShadeLightsShift) + ";\n";
}

/// The leanest shading variant `params` needs, with polarization when
/// its frame measures Stokes vectors (`stokes`): lights in powers of
/// two from 2 up to shading.slang's 16.
std::uint32_t ShadingVariantOf(const FrameParams& params,
                               bool stokes) {
    std::uint32_t variant = 0;
    if (params.atmosphere || !params.bodies.empty()) {
        variant |= kShadeAir;
    }
    if (stokes) {
        variant |= kShadePolarization;
    }
    std::uint32_t lights = 2;
    for (const LightParams& light : params.lights) {
        if (light.type == LightType::Laser) {
            variant |= kShadeLasers;
        }
    }
    while (lights < std::min<std::size_t>(params.lights.size(), 16)) {
        lights *= 2;
    }
    return variant | (lights << kShadeLightsShift);
}

/// The modules of the shading and sky passes with `brdf` as the
/// active model and `patterns`' light patterns, for a frame camera's
/// views or a pushbroom's, shading.slang compiled as shading variant
/// `variant`.
std::vector<ShaderModuleSource>
ShadingModules(const BrdfModel& brdf,
               const LightPatternRegistry& patterns, bool pushbroom,
               std::uint32_t variant = kFullShading) {
    std::vector<ShaderModuleSource> modules =
        LightPatternModules(patterns);
    modules.insert(modules.end(),
                   {{"brdf", std::string(BrdfSlang())},
                    {BrdfModuleName(brdf), brdf.slang},
                    ProjectionModule(pushbroom),
                    {"bindless", shaders::kBindlessSlang},
                    {"atmosphere", shaders::kAtmosphereSlang},
                    {"virtual_shadow", shaders::kVirtualShadowSlang},
                    {"shading", "import " + BrdfModuleName(brdf) +
                                    ";\ntypealias ActiveBrdf = " +
                                    brdf.typeName + ";\n" +
                                    ShadingVariantSource(variant) +
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
        {"bindless", shaders::kBindlessSlang},
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
    // The poses at the last frame, for motion blur.
    {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
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
    {11, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
     VK_SHADER_STAGE_COMPUTE_BIT},
};

/// minimal_resolve.slang's push constants.
struct ResolveConstants {
    std::uint32_t size[4];    ///< Width, height, views, first view.
    std::uint32_t layout[4];  ///< Held vars and the blocks' layout.
    float toDome[3][4];       ///< Rows of the world-to-dome rotation.
    /// The dome's luminance in minimal units (0 unseen), 1 with a
    /// map, and the map's width and height.
    float sky[4];
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

/// meter.slang's push constants, for metering and for automatic
/// exposure (`adaptMain`, `applyMain`), which reach their buffers by
/// address.
struct MeterConstants {
    std::uint32_t size[4];
    float range[4];
    VkDeviceAddress histogram = 0;
    VkDeviceAddress states = 0;
    VkDeviceAddress lens = 0;
    VkDeviceAddress eyes = 0;
    VkDeviceAddress gain = 0;
    std::uint32_t padding[2]{};
    std::uint32_t layout[4];
};
static_assert(sizeof(MeterConstants) == 96,
              "MeterConstants must match meter.slang");

/// Automatic exposures kept at once, one per metering key.
constexpr std::uint32_t kMaxExposureStates = 64;

/// How far a view may move from the one whose depths a cascade fit
/// reuses (`ShadowFit::Visible`): its eye by this fraction of the
/// nearest depth seen, its direction by this cosine.
constexpr double kVisibleFitEyeShift = 0.1;
constexpr float kVisibleFitTurn = 0.995f;

/// depth_range.slang's `CascadeJob`: a distant light's cascades the
/// GPU fits (`RasterRenderer::AddCascadeJob`), matrices by column.
struct CascadeJob {
    float lightView[16];
    float cameraToWorld[16];
    float toView[16];
    float camera[4];
    float fit[4];
    float scene[4];
    float seen[4];
    float volumes[4];
    std::uint32_t out[4];
};
static_assert(sizeof(CascadeJob) == 288,
              "CascadeJob must match depth_range.slang");

/// depth_range.slang's push constants: the depth range's, then the
/// cascade fit's buffers by address.
struct DepthRangeConstants {
    std::uint32_t size[4];
    float clip[4];
    VkDeviceAddress jobs = 0;
    VkDeviceAddress seen = 0;
    VkDeviceAddress matrices = 0;
    VkDeviceAddress blocks = 0;
    VkDeviceAddress frame = 0;
    VkDeviceAddress subframes = 0;
    std::uint32_t layout[4];
};
static_assert(sizeof(DepthRangeConstants) == 96,
              "DepthRangeConstants must match depth_range.slang");

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

constexpr std::array<PresentMode, 14> kPresentModes{{
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
    {render_vars::kDegreeOfLinearPolarization, 9, false},
    {render_vars::kAngleOfLinearPolarization, 10, false},
    {render_vars::kLinearStokes, 11, false},
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

/// Copies `from`, which compute just wrote on the device, into `to`
/// for the host to read once the frame is done. Results reduced with
/// atomics belong in device memory: on host memory each atomic
/// crosses the bus.
void CopyForHost(VkCommandBuffer commands, const GpuBuffer& from,
                 const GpuBuffer& to) {
    ShaderBarrier(commands);
    const VkBufferCopy region{0, 0, std::min(from.Size(), to.Size())};
    vkCmdCopyBuffer(commands, from.Handle(), to.Handle(), 1, &region);
    ShaderBarrier(commands);
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

/// The ambient light's colour as linear RGB of luminance 1, or none
/// (black) for a colour without luminance.
glm::vec3 UnitLuminanceColor(const glm::vec3& rgb) {
    const float luminance =
        0.2126f * rgb.r + 0.7152f * rgb.g + 0.0722f * rgb.b;
    return luminance > 0.0f ? rgb / luminance : glm::vec3(0.0f);
}

/// The luminance (cd/m^2) of a white Lambertian surface facing
/// distant light `light`: its illuminance over pi.
double WhiteLuminance(const LightParams& light,
                      const SpectralBands& bands) {
    double white = 0.0;
    for (int b = 0; b < bands.count &&
                    b < static_cast<int>(light.spectrum.size());
         ++b) {
        white += light.spectrum[static_cast<std::size_t>(b)] *
                 CieXyz(bands.Centre(b)).y * bands.Width();
    }
    return white * kLuminousEfficacy / M_PI;
}

/// The frame's brightest distant light (`WhiteLuminance`), not a
/// dome's direction, of the shared ones or world `world`'s (all for
/// -1); none without one. Unlit surfaces and minimal shading show
/// their colours as a white surface facing it.
const LightParams* BrightestDistant(const FrameParams& params,
                                    const SpectralBands& bands,
                                    std::int32_t world = -1) {
    const LightParams* brightest = nullptr;
    double most = 0.0;
    for (const LightParams& light : params.lights) {
        if (light.type != LightType::Distant || light.domeDiffuse ||
            !(light.world == -1 || world == -1 ||
              light.world == world)) {
            continue;
        }
        const double white = WhiteLuminance(light, bands);
        if (brightest == nullptr || white > most) {
            brightest = &light;
            most = white;
        }
    }
    return brightest;
}

/// The scene's ambient light for minimal shading of view world
/// `world`, which is lit in units of its brightest distant light, per
/// RGB channel: its colour of unit luminance times its luminance over
/// a white surface's facing that light, or without one, of 1 when
/// there is any.
glm::vec3 MinimalAmbient(const FrameParams& params,
                         const SpectralBands& bands,
                         std::int32_t world) {
    if (!(params.ambient > 0.0f)) {
        return glm::vec3(0.0f);
    }
    const glm::vec3 colour = UnitLuminanceColor(params.ambientColor);
    const LightParams* brightest =
        BrightestDistant(params, bands, world);
    const double white = brightest != nullptr
                             ? WhiteLuminance(*brightest, bands)
                             : 0.0;
    return colour * (white > 0.0
                         ? static_cast<float>(params.ambient / white)
                         : 1.0f);
}

/// The dome's luminance as minimal shading's background shows it, in
/// `MinimalAmbient`'s units (a white surface facing the brightest
/// distant light), or 1 per cd/m^2 without one; 0 when camera rays do
/// not see it.
float MinimalSky(const FrameParams& params,
                 const SpectralBands& bands) {
    if (!params.dome || !params.dome->visible) {
        return 0.0f;
    }
    double luminance = 0.0;
    for (int b = 0;
         b < bands.count &&
         b < static_cast<int>(params.dome->radiance.size());
         ++b) {
        luminance +=
            params.dome->radiance[static_cast<std::size_t>(b)] *
            CieXyz(bands.Centre(b)).y * bands.Width();
    }
    luminance *= kLuminousEfficacy;
    const LightParams* brightest = BrightestDistant(params, bands);
    const double white = brightest != nullptr
                             ? WhiteLuminance(*brightest, bands)
                             : 0.0;
    return static_cast<float>(white > 0.0 ? luminance / white
                                          : luminance);
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
                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
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
      filterLayout_(context, kFilterBindings,
                    sizeof(SensorConstants)),
      presentLayout_(context, kPresentBindings, 48),
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
      shadingUniforms_(context, kShadingShared + kShadingStride,
                       kViewUniformUsage, BufferMemory::Device),
      sensorUniforms_(context, kSensorShared + kSensorStride,
                      kViewUniformUsage, BufferMemory::Device),
      lightTable_(
          context, kMaxLightSubframes * kMaxLights * sizeof(LightGpu),
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device),
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
      patternData_(std::make_shared<GpuBuffer>(
          context, sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          BufferMemory::Device)),
      lightTiming_(std::make_unique<GpuBuffer>(
          context, sizeof(glm::vec2),
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device)),
      patternMipLayout_(context, kPatternMipBindings,
                        sizeof(PatternMipStep)),
      lightViewMatrices_(std::make_unique<GpuBuffer>(
          context, kMaxLightShadowViews * sizeof(glm::mat4),
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device)),
      lightShadowBlocks_(
          context,
          (kMaxLightShadowViews + VirtualShadowMap::kMaxLevels) *
              kLightShadowBlockStride,
          kViewUniformUsage, BufferMemory::Device),
      lightsLayout_(context, kLightsBindings, 64),
      depthRangeLayout_(context, kDepthRangeBindings,
                        sizeof(DepthRangeConstants)),
      meterLayout_(context, kMeterBindings, sizeof(MeterConstants)),
      resolveLayout_(context, kResolveBindings,
                     sizeof(ResolveConstants)),
      unheldVar_(context, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 BufferMemory::Device),
      exposureLayout_(context, {}, sizeof(MeterConstants)),
      exposureStates_(context, kMaxExposureStates * 4 * sizeof(float),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      BufferMemory::Device),
      exposureGain_(context, sizeof(float),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
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
    // A texel for the shadow passes' shadow map binding, in place of
    // the light shadow maps they draw into.
    shadowMaps_ = std::make_unique<GpuImage>(
        context_,
        ImageDesc{1, 1, 1, 1, VK_FORMAT_D32_SFLOAT,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT});
    context_.Submit([this](VkCommandBuffer commands) {
        shadowMaps_->Transition(
            commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
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
    volumes_ = std::make_unique<VolumePass>(context_, compiler_);
    tableSamplerSlot_ =
        context_.Bindless().AddSampler(tableSampler_.Handle());
    shadowSamplerSlot_ =
        context_.Bindless().AddSampler(shadowSampler_.Handle(), true);
    shadingResources_ = std::make_unique<GpuBuffer>(
        context_, sizeof(ShadingResources),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, BufferMemory::Device);

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
    presentOverlay_ = presentPipeline("presentOverlay", true);
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
    depthRangeSums_ = std::make_unique<GpuBuffer>(
        context_, 2 * sizeof(std::uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
    cascadeFitPipeline_ = std::make_unique<ComputePipeline>(
        context_, depthRangeLayout_,
        compiler_.CompileSpirv(depthRangeModules, "cascadeFitMain"));
    const std::vector<ShaderModuleSource> meterModules{
        {"meter", shaders::kMeterSlang}};
    meterPipeline_ = std::make_unique<ComputePipeline>(
        context_, meterLayout_,
        compiler_.CompileSpirv(meterModules, "meterMain"));
    exposureAdaptPipeline_ = std::make_unique<ComputePipeline>(
        context_, exposureLayout_,
        compiler_.CompileSpirv(meterModules, "adaptMain"));
    exposureApplyPipeline_ = std::make_unique<ComputePipeline>(
        context_, exposureLayout_,
        compiler_.CompileSpirv(meterModules, "applyMain"));
    {
        // No key has metered yet (`z` 0).
        const std::vector<float> zeros(kMaxExposureStates * 4, 0.0f);
        exposureStates_.Write(std::as_bytes(std::span(zeros)));
    }
    for (std::unique_ptr<GpuBuffer>& bins : meterBins_) {
        bins = std::make_unique<GpuBuffer>(
            context_, kMeterBins * sizeof(std::uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            BufferMemory::Readback);
    }
    meterSums_ = std::make_unique<GpuBuffer>(
        context_, kMeterBins * sizeof(std::uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
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

RasterRenderer::PatternArrayRegion
RasterRenderer::EnsurePatternArray(const std::string& key,
                                   std::uint32_t width,
                                   std::uint32_t height) {
    const auto found = patternArrayRegions_.find(key);
    if (found != patternArrayRegions_.end() &&
        found->second.width == width &&
        found->second.height == height) {
        return found->second;
    }
    // Regions never move, so the frame's offsets already written stay
    // where they are. One at another size frees its old space, which
    // new regions take (the best fit) before the end grows: writes
    // into a region wait for the frames already submitted, which read
    // what it held before.
    if (found != patternArrayRegions_.end()) {
        patternArrayFree_.emplace_back(
            found->second.offset,
            MipTexels(found->second.width, found->second.height));
        std::sort(patternArrayFree_.begin(), patternArrayFree_.end());
        // Neighbours merged.
        std::vector<std::pair<std::size_t, std::size_t>> merged;
        for (const auto& [offset, floats] : patternArrayFree_) {
            if (!merged.empty() &&
                merged.back().first + merged.back().second ==
                    offset) {
                merged.back().second += floats;
            } else {
                merged.emplace_back(offset, floats);
            }
        }
        patternArrayFree_ = std::move(merged);
    }
    const std::size_t floats = MipTexels(width, height);
    auto best = patternArrayFree_.end();
    for (auto it = patternArrayFree_.begin();
         it != patternArrayFree_.end(); ++it) {
        if (it->second >= floats &&
            (best == patternArrayFree_.end() ||
             it->second < best->second)) {
            best = it;
        }
    }
    PatternArrayRegion region{patternArrayFloats_, width, height};
    if (best != patternArrayFree_.end()) {
        region.offset = best->first;
        best->first += floats;
        best->second -= floats;
        if (best->second == 0) {
            patternArrayFree_.erase(best);
        }
    } else {
        patternArrayFloats_ += floats;
    }
    patternArrayRegions_[key] = region;
    EnsurePatternData(patternArrayFloats_ + patternHostFloats_);
    return region;
}

void RasterRenderer::EnsurePatternData(std::size_t floats) {
    const VkDeviceSize bytes =
        std::max<std::size_t>(floats, 1) * sizeof(float);
    const BufferMemory memory = patternArrayFloats_ > 0
                                    ? BufferMemory::Shared
                                    : BufferMemory::Device;
    if (patternData_->Size() >= bytes &&
        patternData_->Memory() == memory) {
        return;
    }
    // A view of the old buffer (`MapArray`) keeps it alive, and its
    // values are copied from it; the host's part uploads again.
    vkDeviceWaitIdle(context_.Device());
    patternData_ = std::make_shared<GpuBuffer>(
        context_, std::max(bytes, 2 * patternData_->Size()),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memory);
    patternDataHeld_.clear();
}

void RasterRenderer::RecordPatternMips(
    VkCommandBuffer commands) const {
    if (patternMipRegions_.empty()) {
        return;
    }
    ERNIE_GPU_ZONE(context_, commands, "Light pattern arrays");
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      patternMipPipeline_->Handle());
    const DescriptorResource resources[] = {
        DescriptorResource::Of(*patternData_)};
    patternMipLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                           resources);
    // Each level from the one above, regions side by side.
    bool more = true;
    for (std::uint32_t level = 1; more; ++level) {
        more = false;
        for (const PatternArrayRegion& region : patternMipRegions_) {
            std::uint32_t width = region.width;
            std::uint32_t height = region.height;
            std::size_t offset = region.offset;
            for (std::uint32_t i = 1; i < level; ++i) {
                offset += static_cast<std::size_t>(width) * height;
                width = std::max(width / 2, 1u);
                height = std::max(height / 2, 1u);
            }
            if (width == 1 && height == 1) {
                continue;
            }
            more = true;
            const PatternMipStep step{
                {static_cast<std::uint32_t>(offset), width, height,
                 0},
                {static_cast<std::uint32_t>(
                     offset +
                     static_cast<std::size_t>(width) * height),
                 std::max(width / 2, 1u), std::max(height / 2, 1u),
                 0}};
            patternMipLayout_.PushConstants(commands, &step,
                                            sizeof(step));
            vkCmdDispatch(commands, (step.target[1] + 15) / 16,
                          (step.target[2] + 15) / 16, 1);
        }
        if (more) {
            ShaderBarrier(commands);
        }
    }
}

void RasterRenderer::WriteLightTiming(const FrameParams& frame,
                                      const FrameParams& params,
                                      std::uint32_t views) {
    constexpr std::size_t kLit = LightParams::Beam::kMaxLit;
    std::vector<glm::vec2> timing(static_cast<std::size_t>(views) *
                                      kMaxLights * kLit,
                                  glm::vec2(0.0f));
    const CameraParams& camera = frame.camera;
    // A sub-frame's share of the shutter, seconds.
    const double spacing =
        subframes_ > 1
            ? std::abs(frame.TimeAt(camera.SubframeTime(1)) -
                       frame.TimeAt(camera.SubframeTime(0)))
            : 0.0;
    bool timed = false;
    // Where the frame's exposure opens (its top row's, with
    // sub-frames the first's), which a flash triggered by it counts
    // from.
    const double framePeriod = frame.time - frame.startTime;
    const double exposure =
        subframes_ > 1
            ? frame.TimeAt(camera.SubframeTime(0))
            : frame.TimeAt(camera.shutterClose) - camera.exposureTime;
    for (std::uint32_t v = 0; v < views; ++v) {
        double start = 0.0;
        double end = 0.0;
        if (subframes_ > 1) {
            const double at = frame.TimeAt(camera.SubframeTime(
                static_cast<int>(v / (outputViews_ * faces_))));
            start = at - 0.5 * spacing;
            end = at + 0.5 * spacing;
        } else {
            end = frame.TimeAt(camera.shutterClose);
            start = end - camera.exposureTime;
        }
        for (std::size_t i = 0;
             i < params.lights.size() && i < kMaxLights; ++i) {
            const LightParams& light = params.lights[i];
            // Lasers' patterns and pulses, and other lights' pulses
            // (`LightGpu::links` w).
            if (light.type != LightType::Laser &&
                !light.beam.Pulsed()) {
                continue;
            }
            const std::vector<LightParams::Beam::Lit> lit =
                light.beam.LitBetween(start, end, exposure,
                                      framePeriod);
            for (std::size_t k = 0; k < lit.size() && k < kLit; ++k) {
                timing[(v * kMaxLights + i) * kLit + k] = glm::vec2(
                    static_cast<float>(lit[k].frame), lit[k].weight);
            }
            timed = true;
        }
    }
    if (!timed) {
        return;
    }
    const VkDeviceSize bytes = timing.size() * sizeof(glm::vec2);
    if (lightTiming_->Size() < bytes) {
        vkDeviceWaitIdle(context_.Device());
        lightTiming_ = std::make_unique<GpuBuffer>(
            context_, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            BufferMemory::Device);
    }
    uploads_.Write(*lightTiming_, std::as_bytes(std::span(timing)));
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
    const std::vector<float> params =
        ResolveBrdfParams(*model, settings.brdfParams);
    // Minimal shading's glossy highlights take the model's roughness
    // and metallic, when it has a roughness.
    std::optional<glm::vec2> gloss;
    const auto param =
        [&](std::string_view name) -> std::optional<float> {
        for (std::size_t i = 0; i < model->params.size(); ++i) {
            if (i < params.size() && model->params[i].name == name) {
                return params[i];
            }
        }
        return std::nullopt;
    };
    if (const std::optional<float> roughness = param("roughness")) {
        gloss =
            glm::vec2(*roughness, param("metallic").value_or(0.0f));
    }
    Material(id).Update(settings, params, gloss);
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

void RasterRenderer::CreateLightShadowMaps(std::uint32_t resolution,
                                           std::uint32_t layers) {
    vkDeviceWaitIdle(context_.Device());
    // Copied within: a batch's world maps start as their light's
    // shared one.
    lightShadowMaps_ = std::make_unique<GpuImage>(
        context_,
        ImageDesc{resolution, resolution, layers, 1,
                  VK_FORMAT_D32_SFLOAT,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT,
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
    const std::uint32_t distantResolution =
        std::max(params.shadows.resolution, 1u);
    lightViews_.clear();
    firstVirtualView_ = 0;
    cascadeJobs_.clear();
    cascadeJobCount_ = 0;
    splitsFitted_ = false;
    subframeFitCount_ = 0;
    subframeCopies_.assign(subframes_, {});
    subframeLightViews_.assign(subframes_, {});
    std::vector<std::size_t>
        perWorld;  ///< Lights with per-world maps.
    std::vector<LightGpu> table(params.lights.size());
    std::vector<glm::vec4>
        lightRecords;  ///< The light pass's, for lights posed on the
                       ///< GPU.
    std::vector<PatternJob> patternJobs;
    patternArrayWrites_.clear();
    patternMipRegions_.clear();
    // The regions of images written as GPU arrays first, so the
    // host's part of `patternData_` starts after them all.
    for (const LightParams& light : params.lights) {
        const Tensor& array = light.beam.patternArray;
        if (light.type == LightType::Laser && !array.Empty()) {
            EnsurePatternArray(
                light.beam.patternArrayKey,
                static_cast<std::uint32_t>(array.Shape()[1]),
                static_cast<std::uint32_t>(array.Shape()[0]));
        }
    }
    const std::size_t hostStart = patternArrayFloats_;
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
        entry.links[0] = light.lightMask;
        entry.links[1] = light.shadowMask;
        // Whether it casts shadows at all, which volumes' shadows
        // follow too (volume.slang).
        entry.links[2] = light.shadows ? 1u : 0u;
        // A pulsed light other than a laser, scaled by its timing
        // (`WriteLightTiming`); lasers always read theirs.
        entry.links[3] =
            light.type != LightType::Laser && light.beam.Pulsed()
                ? 1u
                : 0u;
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
        entry.polarization[0] = light.polarizationAxis.x;
        entry.polarization[1] = light.polarizationAxis.y;
        entry.polarization[2] = light.polarizationAxis.z;
        entry.polarization[3] = light.polarization;
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
        // A batch's distant lights have maps per world, after the
        // others'.
        if (worldMaps_ && bounds && distant && light.shadows &&
            !light.domeDiffuse && !light.posedOnGpu) {
            perWorld.push_back(i);
        } else if (bounds && params.shadows.enabled) {
            // A distant light over a single view casts cascades over
            // it, as the sun does, when its maps have room for them;
            // otherwise one map over the whole scene, as a dome's
            // many directions do (`ApplyDomeDirections`).
            // A distant light's maps are as fine as the sun's
            // cascades' (`ShadowSettings::resolution`); others',
            // and a dome's many directions', are
            // `ShadowSettings::lightResolution` across.
            // In a batch of per-world maps, every map is as fine as
            // the worlds', so the layers stay that size.
            const std::uint32_t own = std::min(
                distant && !light.domeDiffuse ? distantResolution
                                              : resolution,
                worldMaps_
                    ? worldMaps_->resolution
                    : std::numeric_limits<std::uint32_t>::max());
            std::vector<ShadowCascade> views;
            bool cascaded = false;
            if (distant && light.shadows && !light.posedOnGpu &&
                !light.domeDiffuse && cascadeFit_ && axis.z > 0.0f) {
                views = FitViewCascades(axis, own);
                if (lightViews_.size() + views.size() >
                    kMaxLightShadowViews) {
                    views.clear();
                }
                cascaded = !views.empty();
            }
            if (views.empty()) {
                views = FitLightShadowViews(
                    light, params.Relative(light.position),
                    bounds->first, bounds->second, own);
            }
            if (!views.empty() && lightViews_.size() + views.size() <=
                                      kMaxLightShadowViews) {
                for (ShadowCascade& view : views) {
                    view.resolution = own;
                }
                entry.map[0] = static_cast<float>(own);
                entry.cone[2] =
                    static_cast<float>(lightViews_.size());
                entry.cone[3] = static_cast<float>(views.size());
                entry.shadow[0] = views.front().texelWorldSize;
                // Cascades over the view the GPU fits, for a visible
                // fit to the depths the view saw with no wait for the
                // host to read them; the host's fit is their
                // estimate, the same but for the depths seen. A light
                // turning across the shutter is fitted at each
                // sub-frame.
                const bool moving =
                    subframes_ > 1 && light.previous &&
                    !light.posedOnGpu && !light.domeDiffuse;
                const bool gpuCascades = cascaded && !moving;
                if (gpuCascades) {
                    AddCascadeJob(params, axis, own,
                                  static_cast<std::uint32_t>(
                                      lightViews_.size()));
                }
                // What culls by them follows the GPU's where it
                // differs.
                const bool seenFit =
                    gpuCascades && cascadeFit_->visible;
                if (light.posedOnGpu || seenFit) {
                    for (std::size_t v = 0; v < views.size(); ++v) {
                        views[v].gpuFitted = true;
                        views[v].gpuMatrix =
                            static_cast<std::uint32_t>(
                                lightViews_.size() + v);
                    }
                }
                // Posed on the GPU per sub-frame with motion blur
                // (`RecordShadowMaps`): nothing culls by the frame's
                // pose.
                if (light.posedOnGpu && subframes_ > 1) {
                    for (ShadowCascade& view : views) {
                        view.moving = true;
                    }
                }
                // Terrain and planets cull each by the host's
                // estimate, the GPU's fit but for the depths seen:
                // drawing them whole into a cascade shadows flat
                // ground (as the lights posed on the GPU still do).
                if (seenFit) {
                    for (ShadowCascade& view : views) {
                        view.cullViewProjection =
                            view.lightViewProjection;
                    }
                }
                // A light moving across the shutter: its views fitted
                // again at each sub-frame's pose (`SubframeLights`),
                // which that sub-frame's shadows draw from
                // (`RecordShadowMaps`).
                if (subframes_ > 1 && light.previous &&
                    !light.posedOnGpu && !light.domeDiffuse) {
                    for (ShadowCascade& view : views) {
                        view.moving = true;
                    }
                    const glm::dmat4 now =
                        LightParams::Pose{light.position,
                                          light.direction,
                                          light.tangent}
                            .Matrix();
                    for (std::uint32_t k = 0; k < subframes_; ++k) {
                        const double s =
                            1.0 + params.camera.SubframeTime(
                                      static_cast<int>(k));
                        LightParams posed = light;
                        if (distant) {
                            posed.direction =
                                glm::vec3(TurnedDirection(light, s));
                        } else {
                            const glm::dmat4 then = InterpolatePose(
                                light.previous->Matrix(), now, s);
                            posed.position = glm::dvec3(then[3]);
                            posed.direction = glm::vec3(
                                -glm::normalize(glm::dvec3(then[2])));
                            posed.tangent = glm::vec3(
                                glm::normalize(glm::dvec3(then[0])));
                        }
                        // As the frame's were fitted: cascades over
                        // the view, on the GPU, copied in by the
                        // sub-frame's shadow passes, or one map over
                        // the scene.
                        if (cascaded) {
                            AddCascadeJob(params, posed.direction,
                                          own, subframeFitCount_,
                                          true);
                            for (std::size_t v = 0; v < views.size();
                                 ++v) {
                                subframeCopies_[k].emplace_back(
                                    subframeFitCount_ +
                                        static_cast<std::uint32_t>(v),
                                    static_cast<std::uint32_t>(
                                        lightViews_.size() + v));
                            }
                            subframeFitCount_ +=
                                static_cast<std::uint32_t>(
                                    views.size());
                            continue;
                        }
                        const std::vector<ShadowCascade> moved =
                            FitLightShadowViews(
                                posed,
                                params.Relative(posed.position),
                                bounds->first, bounds->second, own);
                        for (std::size_t v = 0;
                             v < moved.size() && v < views.size();
                             ++v) {
                            subframeLightViews_[k].emplace_back(
                                static_cast<std::uint32_t>(
                                    lightViews_.size() + v),
                                moved[v].lightViewProjection);
                        }
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
            entry.map[2] = beam.speckle ? 1.0f : 0.0f;
            entry.map[3] = beam.linewidth;
            const auto index = static_cast<std::uint32_t>(
                lightPatterns_.IndexOf(model->id));
            entry.pattern[0] = static_cast<float>(index);
            entry.pattern[1] = beam.wavelength;
            entry.pattern[2] = static_cast<float>(params.time);
            entry.pattern[3] =
                static_cast<float>(std::max(beam.sequenceCount, 1));
            entry.patternExtent[0] = beam.extent.x;
            entry.patternExtent[1] = beam.extent.y;
            entry.patternLayout[0] = static_cast<std::uint32_t>(
                hostStart + patternData.size());
            entry.patternLayout[1] =
                static_cast<std::uint32_t>(beam.patternData.size());
            patternData.insert(patternData.end(),
                               beam.patternData.begin(),
                               beam.patternData.end());
            if (const LightPatternImage* image =
                    beam.patternImage.get()) {
                entry.patternLayout[2] = static_cast<std::uint32_t>(
                    hostStart + patternData.size());
                entry.patternLayout[3] =
                    image->width | (image->height << 16);
                patternData.insert(patternData.end(),
                                   image->texels.begin(),
                                   image->texels.end());
            } else if (!beam.patternArray.Empty()) {
                // An array on the GPU: in its region of
                // `patternData_`, read there before the frame is
                // submitted, its mip levels built on the GPU.
                const auto width = static_cast<std::uint32_t>(
                    beam.patternArray.Shape()[1]);
                const auto height = static_cast<std::uint32_t>(
                    beam.patternArray.Shape()[0]);
                const PatternArrayRegion region = EnsurePatternArray(
                    beam.patternArrayKey, width, height);
                entry.patternLayout[2] =
                    static_cast<std::uint32_t>(region.offset);
                entry.patternLayout[3] = width | (height << 16);
                patternArrayWrites_.push_back(
                    {beam.patternArray, beam.patternArrayStream,
                     region.offset * sizeof(float)});
                patternMipRegions_.push_back(region);
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
            // Beams volumes integrate across rays (volume.slang's
            // `BeamCrossings`): a spot's, or a grid's of at most 64
            // beamlets.
            const bool smallGrid = beam.pattern == "grid" &&
                                   values.size() >= 2 &&
                                   values[0] * values[1] <= 64.0f;
            entry.patternExtent[2] = beam.IsSpot() ? 1.0f
                                     : smallGrid   ? 2.0f
                                                   : 0.0f;
            // Its integral to compute (`patternNormMain`), unless
            // it is 1.
            if (!model->normalized) {
                PatternJob job{};
                job.info[0] = index;
                job.info[1] = static_cast<std::uint32_t>(i);
                job.info[2] = static_cast<std::uint32_t>(
                    std::max(beam.sequenceCount, 1));
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
            const bool spot = laser && light.beam.IsSpot();
            const auto kind =
                distant ? 0u
                : spot  ? 3u
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
            // A laser's spot: its beam's widest waist radius,
            // shortest Rayleigh range and waist distance, which size
            // its view.
            lightRecords.push_back(
                spot ? glm::vec4(std::max(light.beam.waist.x,
                                          light.beam.waist.y),
                                 std::min(light.beam.rayleigh.x,
                                          light.beam.rayleigh.y),
                                 light.beam.waistDistance, 0.0f)
                     : glm::vec4(2.0f * glm::radians(std::max(
                                            light.coneAngle, 0.5f)),
                                 light.radius, 0.0f, 0.0f));
        }
    }
    if (!patternNorms.empty()) {
        uploads_.Write(patternNorms_,
                       std::as_bytes(std::span(patternNorms)));
    }
    // The host's pattern data and images after the arrays' regions,
    // when they change or move.
    patternHostFloats_ = patternData.size();
    EnsurePatternData(hostStart + patternData.size());
    if ((patternDataHeld != patternDataHeld_ ||
         hostStart != patternDataBase_) &&
        !patternData.empty()) {
        uploads_.Write(*patternData_,
                       std::as_bytes(std::span(patternData)),
                       hostStart * sizeof(float));
    }
    patternDataBase_ = hostStart;
    patternDataHeld_ = std::move(patternDataHeld);
    if (!patternMipRegions_.empty() &&
        patternMipPipeline_ == nullptr) {
        const std::vector<ShaderModuleSource> modules{
            {"pattern_mips", kPatternMipSlang}};
        patternMipPipeline_ = std::make_unique<ComputePipeline>(
            context_, patternMipLayout_,
            compiler_.CompileSpirv(modules, "patternMipMain"));
    }
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
        casters_.shadowView = lightViews_.size();
        casters_.resolution = resolution;
        casters_.top = top;
        casters_.span = span;
        view.resolution = resolution;
        lightViews_.push_back(view);
    }
    if (casters_.valid != before.valid ||
        casters_.layer != before.layer ||
        casters_.resolution != before.resolution ||
        casters_.top != before.top || casters_.span != before.span) {
        castersChanged_ = true;
    }
    // A batch's distant lights: each a map of what every world
    // shares, then one per world of the batch's views, as many as the
    // layers hold, starting as a copy of it with the world's casters
    // drawn on top, or fitted to the world's own casters.
    firstLayeredView_ = lightViews_.size();
    layeredPasses_.clear();
    worldSlots_.clear();
    if (!perWorld.empty()) {
        const WorldMaps& maps = *worldMaps_;
        const std::size_t layers =
            context_.Limits().maxImageArrayLayers;
        const std::size_t each =
            layers > lightViews_.size()
                ? (layers - lightViews_.size()) / perWorld.size()
                : 0;
        const std::size_t slots =
            std::min(maps.worlds.size(), each > 0 ? each - 1 : 0);
        for (std::size_t s = 0; s < slots; ++s) {
            worldSlots_[maps.worlds[s]] =
                static_cast<std::uint32_t>(s + 1);
        }
        for (const std::size_t i : perWorld) {
            const LightParams& light = params.lights[i];
            const auto fit = [&](const glm::vec3& low,
                                 const glm::vec3& high) {
                ShadowCascade view =
                    FitLightShadowViews(light, glm::vec3(0.0f), low,
                                        high, maps.resolution)
                        .front();
                view.resolution = maps.resolution;
                return view;
            };
            const auto base =
                static_cast<std::uint32_t>(lightViews_.size());
            lightViews_.push_back(fit(bounds->first, bounds->second));
            layeredPasses_.push_back(
                {ShadowCasters{true, ShadowCasters::kNoWorld}, base,
                 false});
            for (std::size_t s = 0; s < slots; ++s) {
                const std::int32_t world = maps.worlds[s];
                const auto fitted = maps.fitted.find(world);
                if (fitted == maps.fitted.end()) {
                    // Its world's casters only, culled as such.
                    ShadowCascade copy = lightViews_[base];
                    copy.world = world;
                    lightViews_.push_back(copy);
                    layeredPasses_.push_back(
                        {ShadowCasters{false, world}, base, true});
                    continue;
                }
                const glm::vec3 margin(maps.margin);
                ShadowCascade own =
                    fit(fitted->second.first - margin,
                        fitted->second.second + margin);
                own.world = world;
                lightViews_.push_back(own);
                layeredPasses_.push_back(
                    {ShadowCasters{true, world}, base, false});
            }
            LightGpu& entry = table[i];
            entry.cone[2] = static_cast<float>(base);
            entry.cone[3] = static_cast<float>(slots + 1);
            entry.shadow[0] = lightViews_[base].texelWorldSize;
            entry.map[0] = static_cast<float>(maps.resolution);
            entry.map[1] = 1.0f;
        }
    }
    // The first shadowed distant light of a single view also shadows
    // through a virtual shadow map: its clipmap's levels as light
    // views after every other, with no layers of the maps
    // (`RecordVirtualShadows`).
    firstVirtualView_ = lightViews_.size();
    virtualLight_ = -1;
    virtualDebug_ = params.shadows.virtualDebug;
    if (params.shadows.enabled && params.shadows.virtualMaps &&
        params.minimal == MinimalMode::Off && outputViews_ == 1 &&
        !posesOnGpu_ && faces_ == 1 && subframes_ == 1 &&
        !params.camera.pushbroom && SupportsVirtualShadows()) {
        for (std::size_t i = 0; i < params.lights.size(); ++i) {
            const LightParams& light = params.lights[i];
            if (light.type == LightType::Distant && light.shadows &&
                !light.posedOnGpu && !light.domeDiffuse) {
                virtualLight_ = static_cast<std::int32_t>(i);
                break;
            }
        }
    }
    if (virtualLight_ >= 0) {
        if (virtualShadow_ == nullptr ||
            virtualShadow_->Pages() != params.shadows.virtualPages) {
            vkDeviceWaitIdle(context_.Device());
            virtualShadow_ = std::make_unique<VirtualShadowMap>(
                context_, compiler_, kVirtualFirstLevel,
                kVirtualLastLevel, params.shadows.virtualPages);
        }
        const LightGpu& entry =
            table[static_cast<std::size_t>(virtualLight_)];
        const glm::dvec3 toLight(entry.position[0], entry.position[1],
                                 entry.position[2]);
        const CameraView view = params.View(0);
        const glm::mat4 viewProjection = view.ViewProjection();
        virtualInverse_ = glm::inverse(viewProjection);
        virtualEye_ = view.eye;
        const std::vector<VirtualShadowLevel> levels =
            VirtualShadowMap::FitLevels(
                toLight, params.origin + glm::dvec3(view.eye),
                params.origin, kVirtualFirstLevel, kVirtualLastLevel);
        // About a texel a pixel: a level's texel, 2^(level - 11) m,
        // matches a pixel's width 2 d tan(half field) / width at
        // distance d when level = log2 d + log2(4096 tan / width).
        // Narrower images take a width of `kVirtualMinWidth`'s
        // texels, whose bias of a few texels would otherwise reach
        // past casters a pixel or two above their shadows.
        const float tanHalf =
            std::abs(viewProjection[0][0]) > 0.0f
                ? 1.0f / std::abs(viewProjection[0][0])
                : 1.0f;
        const float bias =
            std::log2(4096.0f * tanHalf /
                      static_cast<float>(std::max(
                          params.camera.width, kVirtualMinWidth))) +
            params.shadows.virtualBias;
        virtualShadow_->SetLevels(levels, bias, &uploads_);
        virtualBias_ = bias;
        for (const VirtualShadowLevel& level : levels) {
            ShadowCascade cascade;
            cascade.lightViewProjection = level.viewProjection;
            cascade.texelWorldSize =
                static_cast<float>(level.texelWorldSize);
            cascade.resolution = VirtualShadowMap::kVirtualTexels;
            // Culled on the GPU from its matrix in
            // `lightViewMatrices_` (meshes, points), as instances'
            // bounds may be known only there; terrain and planets
            // cull by the same matrix.
            cascade.gpuFitted = true;
            cascade.gpuMatrix =
                static_cast<std::uint32_t>(lightViews_.size());
            cascade.cullViewProjection = level.viewProjection;
            cascade.shadowMask =
                params.lights[static_cast<std::size_t>(virtualLight_)]
                    .shadowMask;
            lightViews_.push_back(cascade);
        }
        // Its own cascades then hold no casters, as shading reads the
        // pages instead, unless volumes march through them: the pages
        // back only what the G-buffer saw, not the air in front of
        // it.
        if (params.volumes.empty()) {
            const auto first =
                static_cast<std::size_t>(entry.cone[2]);
            const auto count =
                static_cast<std::size_t>(entry.cone[3]);
            for (std::size_t v = first;
                 v < first + count && v < firstVirtualView_; ++v) {
                lightViews_[v].drawn = false;
            }
        }
    }
    // Shadow linking: each light's views draw only the geometry its
    // shadow link holds.
    for (std::size_t i = 0;
         i < table.size() && i < params.lights.size(); ++i) {
        const auto first = static_cast<std::size_t>(table[i].cone[2]);
        const auto count = static_cast<std::size_t>(table[i].cone[3]);
        for (std::size_t v = first;
             v < first + count && v < lightViews_.size(); ++v) {
            lightViews_[v].shadowMask = params.lights[i].shadowMask;
        }
    }
    // Minimal shading's main light, the brightest distant light, and
    // its maps, which it shadows with.
    minimalKey_.reset();
    if (const LightParams* brightest =
            BrightestDistant(params, params.camera.bands)) {
        const auto i = static_cast<std::size_t>(brightest -
                                                params.lights.data());
        const LightGpu& entry = table[i];
        if (entry.cone[3] > 0.0f) {
            minimalKey_ = MinimalKey{
                i, static_cast<std::uint32_t>(entry.cone[2]),
                static_cast<std::uint32_t>(entry.cone[3]),
                entry.map[1] != 0.0f};
        }
    }
    // The cascades the GPU fits, minimal shading's main light's
    // written to the frame's Shadows block there too.
    for (std::uint32_t j = 0; j < cascadeJobCount_; ++j) {
        auto* job = reinterpret_cast<CascadeJob*>(
            cascadeJobs_.data() + j * sizeof(CascadeJob));
        job->out[3] = minimalKey_ && !minimalKey_->perWorld &&
                              job->out[0] == minimalKey_->first
                          ? 1u
                          : 0u;
    }
    if (subframeFitCount_ > 0 &&
        (!subframeFits_ ||
         subframeFits_->Size() <
             subframeFitCount_ * sizeof(glm::mat4))) {
        vkDeviceWaitIdle(context_.Device());
        subframeFits_ = std::make_unique<GpuBuffer>(
            context_, subframeFitCount_ * sizeof(glm::mat4),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
    }
    if (!table.empty()) {
        uploads_.Write(lightTable_, std::as_bytes(std::span(table)));
    }
    // With sub-frames, a copy of the lights per sub-frame, those that
    // moved since the last frame posed at its time.
    const std::vector<LightGpu> moving =
        SubframeLights(params, table, subframes_);
    lightSubframes_ = !moving.empty();
    if (lightSubframes_) {
        uploads_.Write(lightTable_, std::as_bytes(std::span(moving)));
    } else {
        subframeLightViews_.clear();
    }
    std::vector<glm::mat4> matrices;
    std::vector<std::byte> blocks(firstLayeredView_ *
                                  kLightShadowBlockStride);
    for (std::size_t v = 0; v < lightViews_.size(); ++v) {
        matrices.push_back(lightViews_[v].lightViewProjection);
        if (v < firstLayeredView_) {
            std::memcpy(
                blocks.data() + v * kLightShadowBlockStride,
                glm::value_ptr(lightViews_[v].lightViewProjection),
                sizeof(glm::mat4));
        }
    }
    if (!matrices.empty()) {
        const VkDeviceSize matrixBytes =
            matrices.size() * sizeof(glm::mat4);
        if (lightViewMatrices_->Size() < matrixBytes) {
            // Grown only once no frame still reads it.
            vkDeviceWaitIdle(context_.Device());
            lightViewMatrices_ = std::make_unique<GpuBuffer>(
                context_, matrixBytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                BufferMemory::Device);
        }
        uploads_.Write(*lightViewMatrices_,
                       std::as_bytes(std::span(matrices)));
        if (!blocks.empty()) {
            uploads_.Write(lightShadowBlocks_, blocks);
        }
        // The virtual shadow map's levels' blocks: each its matrix
        // and the map's block with the level (`Shadows::virtualMap`).
        if (virtualLight_ >= 0) {
            const VkDeviceAddress address =
                virtualShadow_->Block().Address();
            std::vector<std::byte> levels(
                (lightViews_.size() - firstVirtualView_) *
                kLightShadowBlockStride);
            for (std::size_t v = firstVirtualView_;
                 v < lightViews_.size(); ++v) {
                ShadowUniforms block{};
                std::memcpy(block.lightViewProjection[0],
                            glm::value_ptr(
                                lightViews_[v].lightViewProjection),
                            sizeof(glm::mat4));
                block.virtualMap[0] =
                    static_cast<std::uint32_t>(address);
                block.virtualMap[1] =
                    static_cast<std::uint32_t>(address >> 32);
                block.virtualMap[2] =
                    static_cast<std::uint32_t>(v - firstVirtualView_);
                block.virtualMap[3] = 1;
                std::memcpy(levels.data() +
                                (v - firstVirtualView_) *
                                    kLightShadowBlockStride,
                            &block, sizeof(block));
            }
            uploads_.Write(lightShadowBlocks_, levels,
                           firstVirtualView_ *
                               kLightShadowBlockStride);
        }
        // Layers as wide as the finest view's; coarser views draw in
        // a corner of theirs. Virtual views have none.
        const auto layers = static_cast<std::uint32_t>(
            std::max<std::size_t>(firstVirtualView_, 1));
        std::uint32_t widest = 1;
        for (std::size_t v = 0; v < firstVirtualView_; ++v) {
            widest = std::max(widest, lightViews_[v].resolution);
        }
        if (widest != lightResolution_ ||
            layers > lightShadowMaps_->Desc().layers) {
            CreateLightShadowMaps(widest, layers);
        }
    }
    const std::uint32_t lastGpuLights = gpuLights_;
    gpuLights_ = static_cast<std::uint32_t>(lightRecords.size() / 2);
    // With motion blur, the light pass blends each sub-frame's poses
    // from the last frame's, which `lightPoses_` still holds when the
    // frame starts recording: the same lights, in buffers kept and
    // relative to the same origin (`lightPosesPrevious_`).
    bool keptPoses = gpuLights_ > 0 && gpuLights_ == lastGpuLights &&
                     params.origin == lastPoseOrigin_;
    lastPoseOrigin_ = params.origin;
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
            lightPosesPrevious_ = std::make_unique<GpuBuffer>(
                context_, poseBytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                BufferMemory::Device);
            keptPoses = false;
        }
        uploads_.Write(*lightRecords_,
                       std::as_bytes(std::span(lightRecords)));
        lightBoundsLow_ =
            bounds ? glm::vec4(bounds->first, 0.0f) : glm::vec4(0.0f);
        lightBoundsHigh_ = bounds ? glm::vec4(bounds->second, 0.0f)
                                  : glm::vec4(0.0f);
    }
    blendGpuLights_ = keptPoses && lightSubframes_;
}

void RasterRenderer::DispatchLights(VkCommandBuffer commands,
                                    std::uint32_t tableOffset,
                                    float share, bool blend) const {
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      lightsPipeline_->Handle());
    const DescriptorResource resources[] = {
        DescriptorResource::Of(*lightRecords_),
        DescriptorResource::Of(*lightPoses_),
        DescriptorResource::Of(lightTable_),
        DescriptorResource::Of(*lightViewMatrices_),
        DescriptorResource::Of(lightShadowBlocks_),
        DescriptorResource::Of(lightPosesPrevious_ != nullptr
                                   ? *lightPosesPrevious_
                                   : *lightPoses_)};
    lightsLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                       resources);
    struct {
        std::uint32_t layout[4];
        glm::vec4 low;
        glm::vec4 high;
        glm::vec4 motion;
    } constants{{gpuLights_,
                 static_cast<std::uint32_t>(sizeof(LightGpu) /
                                            sizeof(glm::vec4)),
                 static_cast<std::uint32_t>(kLightShadowBlockStride /
                                            sizeof(glm::vec4)),
                 tableOffset},
                lightBoundsLow_,
                lightBoundsHigh_,
                glm::vec4(share, blend ? 1.0f : 0.0f, 0.0f, 0.0f)};
    lightsLayout_.PushConstants(commands, &constants,
                                sizeof(constants));
    vkCmdDispatch(commands, (gpuLights_ + 63) / 64, 1, 1);
    ShaderBarrier(commands);
}

void RasterRenderer::RecordLights(
    VkCommandBuffer commands,
    std::span<const RasterDrawable* const> drawables) const {
    // Distant lights' cascades fitted on the GPU, then lights posed
    // there; then the views' cull views follow their matrices.
    RecordCascadeFit(commands);
    if (gpuLights_ == 0) {
        if (cascadeJobCount_ > 0) {
            for (const RasterDrawable* drawable : drawables) {
                ERNIE_GPU_ZONE_NAMED(context_, commands,
                                     drawable->ProfileName());
                drawable->RecordShadowViews(commands);
            }
            ShaderBarrier(commands);
        }
        return;
    }
    ERNIE_GPU_ZONE(context_, commands, "Lights posed on the GPU");
    // The frame's poses into the light table and the views; with
    // motion blur, each sub-frame's again (`RecordShadowMaps`).
    DispatchLights(commands, 0, 1.0f, false);
    // The views' cull views follow their fitted matrices.
    for (const RasterDrawable* drawable : drawables) {
        ERNIE_GPU_ZONE_NAMED(context_, commands,
                             drawable->ProfileName());
        drawable->RecordShadowViews(commands);
    }
    ShaderBarrier(commands);
}

void RasterRenderer::Update(
    const FrameParams& frame,
    std::span<RasterDrawable* const> drawables,
    std::span<const Overlay* const> overlays) {
    ERNIE_ZONE("Renderer::Update");
    ERNIE_PLOT("Views", frame.Views());
    // Heap slots freed frames ago may be reused.
    context_.Bindless().AdvanceFrame();
    // With motion blur, a view per sub-frame and view
    // (`FrameParams::Subframes`): the drawables cull and draw each,
    // and the sensor reads their sum.
    frame.camera.Check();
    subframes_ = static_cast<std::uint32_t>(frame.camera.Subframes());
    subframeShares_.clear();
    for (std::uint32_t k = 0; k < subframes_; ++k) {
        // The last frame is at time -1, this one at 0.
        subframeShares_.push_back(static_cast<float>(
            1.0 + frame.camera.SubframeTime(static_cast<int>(k))));
    }
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
    if (frame.camera.polarizer.type != Polarizer::Type::None &&
        faces_ > 1) {
        throw std::invalid_argument(
            "A polarizer needs a camera drawn in one view, not on "
            "cube "
            "faces, whose Stokes frames differ");
    }
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
                                    " lights");
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
    minimalSky_ = MinimalSky(params, params.camera.bands);
    minimalToDome_ =
        params.dome ? params.dome->toDome : glm::mat3(1.0f);
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
                    "Minimal shading draws meshes, the ground plane "
                    "and point clouds, not yet terrain or planets");
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
    // The Distance display spans the near plane to the far, or with
    // an infinite far plane to the scene's farthest corner.
    if (!std::isfinite(presentRange_[1])) {
        const glm::vec3 eye(glm::dvec3(params.camera.transform[3]) -
                            params.origin);
        float farthest = 1e3f * presentRange_[0];
        if (bounds) {
            farthest = 2.0f * presentRange_[0];
            for (int corner = 0; corner < 8; ++corner) {
                const glm::vec3 point(
                    (corner & 1) != 0 ? bounds->second.x
                                      : bounds->first.x,
                    (corner & 2) != 0 ? bounds->second.y
                                      : bounds->first.y,
                    (corner & 4) != 0 ? bounds->second.z
                                      : bounds->first.z);
                farthest =
                    std::max(farthest, glm::length(point - eye));
            }
        }
        presentRange_[1] = farthest;
    }
    const ShadowSettings& shadows = params.shadows;
    // The lights' maps hold what a single view sees, or in a batch
    // what every world shares (every world's casters when merged; a
    // batch's distant lights add a map per world when separate).
    lightCasters_ =
        !batch ? ShadowCasters{true, params.ViewWorld(0)}
               : ShadowCasters{true,
                               shadows.worlds == WorldShadows::Merged
                                   ? ShadowCasters::kAllWorlds
                                   : ShadowCasters::kNoWorld};
    // What a single view's distant lights' cascades fit to
    // (`FitViewCascades`).
    const bool distantCascades = std::any_of(
        params.lights.begin(), params.lights.end(),
        [](const LightParams& light) {
            return light.type == LightType::Distant && light.shadows;
        });
    cascadeFit_.reset();
    if (!batch && bounds && shadows.enabled && distantCascades) {
        CascadeFit fit;
        fit.view = view;
        fit.around = around;
        // Where the shadows land counts too: on a ground plane, which
        // has no bounds, they reach past the casters, and down to it
        // when the casters float above it.
        fit.casterLow = bounds->first;
        fit.casterHigh = bounds->second;
        for (const RasterDrawable* drawable : drawables) {
            if (const std::optional<float> floor =
                    drawable->ShadowFloor()) {
                fit.casterLow.z = std::min(fit.casterLow.z, *floor);
            }
        }
        fit.count = std::clamp(shadows.cascades, 1, kMaxCascades);
        fit.lambda = shadows.splitLambda;
        if (shadows.fit == ShadowFit::Visible) {
            fit.visible = true;
        }
        cascadeFit_ = fit;
    }
    // A batch's per-world maps of its distant lights, at the
    // resolution of its worlds' maps: the worlds of its views, and
    // the bounds of each world's casters for maps fitted to them.
    worldMaps_.reset();
    if (batch && bounds && shadows.enabled &&
        shadows.worlds == WorldShadows::Separate) {
        WorldMaps maps;
        for (std::uint32_t v = 0; v < views_; ++v) {
            maps.worlds.push_back(raster.ViewWorld(v));
        }
        std::sort(maps.worlds.begin(), maps.worlds.end());
        maps.worlds.erase(
            std::unique(maps.worlds.begin(), maps.worlds.end()),
            maps.worlds.end());
        if (maps.worlds != std::vector<std::int32_t>{-1}) {
            maps.resolution =
                shadows.worldResolution > 0
                    ? shadows.worldResolution
                    : ShadowSettings::AutoWorldResolution(
                          params.camera.width, params.camera.height);
            maps.margin = shadows.worldMargin;
            if (shadows.worldFit == WorldShadowFit::World) {
                for (const RasterDrawable* drawable : drawables) {
                    for (const auto& [world, box] :
                         drawable->WorldsBounds()) {
                        if (world == -1) {
                            continue;
                        }
                        const auto [it, added] =
                            maps.fitted.try_emplace(world, box);
                        if (!added) {
                            it->second = Bounds3(
                                glm::min(it->second.first, box.first),
                                glm::max(it->second.second,
                                         box.second));
                        }
                    }
                }
            }
            worldMaps_ = std::move(maps);
        }
    }
    // The lights first: the views' uniforms take minimal shading's
    // main light's maps, and shading the size of the maps they make.
    // Minimal shading's None mode lights nothing, so it draws no
    // shadow maps: without bounds, no light fits a view.
    WriteLights(params, minimal_ == MinimalMode::None ? std::nullopt
                                                      : bounds);
    // A metering frame counts its luminance for later frames; this
    // slot's last one has finished.
    ReadMeter(uploads_.Slot());
    meter_ = params.meter && !batch;
    metered_[uploads_.Slot()] = meter_;
    meterKeys_[uploads_.Slot()] = meter_ ? meterKey_ : std::string();
    // Automatic exposure, on the GPU, in its key's state: the slot
    // given at the key's first frame (the oldest's taken once all
    // are, starting it afresh).
    exposing_.reset();
    if (meter_ && autoExposure_) {
        auto found = exposureSlots_.find(meterKey_);
        if (found == exposureSlots_.end()) {
            if (exposureSlots_.size() >= kMaxExposureStates) {
                exposureSlots_.clear();
            }
            const auto slot =
                static_cast<std::uint32_t>(exposureSlots_.size());
            found = exposureSlots_.emplace(meterKey_, slot).first;
            // Not metered yet: its first metering sets it at once.
            const std::array<float, 4> fresh{};
            uploads_.Write(exposureStates_,
                           std::as_bytes(std::span(fresh)),
                           slot * sizeof(fresh));
        }
        exposing_ = Exposing{
            found->second, std::clamp(*autoExposure_, 0.0f, 1.0f),
            static_cast<float>(params.camera.LinearExposureScale())};
    }
    autoExposure_.reset();
    meterKey_.clear();
    // A visible fit's frame measures what it sees, for later frames'
    // cascades.
    measureDepth_ = cascadeFit_.has_value() &&
                    params.shadows.fit == ShadowFit::Visible;
    if (measureDepth_) {
        lastMeasured_ = Measured{
            params.stream, glm::dvec3(params.camera.transform[3]),
            view.forward};
    }

    {
        ERNIE_ZONE("Frame uniforms");
        // Every view's in one upload: the array frame.slang reads
        // (`kUniformsStride` apart).
        static_assert(kUniformsStride == sizeof(Uniforms),
                      "Views' uniforms are an array");
        std::vector<Uniforms> viewUniforms(views_);
        for (std::uint32_t v = 0; v < views_; ++v) {
            Uniforms& uniforms = viewUniforms[v];
            // With the poses on the GPU, it writes the camera
            // (`RecordViewPoses`).
            if (!posesOnGpu_) {
                const CameraView viewOf = raster.View(v);
                // A pushbroom projects through its strip in place of
                // a view-projection (`projectWorld`).
                if (params.camera.pushbroom) {
                    const PushbroomStrip strip = params.Strip(v);
                    CopyMatrix(uniforms.viewProjection,
                               strip.Packed());
                    const glm::vec4 intrinsics = strip.Intrinsics();
                    std::memcpy(uniforms.strip,
                                glm::value_ptr(intrinsics),
                                sizeof(uniforms.strip));
                } else {
                    CopyMatrix(uniforms.viewProjection,
                               viewOf.ViewProjection());
                }
                uniforms.forward[0] = viewOf.forward.x;
                uniforms.forward[1] = viewOf.forward.y;
                uniforms.forward[2] = viewOf.forward.z;
                uniforms.forward[3] =
                    glm::dot(viewOf.eye, viewOf.forward);
                uniforms.eye[0] = viewOf.eye.x;
                uniforms.eye[1] = viewOf.eye.y;
                uniforms.eye[2] = viewOf.eye.z;
            }
            uniforms.eye[3] = params.exposure;
            uniforms.flags[0] =
                static_cast<std::uint32_t>(params.wireframe);
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
            // Minimal shading's light: the brightest distant light
            // the view's world sees, then the next three, each as a
            // share of the brightest's, and the ambient light, per
            // channel.
            const SpectralBands& bands = params.camera.bands;
            const std::int32_t world = raster.ViewWorld(v);
            const glm::vec3 ambient =
                MinimalAmbient(params, bands, world);
            uniforms.motion[1] = ambient.r;
            uniforms.motion[2] = ambient.g;
            uniforms.motion[3] = ambient.b;
            if (const LightParams* brightest =
                    BrightestDistant(params, bands, world)) {
                const double unit = WhiteLuminance(*brightest, bands);
                std::vector<const LightParams*> order{brightest};
                for (const LightParams& light : params.lights) {
                    if (order.size() < 4 && &light != brightest &&
                        light.type == LightType::Distant &&
                        !light.domeDiffuse &&
                        (light.world == -1 || world == -1 ||
                         light.world == world)) {
                        order.push_back(&light);
                    }
                }
                for (std::size_t i = 0; i < order.size(); ++i) {
                    const glm::vec3 towards =
                        glm::normalize(order[i]->direction);
                    float* entry = uniforms.minimalLights[i];
                    entry[0] = towards.x;
                    entry[1] = towards.y;
                    entry[2] = towards.z;
                    entry[3] =
                        unit > 0.0
                            ? static_cast<float>(
                                  WhiteLuminance(*order[i], bands) /
                                  unit)
                            : 0.0f;
                }
                // Its shadow, when it is the frame's main light with
                // maps: in a batch, the view's world's map, placed in
                // the shared one's clip space.
                if (minimalKey_ &&
                    brightest == &params.lights[minimalKey_->light]) {
                    std::uint32_t slot = 0;
                    if (minimalKey_->perWorld) {
                        const auto found = worldSlots_.find(world);
                        slot = found != worldSlots_.end()
                                   ? found->second
                                   : 0;
                    }
                    const glm::mat4 place =
                        lightViews_[minimalKey_->first + slot]
                            .lightViewProjection *
                        glm::inverse(lightViews_[minimalKey_->first]
                                         .lightViewProjection);
                    uniforms.minimalShadow[0][0] = place[0][0];
                    uniforms.minimalShadow[0][1] = place[1][1];
                    uniforms.minimalShadow[0][2] = place[3][0];
                    uniforms.minimalShadow[0][3] = place[3][1];
                    uniforms.minimalShadow[1][0] = place[2][2];
                    uniforms.minimalShadow[1][1] = place[3][2];
                    uniforms.minimalShadow[1][2] =
                        static_cast<float>(slot);
                    uniforms.minimalShadow[1][3] = 1.0f;
                }
            }
            // Minimal shading's colour and mode (surface.slang's
            // kMinimal*).
            uniforms.minimal[0] = params.minimalColor.r;
            uniforms.minimal[1] = params.minimalColor.g;
            uniforms.minimal[2] = params.minimalColor.b;
            const std::uint32_t mode =
                params.minimal == MinimalMode::Constant   ? 1u
                : params.minimal == MinimalMode::Textured ? 2u
                : params.minimal == MinimalMode::Glossy   ? 3u
                                                          : 0u;
            // With glossy highlights and emission, which keep no
            // constant colour, x scales the emission (cd/m^2) to the
            // lights' shares: per what the brightest gives a white
            // surface facing it, or 1 without one.
            if (params.minimal == MinimalMode::Glossy) {
                const LightParams* brightest =
                    BrightestDistant(params, bands, world);
                const double unit =
                    brightest != nullptr
                        ? WhiteLuminance(*brightest, bands)
                        : 0.0;
                uniforms.minimal[0] =
                    unit > 0.0 ? static_cast<float>(1.0 / unit)
                               : 1.0f;
            }
            uniforms.minimal[3] = static_cast<float>(mode);
        }
        uploads_.Write(uniforms_,
                       std::as_bytes(std::span(viewUniforms)));
        exposureViews_ =
            static_cast<std::uint32_t>(viewUniforms.size());
    }

    // The G-buffer passes' Shadows block: the view's cascade splits,
    // which every distant light's cascades share, for tinting by
    // cascade (`ShadowSettings::showCascades`).
    ShadowUniforms shadowUniforms{};
    std::size_t splits = 0;
    if (cascadeFit_) {
        const std::vector<ShadowCascade> cascades =
            FitViewCascades(glm::vec3(0.0f, 0.0f, 1.0f), 1);
        splits = std::min<std::size_t>(cascades.size(), kMaxCascades);
        for (std::size_t c = 0; c < splits; ++c) {
            shadowUniforms.splits[c] = cascades[c].splitFar;
        }
    }
    // Minimal shading's main light's maps: its cascades over a single
    // view, or a batch's shared map, which each view's world's
    // places.
    if (minimalKey_) {
        const std::uint32_t maps =
            minimalKey_->perWorld
                ? 1u
                : std::min<std::uint32_t>(minimalKey_->count,
                                          kMaxCascades);
        for (std::uint32_t c = 0; c < maps; ++c) {
            const ShadowCascade& map =
                lightViews_[minimalKey_->first + c];
            std::memcpy(shadowUniforms.lightViewProjection[c],
                        glm::value_ptr(map.lightViewProjection), 64);
            shadowUniforms.texelWorld[c] = map.texelWorldSize;
        }
        const ShadowCascade& first = lightViews_[minimalKey_->first];
        shadowUniforms.minimal[0] =
            static_cast<float>(minimalKey_->first);
        shadowUniforms.minimal[1] =
            static_cast<float>(first.resolution) /
            static_cast<float>(lightResolution_);
        shadowUniforms.minimal[2] = static_cast<float>(maps);
        shadowUniforms.minimal[3] = 1.0f;
    }
    shadowUniforms.params[0] = shadows.softness;
    shadowUniforms.params[1] = shadows.normalOffset;
    shadowUniforms.params[2] = static_cast<float>(splits);
    shadowUniforms.params[3] = splits > 0 ? 1.0f : 0.0f;
    shadowUniforms.debug[0] = shadows.showCascades ? 1u : 0u;
    // The virtual shadow map's levels shown in the G-buffer
    // (`cascadeTint`): the level bias and first level.
    if (virtualLight_ >= 0 && virtualDebug_ == 1) {
        shadowUniforms.virtualMap[0] =
            std::bit_cast<std::uint32_t>(virtualBias_);
        shadowUniforms.virtualMap[1] =
            std::bit_cast<std::uint32_t>(kVirtualFirstLevel);
        shadowUniforms.virtualMap[3] = 2;
    }
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
    // A polarizer, and the polarization vars, read the raster
    // radiance's Stokes Q and U; without them it carries none.
    stokes_ = frame.camera.polarizer.type != Polarizer::Type::None;
    for (const std::string_view var :
         {render_vars::kDegreeOfLinearPolarization,
          render_vars::kAngleOfLinearPolarization,
          render_vars::kLinearStokes}) {
        if (std::find(requiredVars_.begin(), requiredVars_.end(),
                      var) != requiredVars_.end()) {
            if (faces_ > 1) {
                throw std::invalid_argument(
                    "Render var '" + std::string(var) +
                    "' needs a camera drawn in one view, not on cube "
                    "faces, whose Stokes frames differ");
            }
            stokes_ = true;
        }
    }
    lensTable_ = {};
    if (const auto& table = params.camera.lens.rays) {
        lensTable_ = {0u, table->step, table->columns, table->rows};
    }
    filters_ = sensorPasses_.front().plan.filters;
    bands_ = params.camera.bands;
    thermal_ = params.thermal;
    extent_ = {params.camera.width, params.camera.height};
    // Overlay batches with worlds sort by the views' worlds.
    std::uint32_t overlayWorlds = 0;
    for (std::size_t v = 0;
         v < std::min<std::size_t>(raster.worlds.size(), views_);
         ++v) {
        overlayWorlds = std::max(
            overlayWorlds, static_cast<std::uint32_t>(
                               std::max(raster.worlds[v], -1) + 1));
    }
    overlays_->Prepare(overlays, params.origin,
                       params.camera.pushbroom, overlayWorlds);
    // A cube camera's overlays resample through its lens as the
    // sensor does, and with a rolling shutter its faces' texels take
    // the times of the sensor rows the lens maps them to.
    faceLens_.clear();
    if (faces_ > 1) {
        faceLens_ = sensorPasses_.front().plan.LensSource();
        if (!overlays_->Empty()) {
            if (overlayLensLayout_ == nullptr) {
                // The sensor's constants (sensor.slang).
                overlayLensLayout_ = std::make_unique<PipelineLayout>(
                    context_, kOverlayLensBindings,
                    sizeof(SensorConstants));
            }
            LensPipeline(faceLens_, "overlayLensMain",
                         *overlayLensLayout_);
        }
        if (RollsFaces()) {
            if (faceRowsLayout_ == nullptr) {
                faceRowsLayout_ = std::make_unique<PipelineLayout>(
                    context_, kFaceRowsBindings,
                    sizeof(SensorConstants));
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
    shadingVariant_ = ShadingVariantOf(params, stokes_);
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
    if (!params.bodies.empty()) {
        sky_ |= kSkyBodies;
    }
    // The planet's shadow table takes the size of the brightest
    // distant light, the sun's for Earth, for every light.
    const LightParams* brightest = BrightestDistant(params, bands_);
    atmosphere_->Set(params.atmosphere, bands_,
                     brightest != nullptr ? brightest->angle : 0.53f);
    WriteLightTiming(frame, params,
                     static_cast<std::uint32_t>(raster.Views()));
    WriteShading(raster);
    {
        // Each view's frustum culls its volumes, unless the views'
        // poses are on the GPU.
        std::vector<glm::mat4> viewProjections;
        if (!raster.posesOnGpu && !pushbroom_) {
            for (std::uint32_t v = 0; v < views_; ++v) {
                viewProjections.push_back(
                    raster.View(v).ViewProjection());
            }
        }
        // Each light shadow view's volumes cast as its geometry does.
        std::vector<std::int32_t> lightViewWorlds(
            lightViews_.size(), lightCasters_.world);
        for (std::size_t k = 0; k < layeredPasses_.size(); ++k) {
            lightViewWorlds[firstLayeredView_ + k] =
                layeredPasses_[k].casters.world;
        }
        // No sky's light, nor its steps, without an atmosphere.
        VolumeSettings volumeSettings = params.volumeSettings;
        if (!params.atmosphere) {
            volumeSettings.skyLightSteps = 0;
        }
        volumes_->Set(raster.volumes, raster.origin, bands_,
                      viewWorlds_, viewProjections, lightViewWorlds,
                      volumeSettings, params.frame, uploads_);
    }
    WriteCascadeJobs();
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

    shadowViews_ = lightViews_;
    for (RasterDrawable* drawable : drawables) {
        drawable->Prepare(raster, shadowViews_);
    }
    // Overlays posed under instances, once the meshes placed them.
    overlays_->PoseAnchors(params.origin);
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
        {"bindless", shaders::kBindlessSlang},
        {"sensor", shaders::kSensorSlang}};
    auto pipeline = std::make_unique<ComputePipeline>(
        context_, layout, compiler_.CompileSpirv(modules, entry));
    return *lensPipelines_.emplace(key, std::move(pipeline))
                .first->second;
}

const ComputePipeline&
RasterRenderer::ShadingPipeline(std::size_t model, bool pushbroom) {
    const auto key = std::tuple(model, pushbroom, shadingVariant_);
    if (const auto it = shadingPipelines_.find(key);
        it != shadingPipelines_.end()) {
        return *it->second;
    }
    std::vector<std::uint32_t> spirv;
    const BrdfModel& brdf = brdfs_.Models()[model];
    try {
        spirv = compiler_.CompileSpirv(
            ShadingModules(brdf, lightPatterns_, pushbroom,
                           shadingVariant_),
            "shadeMain");
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(
            "BRDF " + brdf.id +
            " failed to compile for shading: " + error.what());
    }
    auto pipeline = std::make_unique<ComputePipeline>(
        context_, shadingLayout_, spirv);
    return *shadingPipelines_.emplace(key, std::move(pipeline))
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
        entry.info[3] = std::bit_cast<std::uint32_t>(
            settings.polarizationRetention);
        entry.surface[0] = settings.speckleRoughness;
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
        // the equal-energy illuminant of 1 cd/m^2 per unit, over the
        // visible range the colour describes (360 to 830 nm) and
        // nothing outside it.
        const glm::vec3 emission = settings.emission;
        const float peak =
            std::max({emission.r, emission.g, emission.b});
        if (peak > 0.0f) {
            entry.info[2] = 1;
            const Spectrum emitted =
                RgbToSpectrum(glm::dvec3(emission / peak));
            const double perNit =
                1.0 / (kLuminousEfficacy * LuminanceIntegral());
            for (int b = 0; b < bands_.count; ++b) {
                const double low = std::max(bands_.Lower(b), 360.0);
                const double high =
                    std::min(bands_.Lower(b) + bands_.Width(), 830.0);
                const double visible =
                    high > low ? emitted.Average(low, high) *
                                     (high - low) / bands_.Width()
                               : 0.0;
                entry.emission[b] =
                    static_cast<float>(visible * peak * perNit);
            }
        }
        // Thermal emission, when the frame has it on: its emissivity,
        // or 1 minus its reflectance (Kirchhoff), times Planck's
        // radiance at its temperature or else the frame's default.
        const float temperature = settings.temperature > 0.0f
                                      ? settings.temperature
                                      : thermal_.defaultTemperature;
        if (thermal_.enable && temperature > 0.0f) {
            entry.info[2] = 1;
            const Spectrum& reflectance = material->Reflectance();
            const std::vector<float> thermal =
                ThermalEmission(bands_, temperature, [&](double nm) {
                    return settings.emissivity.Empty()
                               ? std::clamp(1.0 - reflectance(nm),
                                            0.0, 1.0)
                               : settings.emissivity(nm);
                });
            for (std::size_t b = 0; b < thermal.size(); ++b) {
                entry.emission[b] += thermal[b];
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
    // The scene's ambient light: white of its luminance, the same
    // spectral radiance in every band, which its colour of unit
    // luminance shapes.
    const glm::vec3 ambientColour =
        UnitLuminanceColor(params.ambientColor);
    shading.sun[3] = glm::length(ambientColour) > 0.0f
                         ? static_cast<float>(params.ambient /
                                              (kLuminousEfficacy *
                                               LuminanceIntegral()))
                         : 0.0f;
    shading.ambient[0] = ambientColour.r;
    shading.ambient[1] = ambientColour.g;
    shading.ambient[2] = ambientColour.b;
    shading.range[0] = static_cast<float>(bands_.minNm);
    shading.range[1] = static_cast<float>(bands_.Width());
    shading.range[2] = static_cast<float>(bands_.count);
    shading.range[3] = params.camera.orthographic ? 1.0f : 0.0f;
    shading.size[0] = extent_.width;
    shading.size[1] = extent_.height;
    shading.size[2] = kTableResolution;
    shading.size[3] =
        static_cast<std::uint32_t>(params.lights.size());
    lightCount_ = shading.size[3];
    // Unlit surfaces show their colour as a white one facing the
    // brightest distant light, or in relative units without one.
    const LightParams* brightest = BrightestDistant(params, bands_);
    for (int i = 0; i < bands_.count; ++i) {
        shading.irradiance[i] =
            brightest != nullptr &&
                    i < static_cast<int>(brightest->spectrum.size())
                ? brightest->spectrum[static_cast<std::size_t>(i)]
                : static_cast<float>(M_PI / (kLuminousEfficacy *
                                             LuminanceIntegral()));
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
        shading.atmosphere[1][3] =
            static_cast<float>(atmosphere_->Params().polarization);
    }
    // Bodies in kilometres likewise, the Moon's 384,000 km away.
    for (std::size_t i = 0;
         i < std::min(params.bodies.size(), FrameParams::kMaxBodies);
         ++i) {
        const BodyParams& body = params.bodies[i];
        const glm::dvec3 centre =
            (body.centre - params.origin) * 1e-3;
        shading.bodies[i][0] = static_cast<float>(centre.x);
        shading.bodies[i][1] = static_cast<float>(centre.y);
        shading.bodies[i][2] = static_cast<float>(centre.z);
        shading.bodies[i][3] = static_cast<float>(body.radius * 1e-3);
        shading.bodyAlbedo[i][0] = body.albedo.x;
        shading.bodyAlbedo[i][1] = body.albedo.y;
        shading.bodyAlbedo[i][2] = body.albedo.z;
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
    // The camera's aperture diameter (m), which sizes lasers'
    // speckle: its lens's f-number, or the exposure's for a pinhole.
    const double fNumber = params.camera.lensFStop > 0.0
                               ? params.camera.lensFStop
                               : params.camera.fStop;
    const auto aperture = std::bit_cast<std::uint32_t>(
        static_cast<float>(params.camera.focalLengthMm * 1e-3 /
                           std::max(fNumber, 1e-3)));
    // The uniforms every view shares, then each view's own block, in
    // one upload.
    std::vector<ShadingView> views(views_);
    for (std::uint32_t v = 0; v < views_; ++v) {
        ShadingView& block = views[v];
        // With the poses on the GPU, it writes the camera
        // (`RecordViewPoses`).
        if (!posesOnGpu_) {
            const CameraView view = params.View(v);
            if (params.camera.pushbroom) {
                // Its pixels' rays come from its strip
                // (projection_pushbroom.slang).
                const PushbroomStrip strip = params.Strip(v);
                CopyMatrix(block.invViewProj, strip.Packed());
                const glm::vec4 intrinsics = strip.Intrinsics();
                std::memcpy(block.strip, glm::value_ptr(intrinsics),
                            sizeof(block.strip));
            } else {
                CopyMatrix(block.invViewProj,
                           glm::inverse(view.ViewProjection()));
            }
            block.eye[0] = view.eye.x;
            block.eye[1] = view.eye.y;
            block.eye[2] = view.eye.z;
        }
        // The lights the view sees: the shared ones and its world's,
        // or all for every world.
        const std::int32_t world = params.ViewWorld(v);
        for (std::size_t k = 0; k < params.lights.size(); ++k) {
            const std::int32_t of = params.lights[k].world;
            if (of == -1 || world == -1 || of == world) {
                block.view[0] |= 1u << k;
            }
        }
        // Its world's slot among the distant lights' per-world maps.
        const auto slot = worldSlots_.find(world);
        block.view[1] = slot != worldSlots_.end() ? slot->second : 0;
        // Its sub-frame's lights, when they move between sub-frames
        // (`SubframeLights`).
        const std::size_t subframe = v / (outputViews_ * faces_);
        block.view[2] = static_cast<std::uint32_t>(
            lightSubframes_ ? subframe * kMaxLights : 0);
        block.view[3] = aperture;
    }
    std::vector<std::byte> bytes(kShadingShared +
                                 views.size() * kShadingStride);
    std::memcpy(bytes.data(), &shading, kShadingShared);
    std::memcpy(bytes.data() + kShadingShared, views.data(),
                views.size() * kShadingStride);
    uploads_.Write(shadingUniforms_, bytes);
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
    // Volumes write the background's without a sky (`VolumePass`).
    sensor.clear[3] = sky_ != 0            ? 1.0f
                      : subframes_ > 1     ? 2.0f
                      : volumes_->Active() ? 1.0f
                                           : 0.0f;
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
    // A sensor's own photographic response: each channel's spectral
    // response over its peak, taken as the CIE observer's luminous
    // efficiency (whose peak is 1), so a response shaped as the eye's
    // shows the luminance the eye sees, and one beyond the visible
    // (the near infrared, say) shows the light it sees, which the eye
    // would not. A monochrome sensor's outputs are grey, its
    // channel's response; a colour sensor's are its channels' as red,
    // green and blue when the eye sees less than half the light they
    // do (false colour), else the eye's colour.
    std::vector<std::vector<float>> responses;
    float peak = 0.0f;
    for (const SensorChannel& channel : model.channels) {
        responses.push_back(channel.quantumEfficiency.Bin(bands_));
        for (const float value : responses.back()) {
            peak = std::max(peak, value);
        }
    }
    const auto own = [&](std::size_t c, int i) {
        return peak > 0.0f
                   ? kLuminousEfficacy *
                         responses[c][static_cast<std::size_t>(i)] /
                         peak * bands_.Width()
                   : 0.0;
    };
    if (model.channels.size() == 1) {
        for (int i = 0; i < bands_.count; ++i) {
            colours[static_cast<std::size_t>(i)] =
                glm::dvec3(own(0, i));
        }
    } else if (model.channels.size() == 3) {
        double eye = 0.0;
        double channels = 0.0;
        for (int i = 0; i < bands_.count; ++i) {
            eye += kLuminousEfficacy *
                   weights[static_cast<std::size_t>(i)].y;
            channels += std::max({own(0, i), own(1, i), own(2, i)});
        }
        if (eye < 0.5 * channels) {
            for (int i = 0; i < bands_.count; ++i) {
                colours[static_cast<std::size_t>(i)] =
                    glm::dvec3(own(0, i), own(1, i), own(2, i));
            }
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
    // Distant lights for stray light, such as the sun: the first few,
    // not a dome's directions (its diffuse light); each irradiance as
    // the radiance whose image puts it on the focal plane, through
    // the camera equation (`CameraParams::IrradiancePerRadiance` on
    // axis).
    const double focal = camera.IrradiancePerRadiance({0.0, 0.0});
    std::vector<const LightParams*> sources;
    for (const LightParams& light : params.lights) {
        if (light.type == LightType::Distant && !light.domeDiffuse &&
            sources.size() < kStraySources) {
            sources.push_back(&light);
        }
    }
    for (std::size_t s = 0; s < sources.size(); ++s) {
        const std::vector<float>& spectrum = sources[s]->spectrum;
        for (int i = 0; i < bands_.count &&
                        i < static_cast<int>(spectrum.size()) &&
                        focal > 0.0 && std::isfinite(focal);
             ++i) {
            sensor.sourceRadiance[s][i] = static_cast<float>(
                static_cast<double>(
                    spectrum[static_cast<std::size_t>(i)]) /
                focal);
        }
        sensor.sources[s][3] = 1.0f;
    }
    // The polarizer, which reads the raster's Stokes planes.
    const Polarizer& polarizer = camera.polarizer;
    const bool polarized = polarizer.type != Polarizer::Type::None;
    sensor.polarizer[0] = static_cast<float>(polarizer.angle);
    sensor.polarizer[1] =
        polarizer.type == Polarizer::Type::Mosaic ? 1.0f : 0.0f;
    sensor.polarizer[2] =
        polarized ? static_cast<float>(polarizer.AlongAxis()) : 1.0f;
    sensor.polarizer[3] =
        polarized ? static_cast<float>(polarizer.AcrossAxis()) : 1.0f;
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
    // Sources in the world's frame; the sensor turns them into each
    // view's (`viewUniforms`).
    for (std::size_t s = 0; s < sources.size(); ++s) {
        const glm::vec3 direction = sources[s]->direction;
        const glm::vec3 unit = glm::length(direction) > 0.0f
                                   ? glm::normalize(direction)
                                   : direction;
        sensor.sources[s][0] = unit.x;
        sensor.sources[s][1] = unit.y;
        sensor.sources[s][2] = unit.z;
    }
    if (params.camera.pushbroom) {
        // Its tangential terms (lens.slang's Pushbroom); its k1 to k3
        // and line are each view's distortion.
        sensor.pushbroom[0] = static_cast<float>(lens.p1);
        sensor.pushbroom[1] = static_cast<float>(lens.p2);
    }
    // The sensor filters the output views, reading the G-buffer of
    // their last sub-frame (`params` holds a view per sub-frame and
    // output view): the shared uniforms, then every view's block, in
    // one upload. With the poses on the GPU, it writes the blocks'
    // cameras (`RecordViewPoses`).
    std::vector<SensorView> views(outputViews_);
    for (std::uint32_t v = 0; v < outputViews_; ++v) {
        SensorView& block = views[v];
        std::memcpy(block.distortion, sensor.distortion,
                    sizeof(block.distortion));
        // Every view of a batch has noise of its own; view 0's is the
        // single view's.
        block.seed[0] =
            FrameSeed(model.seed, params.stream ^ (v * 0x9E3779B9u),
                      params.frame);
        if (posesOnGpu_) {
            continue;
        }
        const std::uint32_t last =
            (subframes_ - 1) * outputViews_ + v;
        const CameraView view = params.View(last);
        if (params.camera.pushbroom) {
            // Its strip; its distortion, the line (lens.slang's
            // Pushbroom).
            const PushbroomStrip strip = params.Strip(last);
            CopyMatrix(block.invViewProj, strip.Packed());
            block.distortion[0] = static_cast<float>(lens.k1);
            block.distortion[1] = static_cast<float>(lens.k2);
            block.distortion[2] = static_cast<float>(lens.k3);
            block.distortion[3] = strip.line;
        } else {
            CopyMatrix(block.invViewProj,
                       glm::inverse(view.ViewProjection()));
        }
        CopyMatrix(block.pose, glm::inverse(view.view));
        block.eye[0] = view.eye.x;
        block.eye[1] = view.eye.y;
        block.eye[2] = view.eye.z;
        block.forward[0] = view.forward.x;
        block.forward[1] = view.forward.y;
        block.forward[2] = view.forward.z;
    }
    std::vector<std::byte> bytes(kSensorShared +
                                 views.size() * kSensorStride);
    std::memcpy(bytes.data(), &sensor, kSensorShared);
    std::memcpy(bytes.data() + kSensorShared, views.data(),
                views.size() * kSensorStride);
    uploads_.Write(sensorUniforms_, bytes);
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
        GpuBuffer(context_, kShadingShared + views * kShadingStride,
                  kViewUniformUsage, BufferMemory::Device);
    sensorUniforms_ =
        GpuBuffer(context_, kSensorShared + views * kSensorStride,
                  kViewUniformUsage, BufferMemory::Device);
}

RasterFrame RasterRenderer::Frame(std::uint32_t view) const {
    // The light shadow maps, which minimal shading reads; shadow
    // passes, which draw into them, bind a placeholder instead
    // (`RecordShadowMaps`).
    return {DescriptorResource::Of(uniforms_),
            view,
            DescriptorResource::Of(shadowUniforms_),
            DescriptorResource::Of(lightShadowMaps_->View()),
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
    // lens, onto LdrColor when it shows that, else resampled through
    // the lens and blended over the presented image.
    const bool lensOverlays = faces_ > 1 && present != nullptr &&
                              var == render_vars::kLdrColor;
    RecordViews(commands, targets, drawables, 0,
                present == nullptr || lensOverlays);
    if (present != nullptr) {
        {
            ERNIE_GPU_ZONE(context_, commands, "Present");
            Present(commands, targets, *present, var);
        }
        if (!overlays_->Empty() && faces_ > 1 && !lensOverlays) {
            ERNIE_GPU_ZONE(context_, commands, "Overlays");
            targets.Depth().Transition(
                commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            PresentFaceOverlays(commands, targets, *present);
        } else if (!overlays_->Empty() && faces_ == 1) {
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
    // GPU zones: the frame's setup, then its shadows, as the
    // profiler's tree shows them.
    {
        ERNIE_GPU_ZONE(context_, commands, "Setup");
        {
            ERNIE_GPU_ZONE(context_, commands, "Uploads");
            // The frame's host data first, after every earlier
            // frame's use of the buffers it fills.
            uploads_.Record(commands);
        }
        // The atmosphere's tables, when it or the bands changed,
        // before any view shades with them.
        atmosphere_->Record(commands);
        if (blendGpuLights_) {
            // The last frame's poses of the lights posed on the GPU,
            // which the sub-frames' light passes blend from, before
            // the prepass writes this frame's over them.
            VkMemoryBarrier2 before{
                VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
            before.srcStageMask =
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            before.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT |
                                   VK_ACCESS_2_SHADER_WRITE_BIT;
            before.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            before.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT |
                                   VK_ACCESS_2_TRANSFER_WRITE_BIT;
            VkDependencyInfo dependency{
                VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.memoryBarrierCount = 1;
            dependency.pMemoryBarriers = &before;
            vkCmdPipelineBarrier2(commands, &dependency);
            const VkBufferCopy copy{
                0, 0, VkDeviceSize{gpuLights_} * sizeof(glm::mat4)};
            vkCmdCopyBuffer(commands, lightPoses_->Handle(),
                            lightPosesPrevious_->Handle(), 1, &copy);
            VkMemoryBarrier2 after{
                VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
            after.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            after.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT |
                                  VK_ACCESS_2_TRANSFER_WRITE_BIT;
            after.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            after.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT |
                                  VK_ACCESS_2_SHADER_WRITE_BIT;
            dependency.pMemoryBarriers = &after;
            vkCmdPipelineBarrier2(commands, &dependency);
        }
        {
            ERNIE_GPU_ZONE(context_, commands, "Prepass");
            // Poses composed on the GPU, and views posed under them,
            // before anything reads them.
            for (const RasterDrawable* drawable : drawables) {
                ERNIE_GPU_ZONE_NAMED(context_, commands,
                                     drawable->ProfileName());
                drawable->RecordPrepass(commands);
            }
            // The pattern arrays' mip levels, then the light
            // patterns' integrals, which shading reads.
            RecordPatternMips(commands);
            if (!patternMipRegions_.empty()) {
                ShaderBarrier(commands);
            }
            RecordPatternNorms(commands);
            ShaderBarrier(commands);
        }
        RecordLights(commands, drawables);
        if (posesOnGpu_) {
            ERNIE_GPU_ZONE(context_, commands, "View poses");
            RecordViewPoses(commands, drawables);
        }
        RecordExposure(commands);
        {
            ERNIE_GPU_ZONE(context_, commands, "Cull");
            // Culling and other compute the draws read, once for
            // every view and shadow view.
            for (const RasterDrawable* drawable : drawables) {
                ERNIE_GPU_ZONE_NAMED(context_, commands,
                                     drawable->ProfileName());
                drawable->RecordCull(commands);
            }
            ShaderBarrier(commands);
        }
    }
    ERNIE_GPU_ZONE(context_, commands, "Shadows");
    // The first sub-frame's (the frame's own without motion blur);
    // `RecordViews` the others'.
    RecordShadowMaps(commands, drawables, Frame(0), 0);
    // The volumes' shadow maps, the light views now known.
    volumes_->Prepare(commands, lightTable_, *lightViewMatrices_,
                      lightCount_, lightPatterns_, *patternData_,
                      static_cast<std::uint32_t>(lightViews_.size()));
    {
        ERNIE_GPU_ZONE(context_, commands, "After shadows");
        // What reads the shadow maps before the views draw, such as
        // the terrain's horizon over what stands on it.
        for (const RasterDrawable* drawable : drawables) {
            ERNIE_GPU_ZONE_NAMED(context_, commands,
                                 drawable->ProfileName());
            drawable->RecordAfterShadows(commands);
        }
        ShaderBarrier(commands);
        castersChanged_ = false;
    }
}

void RasterRenderer::RecordShadowMaps(
    VkCommandBuffer commands,
    std::span<const RasterDrawable* const> drawables,
    const RasterFrame& given, std::uint32_t subframe) const {
    // Drawing into the light shadow maps, the passes bind a texel in
    // their place.
    RasterFrame frame = given;
    frame.shadowMap = DescriptorResource::Of(shadowMaps_->View());
    // Moving lights' views at the sub-frame's time, written over the
    // frame's where the last sub-frame's draws and shading have read
    // them.
    std::vector<glm::mat4> matrices;
    for (const ShadowCascade& view : lightViews_) {
        matrices.push_back(view.lightViewProjection);
    }
    // Lights posed on the GPU, posed at the sub-frame's time into its
    // copy of the table and over their views' matrices, once the last
    // sub-frame's draws and shading have read them.
    if (lightSubframes_ && gpuLights_ > 0 &&
        subframe < subframeShares_.size()) {
        ShaderBarrier(commands);
        DispatchLights(commands, subframe * kMaxLights,
                       subframeShares_[subframe], blendGpuLights_);
    }
    const bool hostMoved = subframe < subframeLightViews_.size() &&
                           !subframeLightViews_[subframe].empty();
    const bool gpuMoved = subframe < subframeCopies_.size() &&
                          !subframeCopies_[subframe].empty();
    if (hostMoved || gpuMoved) {
        VkMemoryBarrier2 before{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        before.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        // After the reads of the views, and the cascade fit's writes.
        before.srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT |
                               VK_ACCESS_2_UNIFORM_READ_BIT |
                               VK_ACCESS_2_SHADER_WRITE_BIT;
        before.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        before.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT |
                               VK_ACCESS_2_TRANSFER_READ_BIT;
        VkDependencyInfo dependency{
            VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &before;
        vkCmdPipelineBarrier2(commands, &dependency);
        for (std::size_t i = 0;
             hostMoved && i < subframeLightViews_[subframe].size();
             ++i) {
            const auto& [v, matrix] =
                subframeLightViews_[subframe][i];
            matrices[v] = matrix;
            vkCmdUpdateBuffer(commands, lightViewMatrices_->Handle(),
                              v * sizeof(glm::mat4),
                              sizeof(glm::mat4),
                              glm::value_ptr(matrix));
            if (v < firstLayeredView_) {
                vkCmdUpdateBuffer(
                    commands, lightShadowBlocks_.Handle(),
                    v * kLightShadowBlockStride, sizeof(glm::mat4),
                    glm::value_ptr(matrix));
            }
        }
        // Cascades the GPU fitted at this sub-frame
        // (`RecordCascadeFit`).
        for (std::size_t i = 0;
             gpuMoved && i < subframeCopies_[subframe].size(); ++i) {
            const auto& [from, v] = subframeCopies_[subframe][i];
            const VkBufferCopy matrix{from * sizeof(glm::mat4),
                                      v * sizeof(glm::mat4),
                                      sizeof(glm::mat4)};
            vkCmdCopyBuffer(commands, subframeFits_->Handle(),
                            lightViewMatrices_->Handle(), 1, &matrix);
            if (v < firstLayeredView_) {
                const VkBufferCopy block{from * sizeof(glm::mat4),
                                         v * kLightShadowBlockStride,
                                         sizeof(glm::mat4)};
                vkCmdCopyBuffer(commands, subframeFits_->Handle(),
                                lightShadowBlocks_.Handle(), 1,
                                &block);
            }
        }
        VkMemoryBarrier2 after{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        after.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        after.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        after.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        after.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT |
                              VK_ACCESS_2_UNIFORM_READ_BIT;
        dependency.pMemoryBarriers = &after;
        vkCmdPipelineBarrier2(commands, &dependency);
    }
    if (!lightViews_.empty()) {
        ERNIE_GPU_ZONE(context_, commands, "Light shadows");
        lightShadowMaps_->Transition(
            commands, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
        for (std::size_t v = 0; v < firstLayeredView_; ++v) {
            if (!lightViews_[v].drawn) {
                continue;
            }
            RenderingTarget shadow;
            shadow.depth = lightShadowMaps_->LayerView(
                static_cast<std::uint32_t>(v));
            // In its corner of the layer, as wide as its own map.
            const std::uint32_t texels =
                lightViews_[v].resolution > 0
                    ? lightViews_[v].resolution
                    : lightResolution_;
            shadow.extent = {texels, texels};
            BeginRendering(commands, shadow);
            // Each view draws with its own Shadows block's matrix,
            // which the light pass fits for lights posed on the GPU.
            RasterFrame own = frame;
            own.shadows = DescriptorResource::Of(
                lightShadowBlocks_, v * kLightShadowBlockStride,
                sizeof(ShadowUniforms));
            own.shadowFromFrame = true;
            // Only the geometry its light's shadow link holds.
            ShadowCasters casters = lightCasters_;
            casters.linkGroups = lightViews_[v].shadowMask;
            for (const RasterDrawable* drawable : drawables) {
                ERNIE_GPU_ZONE_NAMED(context_, commands,
                                     drawable->ProfileName());
                drawable->RecordShadow(commands, own, v, matrices[v],
                                       casters);
            }
            vkCmdEndRendering(commands);
        }
        RecordLayeredLightShadows(commands, frame, drawables);
        lightShadowMaps_->Transition(
            commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
}

bool RasterRenderer::SupportsVirtualShadows() const {
    const VkPhysicalDeviceLimits& limits = context_.Limits();
    const std::uint32_t texels = VirtualShadowMap::kVirtualTexels;
    return context_.SupportsFragmentAtomics() &&
           limits.maxViewportDimensions[0] >= texels &&
           limits.maxViewportDimensions[1] >= texels &&
           limits.maxFramebufferWidth >= texels &&
           limits.maxFramebufferHeight >= texels;
}

void RasterRenderer::RecordVirtualShadows(
    VkCommandBuffer commands, RenderTargets& targets,
    std::span<const RasterDrawable* const> drawables,
    const RasterFrame& given) const {
    if (virtualLight_ < 0 || virtualShadow_ == nullptr) {
        return;
    }
    ERNIE_GPU_ZONE(context_, commands, "Virtual shadows");
    // The pages the first view's points need, from its depth.
    GpuImage& depth = targets.Depth();
    depth.Transition(commands,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    virtualShadow_->RecordMark(
        commands, depth.LayerView(0), targets.Extent().width,
        targets.Extent().height, virtualInverse_, virtualEye_);
    virtualShadow_->RecordAllocate(commands);
    virtualShadow_->RecordClear(commands);
    ShaderBarrier(commands);
    // Each level drawn into no attachment over its 16384² virtual
    // map, the casters' fragments writing through its pages.
    RenderingTarget target;
    target.extent = {VirtualShadowMap::kVirtualTexels,
                     VirtualShadowMap::kVirtualTexels};
    ShadowCasters casters = lightCasters_;
    for (std::size_t v = firstVirtualView_; v < lightViews_.size();
         ++v) {
        BeginRendering(commands, target);
        RasterFrame own = given;
        own.shadows = DescriptorResource::Of(
            lightShadowBlocks_, v * kLightShadowBlockStride,
            sizeof(ShadowUniforms));
        own.shadowFromFrame = false;
        own.shadowLayer = 0;
        own.virtualShadow = true;
        casters.linkGroups = lightViews_[v].shadowMask;
        for (const RasterDrawable* drawable : drawables) {
            if (!drawable->DrawsVirtualShadows()) {
                continue;
            }
            ERNIE_GPU_ZONE_NAMED(context_, commands,
                                 drawable->ProfileName());
            drawable->RecordShadow(commands, own, v,
                                   lightViews_[v].lightViewProjection,
                                   casters);
        }
        vkCmdEndRendering(commands);
    }
    ShaderBarrier(commands);
}

void RasterRenderer::RecordLayeredLightShadows(
    VkCommandBuffer commands, const RasterFrame& frame,
    std::span<const RasterDrawable* const> drawables) const {
    if (layeredPasses_.empty()) {
        return;
    }
    // As the batch's sun maps: one layered pass for the passes of a
    // kind, each drawing into its layer (`RasterFrame::shadowLayer`),
    // the worlds' copies of their light's shared map after the rest.
    const std::uint32_t texels =
        lightViews_[firstLayeredView_].resolution;
    const auto record = [&](bool fromBase) {
        RenderingTarget shadow;
        shadow.depth = lightShadowMaps_->View();
        shadow.extent = {texels, texels};
        shadow.layers = lightShadowMaps_->Desc().layers;
        shadow.loadDepth = fromBase;
        BeginRendering(commands, shadow);
        for (std::size_t p = 0; p < layeredPasses_.size(); ++p) {
            const LayeredPass& pass = layeredPasses_[p];
            if (pass.fromBase != fromBase) {
                continue;
            }
            const std::size_t v = firstLayeredView_ + p;
            RasterFrame layered = frame;
            layered.shadowLayer = static_cast<std::uint32_t>(v);
            ShadowCasters casters = pass.casters;
            casters.linkGroups = lightViews_[v].shadowMask;
            for (const RasterDrawable* drawable : drawables) {
                drawable->RecordShadow(
                    commands, layered, v,
                    lightViews_[v].lightViewProjection, casters);
            }
        }
        vkCmdEndRendering(commands);
    };
    {
        ERNIE_GPU_ZONE(context_, commands, "Layered light maps");
        record(false);
    }
    std::vector<VkImageCopy> copies;
    for (std::size_t p = 0; p < layeredPasses_.size(); ++p) {
        if (layeredPasses_[p].fromBase) {
            VkImageCopy copy{};
            copy.srcSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0,
                                   layeredPasses_[p].base, 1};
            copy.dstSubresource = {
                VK_IMAGE_ASPECT_DEPTH_BIT, 0,
                static_cast<std::uint32_t>(firstLayeredView_ + p), 1};
            copy.extent = {texels, texels, 1};
            copies.push_back(copy);
        }
    }
    if (!copies.empty()) {
        {
            ERNIE_GPU_ZONE(context_, commands, "Layered map copies");
            lightShadowMaps_->Transition(commands,
                                         VK_IMAGE_LAYOUT_GENERAL);
            vkCmdCopyImage(commands, lightShadowMaps_->Handle(),
                           VK_IMAGE_LAYOUT_GENERAL,
                           lightShadowMaps_->Handle(),
                           VK_IMAGE_LAYOUT_GENERAL,
                           static_cast<std::uint32_t>(copies.size()),
                           copies.data());
            lightShadowMaps_->Transition(
                commands, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);
        }
        ERNIE_GPU_ZONE(context_, commands,
                       "Layered maps over copies");
        record(true);
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
    // A polarizer's sensor reads the radiance's Stokes planes.
    targets.SetStokes(stokes_);
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
    ERNIE_GPU_ZONE(context_, commands, "Views");
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
                             Frame(subframe * outputViews_ * faces_),
                             subframe);
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
                ERNIE_GPU_ZONE_NAMED(context_, commands,
                                     drawable->ProfileName());
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
                    ERNIE_GPU_ZONE_NAMED(context_, commands,
                                         drawable->ProfileName());
                    drawable->RecordOcclusionCull(
                        commands, targets.Hzb(), first, layers);
                }
                ShaderBarrier(commands);
            }
            ERNIE_GPU_ZONE(context_, commands, "G-buffer occludees");
            draw(DrawPhase::Occludees, true);
        }
        if (subframe == 0 && firstView == 0) {
            RecordVirtualShadows(commands, targets, drawables,
                                 Frame(first));
        }
        Shade(commands, targets, first);
        if (volumes_->Active()) {
            // Shading's resources as Shade wrote them just before.
            VolumePass::Shading shading;
            shading.resources = shadingResources_.get();
            shading.patternData = patternData_.get();
            shading.patterns = &lightPatterns_;
            volumes_->Record(commands, targets, shading,
                             VolumeLayout(), first, extent_.width,
                             extent_.height, pushbroom_, sky_ != 0);
            ShaderBarrier(commands);
        }
        if (subframes_ > 1) {
            Accumulate(commands, targets, subframe);
        }
    }
    // The MotionVector var's passes, over the last sub-frame's
    // G-buffer, only when it is held.
    if (motionVectors_) {
        RecordMotion(commands, targets, drawables, firstView);
    }
    // The vars the sensor writes linear too: all the frame holds so
    // but LdrColor when overlays draw over it afterwards.
    if (RenderTargets::LinearVars* linear = targets.GetLinearVars()) {
        const auto ldr = static_cast<std::uint32_t>(
            std::find(RenderTargets::VarNames().begin(),
                      RenderTargets::VarNames().end(),
                      render_vars::kLdrColor) -
            RenderTargets::VarNames().begin());
        linear->written = linear->held & ~(overlays ? 1u << ldr : 0u);
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
    ERNIE_GPU_ZONE(context_, commands, "Minimal");
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
                    ERNIE_GPU_ZONE_NAMED(context_, commands,
                                         drawable->ProfileName());
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
        resources.push_back(DescriptorResource::Of(
            domeMap_->View(), domeMap_->Layout()));
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                          resolvePipeline_->Handle());
        resolveLayout_.Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                            resources);
        ResolveConstants constants{
            {extent_.width, extent_.height, views, first},
            {holds, static_cast<std::uint32_t>(kUniformsStride),
             static_cast<std::uint32_t>(kShadingStride),
             static_cast<std::uint32_t>(kShadingShared)},
            {},
            {minimalSky_, hasDomeMap_ ? 1.0f : 0.0f,
             static_cast<float>(domeMap_->Desc().width),
             static_cast<float>(domeMap_->Desc().height)}};
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                constants.toDome[r][c] = minimalToDome_[c][r];
            }
        }
        resolveLayout_.PushConstants(commands, &constants,
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
    const GpuBuffer& range = *depthRangeSums_;
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
    // The next frame's cascades fit to it on the GPU
    // (`RecordCascadeFit`).
    ShaderBarrier(commands);
}

void RasterRenderer::RecordMeter(VkCommandBuffer commands,
                                 RenderTargets& targets) const {
    GpuImage* hdr = targets.Var(render_vars::kHdrColor);
    if (hdr == nullptr) {
        return;
    }
    ERNIE_GPU_ZONE(context_, commands, "Exposure meter");
    const GpuBuffer& bins = *meterSums_;
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
    MeterConstants constants{
        {extent_.width, extent_.height, 0, 0},
        {kMeterLow, kMeterBinsPerStop, 0.0f, 0.0f}};
    meterLayout_.PushConstants(commands, &constants,
                               sizeof(constants));
    vkCmdDispatch(commands, (extent_.width + 15) / 16,
                  (extent_.height + 15) / 16, 1);
    // The automatic exposure adapts to it on the GPU, for the next
    // frame of its key (`RecordExposure`).
    if (exposing_) {
        ShaderBarrier(commands);
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                          exposureAdaptPipeline_->Handle());
        constants.range[2] = exposing_->blend;
        constants.range[3] = exposing_->scale;
        constants.histogram = bins.Address();
        constants.states = exposureStates_.Address();
        constants.layout[0] = exposing_->slot;
        exposureLayout_.PushConstants(commands, &constants,
                                      sizeof(constants));
        vkCmdDispatch(commands, 1, 1, 1);
    }
    // Read back only to show the metered luminance.
    CopyForHost(commands, bins, *meterBins_[uploads_.Slot()]);
    hdr->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
}

void RasterRenderer::RecordExposure(VkCommandBuffer commands) const {
    if (!exposing_) {
        return;
    }
    ERNIE_GPU_ZONE(context_, commands, "Exposure");
    // After the frame's uploads (and views posed on the GPU) wrote
    // the exposures it scales.
    ShaderBarrier(commands);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      exposureApplyPipeline_->Handle());
    MeterConstants constants{};
    constants.states = exposureStates_.Address();
    constants.lens = sensorUniforms_.Address() +
                     offsetof(SensorUniforms, lens) +
                     2 * sizeof(float);
    constants.eyes = uniforms_.Address() + offsetof(Uniforms, eye) +
                     3 * sizeof(float);
    constants.gain = exposureGain_.Address();
    constants.layout[0] = exposing_->slot;
    constants.layout[1] = exposureViews_;
    constants.layout[2] =
        static_cast<std::uint32_t>(kUniformsStride / sizeof(float));
    exposureLayout_.PushConstants(commands, &constants,
                                  sizeof(constants));
    vkCmdDispatch(commands, std::max(1u, (exposureViews_ + 63) / 64),
                  1, 1);
    ShaderBarrier(commands);
}

void RasterRenderer::ReadFinishedMeters() {
    // Oldest first, so the latest frame's reading is the one kept.
    for (std::uint32_t k = 1; k <= FrameUploads::kSlots; ++k) {
        ReadMeter((uploads_.Slot() + k) % FrameUploads::kSlots);
    }
}

void RasterRenderer::ReadMeter(std::uint32_t slot) {
    if (!metered_[slot]) {
        return;
    }
    metered_[slot] = false;
    const std::string key = std::move(meterKeys_[slot]);
    meterKeys_[slot].clear();
    std::array<std::uint32_t, kMeterBins> bins{};
    meterBins_[slot]->Read(std::as_writable_bytes(std::span(bins)));
    std::uint64_t total = 0;
    for (const std::uint32_t count : bins) {
        total += count;
    }
    if (total == 0) {
        meteredLuminance_.reset();
        meteredByKey_.erase(key);
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
    if (meteredLuminance_) {
        meteredByKey_[key] = *meteredLuminance_;
    } else {
        meteredByKey_.erase(key);
    }
}

std::vector<ShadowCascade>
RasterRenderer::FitViewCascades(const glm::vec3& towards,
                                std::uint32_t resolution) const {
    const CascadeFit& fit = *cascadeFit_;
    const CameraView& view = fit.view;
    const auto [low, high] =
        WithShadows(fit.casterLow, fit.casterHigh, towards);
    // How far the casters reach along the view; a cube camera's faces
    // look every way.
    float farthest = 0.0f;
    for (int i = 0; i < 8; ++i) {
        const glm::vec3 corner((i & 1) ? high.x : low.x,
                               (i & 2) ? high.y : low.y,
                               (i & 4) ? high.z : low.z);
        farthest =
            std::max(farthest,
                     fit.around > 0.0f
                         ? glm::length(corner - view.eye)
                         : glm::dot(corner - view.eye, view.forward));
    }
    // A margin keeps surfaces at exactly the farthest depth, such as
    // flat ground under a nadir camera, inside the last cascade. Over
    // every depth the casters reach: the GPU fits them to those the
    // view saw (`AddCascadeJob`).
    const float shadowDistance =
        std::min(view.farPlane, 1.01f * farthest);
    return FitCascades(view, shadowDistance, towards, low, high,
                       fit.count, fit.lambda, resolution,
                       view.nearPlane, fit.around,
                       context_.SupportsDepthClamp());
}

void RasterRenderer::AddCascadeJob(const FrameParams& params,
                                   const glm::vec3& towards,
                                   std::uint32_t resolution,
                                   std::uint32_t first,
                                   bool subframe) {
    const CascadeFit& fit = *cascadeFit_;
    const CameraView& view = fit.view;
    const auto [low, high] =
        WithShadows(fit.casterLow, fit.casterHigh, towards);
    float farthest = 0.0f;
    for (int i = 0; i < 8; ++i) {
        const glm::vec3 corner((i & 1) ? high.x : low.x,
                               (i & 2) ? high.y : low.y,
                               (i & 4) ? high.z : low.z);
        farthest =
            std::max(farthest,
                     fit.around > 0.0f
                         ? glm::length(corner - view.eye)
                         : glm::dot(corner - view.eye, view.forward));
    }
    // The light's view and the scene's depths along it, as
    // `FitCascades` takes them.
    const glm::vec3 up = std::fabs(towards.z) > 0.99f
                             ? glm::vec3(0.0f, 1.0f, 0.0f)
                             : glm::vec3(0.0f, 0.0f, 1.0f);
    const glm::mat4 lightView =
        glm::lookAtRH(towards, glm::vec3(0.0f), up);
    float sceneNear = std::numeric_limits<float>::max();
    float sceneFar = std::numeric_limits<float>::lowest();
    for (int i = 0; i < 8; ++i) {
        const glm::vec3 corner((i & 1) ? high.x : low.x,
                               (i & 2) ? high.y : low.y,
                               (i & 4) ? high.z : low.z);
        const float z = (lightView * glm::vec4(corner, 1.0f)).z;
        sceneNear = std::min(sceneNear, -z);
        sceneFar = std::max(sceneFar, -z);
    }
    const float padding = 0.01f * (sceneFar - sceneNear) + 1e-3f;
    CascadeJob job{};
    std::memcpy(job.lightView, glm::value_ptr(lightView), 64);
    const glm::mat4 cameraToWorld = glm::inverse(view.view);
    std::memcpy(job.cameraToWorld, glm::value_ptr(cameraToWorld), 64);
    const glm::mat4 toView = glm::inverse(view.projection);
    std::memcpy(job.toView, glm::value_ptr(toView), 64);
    const float tanY = std::tan(0.5f * view.verticalFov);
    job.camera[0] = tanY * view.aspect;
    job.camera[1] = tanY;
    job.camera[2] = view.nearPlane;
    job.camera[3] = fit.around;
    job.fit[0] = std::min(view.farPlane, 1.01f * farthest);
    job.fit[1] = fit.lambda;
    job.fit[2] = static_cast<float>(resolution);
    job.fit[3] = static_cast<float>(fit.count);
    job.scene[0] = sceneNear - padding;
    job.scene[1] = sceneFar + padding;
    job.scene[2] = context_.SupportsDepthClamp() ? 1.0f : 0.0f;
    job.scene[3] = view.IsOrthographic() ? 1.0f : 0.0f;
    // The depths the last measured frame saw, when it was of this
    // stream and its view close to this one; the GPU checks how far
    // the eye moved against the nearest of them.
    const glm::dvec3 eye(params.camera.transform[3]);
    if (fit.visible && lastMeasured_ &&
        lastMeasured_->stream == params.stream &&
        glm::dot(params.View().forward, lastMeasured_->forward) >=
            kVisibleFitTurn) {
        job.seen[0] = 1.0f;
        job.seen[1] =
            static_cast<float>(glm::length(eye - lastMeasured_->eye));
        job.seen[2] = static_cast<float>(kVisibleFitEyeShift);
    }
    job.out[0] = first;
    // Cascades precede the layered views: each has its Shadows block.
    // A sub-frame's are copied into them (`RecordShadowMaps`).
    job.out[1] =
        subframe ? 0u : std::numeric_limits<std::uint32_t>::max();
    // The first cascaded light's splits are the frame's, which the
    // G-buffer passes tint by.
    job.out[2] = subframe ? 2u : 0u;
    if (!subframe && !splitsFitted_) {
        job.out[2] |= 1u;
        splitsFitted_ = true;
    }
    job.out[3] = 0u;
    const auto* bytes = reinterpret_cast<const std::byte*>(&job);
    cascadeJobs_.insert(cascadeJobs_.end(), bytes,
                        bytes + sizeof(job));
    ++cascadeJobCount_;
}

VkDeviceSize RasterRenderer::ViewUniformsStride() {
    return kUniformsStride;
}

void RasterRenderer::WriteCascadeJobs() {
    if (cascadeJobCount_ == 0) {
        return;
    }
    // Volumes write no depth for the views to see: their bounds'
    // depths along the camera's axis, or from its eye for a cube
    // camera, widen those seen.
    float volumeNear = 0.0f;
    float volumeFar = 0.0f;
    if (const std::optional<Bounds3>& box = volumes_->Bounds();
        box && cascadeFit_) {
        const CameraView& view = cascadeFit_->view;
        volumeNear = std::numeric_limits<float>::max();
        for (int i = 0; i < 8; ++i) {
            const glm::vec3 corner(
                (i & 1) ? box->second.x : box->first.x,
                (i & 2) ? box->second.y : box->first.y,
                (i & 4) ? box->second.z : box->first.z);
            const float depth =
                cascadeFit_->around > 0.0f
                    ? glm::length(corner - view.eye)
                    : glm::dot(corner - view.eye, view.forward);
            volumeNear = std::min(volumeNear, depth);
            volumeFar = std::max(volumeFar, depth);
        }
        if (cascadeFit_->around > 0.0f &&
            glm::all(glm::greaterThanEqual(view.eye, box->first)) &&
            glm::all(glm::lessThanEqual(view.eye, box->second))) {
            volumeNear = 0.0f;
        }
    }
    for (std::uint32_t j = 0; j < cascadeJobCount_; ++j) {
        auto* job = reinterpret_cast<CascadeJob*>(
            cascadeJobs_.data() + j * sizeof(CascadeJob));
        job->volumes[0] = volumeNear;
        job->volumes[1] = volumeFar;
    }
    if (!cascadeJobBuffer_ ||
        cascadeJobBuffer_->Size() < cascadeJobs_.size()) {
        vkDeviceWaitIdle(context_.Device());
        cascadeJobBuffer_ = std::make_unique<GpuBuffer>(
            context_, cascadeJobs_.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferMemory::Device);
    }
    uploads_.Write(*cascadeJobBuffer_, cascadeJobs_);
}

std::array<float, 4> RasterRenderer::ShadowSplits() const {
    ShadowUniforms uniforms{};
    shadowUniforms_.Read(
        std::as_writable_bytes(std::span(&uniforms, 1)));
    return {uniforms.splits[0], uniforms.splits[1],
            uniforms.splits[2], uniforms.splits[3]};
}

void RasterRenderer::RecordCascadeFit(
    VkCommandBuffer commands) const {
    if (cascadeJobCount_ == 0) {
        return;
    }
    ERNIE_GPU_ZONE(context_, commands, "Cascade fit");
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                      cascadeFitPipeline_->Handle());
    DepthRangeConstants constants{};
    constants.jobs = cascadeJobBuffer_->Address();
    constants.seen = depthRangeSums_->Address();
    constants.matrices = lightViewMatrices_->Address();
    constants.blocks = lightShadowBlocks_.Address();
    constants.frame = shadowUniforms_.Address();
    if (subframeFitCount_ > 0) {
        constants.subframes = subframeFits_->Address();
    }
    constants.layout[0] = cascadeJobCount_;
    constants.layout[1] = static_cast<std::uint32_t>(
        kLightShadowBlockStride / sizeof(glm::vec4));
    depthRangeLayout_.PushConstants(commands, &constants,
                                    sizeof(constants));
    vkCmdDispatch(commands, (cascadeJobCount_ + 63) / 64, 1, 1);
    ShaderBarrier(commands);
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
        // Screen items over the image the lens made.
        if (!overlays_->HasScreenItems()) {
            return;
        }
    }
    color->Transition(commands,
                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    for (std::uint32_t v = 0; v < targets.Views(); ++v) {
        OverlayPass::Target target = OverlayTarget(
            targets, LastSubframeView(firstView + v), v);
        target.color = color->LayerView(v);
        target.format = VK_FORMAT_R8G8B8A8_UNORM;
        overlays_->Record(commands, target,
                          faces_ > 1 ? OverlayPass::Layer::Screen
                                     : OverlayPass::Layer::All);
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
        DescriptorResource::Of(faceRows_->View(),
                               VK_IMAGE_LAYOUT_GENERAL)};
    const SensorConstants constants{
        {firstView, 0u, 0u, 0u},
        {sensorPasses_.front().tableOffset, lensTable_[1],
         lensTable_[2], lensTable_[3]},
        {},
        sensorPasses_.front().params->Address(),
        0u,
        0u};
    // Cleared, then each sensor pixel's row scattered to the face
    // texels its tap reads.
    for (const char* entry : {"faceRowsClearMain", "faceRowsMain"}) {
        const ComputePipeline& pipeline =
            *lensPipelines_.at(entry + ("\n" + faceLens_));
        vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipeline.Handle());
        faceRowsLayout_->Push(
            commands, VK_PIPELINE_BIND_POINT_COMPUTE, resources);
        faceRowsLayout_->PushConstants(commands, &constants,
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
                                        GpuImage& color,
                                        bool store) const {
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
            overlays_->Record(commands, target,
                              OverlayPass::Layer::World);
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
        DescriptorResource::Of(
            faceOverlays_[0]->View(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        DescriptorResource::Of(
            faceOverlays_[1]->View(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)};
    overlayLensLayout_->Push(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                             resources);
    const SensorConstants constants{
        {firstView, store ? 1u : 0u, 0u, 0u},
        {sensorPasses_.front().tableOffset, lensTable_[1],
         lensTable_[2], lensTable_[3]},
        {},
        sensorPasses_.front().params->Address(),
        0u,
        0u};
    overlayLensLayout_->PushConstants(commands, &constants,
                                      sizeof(constants));
    vkCmdDispatch(commands, Groups(extent_.width),
                  Groups(extent_.height),
                  store ? 1u : targets.Views());
    ShaderBarrier(commands);
}

void RasterRenderer::PresentFaceOverlays(
    VkCommandBuffer commands, RenderTargets& targets,
    const RenderingTarget& present) const {
    if (lensOverlays_ == nullptr ||
        lensOverlays_->Desc().width != extent_.width ||
        lensOverlays_->Desc().height != extent_.height) {
        vkDeviceWaitIdle(context_.Device());
        lensOverlays_ = std::make_unique<GpuImage>(
            context_, ImageDesc{extent_.width, extent_.height, 1, 1,
                                VK_FORMAT_R8G8B8A8_UNORM,
                                VK_IMAGE_USAGE_STORAGE_BIT |
                                    VK_IMAGE_USAGE_SAMPLED_BIT,
                                false, true});
    }
    RecordFaceOverlays(commands, targets, 0, *lensOverlays_, true);
    lensOverlays_->Transition(
        commands, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    // Blended over what the present pass drew, as the overlay pass
    // draws a frame camera's.
    RenderingTarget target = present;
    target.depth = VK_NULL_HANDLE;
    target.loadColor = true;
    BeginRendering(commands, target);
    vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      presentOverlay_->Handle());
    // As `Present`'s bindings: the image at 0 (and 1), its array view
    // at 2.
    const DescriptorResource image = DescriptorResource::Of(
        lensOverlays_->LayerView(0),
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    const DescriptorResource array = DescriptorResource::Of(
        lensOverlays_->View(),
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    const std::array<DescriptorResource, 4> views{
        image, image, array, DescriptorResource::Of(*highlight_)};
    presentLayout_.Push(commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        views);
    const bool srgb = presentFormat_ == VK_FORMAT_R8G8B8A8_SRGB ||
                      presentFormat_ == VK_FORMAT_B8G8R8A8_SRGB;
    struct {
        std::uint32_t flags[4];
        float range[4];
    } constants{{srgb ? 1u : 0u, 0u, 0u, 0u},
                {0.0f, 0.0f, 0.0f, 0.0f}};
    presentLayout_.PushConstants(commands, &constants,
                                 sizeof(constants));
    vkCmdDraw(commands, 3, 1, 0, 0);
    vkCmdEndRendering(commands);
    lensOverlays_->Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
    // Then the screen items, over the image as presented.
    if (overlays_->HasScreenItems()) {
        OverlayPass::Target screen =
            OverlayTarget(targets, LastSubframeView(0), 0);
        screen.color = present.color;
        screen.format = presentFormat_;
        screen.extent = present.extent;
        overlays_->Record(commands, screen,
                          OverlayPass::Layer::Screen);
    }
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
        offsetof(ShadingView, invViewProj);
    shading.eye = offsetof(ShadingView, eye);
    PatchViews(commands, shadingUniforms_, kShadingShared,
               kShadingStride, shading);
    ViewFields sensor;
    sensor.inverseViewProjection = offsetof(SensorView, invViewProj);
    sensor.eye = offsetof(SensorView, eye);
    sensor.forward = offsetof(SensorView, forward);
    sensor.pose = offsetof(SensorView, pose);
    sensor.outputViews = true;
    PatchViews(commands, sensorUniforms_, kSensorShared,
               kSensorStride, sensor);
    // The poses rendered, for the CPU's estimates of the next
    // frame's.
    ViewFields history;
    history.pose = 0;
    history.outputViews = true;
    PatchViews(commands, *poseHistory_[uploads_.Slot()], 0,
               sizeof(glm::mat4), history);
    for (const RasterDrawable* drawable : drawables) {
        ERNIE_GPU_ZONE_NAMED(context_, commands,
                             drawable->ProfileName());
        drawable->RecordViewPoses(commands);
    }
    ShaderBarrier(commands);
}

void RasterRenderer::BuildHzb(VkCommandBuffer commands,
                              RenderTargets& targets) const {
    ERNIE_ZONE("Record Hi-Z");
    ERNIE_GPU_ZONE(context_, commands, "Depth pyramid");
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

void RasterRenderer::WriteShadingResources(
    VkCommandBuffer commands) const {
    BindlessHeap& heap = context_.Bindless();
    using Kind = BindlessHeap::Kind;
    // The last recording's slots retire; nothing in flight sees them
    // change.
    shadingSlots_.clear();
    const auto texture = [&](Kind kind, const GpuImage& image) {
        shadingSlots_.push_back(heap.AddImage(kind, image.View()));
        return shadingSlots_.back().Index();
    };
    ShadingResources resources{};
    resources.views = shadingUniforms_.Address();
    resources.viewBlocks =
        shadingUniforms_.Address() + kShadingShared;
    resources.materials = materialTable_->Address();
    resources.lights = lightTable_.Address();
    resources.lightViews = lightViewMatrices_->Address();
    resources.virtualShadow =
        virtualLight_ >= 0 ? virtualShadow_->Block().Address() : 0;
    resources.virtualLight =
        virtualLight_ >= 0 ? static_cast<std::uint32_t>(virtualLight_)
                           : ~0u;
    resources.virtualDebug = virtualDebug_;
    resources.lightProfiles = lightProfiles_.Address();
    resources.patternNorms = patternNorms_.Address();
    resources.lightTiming = lightTiming_->Address();
    resources.stars = stars_->Address();
    resources.starCells = starCells_->Address();
    resources.atmosphereMedium = atmosphere_->Medium().Address();
    resources.atmosphereDensities =
        atmosphere_->Densities().Address();
    resources.atmospherePhases = atmosphere_->Phases().Address();
    resources.sigmoidTable =
        texture(Kind::Texture2DArray, sigmoidTable_);
    resources.tableSampler = tableSamplerSlot_.Index();
    resources.lightShadowMaps =
        texture(Kind::Texture2DArray, *lightShadowMaps_);
    resources.lightShadowSampler = shadowSamplerSlot_.Index();
    resources.domeMap = texture(Kind::Texture2D, *domeMap_);
    resources.domeLighting = texture(
        Kind::Texture2D,
        domeLighting_ != nullptr ? *domeLighting_ : *domeMap_);
    resources.horizonMap =
        texture(Kind::Texture2DArray, terrainHorizon_.image != nullptr
                                          ? *terrainHorizon_.image
                                          : *noHorizon_);
    resources.terrainHeights =
        texture(Kind::Texture2D, terrainHorizon_.heights != nullptr
                                     ? *terrainHorizon_.heights
                                     : *noHeights_);
    resources.atmosphereTransmittance =
        texture(Kind::Texture2DArray, atmosphere_->Transmittance());
    resources.atmosphereMultiple = texture(
        Kind::Texture2DArray, atmosphere_->MultipleScattering());
    resources.atmosphereSkyIrradiance =
        texture(Kind::Texture2DArray, atmosphere_->SkyIrradiance());
    resources.atmosphereRefraction =
        texture(Kind::Texture2DArray, atmosphere_->Refraction());
    // The volumes' shadow maps, which the lights' shadow views see
    // through (`VolumePass::Prepare`).
    const GpuImage* volumeShadows = volumes_->ShadowMap();
    resources.volumeShadows =
        volumeShadows != nullptr
            ? texture(Kind::Texture2DArray, *volumeShadows)
            : 0xFFFFFFFFu;
    resources.volumeShadowSize = volumes_->ShadowSize();

    // In the command stream, after the dispatches reading the last
    // and before those reading this.
    VkMemoryBarrier2 before{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    before.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    before.srcAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;
    before.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    before.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &before;
    vkCmdPipelineBarrier2(commands, &dependency);
    vkCmdUpdateBuffer(commands, shadingResources_->Handle(), 0,
                      sizeof(resources), &resources);
    VkMemoryBarrier2 after{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    after.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    after.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    after.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    after.dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;
    dependency.pMemoryBarriers = &after;
    vkCmdPipelineBarrier2(commands, &dependency);
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
        WriteShadingResources(commands);
        const DescriptorResource shading[] = {
            DescriptorResource::Of(*shadingResources_),
            sampled(targets.GBuffer(0)),
            sampled(targets.GBuffer(1)),
            sampled(targets.GBuffer(2)),
            sampled(targets.Depth()),
            general(targets.RasterRadiance()),
            general(occlusion_[0] != nullptr ? *occlusion_[0]
                                             : *noOcclusion_),
            general(occlusion_[1] != nullptr ? *occlusion_[1]
                                             : *noOcclusion_),
            DescriptorResource::Of(*patternData_)};
        // The dispatch's model and first view, and the terrain
        // horizon (none without its image).
        const auto constantsFor = [&](std::uint32_t model) {
            ShadingConstants constants{};
            constants.model[0] = model;
            constants.model[1] = firstView;
            constants.model[2] =
                targets.Stokes() ? targets.StokesOffset() : 0u;
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
            // A zone per material model, named for it.
            ERNIE_GPU_ZONE_NAMED(
                context_, commands,
                profiler::Intern(brdfs_.Models()[model].name));
            vkCmdBindPipeline(
                commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                shadingPipelines_
                    .at({model, pushbroom_, shadingVariant_})
                    ->Handle());
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
            // Its parts in turn (shading.slang's kSkyPart*), each
            // adding to what the one before stored: the dome, which
            // clears the pixels too, so it always runs.
            const auto part = [&](std::uint32_t index) {
                ShadingConstants constants = constantsFor(0);
                constants.model[3] = index;
                shadingLayout_.PushConstants(commands, &constants,
                                             sizeof(constants));
                vkCmdDispatch(commands, groupsX, groupsY, views);
                ShaderBarrier(commands);
            };
            {
                ERNIE_GPU_ZONE(context_, commands, "Dome");
                part(0);
            }
            if ((sky_ & kSkyStars) != 0) {
                ERNIE_GPU_ZONE(context_, commands, "Stars");
                part(1);
            }
            if ((sky_ & kSkyBodies) != 0) {
                ERNIE_GPU_ZONE(context_, commands,
                               "Celestial bodies");
                part(2);
            }
            if ((sky_ & kSkyAtmosphere) != 0) {
                ERNIE_GPU_ZONE(context_, commands, "Atmosphere");
                part(3);
            }
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
                 static_cast<float>(targets.StokesOffset()), 0.0f}};
    motionLayout_.PushConstants(commands, &constants,
                                sizeof(constants));
    // Every layer of the radiance: its intensity, and with a
    // polarizer its Stokes Q and U.
    vkCmdDispatch(
        commands, Groups(extent_.width), Groups(extent_.height),
        targets.StokesOffset() * (targets.Stokes() ? 3u : 1u));
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
        general(
            targets.Bound(render_vars::kDegreeOfLinearPolarization)),
        sampled(targets.GBuffer(2)),
        general(
            targets.Bound(render_vars::kAngleOfLinearPolarization))};
    // Bindings 5 to 14, the vars before MotionVector, which binds at
    // 30.
    for (const std::string_view name : RenderTargets::VarNames()) {
        if (name == render_vars::kMotionVector) {
            break;
        }
        filters.push_back(general(targets.Bound(name)));
    }
    // The G-buffer's normal and depth from the bindless heap, in
    // fresh slots: the last recording's retire, so nothing in flight
    // sees them change.
    BindlessHeap& heap = context_.Bindless();
    sensorSlots_.clear();
    sensorSlots_.push_back(
        heap.AddImage(BindlessHeap::Kind::Texture2DArray,
                      targets.GBuffer(0).View()));
    sensorSlots_.push_back(heap.AddImage(
        BindlessHeap::Kind::Texture2DArray, targets.Depth().View()));
    // Each chain's kernels in turn over the one raster image, each
    // writing its own vars.
    for (const SensorPass& pass : sensorPasses_) {
        const SensorPlan& plan = pass.plan;
        std::vector<DescriptorResource> bound = filters;
        bound.push_back(
            general(targets.Bound(render_vars::kLinearStokes)));
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
        // pass's parameters; the layers from the raster radiance to
        // its Stokes Q and on to U (0 without them); the parameters'
        // address, and the normal's and depth's heap slots.
        // And the vars written to linear buffers too, with their
        // addresses (`RenderTargets::LinearVars`).
        const RenderTargets::LinearVars* linear =
            targets.GetLinearVars();
        const SensorConstants constants{
            {firstView, plan.aovs, state && pass.history ? 1u : 0u,
             state ? plan.states : 0u},
            {pass.tableOffset, lensTable_[1], lensTable_[2],
             lensTable_[3]},
            {targets.Stokes() ? targets.StokesOffset() : 0u,
             linear != nullptr ? linear->written & kLinearVarBits
                               : 0u,
             linear != nullptr
                 ? linear->written >> RenderTargets::LinearVars::kVars
                 : 0u,
             0u},
            pass.params->Address(),
            sensorSlots_[0].Index(),
            sensorSlots_[1].Index(),
            linear != nullptr ? linear->table->Address() : 0u};
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
            // Each kernel's own time, in the order they run.
            ERNIE_GPU_ZONE(context_, commands, "Sensor kernel");
            vkCmdBindPipeline(
                commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                sensorPipelines_.at(SensorKernelKey(plan, segment))
                    ->Handle());
            filterLayout_.Push(
                commands, VK_PIPELINE_BIND_POINT_COMPUTE, resources);
            filterLayout_.PushConstants(commands, &constants,
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
        VkDeviceAddress gain = 0;
        std::uint32_t hasGain = 0;
        std::uint32_t padding = 0;
    } constants{{srgb ? 1u : 0u, mode.show, 0, 0},
                {presentRange_[0], presentRange_[1], presentRange_[2],
                 presentRange_[3]}};
    // The automatic exposure's scale, where the frame has one.
    if (exposing_) {
        constants.gain = exposureGain_.Address();
        constants.hasGain = 1;
    }
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

void RasterRenderer::SetOverlayComputeInterop(
    ComputeInterop* compute) {
    overlays_->SetComputeInterop(compute);
}

void RasterRenderer::SetVolumeComputeInterop(
    ComputeInterop* compute) {
    volumes_->SetComputeInterop(compute);
}

void RasterRenderer::SetVolumeFieldWriter(
    const std::string& path, std::array<std::uint32_t, 3> shape,
    FieldWriter writer) {
    volumes_->SetFieldWriter(path, shape, std::move(writer));
}

std::uint64_t RasterRenderer::GpuBytes() const {
    std::uint64_t bytes =
        uniforms_.Size() + shadowUniforms_.Size() +
        shadowMaps_->Bytes() + lightShadowMaps_->Bytes() +
        lightTable_.Size() + lightProfiles_.Size() +
        lightViewMatrices_->Size() + sigmoidTable_.Bytes() +
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
