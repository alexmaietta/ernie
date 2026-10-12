// SPDX-License-Identifier: BSD-3-Clause
// Volumes (UsdVol): per pixel, its view ray marched through each Volume its view sees, up to the
// surface behind it or on to infinity, after shading. The radiance behind is dimmed by the
// volumes' transmittance and what they emit is added: L = L_behind T + sum of T(t) j(t) dt, with
// grey extinction (sigma_t, from the density field) and per-band emission j: the material's
// emission times its field, what absorbs glowing at its temperature, sigma_a B_lambda(T)
// (Kirchhoff's law), and the light of the view's lights scattered once towards the eye,
// sigma_s p(g) E. Each light's E at a step is what shading would take there (light.slang's
// conventions in shading.slang): distant lights dimmed through the atmosphere, sphere lights
// with their cones, rect, disk and cylinder lights as sources of their projected area (small
// beside their distance), and lasers by their beams and patterns, so beams show in the medium.
// It is shadowed by geometry (one comparison in the light's shadow map, which the jitter between
// pixels and frames softens) and by every volume the view sees between the step and the light (a
// short march towards it). Light linking holds for volumes as for surfaces: a light lights a volume
// only when its light mask holds the volume's group, which skips its shadow march too, and a
// volume shadows the medium from a light only when the light's shadow mask holds it. Each step
// integrates its emission exactly for constant coefficients, (1 - exp(-sigma_t dt)) / sigma_t.
//
// The phase functions take the phase angle g, between the directions to the light and to the eye,
// as the BRDFs' do (brdf.slang): Henyey-Greenstein, Hapke's double Henyey-Greenstein (the same b
// and c a Hapke surface reads, so lofted dust and the regolith it rose from scatter alike),
// Rayleigh or isotropic, each over 4 pi steradians.
//
// The fields are NanoVDB grids in one buffer, which PNanoVDB walks in place; this module is
// compiled after pnanovdb_prelude.slang and PNanoVDB.h. A volume's fields each map its local
// frame to their grid's index space, where values sit at integer coordinates (trilinear between
// them).
//
// Bindings: 0 the shading pass's resources (shading.slang's ShadingResources, as it recorded them:
// the views' uniforms, read at the byte offsets in `constants`, the lights, their shadow maps and
// views, profiles, patterns' norms and timing, the atmosphere's transmittance), 1 depth and 2 ids
// of the G-buffer, 3 the spectral radiance (a layer per view and band), 4 the grids, 5 the pass's
// own buffers (`VolumeResources`), 29 the light patterns' data (light_pattern's). Textures and
// samplers come from the bindless heap (bindless.slang), buffers through their addresses.
//
// References:
// - K. Museth, "NanoVDB: A GPU-Friendly and Portable VDB Data Structure For Real-Time Rendering
//   And Simulation", ACM SIGGRAPH 2021 Talks. https://doi.org/10.1145/3450623.3464653
// - M. Pharr, W. Jakob and G. Humphreys, "Physically Based Rendering", 4th ed., MIT Press 2023,
//   chapter 11 (volume scattering) and 14.2 (the equation of transfer).
//   https://pbr-book.org/4ed/Volume_Scattering
// - S. Hillaire, "Physically Based and Unified Volumetric Rendering in Frostbite", SIGGRAPH
//   Advances in Real-Time Rendering, 2015 (the energy-conserving step integration).
// - M. Planck, "Ueber das Gesetz der Energieverteilung im Normalspectrum", Annalen der Physik
//   309(3), 553-563, 1901; radiation constants from CODATA 2018.
// - L. G. Henyey and J. L. Greenstein, "Diffuse radiation in the Galaxy", Astrophysical Journal
//   93, 70-83, 1941.
// - B. Hapke, "Theory of Reflectance and Emittance Spectroscopy", 2nd ed., Cambridge University
//   Press 2012, section 6.4.2 (the double Henyey-Greenstein function).

import projection;
import brdf;
import bindless;
import atmosphere;
import light_pattern;
import light_patterns;

static const uint kNoGrid = 0xFFFFFFFFu;
static const uint kMaxIntervals = 8;
static const uint kMaxGroups = 16;
// The transmittance below which a ray is opaque.
static const float kOpaque = 1e-3;
// Steps of the march towards a light through the volume.
static const uint kShadowSteps = 12;
static const uint kMaxLights = 16;
// Per view and light, the (sequence frame, weight) pairs a laser shows (shading.slang's).
static const uint kMaxLit = 4;
// Lights' IES profiles (shading.slang's): per slot kProfileRows rows from the light's axis to
// opposite it, of kProfileColumns columns around it from its +x towards +y.
static const uint kProfileRows = 129;
static const uint kProfileColumns = 64;
// Light types (`LightGpu::position.w`).
static const uint kDistantLight = 0;
static const uint kSphereLight = 1;
static const uint kRectLight = 2;
static const uint kDiskLight = 3;
static const uint kCylinderLight = 4;
static const uint kLaser = 5;
// Phase functions (`VolumeMaterialGpu::phase.x`).
static const uint kHenyeyGreenstein = 0;
static const uint kDoubleHenyeyGreenstein = 1;
static const uint kRayleigh = 2;
static const uint kIsotropic = 3;

// One field of a volume: rows of the affine map from the volume's local frame to the grid's index
// space, and the grid: its byte offset in `grids` (kNoGrid for none) and PNANOVDB_GRID_TYPE_*.
struct FieldGpu {
    float4 toIndex[3];
    uint4 grid;
}

struct VolumeGpu {
    // Rows of the affine map from world coordinates (relative to the frame's origin) to the
    // volume's local frame.
    float4 toLocal[3];
    // Local bounds of its fields' active values.
    float4 boundsMin;
    float4 boundsMax;
    FieldGpu density;
    FieldGpu temperature;
    FieldGpu emission;
    // x: the density scale (to sigma_t, 1/world unit); y: the temperature without a field
    // (kelvin), z: the temperature field's scale; w: the march's step, world units.
    float4 params;
    // x: its material, y: its world, z: its light linking group (`LinkGroups`); w unused.
    uint4 info;
}

struct VolumeMaterialGpu {
    // Single-scattering albedo per band, four per float4.
    float4 albedo[16];
    // Emitted spectral radiance per world unit per unit field (W m^-2 sr^-1 nm^-1 per unit).
    float4 emission[16];
    // x: Henyey-Greenstein g; y: black-body scale; z: 1 when it emits; w: 1 when it scatters.
    float4 params;
    // x: its phase function (k*), y: double Henyey-Greenstein b, z: its c; w unused.
    float4 phase;
}

struct VolumeConstants {
    // x: the first view's block, y: the block stride in bytes, z: the most steps per volume, w:
    // the frame's seed.
    uint4 layout;
    // Byte offsets in a block: the inverse view-projection, a pushbroom's strip, the bands
    // (`range`) and the image size.
    uint4 offsets;
    // x: 1 when the background holds radiance (a sky); without one, this pass writes the
    // background's, no radiance but what volumes emit, and the sensor reads it as held. y: the
    // layers from a radiance layer to its Stokes Q, and on to U, with a polarizer (0 without
    // them). zw unused.
    uint4 flags;
    // Byte offsets in a block: x `view` (the lights it sees, a bit each; its world's slot in a
    // batch's distant light maps; its first light), y `lightShadow` (z the maps' texels across), z
    // `atmosphere` (two float4s: the planet's centre in km relative to the frame's origin and its
    // radius; the atmosphere's top radius and 1 with one); w unused. Its `size.w` is the light
    // count.
    uint4 lights;
}

// The shading pass's light (shading.slang's LightGpu, which the renderer checks this against).
struct LightGpu {
    float4 position;
    float4 axis;
    float4 cone;
    float4 shadow;
    float4 extentX;
    float4 extentY;
    float4 spectrum[16];
    float4 tinted[16];
    float4 beam;
    float4 beamExtra;
    float4 pattern;
    float4 patternExtent;
    uint4 patternLayout;
    float4 patternParams[4];
    float4 map;
    uint4 links;
    float4 polarization;
}

// The shading pass's resources (shading.slang's ShadingResources, which the renderer checks this
// against): the buffers this pass reads, typed, the rest only for their place.
struct ShadingResources {
    // Every view's uniforms, read in 16-byte pieces at byte offsets (`viewLoad`).
    uint4* views;
    uint* materials;
    LightGpu* lights;
    float4x4* lightViews;
    float* lightProfiles;
    float* patternNorms;
    float2* lightTiming;
    uint* stars;
    uint* starCells;
    uint* atmosphereMedium;
    uint* atmosphereDensities;
    uint* atmospherePhases;
    uint sigmoidTable;
    uint tableSampler;
    uint lightShadowMaps;
    uint lightShadowSampler;
    uint domeMap;
    uint domeLighting;
    uint horizonMap;
    uint terrainHeights;
    uint atmosphereTransmittance;
    uint atmosphereMultiple;
    uint atmosphereSkyIrradiance;
    uint atmosphereRefraction;
    uint volumeShadows;
    uint volumeShadowSize;
}

// The pass's own buffers (`VolumePass`): the volumes, their materials, each view's runs of
// `viewVolumes` (offset and count of those it marches, then of those that shadow them), and the
// volumes each view sees; for the volumes' shadow maps (`volumeShadowMain`), the lights and their
// shadow views' view-projections, and how many volumes, texels across a map and lights there are.
struct VolumeResources {
    VolumeGpu* volumes;
    VolumeMaterialGpu* materials;
    uint4* runs;
    uint* lists;
    LightGpu* shadowLights;
    float4x4* shadowViews;
    uint volumeCount;
    uint shadowSize;
    uint lightCount;
    uint unused;
}

[[vk::binding(0, 0)]] ConstantBuffer<ShadingResources> shading;
[[vk::binding(1, 0)]] Texture2DArray<float> gDepth;
[[vk::binding(2, 0)]] Texture2DArray<uint4> gIds;
[[vk::binding(3, 0)]] RWTexture2DArray<float> spectral;
[[vk::binding(4, 0)]] StructuredBuffer<uint> grids;
[[vk::binding(5, 0)]] ConstantBuffer<VolumeResources> pass;
// The volumes' shadow maps (`volumeShadowMain`'s output, its own pipeline's).
[[vk::image_format("rgba32f")]] [[vk::binding(6, 0)]] RWTexture2DArray<float4> volumeShadowsOut;
[[vk::push_constant]] ConstantBuffer<VolumeConstants> constants;

#define volumes (pass.volumes)
#define volumeMaterials (pass.materials)
#define viewRuns (pass.runs)
#define viewVolumes (pass.lists)
#define lightViews (shading.lightViews)
#define lightProfiles (shading.lightProfiles)
#define patternNorms (shading.patternNorms)
#define lightTiming (shading.lightTiming)
#define lightShadowMaps (heapTextureArrays[shading.lightShadowMaps])
#define lightShadowSampler (heapComparisonSamplers[shading.lightShadowSampler])
#define atmosphereTransmittance (heapTextureArrays[shading.atmosphereTransmittance])
#define tableSampler (heapSamplers[shading.tableSampler])

// 16 bytes of the views' uniforms at byte offset `offset` (a multiple of 16).
uint4 viewLoad(uint offset) {
    return shading.views[offset / 16];
}

static const uint kBackground = 0xFFFFFFFF;

float4x4 loadMatrix(uint offset) {
    // glm's column-major order: four columns.
    let c0 = asfloat(viewLoad(offset));
    let c1 = asfloat(viewLoad(offset + 16));
    let c2 = asfloat(viewLoad(offset + 32));
    let c3 = asfloat(viewLoad(offset + 48));
    return transpose(float4x4(c0, c1, c2, c3));
}

// A field's value at index-space point `index`, trilinear between the values at the integer
// coordinates around it; 0 outside the grid's active values.
float valueAt(uint type, pnanovdb_coord_t ijk, inout pnanovdb_readaccessor_t accessor) {
    uint level = 0;
    let address =
        pnanovdb_readaccessor_get_value_address_and_level(type, grids, accessor, ijk, level);
    switch (type) {
    case PNANOVDB_GRID_TYPE_FLOAT:
        return pnanovdb_read_float(grids, address);
    case PNANOVDB_GRID_TYPE_HALF:
        return pnanovdb_read_half(grids, address);
    case PNANOVDB_GRID_TYPE_FP4:
        return pnanovdb_root_fp4_read_float(grids, address, ijk, level);
    case PNANOVDB_GRID_TYPE_FP8:
        return pnanovdb_root_fp8_read_float(grids, address, ijk, level);
    case PNANOVDB_GRID_TYPE_FP16:
        return pnanovdb_root_fp16_read_float(grids, address, ijk, level);
    case PNANOVDB_GRID_TYPE_FPN:
        return pnanovdb_root_fpn_read_float(grids, address, ijk, level);
    case PNANOVDB_GRID_TYPE_VEC3F: {
        let v = pnanovdb_read_vec3(grids, address);
        return length(float3(v.x, v.y, v.z));
    }
    default:
        return 0.0;
    }
}

float sampleField(FieldGpu field, float3 local, inout pnanovdb_readaccessor_t accessor) {
    let index = float3(dot(field.toIndex[0], float4(local, 1.0)),
                       dot(field.toIndex[1], float4(local, 1.0)),
                       dot(field.toIndex[2], float4(local, 1.0)));
    let base = floor(index);
    let f = index - base;
    let i = int3(base);
    float corners[8];
    for (int c = 0; c < 8; ++c) {
        pnanovdb_coord_t ijk;
        ijk.x = i.x + (c & 1);
        ijk.y = i.y + ((c >> 1) & 1);
        ijk.z = i.z + ((c >> 2) & 1);
        corners[c] = valueAt(field.grid.y, ijk, accessor);
    }
    let x00 = lerp(corners[0], corners[1], f.x);
    let x10 = lerp(corners[2], corners[3], f.x);
    let x01 = lerp(corners[4], corners[5], f.x);
    let x11 = lerp(corners[6], corners[7], f.x);
    return lerp(lerp(x00, x10, f.y), lerp(x01, x11, f.y), f.z);
}

pnanovdb_readaccessor_t accessorOf(FieldGpu field) {
    pnanovdb_grid_handle_t grid;
    grid.address.byte_offset = field.grid.x == kNoGrid ? 0 : field.grid.x;
    let root = pnanovdb_tree_get_root(grids, pnanovdb_grid_get_tree(grids, grid));
    pnanovdb_readaccessor_t accessor;
    pnanovdb_readaccessor_init(accessor, root);
    return accessor;
}

// Planck's spectral radiance at wavelengths `lambda` (nm) and temperature `kelvin`, W m^-2 sr^-1
// nm^-1.
float4 planck(float4 lambda, float kelvin) {
    // 2 h c^2 in W um^4 m^-2 sr^-1 and h c / k in um K.
    static const float c1 = 1.191042972e8;
    static const float c2 = 14387.76877;
    let um = lambda * 1e-3;
    let x = c2 / (um * max(kelvin, 1e-3));
    let scale = c1 / (um * um * um * um * um) * 1e-3;
    // exp overflows past 88: the tail is then exp(-x) itself.
    return select(x > 80.0, scale * exp(-x), scale / (exp(min(x, 80.0)) - 1.0));
}

float hash(uint3 p) {
    // PCG3D (Jarzynski and Olano 2020), the first component.
    var v = p * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    return float(v.x >> 8) / 16777216.0;
}

// The material's phase function at phase angle cosine `cosG` (towards the light and towards the
// eye), per steradian.
float phaseOf(VolumeMaterialGpu material, float cosG) {
    switch (uint(material.phase.x)) {
    case kDoubleHenyeyGreenstein:
        return doubleHenyeyGreenstein(material.phase.y, material.phase.z, cosG) / (4.0 * PI);
    case kRayleigh:
        return 3.0 / (16.0 * PI) * (1.0 + cosG * cosG);
    case kIsotropic:
        return 1.0 / (4.0 * PI);
    default:
        // Forward scattering (g > 0) is towards the light's far side: phase angle near pi.
        return henyeyGreenstein(material.params.x, cosG) / (4.0 * PI);
    }
}

// The transmittance from world point `world` towards a light along unit `towards`, up to world
// distance `far`, through every volume of the view's run `run` whose link group the light's shadow
// mask `casters` holds (`kShadowSteps` through each one the segment crosses).
float volumesShadow(uint2 run, float3 world, float3 towards, float far, uint casters) {
    float depth = 0.0;
    for (uint k = 0; k < run.y; ++k) {
        let volume = volumes[viewVolumes[run.x + k]];
        if (volume.density.grid.x == kNoGrid || ((casters >> volume.info.z) & 1u) == 0) {
            continue;
        }
        let o = float3(dot(volume.toLocal[0], float4(world, 1.0)),
                       dot(volume.toLocal[1], float4(world, 1.0)),
                       dot(volume.toLocal[2], float4(world, 1.0)));
        let d = float3(dot(volume.toLocal[0].xyz, towards), dot(volume.toLocal[1].xyz, towards),
                       dot(volume.toLocal[2].xyz, towards));
        let inverse = 1.0 / d;
        let t0 = (volume.boundsMin.xyz - o) * inverse;
        let t1 = (volume.boundsMax.xyz - o) * inverse;
        let near = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), max(min(t0.z, t1.z), 0.0));
        let exit = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), min(max(t0.z, t1.z), far));
        if (near >= exit) {
            continue;
        }
        var accessor = accessorOf(volume.density);
        let dt = (exit - near) / float(kShadowSteps);
        float sum = 0.0;
        for (uint i = 0; i < kShadowSteps; ++i) {
            sum += max(sampleField(volume.density, o + d * (near + (float(i) + 0.5) * dt),
                                   accessor),
                       0.0);
        }
        depth += sum * volume.params.x * dt;
    }
    return exp(-depth);
}

// The inverse of `matrix` by cofactors, 0 for a singular one (views.slang's `invert`).
//
// Reference: MESA GLU, `__gluInvertMatrixd` (cofactor inverse of a 4x4 matrix).
//   https://gitlab.freedesktop.org/mesa/glu/-/blob/master/src/libutil/project.c
float4x4 invert(float4x4 matrix) {
    float m[16];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            m[r * 4 + c] = matrix[r][c];
        }
    }
    float inv[16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] +
             m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] -
             m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] +
             m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] -
              m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] -
             m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] +
             m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] -
             m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] +
              m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] +
             m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] -
             m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] +
              m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] -
              m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] -
             m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] -
              m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    let scale = det != 0.0 ? 1.0 / det : 0.0;
    float4x4 result;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            result[r][c] = inv[r * 4 + c] * scale;
        }
    }
    return result;
}

// The shadow view of `light`'s maps that sees `world`, as shading picks it (shading.slang's
// `lightView`): a batch's distant light's world's, a distant light's finest cascade holding it,
// a cube's face, or its only one.
uint lightViewOf(LightGpu light, float3 world, uint worldSlot) {
    let first = uint(light.cone.z);
    let count = uint(light.cone.w);
    if (light.map.y != 0.0) {
        return first + min(worldSlot, count - 1);
    }
    if (light.position.w == 0.0 && count > 1) {
        for (uint c = 0; c + 1 < count; ++c) {
            let clip = mul(lightViews[first + c], float4(world, 1.0)).xyz;
            if (all(abs(clip.xy) <= float2(1.0)) && clip.z >= 0.0 && clip.z <= 1.0) {
                return first + c;
            }
        }
        return first + count - 1;
    }
    if (count < 6) {
        return first;
    }
    let d = world - light.position.xyz;
    let a = abs(d);
    if (a.x >= a.y && a.x >= a.z) {
        return first + (d.x >= 0.0 ? 0 : 1);
    }
    if (a.y >= a.z) {
        return first + (d.y >= 0.0 ? 2 : 3);
    }
    return first + (d.z >= 0.0 ? 4 : 5);
}

// Whether geometry hides `light` from `world`: one comparison in its shadow map, lit off the map
// or without one. The maps hold standard depth, near 0.
float geometryShadow(LightGpu light, float3 world, uint worldSlot, float mapTexels) {
    if (light.cone.w <= 0.0) {
        return 1.0;
    }
    let view = lightViewOf(light, world, worldSlot);
    let clip = mul(lightViews[view], float4(world, 1.0));
    if (clip.w <= 0.0) {
        return 1.0;
    }
    let ndc = clip.xyz / clip.w;
    let uv = float2(0.5 * ndc.x + 0.5, 0.5 - 0.5 * ndc.y);
    if (any(uv < float2(0.0)) || any(uv > float2(1.0))) {
        return 1.0;
    }
    // In the light's corner of the layer.
    let corner = light.map.x / mapTexels;
    let at = min(uv * corner, float2(corner - 0.5 / mapTexels));
    return lightShadowMaps.SampleCmpLevelZero(lightShadowSampler, float3(at, float(view)),
                                              min(ndc.z, 0.99999994));
}

// A laser's irradiance at `world` per unit of its spectrum, across a footprint of `footprint`
// (world extents along two axes), at sequence frame `frame` (shading.slang's `laserIrradiance`).
//
// Reference: A. E. Siegman, Lasers, University Science Books, 1986, ch. 17.
float laserIrradiance(LightGpu light, uint slot, float3 world, float3x2 footprint, uint frame) {
    let offset = world - light.position.xyz;
    let along = dot(offset, light.axis.xyz);
    if (along <= 0.0) {
        return 0.0;
    }
    let x = normalize(light.extentX.xyz);
    let y = cross(x, light.axis.xyz);
    let z = along - light.beamExtra.x;
    let zr = max(light.beam.zw, float2(1e-12));
    let w = light.beam.xy * sqrt(1.0 + (z / zr) * (z / zr));
    let spanX = abs(mul(x, footprint));
    let spanY = abs(mul(y, footprint));
    let width = float2(spanX.x + spanX.y, spanY.x + spanY.y);
    let across = float2(dot(offset, x), dot(offset, y));
    LightPatternInput input;
    input.t = across / along;
    input.footprint = width / along;
    input.beam = w / along;
    input.wavelength = light.pattern.y;
    input.time = light.pattern.z;
    input.frame = frame;
    input.frames = max(uint(light.pattern.w), 1);
    let slope = across * z / (z * z + zr * zr);
    let beamOblique = sqrt(1.0 + dot(slope, slope));
    let patternOblique = length(offset) / along;
    var irradiance = light.beamExtra.w * beamOblique * beamlet(float2(0.0), input);
    let efficiency = light.beamExtra.z;
    let norm = patternNorms[slot];
    if (efficiency > 0.0 && norm > 0.0) {
        let extent = light.patternExtent.xy;
        let bounded = extent > 0.0;
        let cover = select(bounded,
                           saturate((extent - abs(input.t)) / max(input.footprint, 1e-9) + 0.5),
                           float2(1.0));
        if (cover.x * cover.y > 0.0) {
            input.t = select(bounded, clamp(input.t, -extent, extent), input.t);
            LightPatternParams p;
            p.values = light.patternParams;
            p.layout = light.patternLayout;
            let emission = max(lightPattern(uint(light.pattern.x), input, p), 0.0);
            let oblique = light.beamExtra.y > 0.5 ? patternOblique : beamOblique;
            irradiance += oblique * efficiency * cover.x * cover.y * emission / norm;
        }
    }
    return irradiance / (along * along);
}

// A light's IES profile towards unit `direction` from it, bilinear (shading.slang's
// `lightProfile`).
float lightProfile(LightGpu light, float3 direction) {
    let base = (uint(light.shadow.w) - 1) * kProfileRows * kProfileColumns;
    let x = normalize(light.extentX.xyz);
    let y = cross(x, light.axis.xyz);
    let theta = acos(clamp(dot(direction, light.axis.xyz), -1.0, 1.0));
    var phi = atan2(dot(direction, y), dot(direction, x));
    phi = phi < 0.0 ? phi + 2.0 * PI : phi;
    let r = theta / PI * float(kProfileRows - 1);
    let c = phi / (2.0 * PI) * float(kProfileColumns);
    let r0 = min(uint(r), kProfileRows - 2);
    let c0 = uint(c) % kProfileColumns;
    let c1 = (c0 + 1) % kProfileColumns;
    let tr = saturate(r - float(r0));
    let tc = frac(c);
    let row0 = base + r0 * kProfileColumns;
    let row1 = row0 + kProfileColumns;
    return lerp(lerp(lightProfiles[row0 + c0], lightProfiles[row0 + c1], tc),
                lerp(lightProfiles[row1 + c0], lightProfiles[row1 + c1], tc), tr);
}

// How a light reaches a point of the medium: towards it, how far it is, the factor on its
// `spectrum` (focus included) and on its `tinted` part, before shadows and the air.
struct LightReach {
    float3 towards;
    float far;
    float scale;
    float focus;
}

// Light `light` (the view's slot `slot` of view `view`) at world point `world`, a footprint
// `footprint` across: shading's irradiance per unit spectrum, an area light as a source of its
// projected area times its radiance (small beside its distance); `scale` 0 when it does not
// light the medium (a dome's direction, which stands for its diffuse light).
LightReach reachOf(LightGpu light, uint slot, uint view, float3 world, float3x2 footprint) {
    LightReach reach;
    reach.towards = light.position.xyz;
    reach.far = 1e30;
    reach.scale = 1.0;
    reach.focus = 1.0;
    let type = uint(light.position.w);
    let offset = light.position.xyz - world;
    let distance = length(offset);
    if (type != kDistantLight && type != kLaser) {
        reach.towards = offset / max(distance, 1e-9);
        reach.far = distance;
    }
    if (type == kDistantLight) {
        reach.scale = light.shadow.y < 0.5 ? 1.0 : 0.0;
    } else if (type == kSphereLight) {
        reach.scale = 1.0 / max(distance * distance, light.axis.w * light.axis.w);
        if (light.cone.x > -1.5) {
            reach.scale *= smoothstep(light.cone.x, max(light.cone.y, light.cone.x + 1e-6),
                                      dot(-reach.towards, light.axis.xyz));
        }
    } else if (type == kRectLight || type == kDiskLight) {
        let area = (type == kRectLight ? 4.0 : PI) * length(light.extentX.xyz) *
                   length(light.extentY.xyz);
        let facing = dot(-reach.towards, light.axis.xyz);
        reach.scale = facing > 0.0 ? area * facing / max(distance * distance, area) : 0.0;
    } else if (type == kCylinderLight) {
        let half = length(light.extentX.xyz);
        let along = light.extentX.xyz / max(half, 1e-9);
        let side = 4.0 * light.axis.w * half;
        let sine = length(cross(reach.towards, along));
        reach.scale = side * sine / max(distance * distance, side);
    } else if (type == kLaser) {
        reach.scale = 0.0;
        for (uint s = 0; s < kMaxLit; ++s) {
            let lit = lightTiming[(view * kMaxLights + slot) * kMaxLit + s];
            if (lit.y > 0.0) {
                reach.scale += lit.y * laserIrradiance(light, slot, world, footprint, uint(lit.x));
            }
        }
        reach.towards = light.beamExtra.y > 0.5 ? offset / max(distance, 1e-9) : -light.axis.xyz;
        reach.far = light.beamExtra.y > 0.5 ? distance : max(dot(-offset, light.axis.xyz), 0.0);
    }
    if (type != kDistantLight && light.shadow.z > 0.0) {
        // UsdLux's focus, by the direction from the light's centre.
        reach.focus = pow(abs(dot(-reach.towards, light.axis.xyz)), light.shadow.z);
    }
    if (type != kDistantLight && light.shadow.w > 0.0) {
        reach.scale *= lightProfile(light, -reach.towards);
    }
    return reach;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void volumeMain(uint3 id : SV_DispatchThreadID) {
    let block = (constants.layout.x + id.z) * constants.layout.y;
    let size = viewLoad(block + constants.offsets.w);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let range = asfloat(viewLoad(block + constants.offsets.z));
    let bands = uint(range.z);
    let groups = (bands + 3) / 4;
    let pixel = int4(id.xyz, 0);
    let background = gIds.Load(pixel).x == kBackground;
    // A background shading left unwritten: no radiance behind the volumes.
    let unwritten = background && constants.flags.x == 0;
    let runs = viewRuns[constants.layout.x + id.z];
    let run = runs.xy;
    let shadowRun = runs.zw;
    if (run.y == 0) {
        if (unwritten) {
            for (uint b = 0; b < bands; ++b) {
                writeNoRadiance(id, id.z * bands + b);
            }
        }
        return;
    }
    let invViewProj = loadMatrix(block + constants.offsets.x);
    let strip = asfloat(viewLoad(block + constants.offsets.y));

    let ndc = float2((float(id.x) + 0.5) / float(size.x) * 2.0 - 1.0,
                     1.0 - (float(id.y) + 0.5) / float(size.y) * 2.0);
    float3 origin;
    float3 toFar;
    cameraRayThrough(invViewProj, strip, ndc, origin, toFar);
    let direction = normalize(toFar);
    float tMax = 1e30;
    if (!background) {
        let surface = cameraPoint(invViewProj, strip, ndc, gDepth.Load(pixel));
        tMax = dot(surface - origin, direction);
    }

    // The volumes the ray passes through, nearest first.
    float2 spans[kMaxIntervals];
    uint which[kMaxIntervals];
    for (uint k = 0; k < kMaxIntervals; ++k) {
        spans[k] = float2(0.0);
        which[k] = 0;
    }
    uint count = 0;
    for (uint k = 0; k < run.y && count < kMaxIntervals; ++k) {
        let index = viewVolumes[run.x + k];
        let volume = volumes[index];
        let o = float3(dot(volume.toLocal[0], float4(origin, 1.0)),
                       dot(volume.toLocal[1], float4(origin, 1.0)),
                       dot(volume.toLocal[2], float4(origin, 1.0)));
        let d = float3(dot(volume.toLocal[0].xyz, direction), dot(volume.toLocal[1].xyz, direction),
                       dot(volume.toLocal[2].xyz, direction));
        let inverse = 1.0 / d;
        let t0 = (volume.boundsMin.xyz - o) * inverse;
        let t1 = (volume.boundsMax.xyz - o) * inverse;
        let near = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), max(min(t0.z, t1.z), 0.0));
        let far = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), min(max(t0.z, t1.z), tMax));
        if (near >= far) {
            continue;
        }
        // Insertion by entry distance.
        var at = count;
        while (at > 0 && spans[at - 1].x > near) {
            spans[at] = spans[at - 1];
            which[at] = which[at - 1];
            --at;
        }
        spans[at] = float2(near, far);
        which[at] = index;
        ++count;
    }
    if (count == 0) {
        if (unwritten) {
            for (uint b = 0; b < bands; ++b) {
                writeNoRadiance(id, id.z * bands + b);
            }
        }
        return;
    }

    // The view's lights: those of its world (a bit each) from its first; its shadow maps' size;
    // the atmosphere.
    let view = constants.layout.x + id.z;
    let viewLights = viewLoad(block + constants.lights.x);
    let lightCount = min(size.w, kMaxLights);
    let mapTexels = asfloat(viewLoad(block + constants.lights.y)).z;
    float4 air[2];
    air[0] = asfloat(viewLoad(block + constants.lights.z));
    air[1] = asfloat(viewLoad(block + constants.lights.z + 16));
    // The pixel's extent per unit distance along the ray, to its neighbours' rays.
    float3 neighbour;
    float3 toNeighbour;
    cameraRayThrough(invViewProj, strip, ndc + float2(2.0 / float(size.x), 0.0), neighbour,
                     toNeighbour);
    let footprintX = normalize(toNeighbour) - direction;
    cameraRayThrough(invViewProj, strip, ndc - float2(0.0, 2.0 / float(size.y)), neighbour,
                     toNeighbour);
    let footprintY = normalize(toNeighbour) - direction;

    // Per band, what the volumes add towards the eye, each step's integrated with the
    // transmittance up to it.
    float4 emitted[kMaxGroups];
    for (uint g = 0; g < groups; ++g) {
        emitted[g] = float4(0.0);
    }
    float transmittance = 1.0;
    let jitter = hash(uint3(id.xy, constants.layout.w ^ (id.z * 7919u)));
    for (uint s = 0; s < count && transmittance > kOpaque; ++s) {
        let volume = volumes[which[s]];
        let materialIndex = volume.info.x;
        let material = volumeMaterials[materialIndex];
        let o = float3(dot(volume.toLocal[0], float4(origin, 1.0)),
                       dot(volume.toLocal[1], float4(origin, 1.0)),
                       dot(volume.toLocal[2], float4(origin, 1.0)));
        let d = float3(dot(volume.toLocal[0].xyz, direction), dot(volume.toLocal[1].xyz, direction),
                       dot(volume.toLocal[2].xyz, direction));
        let span = spans[s];
        let steps = clamp(uint(ceil((span.y - span.x) / volume.params.w)), 1u, constants.layout.z);
        let dt = (span.y - span.x) / float(steps);
        let hasDensity = volume.density.grid.x != kNoGrid;
        let hasTemperature = volume.temperature.grid.x != kNoGrid;
        let hasEmission = volume.emission.grid.x != kNoGrid;
        let emits = material.params.z != 0.0;
        let scatters = material.params.w != 0.0 && hasDensity;
        let kelvinBase = volume.params.y;
        let glows = material.params.y > 0.0 && (hasTemperature || kelvinBase > 0.0);
        var densityAccessor = accessorOf(volume.density);
        var temperatureAccessor = accessorOf(volume.temperature);
        var emissionAccessor = accessorOf(volume.emission);
        for (uint i = 0; i < steps && transmittance > kOpaque; ++i) {
            let t = span.x + (float(i) + jitter) * dt;
            let local = o + d * t;
            let rho = hasDensity ? max(sampleField(volume.density, local, densityAccessor), 0.0)
                                 : 0.0;
            let sigma = rho * volume.params.x;
            let opacity = 1.0 - exp(-sigma * dt);
            // The transmittance integrated over the step, from its start.
            let weight = sigma > 1e-6 ? transmittance * opacity / sigma : transmittance * dt;
            if (emits) {
                let field = hasEmission
                                ? max(sampleField(volume.emission, local, emissionAccessor), 0.0)
                                : rho;
                for (uint g = 0; g < groups; ++g) {
                    emitted[g] += volumeMaterials[materialIndex].emission[g] * (weight * field);
                }
            }
            if (glows && sigma > 0.0) {
                let kelvin =
                    hasTemperature
                        ? sampleField(volume.temperature, local, temperatureAccessor) *
                              volume.params.z
                        : kelvinBase;
                if (kelvin > 0.0) {
                    let scale = weight * sigma * material.params.y;
                    for (uint g = 0; g < groups; ++g) {
                        let lambda =
                            range.x + (float4(g * 4) + float4(0.5, 1.5, 2.5, 3.5)) * range.y;
                        emitted[g] += (1.0 - volumeMaterials[materialIndex].albedo[g]) * scale *
                                      planck(lambda, kelvin);
                    }
                }
            }
            if (scatters && sigma > 0.0) {
                let world = origin + direction * t;
                // The pixel's footprint this far along the ray, for lasers' patterns.
                let footprint = float3x2(float2(footprintX.x, footprintY.x) * t,
                                         float2(footprintX.y, footprintY.y) * t,
                                         float2(footprintX.z, footprintY.z) * t);
                for (uint k = 0; k < lightCount; ++k) {
                    if ((viewLights.x & (1u << k)) == 0) {
                        continue;
                    }
                    let light = shading.lights[viewLights.z + k];
                    // Light linking: only the lights whose links hold the volume light it.
                    if (((light.links.x >> volume.info.z) & 1u) == 0) {
                        continue;
                    }
                    let reach = reachOf(light, k, view, world, footprint);
                    if (reach.scale <= 0.0) {
                        continue;
                    }
                    let lit = weight * sigma *
                              phaseOf(material, dot(reach.towards, -direction)) * reach.scale *
                              geometryShadow(light, world, viewLights.y, mapTexels) *
                              (light.links.z != 0 ? volumesShadow(shadowRun, world,
                                                                  reach.towards, reach.far,
                                                                  light.links.y)
                                                  : 1.0);
                    if (lit <= 0.0) {
                        continue;
                    }
                    // A distant light through the air (not refracted, nor eclipsed).
                    let throughAir =
                        uint(light.position.w) == kDistantLight && air[1].y != 0.0;
                    let tangent = light.extentY.w;
                    let sunSine = tangent * rsqrt(1.0 + tangent * tangent);
                    for (uint g = 0; g < groups; ++g) {
                        var spectrum = light.spectrum[g] * reach.focus + light.tinted[g];
                        if (throughAir) {
                            spectrum *= sunTransmittance(atmosphereTransmittance, tableSampler,
                                                         world * 1e-3 - air[0].xyz,
                                                         reach.towards, sunSine, g, air[0].w,
                                                         air[1].x);
                        }
                        emitted[g] += volumeMaterials[materialIndex].albedo[g] * lit * spectrum;
                    }
                }
            }
            transmittance *= 1.0 - opacity;
        }
    }

    // Absorption dims the light behind as it is, its polarization too; what the volumes emit
    // and scatter is unpolarized.
    let stokes = constants.flags.y;
    for (uint b = 0; b < bands; ++b) {
        let layer = int3(id.xy, id.z * bands + b);
        let behind = unwritten ? 0.0 : spectral[layer];
        spectral[layer] = behind * transmittance + emitted[b / 4][b % 4];
        if (stokes != 0) {
            for (uint plane = 1; plane <= 2; ++plane) {
                let at = int3(layer.xy, layer.z + int(plane * stokes));
                spectral[at] = unwritten ? 0.0 : spectral[at] * transmittance;
            }
        }
    }
}

// No radiance at layer `layer` of pixel `id`, its Stokes Q and U too with a polarizer.
void writeNoRadiance(uint3 id, uint layer) {
    spectral[int3(id.xy, layer)] = 0.0;
    let stokes = constants.flags.y;
    if (stokes != 0) {
        spectral[int3(id.xy, stokes + layer)] = 0.0;
        spectral[int3(id.xy, 2 * stokes + layer)] = 0.0;
    }
}

// The volumes' shadow maps, which shading reads (shading.slang's `volumeTransmittance`): per light
// shadow view `id.z` and texel, along the view's ray through the texel, the depths (the view's
// NDC z, standard depth) where the volumes whose group the view's light's shadow mask holds begin
// and end (layer 3 z, xy; begin past end where there are none), and the transmittance at
// `kShadowSlices` depths evenly between them (layers 3 z + 1 and 3 z + 2), each marched in
// `kShadowSubsteps` steps from the one before.
static const uint kShadowSlices = 8;
static const uint kShadowSubsteps = 4;

// The extinction coefficient at world point `world` of every volume whose group `casters` holds.
float extinctionAt(float3 world, uint casters) {
    float sigma = 0.0;
    for (uint i = 0; i < pass.volumeCount; ++i) {
        let volume = volumes[i];
        if (volume.density.grid.x == kNoGrid || ((casters >> volume.info.z) & 1u) == 0) {
            continue;
        }
        let local = float3(dot(volume.toLocal[0], float4(world, 1.0)),
                           dot(volume.toLocal[1], float4(world, 1.0)),
                           dot(volume.toLocal[2], float4(world, 1.0)));
        if (any(local < volume.boundsMin.xyz) || any(local > volume.boundsMax.xyz)) {
            continue;
        }
        var accessor = accessorOf(volume.density);
        sigma += max(sampleField(volume.density, local, accessor), 0.0) * volume.params.x;
    }
    return sigma;
}

float3 unproject(float4x4 inverseViewProj, float2 ndc, float depth) {
    let p = mul(inverseViewProj, float4(ndc, depth, 1.0));
    return p.xyz / p.w;
}

// World point `world`'s NDC depth in `viewProj`.
float depthIn(float4x4 viewProj, float3 world) {
    let clip = mul(viewProj, float4(world, 1.0));
    return clip.z / clip.w;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void volumeShadowMain(uint3 id : SV_DispatchThreadID) {
    let size = pass.shadowSize;
    if (id.x >= size || id.y >= size) {
        return;
    }
    let view = id.z;
    let none = float4(1.0, 0.0, 0.0, 0.0);
    volumeShadowsOut[int3(id.xy, 3 * view)] = none;
    volumeShadowsOut[int3(id.xy, 3 * view + 1)] = float4(1.0);
    volumeShadowsOut[int3(id.xy, 3 * view + 2)] = float4(1.0);
    // The light whose maps hold this view, when it casts shadows.
    uint casters = 0;
    for (uint k = 0; k < pass.lightCount; ++k) {
        let light = pass.shadowLights[k];
        let first = uint(light.cone.z);
        if (light.cone.w > 0.0 && view >= first && view < first + uint(light.cone.w) &&
            light.links.z != 0) {
            casters = light.links.y;
        }
    }
    if (casters == 0) {
        return;
    }
    let viewProj = pass.shadowViews[view];
    let inverseViewProj = invert(viewProj);
    let uv = (float2(id.xy) + 0.5) / float(size);
    let ndc = float2(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y);
    let start = unproject(inverseViewProj, ndc, 0.0);
    let ray = unproject(inverseViewProj, ndc, 1.0) - start;
    // Where the ray (fraction s of the way from the near plane to the far) is in any caster. An
    // orthographic view's (a distant light's cascade) runs on back towards the light past its near
    // plane, which fits the geometry that casts shadows, not volumes above it; depths there are
    // below 0.
    let orthographic = all(abs(viewProj[3].xyz) < float3(1e-7));
    let back = orthographic ? -1e6 : 0.0;
    float near = 1.0;
    float far = back;
    for (uint i = 0; i < pass.volumeCount; ++i) {
        let volume = volumes[i];
        if (volume.density.grid.x == kNoGrid || ((casters >> volume.info.z) & 1u) == 0) {
            continue;
        }
        let o = float3(dot(volume.toLocal[0], float4(start, 1.0)),
                       dot(volume.toLocal[1], float4(start, 1.0)),
                       dot(volume.toLocal[2], float4(start, 1.0)));
        let d = float3(dot(volume.toLocal[0].xyz, ray), dot(volume.toLocal[1].xyz, ray),
                       dot(volume.toLocal[2].xyz, ray));
        let inverse = 1.0 / d;
        let t0 = (volume.boundsMin.xyz - o) * inverse;
        let t1 = (volume.boundsMax.xyz - o) * inverse;
        let enter = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), max(min(t0.z, t1.z), back));
        let exit = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), min(max(t0.z, t1.z), 1.0));
        if (enter < exit) {
            near = min(near, enter);
            far = max(far, exit);
        }
    }
    if (near >= far) {
        return;
    }
    let nearDepth = depthIn(viewProj, start + ray * near);
    let farDepth = depthIn(viewProj, start + ray * far);
    volumeShadowsOut[int3(id.xy, 3 * view)] = float4(nearDepth, farDepth, 0.0, 0.0);
    float slices[kShadowSlices];
    float opticalDepth = 0.0;
    var from = unproject(inverseViewProj, ndc, nearDepth);
    for (uint i = 0; i < kShadowSlices; ++i) {
        let to = unproject(inverseViewProj, ndc,
                           lerp(nearDepth, farDepth, float(i + 1) / float(kShadowSlices)));
        let step = (to - from) / float(kShadowSubsteps);
        let length = length(step);
        for (uint j = 0; j < kShadowSubsteps; ++j) {
            opticalDepth += extinctionAt(from + step * (float(j) + 0.5), casters) * length;
        }
        slices[i] = exp(-opticalDepth);
        from = to;
    }
    volumeShadowsOut[int3(id.xy, 3 * view + 1)] =
        float4(slices[0], slices[1], slices[2], slices[3]);
    volumeShadowsOut[int3(id.xy, 3 * view + 2)] =
        float4(slices[4], slices[5], slices[6], slices[7]);
}
