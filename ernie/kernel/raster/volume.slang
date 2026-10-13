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
import laser_beams;

static const uint kNoGrid = 0xFFFFFFFFu;
static const uint kMaxIntervals = 8;
// The band groups (four bands each) the march holds: the frame's, compiled in (`VOLUME_GROUPS`), so
// that its arrays of them stay in registers and its loops over them unroll; 16 (64 bands) without.
#ifndef VOLUME_GROUPS
#define VOLUME_GROUPS 16
#endif
static const uint kMaxGroups = VOLUME_GROUPS;
// The transmittance below which a ray is opaque.
static const float kOpaque = 1e-3;
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
static const uint kCloudDroplets = 4;
static const uint kDualHenyeyGreenstein = 5;

// One field of a volume: rows of the affine map from the volume's local frame to the grid's index
// space, and the grid: its byte offset in `grids` (kNoGrid for none) and PNANOVDB_GRID_TYPE_*.
struct FieldGpu {
    float4 toIndex[3];
    // x: the grid's byte offset in the pool (`kNoGrid` for none); y: its value type; z: its
    // distance grid's first word in `pass.distances` (`kNoGrid` for none); w unused.
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
    // x: its material, y: its world, z: its light linking group (`LinkGroups`); w its light map's
    // first float4 in `pass.lightMaps` (`kNoCache` for none).
    uint4 info;
    // Rows of the affine map from its local frame to world coordinates (relative to the frame's
    // origin).
    float4 toWorld[3];
    // Its lighting cache (`lightCacheMain`): its cells along x, y and z across its bounds, and its
    // first cell of all instances' (`kNoCache` for none); each slot's cells follow the last's.
    uint4 cache;
}

static const uint kNoCache = 0xFFFFFFFFu;

struct VolumeMaterialGpu {
    // Single-scattering albedo per band, four per float4.
    float4 albedo[16];
    // Emitted spectral radiance per world unit per unit field (W m^-2 sr^-1 nm^-1 per unit).
    float4 emission[16];
    // x: Henyey-Greenstein g; y: black-body scale; z: 1 when it emits; w: 1 when it scatters.
    float4 params;
    // x: its phase function (k*), y: double Henyey-Greenstein b, z: its c; w: the scale of the
    // sky's and ground's light.
    float4 phase;
    // Wrenninge's multiple-scattering octaves: x their count (1, single scattering), y each one's
    // scattering of the one before's (a), z its extinction (b), w its anisotropy (c).
    float4 scattering;
    // x: the powder effect's strength (0 none), y its length (world units): in-scattered light
    // dimmed by 1 - x e^(-sigma y) where the medium is thin, as at clouds' edges.
    float4 extras;
}

struct VolumeConstants {
    // x: the first view's block, y: the block stride in bytes, z: the most steps per volume, w:
    // the frame's seed.
    uint4 layout;
    // Byte offsets: in a view's block, the inverse view-projection and a pushbroom's strip; from
    // the start (every view's), the bands (`range`) and the image size.
    uint4 offsets;
    // x: 1 when the background holds radiance (a sky); without one, this pass writes the
    // background's, no radiance but what volumes emit, and the sensor reads it as held. y: the
    // layers from a radiance layer to its Stokes Q, and on to U, with a polarizer (0 without
    // them). z: the render settings' steps and octaves (`shadowSteps` and on). w: temporal
    // amortisation (`kTemporal` and on).
    uint4 flags;
    // Byte offsets: in a view's block, x `view` (the lights it sees, a bit each; its world's slot
    // in a batch's distant light maps; its first light); from the start (every view's), y
    // `lightShadow` (z the maps' texels across), z `atmosphere` (two float4s: the planet's centre
    // in km relative to the frame's origin and its radius; the atmosphere's top radius and 1 with
    // one); w the views this dispatch shades (`historyAt`). The shared `size.w` is the light count.
    uint4 lights;
    // Temporal amortisation (`flags.w`): the history this frame writes and the last frame's,
    // which it reads (`historyAt`).
    float* historyNow;
    float* historyLast;
    // Each block's range of what this frame marched about it (`clampRangesMain`).
    float* clampRanges;
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
    float4* atmosphereMedium;
    float* atmosphereDensities;
    float* atmospherePhases;
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
    // Each view's own uniforms and the materials' temperature models, not read here.
    uint2 viewBlocks;
    uint2 temperatureMaterials;
    // The distant lights' long-range visibility maps (an array).
    uint longRangeVisibility;
    uint padding0;
}

// The pass's own buffers (`VolumePass`): the volumes, their materials, each view's runs of
// `viewVolumes` (offset and count of those it marches, then of those that shadow them), and the
// volumes each view sees; for the volumes' shadow maps (`volumeShadowMain`), the lights and their
// shadow views' view-projections and the worlds whose volumes cast into them besides the shared
// ones (`ShadowCasters::world`'s: `kAllWorlds`, `kNoWorld` or one), and how many volumes, texels
// across a map and lights there are.
struct VolumeResources {
    VolumeGpu* volumes;
    VolumeMaterialGpu* materials;
    uint4* runs;
    uint* lists;
    LightGpu* shadowLights;
    float4x4* shadowViews;
    int* shadowWorlds;
    // The volumes' lighting caches, optical depths: an instance's slot s (light s, then up and down
    // for the sky's and the ground's light) cell n at (cache.w `cacheSlots` + s cells + n).
    float* lightCache;
    uint volumeCount;
    uint shadowSize;
    uint lightCount;
    uint cacheSlots;
    // What the caches and shadow maps last read (`volumeChangesMain`), and the two indirect
    // dispatches that rebuild them, the caches' then the maps'.
    uint* previous;
    uint* rebuild;
    uint materialCount;
    uint shadowViewCount;
    // The lights' IES profiles (`kProfileRows` x `kProfileColumns` floats each) and their timing
    // (`timingWords` words), which temporal amortisation compares with the last frame's.
    float* profiles;
    // This frame's origin less the last's: added to a point relative to this frame's origin, the
    // same point relative to the last's (temporal amortisation's reprojection).
    float4 originShift;
    uint* timing;
    uint timingWords;
    uint unused;
    // The grids' distance grids (`stepAfterEmptyBricks`), at their fields' `grid.z` words.
    uint* distances;
    // The march's steps per pixel far off, at most (`VolumeSettings::stepsPerPixel`; 0, any).
    float stepsPerPixel;
    // A brick's optical depth across at most, below which its steps are taken as one
    // (`stepsInThinBrick`; 0, never).
    float thinOpticalDepth;
    // Cloud droplets' phase tables (`dropletAt`): their values, each row's angles, and the bands
    // (the row past them the bands' mean).
    float* dropletTables;
    uint dropletAngles;
    uint dropletBands;
    // Cloud droplets' multiple scattering by collectors (`collectorScattering`): the canonical
    // transport table (`transportAt`) and the volumes' light maps (`lightMapMain`, a volume's from
    // its `info.w`, `lightMapSizeOf` texels across at its finest), the largest `lightMapSize`
    // across; and 1 once the table is ready, else 0 (their octaves meanwhile).
    float4* transport;
    float4* lightMaps;
    uint lightMapSize;
    uint collectors;
    // Procedural cloud layers (`cloudLayersDensity`): their noise textures (`CloudNoise`'s, one after
    // another) and each field's block of `kCloudLayerBlock` float4s, a field's from its `grid.x`.
    float* cloudNoise;
    float4* cloudLayers;
    // Cloud layers' shadow maps from their sun (`cloudShadowMain`), a volume's from its
    // `cloudShadowBase`.
    float4* cloudShadows;
}

[[vk::binding(0, 0)]] ConstantBuffer<ShadingResources> shading;
[[vk::binding(1, 0)]] Texture2DArray<float> gDepth;
[[vk::binding(2, 0)]] Texture2DArray<uint4> gIds;
[[vk::binding(3, 0)]] RWTexture2DArray<float> spectral;
[[vk::binding(4, 0)]] StructuredBuffer<uint> grids;
[[vk::binding(5, 0)]] ConstantBuffer<VolumeResources> pass;
// The volumes' shadow maps (`volumeShadowMain`'s output, its own pipeline's).
[[vk::image_format("rgba32f")]] [[vk::binding(6, 0)]] RWTexture2DArray<float4> volumeShadowsOut;
// The dense grids' scatter (`denseScatterMain`, its own pipeline's): the grids' pool it writes and
// the values it reads.
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> gridsOut;
[[vk::binding(8, 0)]] StructuredBuffer<float> denseValues;
// The volumes' temperature for the Temperature var (`kMediaTemperature`, renderer's
// `RenderTargets::MediaTemperature`): per pixel, x the sum over the march of each step's extinction
// reaching the eye where it has a temperature times its kelvin, y the sum of those shares.
[[vk::image_format("rg32f")]] [[vk::binding(9, 0)]] RWTexture2DArray<float2> mediaTemperatureOut;
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
#define atmosphereMultiple (heapTextureArrays[shading.atmosphereMultiple])
#define atmosphereSkyIrradiance (heapTextureArrays[shading.atmosphereSkyIrradiance])
#define atmosphereMedium (shading.atmosphereMedium)
#define atmosphereDensities (shading.atmosphereDensities)
#define atmospherePhases (shading.atmospherePhases)
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
    if (isCloudLayers(field)) {
        return cloudLayersDensity(field, local);
    }
    let index = float3(dot(field.toIndex[0], float4(local, 1.0)),
                       dot(field.toIndex[1], float4(local, 1.0)),
                       dot(field.toIndex[2], float4(local, 1.0)));
    let base = floor(index);
    let f = index - base;
    let i = int3(base);
    float corners[8];
    // Exactly as below, faster: when the eight corners lie in one leaf (8^3 voxels), it is found
    // once and its float values read where they lie; when they lie in one tile above the leaves,
    // they all hold its value.
    if (field.grid.y == PNANOVDB_GRID_TYPE_FLOAT && all((i & 7) != 7)) {
        pnanovdb_coord_t ijk;
        ijk.x = i.x;
        ijk.y = i.y;
        ijk.z = i.z;
        uint level = 0;
        let address = pnanovdb_readaccessor_get_value_address_and_level(
            PNANOVDB_GRID_TYPE_FLOAT, grids, accessor, ijk, level);
        if (level != 0) {
            return pnanovdb_read_float(grids, address);
        }
        for (int c = 0; c < 8; ++c) {
            let offset = ((c & 1) << 6) | (((c >> 1) & 1) << 3) | ((c >> 2) & 1);
            corners[c] = pnanovdb_read_float(grids, pnanovdb_address_offset(address, 4u * uint(offset)));
        }
        let x00 = lerp(corners[0], corners[1], f.x);
        let x10 = lerp(corners[2], corners[3], f.x);
        let x01 = lerp(corners[4], corners[5], f.x);
        let x11 = lerp(corners[6], corners[7], f.x);
        return lerp(lerp(x00, x10, f.y), lerp(x01, x11, f.y), f.z);
    }
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

// Empty-space skipping, exact: when index-space point `index` (on the ray `origin` + `step` t, in
// march steps `dt` from `start` offset by `jitter`) lies in a tile of `field`'s tree that holds no
// values (a node above the leaves, all background) and is not within a voxel of its far side, so
// that every trilinear sample there reads only its zeros, the first step past where the ray
// leaves that interior; else 0.
//
// Reference: K. Museth, "Hierarchical Digital Differential Analyzer for Efficient Ray-Marching in
//   OpenVDB", SIGGRAPH 2014 Talks. https://doi.org/10.1145/2614106.2614136
uint stepAfterEmptyTile(FieldGpu field, float3 index, float3 origin, float3 step, float start,
                        float dt, float jitter, inout pnanovdb_readaccessor_t accessor) {
    if (isCloudLayers(field)) {
        return stepAfterClearSky(field, index, origin, step, start, dt, jitter);
    }
    pnanovdb_coord_t ijk;
    let cell = int3(floor(index));
    ijk.x = cell.x;
    ijk.y = cell.y;
    ijk.z = cell.z;
    let dim = pnanovdb_readaccessor_get_dim(field.grid.y, grids, accessor, ijk);
    if (dim <= 1 || valueAt(field.grid.y, ijk, accessor) != 0.0) {
        return 0;
    }
    let low = float3(cell & int3(~int(dim - 1)));
    let high = low + float(dim) - 1.0;
    if (any(index >= high)) {
        return 0;
    }
    float exit = 1e30;
    for (int a = 0; a < 3; ++a) {
        if (step[a] > 0.0) {
            exit = min(exit, (high[a] - origin[a]) / step[a]);
        } else if (step[a] < 0.0) {
            exit = min(exit, (low[a] - origin[a]) / step[a]);
        }
    }
    return uint(max(ceil((exit - start) / dt - jitter), 0.0));
}

// Distance grids, for empty-space skipping over many leaves at once: per grid, over its active
// bounds in bricks of `kBrick`^3 voxels (its leaves'), the Chebyshev distance in bricks from each
// to the nearest holding a value other than 0, at most `kMaxBrickDistance`. In
// `pass.distances` from a field's `grid.z`: the first brick (index / `kBrick`) and the bricks
// along x, y and z, two words unused, then a word per brick, x fastest, and its coarser levels
// (`sampleLevel`). Built on the GPU whenever the grid changes (`brickOccupancyMain`,
// `brickDistanceMain`).
//
// References:
// - A. Schneider, "Nubis Cubed: Methods (and Madness) to Model and Render Immersive Real-Time
//   Voxel-Based Clouds", SIGGRAPH Advances in Real-Time Rendering, 2023 (a distance field over a
//   voxel cloud's empty space, the march leaping through it).
// - A. Meijster, J. B. T. M. Roerdink and W. H. Hesselink, "A General Algorithm for Computing
//   Distance Transforms in Linear Time", Mathematical Morphology and its Applications to Image and
//   Signal Processing, 2002 (the transform a dimension at a time, which holds for Chebyshev
//   distances).
static const int kBrick = 8;
static const uint kMaxBrickDistance = 16;
static const uint kDistanceHeader = 8;

// Brick `b` (from a distance grid's first) of `dims`'s word, x fastest.
uint brickIndex(int3 b, int3 dims) {
    return uint((b.z * dims.y + b.y) * dims.x + b.x);
}

// Empty-space skipping, exact, as `stepAfterEmptyTile` but over the bricks around `index` that
// `field`'s distance grid finds empty: where every voxel a trilinear sample there reads is 0, the
// first step past where the ray leaves them; else 0.
uint stepAfterEmptyBricks(FieldGpu field, float3 index, float3 origin, float3 step, float start,
                          float dt, float jitter) {
    if (field.grid.z == kNoGrid) {
        return 0;
    }
    let header = pass.distances + field.grid.z;
    let first = int3(asint(header[0]), asint(header[1]), asint(header[2]));
    let dims = int3(header[3], header[4], header[5]);
    let brick = int3(floor(index / float(kBrick)));
    let b = brick - first;
    if (any(b < int3(0)) || any(b >= dims)) {
        return 0;
    }
    let d = int(header[kDistanceHeader + brickIndex(b, dims)]);
    if (d == 0) {
        return 0;
    }
    // The bricks within d - 1 of this one are empty; a sample reads the voxels at and past its
    // own, so up to a voxel short of their far sides.
    let low = float3((brick - (d - 1)) * kBrick);
    let high = float3((brick + d) * kBrick) - 1.0;
    if (any(index >= high) || any(index < low)) {
        return 0;
    }
    float exit = 1e30;
    for (int a = 0; a < 3; ++a) {
        if (step[a] > 0.0) {
            exit = min(exit, (high[a] - origin[a]) / step[a]);
        } else if (step[a] < 0.0) {
            exit = min(exit, (low[a] - origin[a]) / step[a]);
        }
    }
    return uint(max(ceil((exit - start) / dt - jitter), 0.0));
}

// The coarser levels of a field with a distance grid, after its distances (`B` words, a brick
// each): each brick's largest value (`B`), its mean (level 3, `B`), and its eighths' (level 2, 4^3
// voxels each, `8 B`, 2 per brick along each axis, x fastest).
//
// Reference: L. Williams, "Pyramidal Parametrics", SIGGRAPH 1983 (a prefiltered level for samples
//   farther apart than the finest's).
uint bricksOf(uint* header) {
    return header[3] * header[4] * header[5];
}

// `field`'s point at local point `local`, in its index space.
float3 fieldIndex(FieldGpu field, float3 local) {
    return float3(dot(field.toIndex[0], float4(local, 1.0)), dot(field.toIndex[1], float4(local, 1.0)),
                  dot(field.toIndex[2], float4(local, 1.0)));
}

// `field`'s value at index-space point `index` from its level `level` (2 or 3), trilinear between
// the centres of its blocks of 4^3 or 8^3 voxels; 0 outside its bounds.
float sampleLevel(FieldGpu field, float3 index, uint level) {
    let header = pass.distances + field.grid.z;
    let first = int3(asint(header[0]), asint(header[1]), asint(header[2]));
    let bricks = int3(header[3], header[4], header[5]);
    let count = bricksOf(header);
    let across = level == 3 ? kBrick : kBrick / 2;
    let dims = level == 3 ? bricks : 2 * bricks;
    let values = header + kDistanceHeader + (level == 3 ? 2 * count : 3 * count);
    let at = (index - float3(first * kBrick) - 0.5 * float(across - 1)) / float(across);
    let cell = int3(floor(at));
    let f = at - float3(cell);
    float corners[8];
    for (int k = 0; k < 8; ++k) {
        let p = cell + int3(k & 1, (k >> 1) & 1, k >> 2);
        corners[k] = all(p >= int3(0)) && all(p < dims) ? asfloat(values[brickIndex(p, dims)]) : 0.0;
    }
    let x00 = lerp(corners[0], corners[1], f.x);
    let x10 = lerp(corners[2], corners[3], f.x);
    let x01 = lerp(corners[4], corners[5], f.x);
    let x11 = lerp(corners[6], corners[7], f.x);
    return lerp(lerp(x00, x10, f.y), lerp(x01, x11, f.y), f.z);
}

// The level to read `field` at for steps `voxels` of its voxels long: its own (0) up to 4, its
// blocks of 4^3 (2) up to 8, then of 8^3 (3); its own without a distance grid.
uint levelFor(FieldGpu field, float voxels) {
    return field.grid.z == kNoGrid ? 0u : voxels >= 8.0 ? 3u : voxels >= 4.0 ? 2u : 0u;
}

// `field`'s value at local point `local`, from level `level` (`levelFor`).
float sampleFieldAt(FieldGpu field, float3 local, uint level,
                    inout pnanovdb_readaccessor_t accessor) {
    return level == 0 ? sampleField(field, local, accessor)
                      : sampleLevel(field, fieldIndex(field, local), level);
}

// Adaptive steps: where index-space point `index` of `volume`'s density (on the ray `origin` +
// `step` t, in march steps `dt` from `start` offset by `jitter`, step `i` of `steps`) lies in a
// brick so thin that the most of it the ray could cross holds back less than
// `pass.thinOpticalDepth` (its largest value through a brick's width), the steps as far as the ray
// leaves it, to take as one; else 1.
//
// Reference: J. Novák, I. Georgiev, J. Hanika and W. Jarosz, "Monte Carlo Methods for Volumetric
//   Light Transport Simulation", Eurographics State of the Art Reports, 2018 (bounds on extinction
//   over a grid's cells, majorants, to take longer steps where it is low).
uint stepsInThinBrick(VolumeGpu volume, float3 index, float3 origin, float3 step, float start,
                      float dt, float jitter, uint i, uint steps) {
    let field = volume.density;
    if (field.grid.z == kNoGrid || pass.thinOpticalDepth <= 0.0) {
        return 1;
    }
    let header = pass.distances + field.grid.z;
    let first = int3(asint(header[0]), asint(header[1]), asint(header[2]));
    let bricks = int3(header[3], header[4], header[5]);
    let brick = int3(floor(index / float(kBrick)));
    let b = brick - first;
    if (any(b < int3(0)) || any(b >= bricks)) {
        return 1;
    }
    let largest = asfloat(header[kDistanceHeader + bricksOf(header) + brickIndex(b, bricks)]);
    let across = float(kBrick) / max(length(step), 1e-9);
    if (max(largest, 0.0) * volume.params.x * across >= pass.thinOpticalDepth) {
        return 1;
    }
    let low = float3(brick * kBrick);
    let high = low + float(kBrick);
    float exit = 1e30;
    for (int a = 0; a < 3; ++a) {
        if (step[a] > 0.0) {
            exit = min(exit, (high[a] - origin[a]) / step[a]);
        } else if (step[a] < 0.0) {
            exit = min(exit, (low[a] - origin[a]) / step[a]);
        }
    }
    let past = uint(max(ceil((exit - start) / dt - jitter), 0.0));
    return past > i ? min(past - i, steps - i) : 1u;
}

// A distance grid's build (`constants`): x, y and z its bricks along each axis and w the axis
// (`brickDistanceMain`); offsets xyz its first brick, w its grid's byte offset in the pool; flags
// x the grid's value type, y its first word in `pass.distances`. `historyLast` and `historyNow`
// the words a pass reads and writes.

// A distance grid's bricks, one each: 0 where any voxel holds a value other than 0, else
// `kMaxBrickDistance`; brick 0's thread writes the grid's header.
[shader("compute")]
[numthreads(64, 1, 1)]
void brickOccupancyMain(uint3 id: SV_DispatchThreadID) {
    let dims = int3(constants.layout.xyz);
    if (id.x >= uint(dims.x * dims.y * dims.z)) {
        return;
    }
    let first = asint(constants.offsets.xyz);
    if (id.x == 0) {
        let header = pass.distances + constants.flags.y;
        header[0] = asuint(first.x);
        header[1] = asuint(first.y);
        header[2] = asuint(first.z);
        header[3] = uint(dims.x);
        header[4] = uint(dims.y);
        header[5] = uint(dims.z);
    }
    let b = int3(int(id.x) % dims.x, (int(id.x) / dims.x) % dims.y, int(id.x) / (dims.x * dims.y));
    let low = (first + b) * kBrick;
    FieldGpu field;
    field.toIndex[0] = float4(0.0);
    field.toIndex[1] = float4(0.0);
    field.toIndex[2] = float4(0.0);
    field.grid = uint4(constants.offsets.w, constants.flags.x, kNoGrid, 0);
    var accessor = accessorOf(field);
    let type = constants.flags.x;
    pnanovdb_coord_t ijk;
    ijk.x = low.x;
    ijk.y = low.y;
    ijk.z = low.z;
    // Whether any voxel holds a value other than 0, the largest, and the sums of each eighth.
    bool occupied = false;
    float largest = -1e30;
    float eighths[8];
    for (int e = 0; e < 8; ++e) {
        eighths[e] = 0.0;
    }
    if (pnanovdb_readaccessor_get_dim(type, grids, accessor, ijk) > uint(kBrick)) {
        // A tile above the leaves holds one value for the whole brick.
        let value = valueAt(type, ijk, accessor);
        occupied = value != 0.0;
        largest = value;
        for (int e = 0; e < 8; ++e) {
            eighths[e] = value * float(kBrick * kBrick * kBrick / 8);
        }
    } else {
        for (int v = 0; v < kBrick * kBrick * kBrick; ++v) {
            let at = int3(v % kBrick, (v / kBrick) % kBrick, v / (kBrick * kBrick));
            ijk.x = low.x + at.x;
            ijk.y = low.y + at.y;
            ijk.z = low.z + at.z;
            let value = valueAt(type, ijk, accessor);
            occupied = occupied || value != 0.0;
            largest = max(largest, value);
            let octant = at / (kBrick / 2);
            eighths[octant.x + 2 * octant.y + 4 * octant.z] += value;
        }
    }
    ((uint*)constants.historyNow)[id.x] = occupied ? 0u : kMaxBrickDistance;
    // Its coarser levels (`sampleLevel`), and the largest value (`stepsInThinBrick`).
    let count = uint(dims.x * dims.y * dims.z);
    let values = pass.distances + constants.flags.y + kDistanceHeader;
    values[count + id.x] = asuint(largest);
    float sum = 0.0;
    for (int e = 0; e < 8; ++e) {
        sum += eighths[e];
        let cell = 2 * b + int3(e & 1, (e >> 1) & 1, e >> 2);
        values[3 * count + brickIndex(cell, 2 * dims)] =
            asuint(eighths[e] / float(kBrick * kBrick * kBrick / 8));
    }
    values[2 * count + id.x] = asuint(sum / float(kBrick * kBrick * kBrick));
}

// A pass of a distance grid's transform along axis `constants.layout.w`: each brick's least, over
// the bricks along that axis within `kMaxBrickDistance`, of the larger of the distance to it and
// its value from the pass before. Along x, y then z, from the occupancy, the Chebyshev distance.
[shader("compute")]
[numthreads(64, 1, 1)]
void brickDistanceMain(uint3 id: SV_DispatchThreadID) {
    let dims = int3(constants.layout.xyz);
    if (id.x >= uint(dims.x * dims.y * dims.z)) {
        return;
    }
    let axis = constants.layout.w;
    let input = (uint*)constants.historyLast;
    let b = int3(int(id.x) % dims.x, (int(id.x) / dims.x) % dims.y, int(id.x) / (dims.x * dims.y));
    var best = input[id.x];
    for (uint k = 1; k < best; ++k) {
        for (int sign = -1; sign <= 1; sign += 2) {
            var n = b;
            n[axis] += sign * int(k);
            if (n[axis] >= 0 && n[axis] < dims[axis]) {
                best = min(best, max(k, input[brickIndex(n, dims)]));
            }
        }
    }
    ((uint*)constants.historyNow)[id.x] = best;
}

pnanovdb_readaccessor_t accessorOf(FieldGpu field) {
    pnanovdb_grid_handle_t grid;
    grid.address.byte_offset = field.grid.x == kNoGrid || isCloudLayers(field) ? 0 : field.grid.x;
    let root = pnanovdb_tree_get_root(grids, pnanovdb_grid_get_tree(grids, grid));
    pnanovdb_readaccessor_t accessor;
    pnanovdb_readaccessor_init(accessor, root);
    return accessor;
}

// Planck's law: 2 h c^2 in W um^4 m^-2 sr^-1 and h c / k in um K.
static const float kPlanckC1 = 1.191042972e8;
static const float kPlanckC2 = 14387.76877;

// Planck's spectral radiance from its parts at each wavelength: `scale` 2 h c^2 / lambda^5 (in W
// m^-2 sr^-1 nm^-1, times what else multiplies it), `x` h c / (lambda k), and the temperature's
// reciprocal `inverse`, 1/K.
float4 planckOf(float4 scale, float4 x, float inverse) {
    let exponent = x * inverse;
    // exp overflows past 88: the tail is then exp(-x) itself.
    return select(exponent > 80.0, scale * exp(-exponent),
                  scale / (exp(min(exponent, 80.0)) - 1.0));
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

// Cloud droplets' phase function (`DropletPhaseTable`) of `material` at phase angle cosine `cosG`
// (as `phaseOf`'s, the scattering angle's cosine negated), per steradian: band `row`'s, or the
// bands' mean at row `pass.dropletBands`, linear between the table's angles, theta(u) = pi / 2
// (1 - cos pi u).
float dropletAt(VolumeMaterialGpu material, uint row, float cosG) {
    let theta = acos(clamp(-cosG, -1.0, 1.0));
    let u = acos(clamp(1.0 - 2.0 * theta / PI, -1.0, 1.0)) / PI;
    let at = u * float(pass.dropletAngles - 1);
    let i = min(uint(at), pass.dropletAngles - 2);
    let base = asuint(material.phase.y) + row * pass.dropletAngles + i;
    return lerp(pass.dropletTables[base], pass.dropletTables[base + 1], at - float(i));
}

// The material's phase function at phase angle cosine `cosG` (towards the light and towards the
// eye), per steradian, its anisotropy scaled by `spread` (1 as it is, 0 isotropic).
float phaseOf(VolumeMaterialGpu material, float cosG, float spread = 1.0) {
    switch (uint(material.phase.x)) {
    case kCloudDroplets:
        // The bands' mean, as light scattered many times blurs their colours.
        return lerp(1.0 / (4.0 * PI), dropletAt(material, pass.dropletBands, cosG), spread);
    case kDoubleHenyeyGreenstein:
        return doubleHenyeyGreenstein(material.phase.y * spread, material.phase.z, cosG) /
               (4.0 * PI);
    case kDualHenyeyGreenstein:
        // Two lobes, g (params.x) and g2 (phase.y), the second's share phase.z.
        return lerp(henyeyGreenstein(material.params.x * spread, cosG),
                    henyeyGreenstein(material.phase.y * spread, cosG), material.phase.z) /
               (4.0 * PI);
    case kRayleigh:
        return lerp(1.0, 0.75 * (1.0 + cosG * cosG), spread) / (4.0 * PI);
    case kIsotropic:
        return 1.0 / (4.0 * PI);
    default:
        // Forward scattering (g > 0) is towards the light's far side: phase angle near pi.
        return henyeyGreenstein(material.params.x * spread, cosG) / (4.0 * PI);
    }
}

// The light a light scatters towards the eye per unit of what reaches the medium's surface, its
// optical depth through the media `depth` on the way: single scattering, or Wrenninge's octaves
// (each order of scattering a less, its extinction b less and its anisotropy c less than the
// one before's).
//
// Reference: M. Wrenninge, C. Kulla and V. Lundqvist, "Oz: The Great and Volumetric", SIGGRAPH
//   2013 Talks. https://doi.org/10.1145/2504459.2504518
float octavesOf(VolumeMaterialGpu material, float cosG, float depth) {
    let octaves = min(uint(material.scattering.x), maxOctaves());
    float sum = 0.0;
    float a = 1.0;
    float b = 1.0;
    float c = 1.0;
    for (uint i = 0; i < octaves; ++i) {
        sum += a * exp(-b * depth) * phaseOf(material, cosG, c);
        a *= material.scattering.y;
        b *= material.scattering.z;
        c *= material.scattering.w;
    }
    return sum;
}

// The optical depth from world point `world` towards a light along unit `towards`, up to world
// distance `far`, through every volume of the view's run `run` whose link group the light's shadow
// mask `casters` holds (`steps` through each one the segment crosses).
float volumesDepth(uint2 run, float3 world, float3 towards, float far, uint casters,
                   uint steps) {
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
        let dt = (exit - near) / float(steps);
        float sum = 0.0;
        for (uint i = 0; i < steps; ++i) {
            sum += max(sampleField(volume.density, o + d * (near + (float(i) + 0.5) * dt),
                                   accessor),
                       0.0);
        }
        depth += sum * volume.params.x * dt;
    }
    return depth;
}

// The render settings' steps through each volume towards a light, up and down for the sky's light
// (0: unoccluded) and through the air in front of the volumes (0: none), and the most octaves of
// any material (`VolumeSettings`, a byte each of `constants.flags.z`).
uint shadowSteps() {
    return constants.flags.z & 0xFFu;
}
uint skyLightSteps() {
    return (constants.flags.z >> 8) & 0xFFu;
}
uint airSteps() {
    return (constants.flags.z >> 16) & 0xFFu;
}
uint maxOctaves() {
    return constants.flags.z >> 24;
}

// The most octaves any render settings allow (`VolumeSettings::maxOctaves`), and no distant light
// for a span's march to treat as its sun.
static const uint kMaxOctaves = 8;
static const uint kNoSun = 0xFFFFFFFFu;

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

// The share of distant light `light`'s disc its long-range view's casters (such as moons) leave
// `world`, at a band group's `wavelengths` (nm): shading.slang's `longRangeGroup`.
static const float4 kLongRangeWavelengths = float4(300.0, 450.0, 800.0, 2400.0);

float4 longRangeGroup(LightGpu light, float3 world, float4 wavelengths) {
    if (light.position.w != 0.0 || light.map.z <= 0.0) {
        return float4(1.0);
    }
    let clip = mul(lightViews[uint(light.map.z) - 1], float4(world, 1.0));
    let uv = (float2(0.5 * clip.x + 0.5, 0.5 - 0.5 * clip.y) - light.beamExtra.yz) /
             max(light.beamExtra.w, 1e-30);
    if (clip.z < 1.0 || any(uv < float2(0.0)) || any(uv > float2(1.0))) {
        return float4(1.0);
    }
    let maps = heapTextureArrays[shading.longRangeVisibility];
    let first = 2.0 * light.map.w;
    let t = saturate((clip.z - light.patternExtent.x) /
                     max(light.patternExtent.y - light.patternExtent.x, 1e-30));
    let held = lerp(maps.SampleLevel(tableSampler, float3(uv, first), 0.0),
                    maps.SampleLevel(tableSampler, float3(uv, first + 1.0), 0.0), t);
    float4 alphas;
    for (uint i = 0; i < 4; ++i) {
        alphas[i] = limbDarkening(kLongRangeWavelengths[i]);
    }
    float4 result;
    for (uint k = 0; k < 4; ++k) {
        let alpha = limbDarkening(wavelengths[k]);
        result[k] = alpha >= alphas.x ? held.x : held.w;
        for (uint i = 0; i < 3; ++i) {
            if (alpha < alphas[i] && alpha >= alphas[i + 1]) {
                result[k] = lerp(held[i + 1], held[i],
                                 (alpha - alphas[i + 1]) / (alphas[i] - alphas[i + 1]));
            }
        }
    }
    return result;
}

// `longRangeGroup` for the air's marches (atmosphere.slang's `IAirShadow`): the long-range
// shadows alone, the scene's near shadows left out of the air in front of the volumes.
// Points km from the planet's centre `centre` (km from the frame's origin), the lights in
// `slots` (`skyLightsOf`'s order).
struct VolumeFarCasters : IAirShadow {
    uint slots[kSkyLights];
    float3 centre;

    float atPoint(uint light, float3 point) {
        return 1.0;
    }

    float4 eclipseAt(uint light, float3 point, float4 wavelengths) {
        let world = (point + centre) * 1e3;
        return longRangeGroup(shading.lights[slots[light]], world, wavelengths);
    }
}

// The view's sky lights' (`skyLightsOf`'s order) long-range shadows in the air about the planet
// centred `centre` (km).
VolumeFarCasters volumeFarCasters(uint4 viewLights, uint lightCount, float3 centre) {
    VolumeFarCasters casters;
    casters.centre = centre;
    uint count = 0;
    for (uint k = 0; k < lightCount && count < kSkyLights; ++k) {
        let light = shading.lights[viewLights.z + k];
        if ((viewLights.x & (1u << k)) == 0 || uint(light.position.w) != kDistantLight ||
            light.shadow.y >= 0.5) {
            continue;
        }
        casters.slots[count] = viewLights.z + k;
        ++count;
    }
    return casters;
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

// Lasers' beams integrated across a pixel's ray exactly (laser_beams.slang), up to
// `kMaxBeamCrossings` of them, added at the march's steps holding them.
static const uint kMaxBeamCrossings = 16;

struct BeamCrossings {
    uint count;
    // Along the ray, by distance: where each crossing is, its light (per unit of its laser's
    // spectrum), its laser's slot, and towards the laser (xyz) and how far (w).
    float t[kMaxBeamCrossings];
    float value[kMaxBeamCrossings];
    uint slot[kMaxBeamCrossings];
    float4 towards[kMaxBeamCrossings];
}

// Whether laser `light`'s beams are integrated across rays (`BeamCrossings`), not sampled.
bool analyticBeams(LightGpu light) {
    return uint(light.position.w) == kLaser && light.patternExtent.z > 0.5;
}

// Laser `light`'s beam (shading.slang's `beamOf`).
Beam beamOf(LightGpu light) {
    Beam beam;
    beam.position = light.position.xyz;
    beam.axis = light.axis.xyz;
    beam.x = normalize(light.extentX.xyz);
    beam.beam = light.beam;
    beam.waistAt = light.beamExtra.x;
    beam.kind = light.patternExtent.z;
    beam.params = light.patternParams[0];
    beam.more = light.patternParams[1];
    return beam;
}

// Adds `crossing` of laser slot `slot` to `crossings`, kept by distance along the ray.
void insertCrossing(inout BeamCrossings crossings, BeamCrossing crossing, uint slot) {
    if (!(crossing.value > 0.0) || crossings.count == kMaxBeamCrossings) {
        return;
    }
    var at = crossings.count;
    while (at > 0 && crossings.t[at - 1] > crossing.t) {
        crossings.t[at] = crossings.t[at - 1];
        crossings.value[at] = crossings.value[at - 1];
        crossings.slot[at] = crossings.slot[at - 1];
        crossings.towards[at] = crossings.towards[at - 1];
        --at;
    }
    crossings.t[at] = crossing.t;
    crossings.value[at] = crossing.value;
    crossings.slot[at] = slot;
    crossings.towards[at] = crossing.towards;
    ++crossings.count;
}

// The crossings of the pixel's ray with the beams of the view's lasers integrated across rays,
// at the light each laser shows over the view's span of time.
BeamCrossings beamCrossingsOf(uint4 viewLights, uint lightCount, uint view, float3 origin,
                              float3 direction, float3 footprintX, float3 footprintY) {
    BeamCrossings crossings;
    crossings.count = 0;
    for (uint k = 0; k < lightCount; ++k) {
        if ((viewLights.x & (1u << k)) == 0) {
            continue;
        }
        let light = shading.lights[viewLights.z + k];
        if (!analyticBeams(light)) {
            continue;
        }
        var shown = 0.0;
        for (uint s = 0; s < kMaxLit; ++s) {
            shown += lightTiming[(view * kMaxLights + k) * kMaxLit + s].y;
        }
        let norm = patternNorms[k];
        let efficiency = norm > 0.0 ? light.beamExtra.z / norm : 0.0;
        let beam = beamOf(light);
        var walk = beamWalk(beam, origin, direction, 1e30);
        BeamCrossing crossing;
        while (nextCrossing(walk, beam, shown * light.beamExtra.w, shown * efficiency, origin,
                            direction, footprintX, footprintY, crossing)) {
            insertCrossing(crossings, crossing, k);
        }
    }
    return crossings;
}

// Adds to `emitted` the light the medium scatters towards the eye from the beam crossings of the
// step from `start` to `end` along the ray (`next` the first not yet passed), where the
// transmittance is `transmittance` and the extinction `sigma`: per crossing, as a step lights
// the medium (`octavesOf`, the volumes' depth towards the laser, its geometry's shadow).
void addBeamCrossings(inout float4 emitted[kMaxGroups], inout uint next,
                      BeamCrossings crossings, float start, float end, float transmittance,
                      float sigma, VolumeGpu volume, uint materialIndex, float3 origin,
                      float3 direction, float3 o, float3 d, uint4 viewLights, uint2 shadowRun,
                      float mapTexels, uint groups) {
    while (next < crossings.count && crossings.t[next] < start) {
        ++next;
    }
    let material = volumeMaterials[materialIndex];
    while (next < crossings.count && crossings.t[next] < end) {
        let c = next;
        ++next;
        let slot = crossings.slot[c];
        let light = shading.lights[viewLights.z + slot];
        if (((light.links.x >> volume.info.z) & 1u) == 0) {
            continue;
        }
        let at = origin + direction * crossings.t[c];
        let local = o + d * crossings.t[c];
        let towards = crossings.towards[c].xyz;
        let lightSlot = viewLights.z + slot;
        let depth = light.links.z == 0 ? 0.0
                    : hasCache(volume, lightSlot)
                        ? cachedDepth(volume, lightSlot, local)
                        : volumesDepth(shadowRun, at, towards, crossings.towards[c].w,
                                       light.links.y, shadowSteps());
        // Its geometry's shadow on the beam's own axis there, which its map, as narrow, sees.
        let axis = light.position.xyz - towards * crossings.towards[c].w;
        let lit = transmittance * sigma * octavesOf(material, dot(towards, -direction), depth) *
                  crossings.value[c] *
                  geometryShadow(light, axis, viewLights.y, mapTexels);
        if (lit <= 0.0) {
            continue;
        }
        for (uint g = 0; g < groups; ++g) {
            emitted[g] += material.albedo[g] * lit * (light.spectrum[g] + light.tinted[g]);
        }
    }
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
    } else if (light.links.w != 0) {
        // A strobe or flash: its light over the view's span of the sensor clock (shading.slang's).
        var weight = 0.0;
        for (uint s = 0; s < kMaxLit; ++s) {
            weight += lightTiming[(view * kMaxLights + slot) * kMaxLit + s].y;
        }
        reach.scale *= weight;
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

// Adds to `emitted` the light of the sky and the ground around world point `world` (relative to
// the frame's origin) that the medium there scatters towards the eye, `scale` its scattering
// coefficient times the step's weight and the material's sky light scale. Each of the view's
// distant lights the atmosphere carries (those of `viewLights` that light link group `linkGroup`)
// lights the sky (the sky irradiance table) and, Lambertian with the atmosphere's ground albedo
// with the sky, the ground; the medium sees the sky's irradiance on level ground over the upper
// half of its surround and the ground's over the lower, isotropically, each dimmed by the
// volumes above or below it.
//
// References:
// - E. Bruneton, "Precomputed Atmospheric Scattering: a New Implementation", 2017,
//   `GetSunAndSkyIrradiance`. https://ebruneton.github.io/precomputed_atmospheric_scattering/
// - S. Hillaire, "Physically Based Sky, Atmosphere and Cloud Rendering in Frostbite", SIGGRAPH
//   2016 Course: Physically Based Shading in Theory and Practice (ambient light on clouds).
void emitSkyLight(inout float4 emitted[kMaxGroups], uint materialIndex, float3 world,
                  uint2 shadowRun, uint4 viewLights, uint lightCount, uint linkGroup,
                  float4 air[2],
                  uint groups, float scale, float2 visible) {
    let bottom = air[0].w;
    let top = air[1].x;
    let p = world * 1e-3 - air[0].xyz;
    let r = max(length(p), bottom);
    let up = normalize(p);
    // What the volumes let through from above and below: `visible` (from a lighting cache), else
    // marched.
    let steps = skyLightSteps();
    let marched = visible.x < 0.0 && steps > 0;
    let above = visible.x >= 0.0 ? visible.x
                : marched ? exp(-volumesDepth(shadowRun, world, up, 1e30, ~0u, steps))
                          : 1.0;
    let below = visible.x >= 0.0 ? visible.y
                : marched ? exp(-volumesDepth(shadowRun, world, -up, 1e30, ~0u, steps))
                          : 1.0;
    for (uint k = 0; k < lightCount; ++k) {
        let light = shading.lights[viewLights.z + k];
        if ((viewLights.x & (1u << k)) == 0 || uint(light.position.w) != kDistantLight ||
            light.shadow.y >= 0.5 || ((light.links.x >> linkGroup) & 1u) == 0) {
            continue;
        }
        let towards = light.position.xyz;
        let tangent = light.extentY.w;
        let sine = tangent * rsqrt(1.0 + tangent * tangent);
        let muSun = dot(up, towards);
        for (uint g = 0; g < groups; ++g) {
            let sky = skyIrradiance(atmosphereSkyIrradiance, tableSampler, r, muSun, g, bottom,
                                    top);
            let toSun = sunTransmittance(atmosphereTransmittance, tableSampler, up * bottom,
                                         towards, sine, g, bottom, top);
            let ground = groundAlbedo(atmosphereMedium, g) * (toSun * max(muSun, 0.0) + sky);
            let irradiance = light.spectrum[g] + light.tinted[g];
            emitted[g] += volumeMaterials[materialIndex].albedo[g] * irradiance *
                          (sky * above + ground * below) * (0.5 / PI * scale);
        }
    }
}

// The distant lights the atmosphere carries (shading.slang's `skyLights`): those of `viewLights`
// that are not a dome's directions, with their irradiance in band group `g`.
SkyLights skyLightsOf(uint4 viewLights, uint lightCount, uint g) {
    SkyLights sky;
    sky.count = 0;
    for (uint k = 0; k < lightCount && sky.count < kSkyLights; ++k) {
        let light = shading.lights[viewLights.z + k];
        if ((viewLights.x & (1u << k)) == 0 || uint(light.position.w) != kDistantLight ||
            light.shadow.y >= 0.5) {
            continue;
        }
        let tangent = light.extentY.w;
        sky.towards[sky.count] =
            float4(light.position.xyz, tangent * rsqrt(1.0 + tangent * tangent));
        sky.irradiance[sky.count] = light.spectrum[g] + light.tinted[g];
        ++sky.count;
    }
    return sky;
}

// Temporal amortisation (`constants.flags.w`): on; whether the last frame's history holds for
// this dispatch's views; whether this dispatch marches only this frame's pixel of each block, a
// thread each (the dispatch after reprojects the others, so no group's threads wait on others
// marching); blocks of 4 x 4 pixels rather than 2 x 2; bits 4 to 5 and 6 to 7 this frame's
// pixel's x and y in its block; and whether volumes that changed reproject too, clamped to what
// this frame marched about them (`clampToMarched`).
static const uint kTemporal = 1u;
static const uint kHistoryHeld = 2u;
static const uint kMarchOnly = 4u;
static const uint kBlocksOfFour = 8u;
static const uint kAnimated = 256u;
// The frame holds the Temperature var: every pixel writes `mediaTemperatureOut`, those that
// reproject from the history.
static const uint kMediaTemperature = 512u;

void writeMediaTemperature(uint3 id, float2 media) {
    if ((constants.flags.w & kMediaTemperature) != 0) {
        mediaTemperatureOut[id] = media;
    }
}

// Temporal amortisation's blocks' pixels across, and this frame's pixel in each.
uint temporalBlock() {
    return (constants.flags.w & kBlocksOfFour) != 0 ? 4u : 2u;
}

uint2 marchedInBlock() {
    return uint2((constants.flags.w >> 4) & 3u, (constants.flags.w >> 6) & 3u);
}

// A history's channels per pixel: the volumes' transmittance (-1 where none were in view), the
// distance along the ray of what they hold back, the geometry's (`kNoSurface` for none), the
// Temperature var's two sums (`mediaTemperatureOut`), and what they add towards the eye in each
// band, two bands a channel as halves (`historyBand`).
static const uint kHistoryTransmittance = 0;
static const uint kHistoryFront = 1;
static const uint kHistorySurface = 2;
static const uint kHistoryMediaKelvin = 3;
static const uint kHistoryMediaShare = 4;
static const uint kHistoryBands = 5;
static const float kNoSurface = 1e30;

// Where channel `channel` of view `view`'s pixel `xy` is in a history of images `size` across: past
// each view's inverse view-projection and view-projection (32 floats, by rows), a plane per
// channel of every view's pixels.
uint historyAt(uint channel, uint view, uint2 xy, uint2 size) {
    let views = constants.lights.w;
    return 32 * views + ((channel * views + view) * size.y + xy.y) * size.x + xy.x;
}

// Band `b`'s light added of view `view`'s pixel `xy` in `history`: the pixel's largest band's,
// then each band's share of it in 16 bits, two bands a channel, half the memory of a float each,
// as what is added ends in an image of far fewer bits.
float historyBand(float* history, uint b, uint view, uint2 xy, uint2 size) {
    let largest = history[historyAt(kHistoryBands, view, xy, size)];
    let word = ((uint*)history)[historyAt(kHistoryBands + 1 + b / 2, view, xy, size)];
    let share = (b & 1u) != 0 ? word >> 16 : word & 0xFFFFu;
    return largest * float(share) / 65535.0;
}

// Each band's light added of view `view`'s pixel `xy` into `history` (`historyBand`).
void writeHistoryBands(float* history, uint bands, uint view, uint2 xy, uint2 size,
                       float4 added[kMaxGroups]) {
    float largest = 0.0;
    for (uint b = 0; b < bands; ++b) {
        largest = max(largest, added[b / 4][b % 4]);
    }
    history[historyAt(kHistoryBands, view, xy, size)] = largest;
    let toShare = largest > 0.0 ? 65535.0 / largest : 0.0;
    for (uint b = 0; b < bands; b += 2) {
        let low = uint(clamp(round(added[b / 4][b % 4] * toShare), 0.0, 65535.0));
        let high =
            b + 1 < bands
                ? uint(clamp(round(added[(b + 1) / 4][(b + 1) % 4] * toShare), 0.0, 65535.0))
                : 0u;
        ((uint*)history)[historyAt(kHistoryBands + 1 + b / 2, view, xy, size)] =
            low | (high << 16);
    }
}

// Row `row` of matrix `matrix` (0 the inverse view-projection, 1 the view-projection) of view
// `view` in history `history`.
float4 historyRow(float* history, uint view, uint matrix, uint row) {
    let at = 32 * view + 16 * matrix + 4 * row;
    return float4(history[at], history[at + 1], history[at + 2], history[at + 3]);
}

bool temporal() {
    return (constants.flags.w & kTemporal) != 0;
}

// Half resolution (`VolumeSettings::resolutionScale`): the march's pixels a 2 x 2 of the image's
// each, written only to the history, which `volumeUpsampleMain` composites at the image's own.
static const uint kHalfResolution = 1024u;

bool halfResolution() {
    return (constants.flags.w & kHalfResolution) != 0;
}

// Whether the march writes its history: to reproject (temporal amortisation), or to upsample.
bool keepsHistory() {
    return temporal() || halfResolution();
}

// Whether pixel `id` (its ray from `origin` along unit `direction` to the geometry `tMax` along it)
// shows the last frame's result, reprojected: the volumes' `transmittance`, `front` and what they
// add (`added`) where the point they held back last frame was then. Not where that falls outside
// the image or held no volumes, nor where the geometry behind differs (moved, or uncovered).
//
// References:
// - A. Schneider, "Nubis: Authoring Real-Time Volumetric Cloudscapes with the Decima Engine",
//   SIGGRAPH Advances in Real-Time Rendering, 2017 (a pixel of each block marched a frame, the
//   others reprojected).
bool reproject(uint3 id, uint2 size, float3 origin, float3 direction, float tMax, float middle,
               uint bands, out float transmittance, out float front,
               out float4 added[kMaxGroups], out float2 media) {
    transmittance = 1.0;
    front = 0.0;
    media = float2(0.0);
    for (uint g = 0; g < kMaxGroups; ++g) {
        added[g] = float4(0.0);
    }
    let last = constants.historyLast;
    // How far along the ray the volumes were at this pixel last frame; where they held nothing
    // back, `middle` (the middle of the ray's way through them).
    let held = last[historyAt(kHistoryTransmittance, id.z, id.xy, size)];
    if (held < 0.0) {
        return false;
    }
    let guess = held < 1.0 ? last[historyAt(kHistoryFront, id.z, id.xy, size)] : middle;
    float4x4 lastViewProj;
    for (uint r = 0; r < 4; ++r) {
        lastViewProj[r] = historyRow(last, id.z, 1, r);
    }
    let clip = mul(lastViewProj, float4(origin + direction * guess + pass.originShift.xyz, 1.0));
    if (clip.w <= 0.0) {
        return false;
    }
    let at = float2((clip.x / clip.w + 1.0) * 0.5 * float(size.x),
                    (1.0 - clip.y / clip.w) * 0.5 * float(size.y));
    if (any(at < float2(0.0)) || at.x >= float(size.x) || at.y >= float(size.y)) {
        return false;
    }
    let then = uint2(at);
    transmittance = last[historyAt(kHistoryTransmittance, id.z, then, size)];
    if (transmittance < 0.0) {
        return false;
    }
    let surface = last[historyAt(kHistorySurface, id.z, then, size)];
    if ((surface >= kNoSurface) != (tMax >= kNoSurface)) {
        return false;
    }
    if (tMax < kNoSurface) {
        // The geometry there last frame, and here now, the same within 2% of its distance.
        let ndc = float2((float(then.x) + 0.5) / float(size.x) * 2.0 - 1.0,
                         1.0 - (float(then.y) + 0.5) / float(size.y) * 2.0);
        float4x4 lastInverse;
        for (uint r = 0; r < 4; ++r) {
            lastInverse[r] = historyRow(last, id.z, 0, r);
        }
        float3 lastOrigin;
        float3 lastToFar;
        cameraRayThrough(lastInverse, float4(0.0), ndc, lastOrigin, lastToFar);
        let was = lastOrigin + normalize(lastToFar) * surface - pass.originShift.xyz;
        if (length(was - (origin + direction * tMax)) > 0.02 * tMax) {
            return false;
        }
    }
    front = last[historyAt(kHistoryFront, id.z, then, size)];
    media = float2(last[historyAt(kHistoryMediaKelvin, id.z, then, size)],
                   last[historyAt(kHistoryMediaShare, id.z, then, size)]);
    for (uint b = 0; b < bands; ++b) {
        added[b / 4][b % 4] = historyBand(last, b, id.z, then, size);
    }
    return true;
}

// Whether pixel `id` (the geometry `tMax` along its ray) shows what its block's pixel marched this
// frame shows: where volumes were in view there and the geometry behind is the same within 2% of
// its distance (or both have none).
//
// Reference: B. Karis, "High-Quality Temporal Supersampling", SIGGRAPH Advances in Real-Time
//   Rendering, 2014 (and bilateral upsampling of a lower resolution's results by depth).
bool fromMarched(uint3 id, uint2 size, float tMax, uint bands, out float transmittance,
                 out float front, out float4 added[kMaxGroups], out float2 media) {
    transmittance = 1.0;
    front = 0.0;
    media = float2(0.0);
    for (uint g = 0; g < kMaxGroups; ++g) {
        added[g] = float4(0.0);
    }
    let block = temporalBlock();
    let marched = id.xy / block * block + marchedInBlock();
    if (any(marched >= size)) {
        return false;
    }
    let now = constants.historyNow;
    transmittance = now[historyAt(kHistoryTransmittance, id.z, marched, size)];
    if (transmittance < 0.0) {
        return false;
    }
    let surface = now[historyAt(kHistorySurface, id.z, marched, size)];
    if ((surface >= kNoSurface) != (tMax >= kNoSurface) ||
        (tMax < kNoSurface && abs(surface - tMax) > 0.02 * tMax)) {
        return false;
    }
    front = now[historyAt(kHistoryFront, id.z, marched, size)];
    media = float2(now[historyAt(kHistoryMediaKelvin, id.z, marched, size)],
                   now[historyAt(kHistoryMediaShare, id.z, marched, size)]);
    for (uint b = 0; b < bands; ++b) {
        added[b / 4][b % 4] = historyBand(now, b, id.z, marched, size);
    }
    return true;
}

// Where block `blockAt` of view `view`'s range of channel `channel` (0 the transmittance, then
// each band's) is in `constants.clampRanges`, its lowest then (`high`) its highest: a plane each
// of every view's blocks, `blocks` of them across and down.
uint clampRangeAt(uint channel, bool high, uint view, uint2 blockAt, uint2 blocks) {
    let views = constants.lights.w;
    return (((2 * channel + (high ? 1u : 0u)) * views + view) * blocks.y + blockAt.y) * blocks.x +
           blockAt.x;
}

// Where what the volumes show changed, each block's range, channel by channel, of what this frame
// marched at the pixels of the block and the blocks about it, for the block's other pixels to
// clamp what they reproject to (`clampToMarched`): found once a block rather than at each pixel.
// A thread a block (`id.xy`) of view `id.z`.
//
// Reference: B. Karis, "High-Quality Temporal Supersampling", SIGGRAPH Advances in Real-Time
//   Rendering, 2014 (history clamped to the current frame's neighbourhood).
[shader("compute")]
[numthreads(8, 8, 1)]
void clampRangesMain(uint3 id: SV_DispatchThreadID) {
    if (pass.rebuild[6] != 0 || pass.rebuild[7] == 0) {
        return;
    }
    // The march's size: half the image's across and down at half resolution.
    let imageSize = viewLoad(constants.offsets.w).xy;
    let size = halfResolution() ? (imageSize + 1) / 2 : imageSize;
    let block = temporalBlock();
    let blocks = (size + block - 1) / block;
    if (any(id.xy >= blocks)) {
        return;
    }
    let bands = uint(asfloat(viewLoad(constants.offsets.z)).z);
    let now = constants.historyNow;
    for (uint channel = 0; channel <= bands; ++channel) {
        float low = 1e30;
        float high = -1e30;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                let at = (int2(id.xy) + int2(dx, dy)) * int(block) + int2(marchedInBlock());
                if (any(at < int2(0)) || any(at >= int2(size))) {
                    continue;
                }
                let marched = uint2(at);
                let t = now[historyAt(kHistoryTransmittance, id.z, marched, size)];
                // Volumes not in view there: no light added, all let through.
                let value =
                    channel == 0
                        ? (t >= 0.0 ? t : 1.0)
                        : (t >= 0.0
                               ? historyBand(now, channel - 1, id.z, marched, size)
                               : 0.0);
                low = min(low, value);
                high = max(high, value);
            }
        }
        constants.clampRanges[clampRangeAt(channel, false, id.z, id.xy, blocks)] = low;
        constants.clampRanges[clampRangeAt(channel, true, id.z, id.xy, blocks)] = high;
    }
}

// Pixel `id`'s reprojected `transmittance` and `added` clamped, band by band, to its block's range
// (`clampRangesMain`): what moved on leaves no trail.
void clampToMarched(uint3 id, uint2 size, uint bands, inout float transmittance,
                    inout float4 added[kMaxGroups]) {
    let block = temporalBlock();
    let blocks = (size + block - 1) / block;
    let blockAt = id.xy / block;
    let ranges = constants.clampRanges;
    let lowT = ranges[clampRangeAt(0, false, id.z, blockAt, blocks)];
    let highT = ranges[clampRangeAt(0, true, id.z, blockAt, blocks)];
    if (lowT > highT) {
        return;
    }
    transmittance = clamp(transmittance, lowT, highT);
    for (uint b = 0; b < bands; ++b) {
        added[b / 4][b % 4] =
            clamp(added[b / 4][b % 4], ranges[clampRangeAt(b + 1, false, id.z, blockAt, blocks)],
                  ranges[clampRangeAt(b + 1, true, id.z, blockAt, blocks)]);
    }
}

// Pixel `id`'s history this frame: the volumes' `transmittance` (-1 for none in view), `front`,
// the geometry's distance `tMax` and what they add (`added`).
void writeHistory(uint3 id, uint2 size, float transmittance, float front, float tMax, uint bands,
                  float4 added[kMaxGroups], float2 media) {
    let now = constants.historyNow;
    now[historyAt(kHistoryTransmittance, id.z, id.xy, size)] = transmittance;
    now[historyAt(kHistoryFront, id.z, id.xy, size)] = front;
    now[historyAt(kHistorySurface, id.z, id.xy, size)] = tMax;
    now[historyAt(kHistoryMediaKelvin, id.z, id.xy, size)] = media.x;
    now[historyAt(kHistoryMediaShare, id.z, id.xy, size)] = media.y;
    writeHistoryBands(now, bands, id.z, id.xy, size, added);
}

// Pixel `id`'s radiance through the volumes: the light behind dimmed by their `transmittance`
// (its polarization too), and what they add, unpolarized.
void composite(uint3 id, uint bands, bool unwritten, float transmittance,
               float4 added[kMaxGroups]) {
    let stokes = constants.flags.y;
    for (uint b = 0; b < bands; ++b) {
        let layer = int3(id.xy, id.z * bands + b);
        let behind = unwritten ? 0.0 : spectral[layer];
        spectral[layer] = behind * transmittance + added[b / 4][b % 4];
        if (stokes != 0) {
            for (uint plane = 1; plane <= 2; ++plane) {
                let at = int3(layer.xy, layer.z + int(plane * stokes));
                spectral[at] = unwritten ? 0.0 : spectral[at] * transmittance;
            }
        }
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void volumeMain(uint3 thread : SV_DispatchThreadID) {
    // A thread a pixel. With a history: in one dispatch, this frame's pixel of each block, a
    // thread each; in the one after, the block's others, a thread each (along x, the blocks'
    // others in turn), so that no thread idles in either, whether the others reproject or march.
    let temporalFlags = constants.flags.w;
    let blockSize = temporalBlock();
    let marched = marchedInBlock();
    let marchOnly = (temporalFlags & kMarchOnly) != 0;
    let held = temporal() && (temporalFlags & kHistoryHeld) != 0;
    var id = thread;
    if (marchOnly) {
        id.xy = blockSize * thread.xy + marched;
    } else if (held) {
        let others = blockSize * blockSize - 1;
        let k = thread.x % others;
        let skipped = marched.y * blockSize + marched.x;
        let p = k < skipped ? k : k + 1;
        id.xy = blockSize * uint2(thread.x / others, thread.y) + uint2(p % blockSize, p / blockSize);
    }
    // The view's own fields at their offsets in its block; those every view shares (the bands,
    // size, shadow maps and air) at theirs from the start.
    let block = (constants.layout.x + id.z) * constants.layout.y;
    // The image's size, and the march's: half of it across and down at half resolution.
    let imageSize = viewLoad(constants.offsets.w);
    let size = halfResolution() ? uint4((imageSize.xy + 1) / 2, imageSize.zw) : imageSize;
    if (temporal() && !marchOnly && all(thread.xy == 0)) {
        // The view's camera, for the next frame to reproject through.
        let camera = loadMatrix(block + constants.offsets.x);
        let viewProj = invert(camera);
        for (uint r = 0; r < 4; ++r) {
            for (uint c = 0; c < 4; ++c) {
                constants.historyNow[32 * id.z + 4 * r + c] = camera[r][c];
                constants.historyNow[32 * id.z + 16 + 4 * r + c] = viewProj[r][c];
            }
        }
    }
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let range = asfloat(viewLoad(constants.offsets.z));
    let bands = uint(range.z);
    // The frame's band groups, (bands + 3) / 4, compiled in.
    let groups = kMaxGroups;
    // The image's pixel whose geometry the ray meets: at half resolution, its 2 x 2's nearest the
    // ray through its middle, the lower right.
    let pixel = int4(halfResolution() ? min(2 * id.xy + 1, imageSize.xy - 1) : id.xy, id.z, 0);
    let background = gIds.Load(pixel).x == kBackground;
    // A background shading left unwritten: no radiance behind the volumes.
    let unwritten = background && constants.flags.x == 0;
    let runs = viewRuns[constants.layout.x + id.z];
    let run = runs.xy;
    let shadowRun = runs.zw;
    if (run.y == 0) {
        if (keepsHistory()) {
            constants.historyNow[historyAt(kHistoryTransmittance, id.z, id.xy, size.xy)] = -1.0;
        }
        // At half resolution, the upsampling writes the image.
        if (halfResolution()) {
            return;
        }
        writeMediaTemperature(id, float2(0.0));
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
        if (keepsHistory()) {
            constants.historyNow[historyAt(kHistoryTransmittance, id.z, id.xy, size.xy)] = -1.0;
        }
        // At half resolution, the upsampling writes the image.
        if (halfResolution()) {
            return;
        }
        writeMediaTemperature(id, float2(0.0));
        if (unwritten) {
            for (uint b = 0; b < bands; ++b) {
                writeNoRadiance(id, id.z * bands + b);
            }
        }
        return;
    }
    // With temporal amortisation, a pixel of each 2 x 2 marches a frame, and every pixel when the
    // last frame's history does not hold or what the volumes show changed; the others show the
    // last frame's where it reprojects.
    // Changed at a stroke, every pixel marches; their grids written again, the others reproject
    // clamped to what this frame marched, where animated volumes may; the Temperature var's sums
    // reprojected with the rest.
    let changed = pass.rebuild[7] != 0;
    if (held && !marchOnly && pass.rebuild[6] == 0 &&
        (!changed || (temporalFlags & kAnimated) != 0)) {
        float kept;
        float front;
        float4 added[kMaxGroups];
        float2 media;
        let middle = 0.5 * (spans[0].x + spans[count - 1].y);
        // Else, uncovered or come into view, its block's pixel marched this frame where the
        // geometry behind is as far (an upsampling that keeps to edges).
        if (reproject(id, size.xy, origin, direction, tMax, middle, bands, kept, front, added,
                      media) ||
            fromMarched(id, size.xy, tMax, bands, kept, front, added, media)) {
            if (changed) {
                clampToMarched(id, size.xy, bands, kept, added);
            }
            writeHistory(id, size.xy, kept, front, tMax, bands, added, media);
            if (!halfResolution()) {
                writeMediaTemperature(id, media);
                composite(id, bands, unwritten, kept, added);
            }
            return;
        }
    }

    // The view's lights: those of its world (a bit each) from its first; its shadow maps' size;
    // the atmosphere.
    let view = constants.layout.x + id.z;
    let viewLights = viewLoad(block + constants.lights.x);
    let lightCount = min(size.w, kMaxLights);
    let mapTexels = asfloat(viewLoad(constants.lights.y)).z;
    float4 air[2];
    air[0] = asfloat(viewLoad(constants.lights.z));
    air[1] = asfloat(viewLoad(constants.lights.z + 16));
    // The pixel's extent per unit distance along the ray, to its neighbours' rays.
    float3 neighbour;
    float3 toNeighbour;
    cameraRayThrough(invViewProj, strip, ndc + float2(2.0 / float(size.x), 0.0), neighbour,
                     toNeighbour);
    let footprintX = normalize(toNeighbour) - direction;
    cameraRayThrough(invViewProj, strip, ndc - float2(0.0, 2.0 / float(size.y)), neighbour,
                     toNeighbour);
    let footprintY = normalize(toNeighbour) - direction;
    // The angle a pixel spans, about, for steps no shorter than a pixel far off.
    let pixelAngle = max(length(footprintX), length(footprintY));

    // Per band, what the volumes add towards the eye, each step's integrated with the
    // transmittance up to it.
    float4 emitted[kMaxGroups];
    for (uint g = 0; g < groups; ++g) {
        emitted[g] = float4(0.0);
    }
    float transmittance = 1.0;
    // The distance along the ray of what the volumes hold back, weighted by it: where the air in
    // front of them ends.
    float frontSum = 0.0;
    // The Temperature var's sums (`mediaTemperatureOut`).
    float2 media = float2(0.0);
    let jitter = hash(uint3(id.xy, constants.layout.w ^ (id.z * 7919u)));
    // Lasers' beams the ray crosses, added at the steps holding them (`BeamCrossings`).
    let crossings =
        beamCrossingsOf(viewLights, lightCount, view, origin, direction, footprintX, footprintY);
    for (uint s = 0; s < count && transmittance > kOpaque; ++s) {
        let volume = volumes[which[s]];
        let materialIndex = volume.info.x;
        let material = volumeMaterials[materialIndex];
        uint nextCrossing = 0;
        let o = float3(dot(volume.toLocal[0], float4(origin, 1.0)),
                       dot(volume.toLocal[1], float4(origin, 1.0)),
                       dot(volume.toLocal[2], float4(origin, 1.0)));
        let d = float3(dot(volume.toLocal[0].xyz, direction), dot(volume.toLocal[1].xyz, direction),
                       dot(volume.toLocal[2].xyz, direction));
        let span = spans[s];
        let hasDensity = volume.density.grid.x != kNoGrid;
        let hasTemperature = volume.temperature.grid.x != kNoGrid;
        let hasEmission = volume.emission.grid.x != kNoGrid;
        let emits = material.params.z != 0.0;
        let scatters = material.params.w != 0.0 && hasDensity;
        // The sky's and the ground's light, with an atmosphere.
        let skyLit = material.phase.w > 0.0 && air[1].y != 0.0;
        let kelvinBase = volume.params.y;
        let glows = material.params.y > 0.0 && (hasTemperature || kelvinBase > 0.0);
        var densityAccessor = accessorOf(volume.density);
        var temperatureAccessor = accessorOf(volume.temperature);
        var emissionAccessor = accessorOf(volume.emission);
        // Without an emission field of its own, what the volume adds goes with its density, so the
        // steps in an empty tile of the density's tree add nothing and are skipped: exactly, the
        // steps after the same as without skipping. The ray in the density grid's index space.
        let skips = hasDensity && !hasEmission;
        let indexOrigin = float3(dot(volume.density.toIndex[0], float4(o, 1.0)),
                                 dot(volume.density.toIndex[1], float4(o, 1.0)),
                                 dot(volume.density.toIndex[2], float4(o, 1.0)));
        let indexStep = float3(dot(volume.density.toIndex[0].xyz, d),
                               dot(volume.density.toIndex[1].xyz, d),
                               dot(volume.density.toIndex[2].xyz, d));
        // Steps of the volume's own length (`params.w`), or where a pixel spans more than
        // `stepsPerPixel` of them, farther off, of its span's share: the span in segments, each's
        // steps as long as that at its start, until where that has doubled.
        // The first distant light lighting the volume, which every step sees alike: its phase for
        // each octave towards the eye and its spectrum through the material's albedo, found once
        // here rather than at each step.
        uint sun = kNoSun;
        float sunPhases[kMaxOctaves];
        // Its first octave's phase in each band: cloud droplets' each its own (their fogbow's and
        // glory's colours), else `sunPhases[0]`.
        float4 sunFirst[kMaxGroups];
        float4 sunTints[kMaxGroups];
        uint3 sunLinks = uint3(0);
        float3 sunTowards = float3(0.0);
        float sunSine = 0.0;
        let sunOctaves = min(uint(material.scattering.x), maxOctaves());
        for (uint k = 0; k < lightCount && sun == kNoSun && scatters; ++k) {
            if ((viewLights.x & (1u << k)) == 0) {
                continue;
            }
            let light = shading.lights[viewLights.z + k];
            if (uint(light.position.w) != kDistantLight || light.shadow.y >= 0.5 ||
                light.links.w != 0 || ((light.links.x >> volume.info.z) & 1u) == 0 ||
                analyticBeams(light)) {
                continue;
            }
            sun = k;
            sunLinks = light.links.xyz;
            sunTowards = light.position.xyz;
            let tangent = light.extentY.w;
            sunSine = tangent * rsqrt(1.0 + tangent * tangent);
            let cosG = dot(sunTowards, -direction);
            float spread = 1.0;
            for (uint n = 0; n < kMaxOctaves; ++n) {
                sunPhases[n] = n < sunOctaves ? phaseOf(material, cosG, spread) : 0.0;
                spread *= material.scattering.w;
            }
            let droplets = uint(material.phase.x) == kCloudDroplets;
            for (uint g = 0; g < groups; ++g) {
                sunTints[g] = volumeMaterials[materialIndex].albedo[g] *
                              (light.spectrum[g] + light.tinted[g]);
                sunFirst[g] = float4(sunPhases[0]);
                if (droplets) {
                    for (uint c = 0; c < 4; ++c) {
                        sunFirst[g][c] = 4 * g + c < pass.dropletBands
                                             ? dropletAt(material, 4 * g + c, cosG)
                                             : 0.0;
                    }
                }
            }
        }
        // Cloud droplets' multiple scattering by collectors (`collectorScattering`), with their
        // light map built for this sun: once, where the ray first reaches the surface's density;
        // the octaves before (thin wisps in front of it), single scattering after.
        let collecting = sun != kNoSun && pass.collectors != 0 && volume.info.w != kNoCache &&
                         uint(material.phase.x) == kCloudDroplets &&
                         lightMapFrameOf(volume).slot == viewLights.z + sun;
        var collected = false;
        // Where its light map holds no surface about the ray, the octaves after all.
        var declined = false;
        // The collectors' light, held until the march leaves the span, then as much of it as the
        // cloud behind its surface along the ray holds back (a grazing ray through a thin edge,
        // little): through the transmittance up to the surface, `collectedAt`, already.
        float collectedScale = 0.0;
        float collectedAt = 1.0;
        // Steps since the ray left the piece of cloud it collected for, and the pieces collected.
        uint gapSteps = 0;
        uint pieces = 0;
        // The last step's position and density, for where between it and the next the density
        // reached the surface's.
        float previousT = -1e30;
        float previousRho = 0.0;
        float4 collectedTints[kMaxGroups];
        for (uint g = 0; g < groups; ++g) {
            collectedTints[g] = float4(0.0);
        }
        // A black body's glow, its parts at each band's wavelength found once for the span: what
        // the material absorbs times 2 h c^2 / lambda^5, and h c / (lambda k) (`planckOf`).
        float4 glowScales[kMaxGroups];
        float4 glowExponents[kMaxGroups];
        for (uint g = 0; g < groups; ++g) {
            let um = (range.x + (float4(g * 4) + float4(0.5, 1.5, 2.5, 3.5)) * range.y) * 1e-3;
            glowScales[g] = glows ? (1.0 - volumeMaterials[materialIndex].albedo[g]) * kPlanckC1 /
                                        (um * um * um * um * um) * 1e-3
                                  : float4(0.0);
            glowExponents[g] = kPlanckC2 / um;
        }
        let perPixel = pass.stepsPerPixel > 0.0 ? pixelAngle / pass.stepsPerPixel : 0.0;
        var budget = constants.layout.z;
        var start = span.x;
        var end = span.x;
        var steps = 0u;
        var dt = 0.0;
        for (uint i = 0;; ++i) {
            if (transmittance <= kOpaque) {
                break;
            }
            if (i >= steps) {
                // The next segment.
                start = end;
                if (start >= span.y || budget == 0) {
                    break;
                }
                let stepLength = max(volume.params.w, perPixel * start);
                end = perPixel > 0.0
                          ? min(span.y, max(2.0 * stepLength / perPixel, start + stepLength))
                          : span.y;
                steps = clamp(uint(ceil((end - start) / stepLength)), 1u, budget);
                budget -= steps;
                dt = (end - start) / float(steps);
                i = 0;
            }
            var t = start + (float(i) + jitter) * dt;
            if (skips) {
                // Over the empty bricks around, else the empty tile.
                let index = indexOrigin + indexStep * t;
                var next = stepAfterEmptyBricks(volume.density, index, indexOrigin, indexStep,
                                                start, dt, jitter);
                if (next <= i) {
                    next = stepAfterEmptyTile(volume.density, index, indexOrigin, indexStep,
                                              start, dt, jitter, densityAccessor);
                }
                if (next > i) {
                    i = next - 1;
                    continue;
                }
            }
            // A thin brick's steps taken as one, at their middle (`stepsInThinBrick`); its density
            // read at the level for the step's length (`levelFor`).
            let firstStep = i;
            let merged = skips ? stepsInThinBrick(volume, indexOrigin + indexStep * t, indexOrigin,
                                                  indexStep, start, dt, jitter, i, steps)
                               : 1u;
            i += merged - 1;
            t += 0.5 * float(merged - 1) * dt;
            let stepDt = float(merged) * dt;
            let local = o + d * t;
            float rho = 0.0;
            if (hasDensity && merged > 1) {
                // A thin brick's density over its steps, each read as the march reads it: its
                // depth and what it emits as without merging, its light found once.
                for (uint k = 0; k < merged; ++k) {
                    let at = o + d * (start + (float(firstStep + k) + jitter) * dt);
                    rho += max(sampleField(volume.density, at, densityAccessor), 0.0);
                }
                rho /= float(merged);
            } else if (hasDensity) {
                let level = levelFor(volume.density, length(indexStep) * dt);
                rho = max(sampleFieldAt(volume.density, local, level, densityAccessor), 0.0);
            }
            // Past a gap after the piece of cloud it collected for (two steps below half the
            // surface's density, or empty space skipped), that piece's collectors' light as much
            // as it held back, and the next piece's own from where the ray reaches its surface: a
            // ray grazing one cloud's thin edge then into another behind it takes the far one's,
            // not the edge's all the way.
            if (collected && pieces + 1 < kCollectedPieces) {
                let skipped = previousT < t - 1.5 * stepDt;
                gapSteps = skipped ? 2u : rho < 0.5 * material.phase.z ? gapSteps + 1 : 0u;
                if (gapSteps >= 2) {
                    let held = saturate(1.0 - transmittance / max(collectedAt, 1e-9));
                    for (uint g = 0; g < groups; ++g) {
                        emitted[g] += collectedTints[g] * (collectedScale * held);
                    }
                    collected = false;
                    collectedScale = 0.0;
                    gapSteps = 0;
                    ++pieces;
                }
            }
            let sigma = rho * volume.params.x;
            // The powder effect (`VolumeMaterialGpu::extras`) on the light scattered in.
            let powdered = 1.0 - material.extras.x * exp(-sigma * material.extras.y);
            let opacity = 1.0 - exp(-sigma * stepDt);
            // The transmittance integrated over the step, from its start.
            let weight = sigma > 1e-6 ? transmittance * opacity / sigma : transmittance * stepDt;
            if (emits) {
                let field = hasEmission
                                ? max(sampleField(volume.emission, local, emissionAccessor), 0.0)
                                : rho;
                for (uint g = 0; g < groups; ++g) {
                    emitted[g] += volumeMaterials[materialIndex].emission[g] * (weight * field);
                }
            }
            if (glows && sigma > 0.0) {
                // The field's value scaled, from the material's temperature (what its 0 is).
                let kelvin =
                    hasTemperature
                        ? kelvinBase + sampleField(volume.temperature, local, temperatureAccessor) *
                                           volume.params.z
                        : kelvinBase;
                if (kelvin > 0.0) {
                    // Its share of what reaches the eye, transmittance times opacity.
                    media += float2(kelvin, 1.0) * (weight * sigma);
                    let scale = weight * sigma * material.params.y;
                    let inverse = 1.0 / kelvin;
                    for (uint g = 0; g < groups; ++g) {
                        emitted[g] += scale * planckOf(glowScales[g], glowExponents[g], inverse);
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
                    if (k == sun) {
                        // The span's sun: its phases and tints found for the span.
                        let slot = viewLights.z + k;
                        // Cloud layers from their shadow map and a short march (as three-geospatial's
                        // clouds), else the lighting cache, else marched.
                        let depth = sunLinks.z == 0 ? 0.0
                                    : hasCloudShadow(volume, slot)
                                        ? cloudSunDepth(volume, local, world, sunTowards, jitter)
                                    : hasCache(volume, slot)
                                        ? cachedDepth(volume, slot, local)
                                        : volumesDepth(shadowRun, world, sunTowards, 1e30,
                                                       sunLinks.y, shadowSteps());
                        // The first octave per band (`sunFirst`), the others alike in all.
                        let first = exp(-depth);
                        float others = 0.0;
                        float a = material.scattering.y;
                        float b = material.scattering.z;
                        // The surface reached: what its collectors bring, through the
                        // transmittance up to it, where the light reaches its lit surface.
                        let surfaced =
                            collecting && !declined && (collected || rho >= material.phase.z);
                        float scattered = 0.0;
                        if (surfaced && !collected) {
                            collected = true;
                            float3 litSurface;
                            // Where the density reached the surface's, between the last step
                            // (empty, a step back, where steps were skipped) and this one.
                            let adjacent = previousT > t - 1.5 * stepDt;
                            let behindT = adjacent ? previousT : t - stepDt;
                            let behindRho = adjacent ? previousRho : 0.0;
                            let tSurface =
                                lerp(behindT, t,
                                     saturate((material.phase.z - behindRho) /
                                              max(rho - behindRho, 1e-6)));
                            scattered = collectorScattering(
                                volume, lightMapFrameOf(volume), origin + direction * tSurface,
                                -direction, depth, hasCache(volume, slot) ? slot : kNoSun,
                                litSurface);
                            declined = scattered < 0.0;
                            scattered = declined ? 0.0
                                                 : scattered * transmittance *
                                                       geometryShadow(shading.lights[slot],
                                                                      litSurface, viewLights.y,
                                                                      mapTexels);
                            collectedScale = scattered;
                            collectedAt = transmittance;
                        }
                        // Past the cloud's surface, its collectors' alone.
                        for (uint n = 1; n < sunOctaves && !(surfaced && !declined); ++n) {
                            others += a * exp(-b * depth) * sunPhases[n];
                            a *= material.scattering.y;
                            b *= material.scattering.z;
                        }
                        let lit = weight * sigma * powdered *
                                  geometryShadow(shading.lights[slot], world, viewLights.y,
                                                 mapTexels);
                        if (lit > 0.0 || scattered > 0.0) {
                            let throughAir = air[1].y != 0.0;
                            for (uint g = 0; g < groups; ++g) {
                                var tint = sunTints[g];
                                if (throughAir) {
                                    tint *= sunTransmittance(atmosphereTransmittance,
                                                             tableSampler,
                                                             world * 1e-3 - air[0].xyz,
                                                             sunTowards, sunSine, g, air[0].w,
                                                             air[1].x);
                                }
                                emitted[g] += lit * (first * sunFirst[g] + others) * tint;
                                if (scattered > 0.0) {
                                    collectedTints[g] = tint;
                                }
                            }
                        }
                        continue;
                    }
                    let light = shading.lights[viewLights.z + k];
                    // Light linking: only the lights whose links hold the volume light it; and
                    // lasers' beams integrated across the ray come after.
                    if (((light.links.x >> volume.info.z) & 1u) == 0 || analyticBeams(light)) {
                        continue;
                    }
                    let reach = reachOf(light, k, view, world, footprint);
                    if (reach.scale <= 0.0) {
                        continue;
                    }
                    // From the instance's lighting cache, or marched.
                    let slot = viewLights.z + k;
                    let depth = light.links.z == 0 ? 0.0
                                : hasCache(volume, slot)
                                    ? cachedDepth(volume, slot, local)
                                    : volumesDepth(shadowRun, world, reach.towards, reach.far,
                                                   light.links.y, shadowSteps());
                    let lit = weight * sigma * powdered *
                              octavesOf(material, dot(reach.towards, -direction), depth) *
                              reach.scale *
                              geometryShadow(light, world, viewLights.y, mapTexels);
                    if (lit <= 0.0) {
                        continue;
                    }
                    // A distant light through the air (not refracted, nor eclipsed by the
                    // celestial bodies), and past the casters far towards it, such as moons
                    // (`longRangeGroup`).
                    let throughAir =
                        uint(light.position.w) == kDistantLight && air[1].y != 0.0;
                    let tangent = light.extentY.w;
                    let sunSine = tangent * rsqrt(1.0 + tangent * tangent);
                    for (uint g = 0; g < groups; ++g) {
                        var spectrum = (light.spectrum[g] * reach.focus + light.tinted[g]) *
                                       longRangeGroup(light, world,
                                                      range.x + (float4(g * 4) +
                                                                 float4(0.5, 1.5, 2.5, 3.5)) *
                                                                    range.y);
                        if (throughAir) {
                            spectrum *= sunTransmittance(atmosphereTransmittance, tableSampler,
                                                         world * 1e-3 - air[0].xyz,
                                                         reach.towards, sunSine, g, air[0].w,
                                                         air[1].x);
                        }
                        emitted[g] += volumeMaterials[materialIndex].albedo[g] * lit * spectrum;
                    }
                }
                addBeamCrossings(emitted, nextCrossing, crossings, start + float(firstStep) * dt,
                                 start + float(i + 1) * dt, transmittance, sigma, volume,
                                 materialIndex, origin, direction, o, d, viewLights, shadowRun,
                                 mapTexels, groups);
                if (skyLit) {
                    // How much of the sky above and the ground below the volumes let through:
                    // from the lighting cache's slots past the lights, else marched.
                    float2 visible = float2(-1.0);
                    if (hasCache(volume, pass.lightCount + 1)) {
                        visible = float2(exp(-cachedDepth(volume, pass.lightCount, local)),
                                         exp(-cachedDepth(volume, pass.lightCount + 1, local)));
                    }
                    emitSkyLight(emitted, materialIndex, world, shadowRun, viewLights, lightCount,
                                 volume.info.z, air, groups,
                                 weight * sigma * powdered * material.phase.w,
                                 visible);
                }
            }
            previousT = t;
            previousRho = rho;
            frontSum += transmittance * opacity * t;
            transmittance *= 1.0 - opacity;
        }
        if (collectedScale > 0.0) {
            let held = saturate(1.0 - transmittance / max(collectedAt, 1e-9));
            for (uint g = 0; g < groups; ++g) {
                emitted[g] += collectedTints[g] * (collectedScale * held);
            }
        }
    }

    // The air in front of the volumes, as far as their mean depth: what reaches the eye from them
    // dims across it, and the light it scatters shows in front of them. The light behind already
    // holds the air all the way along (shading's aerial perspective, or the sky), the air in front
    // of the volumes included, so of that air's light only the share the volumes hold back goes:
    // behind T + T_air emitted + S_air (1 - T).
    float4 airLight[kMaxGroups];
    float4 airThrough[kMaxGroups];
    for (uint g = 0; g < groups; ++g) {
        airLight[g] = float4(0.0);
        airThrough[g] = float4(1.0);
    }
    if (air[1].y != 0.0 && !unwritten && transmittance < 1.0 && airSteps() > 0) {
        let front = frontSum / (1.0 - transmittance);
        let bottom = air[0].w;
        let top = air[1].x;
        let p = aboveGround(origin * 1e-3 - air[0].xyz, bottom);
        float start;
        float end;
        if (clipSegment(p, direction, front * 1e-3, top, start, end)) {
            // Eclipsed by the casters far towards the lights (`longRangeGroup`).
            let far = volumeFarCasters(viewLights, lightCount, air[0].xyz);
            for (uint g = 0; g < groups; ++g) {
                var eclipse = noEclipse();
                eclipse.wavelengths =
                    range.x + (float4(g * 4) + float4(0.5, 1.5, 2.5, 3.5)) * range.y;
                scatterAlongLightsShadowed(
                    atmosphereMedium, atmosphereDensities, atmospherePhases,
                    atmosphereTransmittance, atmosphereMultiple, tableSampler, p, direction,
                    max(start, 0.0), end, skyLightsOf(viewLights, lightCount, g), eclipse, g,
                    airSteps(), far, airLight[g], airThrough[g]);
            }
        }
    }

    // Absorption dims the light behind as it is, its polarization too; what the volumes emit
    // and scatter is unpolarized.
    float4 added[kMaxGroups];
    for (uint g = 0; g < groups; ++g) {
        added[g] = airThrough[g] * emitted[g] + airLight[g] * (1.0 - transmittance);
    }
    if (keepsHistory()) {
        let front = transmittance < 1.0 ? frontSum / (1.0 - transmittance) : 0.0;
        writeHistory(id, size.xy, transmittance, front, tMax, bands, added, media);
    }
    if (!halfResolution()) {
        writeMediaTemperature(id, media);
        composite(id, bands, unwritten, transmittance, added);
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

// Half resolution's image (`halfResolution`): pixel `id.xy` of view `id.z` through the volumes as
// the march found them at its four nearest (this frame's history, at the middles of the image's
// 2 x 2s), weighted bilinearly and by whether the geometry behind each is as far as the pixel's
// own, within 5% (or both have none): the volumes keep to the edges of what stands before them.
//
// References:
// - J. Kopf, M. F. Cohen, D. Lischinski and M. Uyttendaele, "Joint Bilateral Upsampling", ACM
//   Transactions on Graphics 26(3), SIGGRAPH 2007.
// - S. Hillaire, "Physically Based Sky, Atmosphere and Cloud Rendering in Frostbite", SIGGRAPH
//   2016 Course (volumetric clouds traced at a lower resolution, upsampled by depth).
[shader("compute")]
[numthreads(8, 8, 1)]
void volumeUpsampleMain(uint3 id: SV_DispatchThreadID) {
    let block = (constants.layout.x + id.z) * constants.layout.y;
    let imageSize = viewLoad(constants.offsets.w);
    if (id.x >= imageSize.x || id.y >= imageSize.y) {
        return;
    }
    let size = (imageSize.xy + 1) / 2;
    let bands = uint(asfloat(viewLoad(constants.offsets.z)).z);
    let pixel = int4(id.xyz, 0);
    let background = gIds.Load(pixel).x == kBackground;
    let unwritten = background && constants.flags.x == 0;
    // How far along its ray the pixel's geometry is (`kNoSurface` for none).
    float tMax = kNoSurface;
    if (!background) {
        let invViewProj = loadMatrix(block + constants.offsets.x);
        let strip = asfloat(viewLoad(block + constants.offsets.y));
        let ndc = float2((float(id.x) + 0.5) / float(imageSize.x) * 2.0 - 1.0,
                         1.0 - (float(id.y) + 0.5) / float(imageSize.y) * 2.0);
        float3 origin;
        float3 toFar;
        cameraRayThrough(invViewProj, strip, ndc, origin, toFar);
        let direction = normalize(toFar);
        tMax = dot(cameraPoint(invViewProj, strip, ndc, gDepth.Load(pixel)) - origin, direction);
    }
    // The march's samples about the pixel, in their own pixels.
    let at = (float2(id.xy) + 0.5) * 0.5 - 0.5;
    let base = int2(floor(at));
    let f = at - float2(base);
    let now = constants.historyNow;
    float total = 0.0;
    float transmittance = 0.0;
    float2 media = float2(0.0);
    float4 added[kMaxGroups];
    for (uint g = 0; g < kMaxGroups; ++g) {
        added[g] = float4(0.0);
    }
    bool seen = false;
    for (int k = 0; k < 4; ++k) {
        let q = uint2(clamp(base + int2(k & 1, k >> 1), int2(0), int2(size) - 1));
        let surface = now[historyAt(kHistorySurface, id.z, q, size)];
        let same = (surface >= kNoSurface) == (tMax >= kNoSurface) &&
                   (tMax >= kNoSurface || abs(surface - tMax) <= 0.05 * tMax);
        let weight = ((k & 1) != 0 ? f.x : 1.0 - f.x) * ((k >> 1) != 0 ? f.y : 1.0 - f.y) *
                     (same ? 1.0 : 1e-3);
        total += weight;
        let t = now[historyAt(kHistoryTransmittance, id.z, q, size)];
        // Volumes not in view there: all let through, nothing added.
        if (t < 0.0) {
            transmittance += weight;
            continue;
        }
        seen = true;
        transmittance += weight * t;
        media += weight * float2(now[historyAt(kHistoryMediaKelvin, id.z, q, size)],
                                 now[historyAt(kHistoryMediaShare, id.z, q, size)]);
        for (uint b = 0; b < bands; ++b) {
            added[b / 4][b % 4] += weight * historyBand(now, b, id.z, q, size);
        }
    }
    if (!seen) {
        writeMediaTemperature(id, float2(0.0));
        if (unwritten) {
            for (uint b = 0; b < bands; ++b) {
                writeNoRadiance(id, id.z * bands + b);
            }
        }
        return;
    }
    let inverse = total > 0.0 ? 1.0 / total : 0.0;
    for (uint g = 0; g < kMaxGroups; ++g) {
        added[g] *= inverse;
    }
    writeMediaTemperature(id, media * inverse);
    composite(id, bands, unwritten, transmittance * inverse, added);
}

// The volumes' shadow maps, which shading reads (shading.slang's `volumeTransmittance`): per light
// shadow view `id.z` and texel, along the view's ray through the texel, the depths (the view's
// NDC z, standard depth) where the volumes whose group the view's light's shadow mask holds begin
// and end (layer 3 z, xy; begin past end where there are none), and the transmittance at
// `kShadowSlices` depths evenly between them (layers 3 z + 1 and 3 z + 2), each marched in
// `kShadowSubsteps` steps from the one before.
static const uint kShadowSlices = 8;
static const uint kShadowSubsteps = 4;

// `VolumeResources::shadowWorlds`' (drawable.h's ShadowCasters'): every world's volumes, or only
// the shared.
static const int kAllWorlds = -1;
static const int kNoWorld = -2;

// Whether `volume` casts into a shadow view of world `viewWorld` (`VolumeResources::shadowWorlds`)
// for a light whose shadow group is `casters`.
bool castsInto(VolumeGpu volume, int viewWorld, uint casters) {
    let world = int(volume.info.y);
    return volume.density.grid.x != kNoGrid && ((casters >> volume.info.z) & 1u) != 0 &&
           (world == -1 || viewWorld == kAllWorlds || world == viewWorld);
}

// The extinction coefficient at world point `world` of every volume that casts into a shadow view
// of world `viewWorld` for shadow group `casters`.
float extinctionAt(float3 world, int viewWorld, uint casters) {
    float sigma = 0.0;
    for (uint i = 0; i < pass.volumeCount; ++i) {
        let volume = volumes[i];
        if (!castsInto(volume, viewWorld, casters)) {
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
    let viewWorld = pass.shadowWorlds[view];
    float near = 1.0;
    float far = back;
    for (uint i = 0; i < pass.volumeCount; ++i) {
        let volume = volumes[i];
        if (!castsInto(volume, viewWorld, casters)) {
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
            opticalDepth +=
                extinctionAt(from + step * (float(j) + 0.5), viewWorld, casters) * length;
        }
        slices[i] = exp(-opticalDepth);
        from = to;
    }
    volumeShadowsOut[int3(id.xy, 3 * view + 1)] =
        float4(slices[0], slices[1], slices[2], slices[3]);
    volumeShadowsOut[int3(id.xy, 3 * view + 2)] =
        float4(slices[4], slices[5], slices[6], slices[7]);
}

// A dense grid's values (a DenseField's on a device, `VolumePass::StageDenseGrids`) scattered into
// its grid of every voxel in the pool (`DenseNanoVdbTopology`): value (i, j, k) of the (x, y, z)
// array goes to its leaf's value table. In 4-byte words, `constants` hold the shape
// (`layout.xyz`) and where its values start (`layout.w`); its first leaf in the pool, the words
// of a leaf, and where a leaf's values start in it (`offsets.xyz`); and its leaves along y and z
// (`offsets.w`, `flags.x`).
[shader("compute")]
[numthreads(4, 4, 4)]
void denseScatterMain(uint3 id : SV_DispatchThreadID) {
    let shape = constants.layout.xyz;
    if (any(id >= shape)) {
        return;
    }
    let value = denseValues[constants.layout.w + (id.x * shape.y + id.y) * shape.z + id.z];
    let leaf = ((id.x >> 3) * constants.offsets.w + (id.y >> 3)) * constants.flags.x + (id.z >> 3);
    let inLeaf = ((id.x & 7u) << 6) | ((id.y & 7u) << 3) | (id.z & 7u);
    gridsOut[constants.offsets.x + leaf * constants.offsets.y + constants.offsets.z + inLeaf] =
        asuint(value);
}

// The parts of the frame `volumeChangesMain` found changed since the last: bit 1 for the lighting
// caches, bit 2 for the shadow maps, bit 4 for what else the volumes show (temporal amortisation's
// history; the caches' changes too).
groupshared uint changedParts;

// The parts the host changed itself (`constants.layout.w`): bits 1 and 2 as `changedParts`'s, and
// whether its volumes changed at a stroke (their grids loaded, the settings) or only their grids
// were written again (fields written each frame).
static const uint kHostGrids = 8u;
static const uint kHostDiscrete = 16u;

// Compares `count` words from `from` with the last frame's at `pass.previous[at]` on, keeping the
// new ones and marking `parts` changed where any differs.
void compareWords(uint thread, uint* from, uint count, uint at, uint parts) {
    for (uint i = thread; i < count; i += 256) {
        let word = from[i];
        if (pass.previous[at + i] != word) {
            pass.previous[at + i] = word;
            InterlockedOr(changedParts, parts);
        }
    }
}

// Whether the volumes' lighting caches and shadow maps need building again, decided on the GPU
// so a still scene's volumes keep last frame's: the volumes, their materials and where the lights
// are and what they light are compared with the last frame's (both), the lights' views and their
// worlds too (the maps', which the GPU fits); and whether what the volumes show changed besides
// (`pass.rebuild[6]`, temporal amortisation's). Writes the two dispatches `pass.rebuild` holds, `constants.layout.xyz` groups for
// the caches and `constants.offsets.xyz` for the maps, or none where nothing they read changed.
// `constants.layout.w` holds the parts the host changed itself (the grids, the settings).
[shader("compute")]
[numthreads(256, 1, 1)]
void volumeChangesMain(uint3 id: SV_DispatchThreadID) {
    if (id.x == 0) {
        changedParts = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    var at = 0u;
    let instances = pass.volumeCount * uint(sizeof(VolumeGpu)) / 4;
    compareWords(id.x, (uint*)volumes, instances, at, 3u);
    at += instances;
    let materials = pass.materialCount * uint(sizeof(VolumeMaterialGpu)) / 4;
    compareWords(id.x, (uint*)pass.materials, materials, at, 3u);
    at += materials;
    // The lights: where they are, their shapes and links, for the caches and maps too; all of
    // them (their spectra, patterns, time) for what the volumes show.
    for (uint k = id.x; k < pass.lightCount; k += 256) {
        let light = pass.shadowLights[k];
        let placed = uint4[7](asuint(light.position), asuint(light.axis), asuint(light.cone),
                              asuint(light.shadow), asuint(light.extentX), asuint(light.extentY),
                              light.links);
        for (uint w = 0; w < 7; ++w) {
            for (uint c = 0; c < 4; ++c) {
                let word = placed[w][c];
                let last = at + 28 * k + 4 * w + c;
                if (pass.previous[last] != word) {
                    pass.previous[last] = word;
                    InterlockedOr(changedParts, 3u);
                }
            }
        }
    }
    at += 28 * pass.lightCount;
    let lights = pass.lightCount * uint(sizeof(LightGpu)) / 4;
    compareWords(id.x, (uint*)pass.shadowLights, lights, at, 4u);
    at += lights;
    if (pass.shadowViewCount > 0) {
        compareWords(id.x, (uint*)pass.shadowViews, 16 * pass.shadowViewCount, at, 2u);
        at += 16 * pass.shadowViewCount;
        compareWords(id.x, (uint*)pass.shadowWorlds, pass.shadowViewCount, at, 2u);
        at += pass.shadowViewCount;
    }
    // What else lights the volumes, for temporal amortisation alone (bit 4): the lights' IES
    // profiles and their timing (lasers' and strobes'), compared; their pattern images and arrays,
    // which change unseen, always.
    let profileWords = kProfileRows * kProfileColumns;
    bool timed = false;
    for (uint k = 0; k < pass.lightCount; ++k) {
        let light = pass.shadowLights[k];
        let profile = uint(light.shadow.w);
        if (profile > 0) {
            compareWords(id.x, (uint*)(pass.profiles + (profile - 1) * profileWords), profileWords,
                         at, 4u);
        }
        at += profileWords;
        timed = timed || uint(light.position.w) == kLaser || light.links.w != 0;
        if (light.patternLayout.w != 0 && id.x == 0) {
            InterlockedOr(changedParts, 4u);
        }
    }
    if (timed) {
        compareWords(id.x, pass.timing, pass.timingWords, at, 4u);
    }
    GroupMemoryBarrierWithGroupSync();
    if (id.x == 0) {
        // With the parts the host changed itself.
        let host = constants.layout.w;
        let parts = changedParts | (host & 3u);
        let caches = (parts & 1u) != 0 ? constants.layout.xyz : uint3(0);
        let maps = (parts & 2u) != 0 ? constants.offsets.xyz : uint3(0);
        pass.rebuild[0] = caches.x;
        pass.rebuild[1] = caches.y;
        pass.rebuild[2] = caches.z;
        pass.rebuild[3] = maps.x;
        pass.rebuild[4] = maps.y;
        pass.rebuild[5] = maps.z;
        // For temporal amortisation: whether what the volumes show changed at a stroke (their
        // volumes, materials, lights or settings), so every pixel marches; and whether only
        // their grids were written again (`kHostGrids`), which pixels may still reproject,
        // clamped (`clampToMarched`).
        pass.rebuild[6] = (changedParts & 5u) != 0 || (host & kHostDiscrete) != 0 ? 1u : 0u;
        pass.rebuild[7] = (host & kHostGrids) != 0 ? 1u : 0u;
        // The light maps (`lightMapMain`), with the caches.
        pass.rebuild[8] = (parts & 1u) != 0 ? 1u : 0u;
    }
}

// The most steps a lighting cache's cell takes through one volume (`lightCacheMain`).
static const uint kMaxCacheSteps = 256;

// The optical depth from world point `world` (relative to the frame's origin) along unit `towards`,
// up to world distance `far`, through every volume that casts into world `viewWorld` for shadow
// group `casters` (`castsInto`): about a voxel a step in each, the steps in empty tiles skipped.
float castersDepth(float3 world, float3 towards, float far, int viewWorld, uint casters) {
    float depth = 0.0;
    for (uint v = 0; v < pass.volumeCount; ++v) {
        let volume = volumes[v];
        if (!castsInto(volume, viewWorld, casters)) {
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
        let indexOrigin = float3(dot(volume.density.toIndex[0], float4(o, 1.0)),
                                 dot(volume.density.toIndex[1], float4(o, 1.0)),
                                 dot(volume.density.toIndex[2], float4(o, 1.0)));
        let indexStep = float3(dot(volume.density.toIndex[0].xyz, d),
                               dot(volume.density.toIndex[1].xyz, d),
                               dot(volume.density.toIndex[2].xyz, d));
        // A voxel of the density's grid a step along the way, whatever the march's steps.
        let steps =
            clamp(uint(ceil((exit - near) * length(indexStep))), 1u, kMaxCacheSteps);
        let dt = (exit - near) / float(steps);
        float sum = 0.0;
        for (uint i = 0; i < steps; ++i) {
            let t = near + (float(i) + 0.5) * dt;
            let index = indexOrigin + indexStep * t;
            var next =
                stepAfterEmptyBricks(volume.density, index, indexOrigin, indexStep, near, dt, 0.5);
            if (next <= i) {
                next = stepAfterEmptyTile(volume.density, index, indexOrigin, indexStep, near, dt,
                                          0.5, accessor);
            }
            if (next > i) {
                i = next - 1;
                continue;
            }
            sum += max(sampleField(volume.density, o + d * t, accessor), 0.0);
        }
        depth += sum * volume.params.x * dt;
    }
    return depth;
}

// The volumes' lighting caches, once a frame for every view and step that reads them: for
// instance `id.z`'s cell `id.x` (a cell per two voxels across its bounds, z fastest), its optical
// depth towards light `id.y` through the volumes that cast its shadows (those of the instance's
// world and the shared ones), or for the two slots past the lights, straight up and straight down
// through them all, for the sky's light and the ground's.
//
// References:
// - T. Lokovic and E. Veach, "Deep Shadow Maps", SIGGRAPH 2000 (transmittance cached once for every
//   lookup).
// - NVIDIA Flow's self-shadow grid (flow/source/nvflowext/shaders/ShadowUpdateCS.hlsl) and Unreal
//   Engine's heterogeneous volumes' lighting cache (HeterogeneousVolumesLiveShadingPipeline.usf).
[shader("compute")]
[numthreads(64, 1, 1)]
void lightCacheMain(uint3 id: SV_DispatchThreadID) {
    if (id.z >= pass.volumeCount || id.y >= pass.cacheSlots) {
        return;
    }
    let volume = volumes[id.z];
    let dims = volume.cache.xyz;
    let cells = dims.x * dims.y * dims.z;
    if (volume.cache.w == kNoCache || id.x >= cells) {
        return;
    }
    let cell = uint3(id.x / (dims.y * dims.z), (id.x / dims.z) % dims.y, id.x % dims.z);
    let local =
        lerp(volume.boundsMin.xyz, volume.boundsMax.xyz, (float3(cell) + 0.5) / float3(dims));
    let world = float3(dot(volume.toWorld[0], float4(local, 1.0)),
                       dot(volume.toWorld[1], float4(local, 1.0)),
                       dot(volume.toWorld[2], float4(local, 1.0)));
    // A world's instance is shadowed by its world's volumes and the shared ones; a shared one by
    // the shared ones.
    let world_ = int(volume.info.y);
    let viewWorld = world_ == -1 ? kNoWorld : world_;
    float depth = 0.0;
    if (id.y < pass.lightCount) {
        let light = pass.shadowLights[id.y];
        if (light.links.z != 0 && ((light.links.x >> volume.info.z) & 1u) != 0) {
            if (uint(light.position.w) == kDistantLight) {
                depth = castersDepth(world, light.position.xyz, 1e30, viewWorld, light.links.y);
            } else {
                let to = light.position.xyz - world;
                let far = length(to);
                depth = castersDepth(world, to / max(far, 1e-6), far, viewWorld, light.links.y);
            }
        }
    } else {
        // Up, the world's z; down, against it.
        let up = float3(0.0, 0.0, id.y == pass.lightCount ? 1.0 : -1.0);
        depth = castersDepth(world, up, 1e30, viewWorld, ~0u);
    }
    pass.lightCache[volume.cache.w * pass.cacheSlots + id.y * cells + id.x] = depth;
}

// Instance `volume`'s cached optical depth in slot `slot` at its local point `local`, trilinear
// between its cells' centres (`lightCacheMain`).
float cachedDepth(VolumeGpu volume, uint slot, float3 local) {
    let dims = int3(volume.cache.xyz);
    let cells = uint(dims.x * dims.y * dims.z);
    let p = (local - volume.boundsMin.xyz) / (volume.boundsMax.xyz - volume.boundsMin.xyz) *
                float3(dims) -
            0.5;
    let c0 = clamp(int3(floor(p)), int3(0), dims - 1);
    let c1 = min(c0 + 1, dims - 1);
    let f = saturate(p - float3(c0));
    let base = volume.cache.w * pass.cacheSlots + slot * cells;
    float corners[8];
    for (int c = 0; c < 8; ++c) {
        let x = (c & 1) != 0 ? c1.x : c0.x;
        let y = (c & 2) != 0 ? c1.y : c0.y;
        let z = (c & 4) != 0 ? c1.z : c0.z;
        corners[c] = pass.lightCache[base + uint((x * dims.y + y) * dims.z + z)];
    }
    let x00 = lerp(corners[0], corners[1], f.x);
    let x10 = lerp(corners[2], corners[3], f.x);
    let x01 = lerp(corners[4], corners[5], f.x);
    let x11 = lerp(corners[6], corners[7], f.x);
    return lerp(lerp(x00, x10, f.y), lerp(x01, x11, f.y), f.z);
}

// Whether instance `volume` has a lighting cache with slot `slot`.
bool hasCache(VolumeGpu volume, uint slot) {
    return volume.cache.w != kNoCache && slot < pass.cacheSlots;
}

// Cloud droplets' multiple scattering by collectors: per pixel, once where the view ray enters a
// cloud, the light scattered twice or more (in sets of orders) found where it entered the cloud's
// lit surface, the cloud there taken as a slab, from the canonical transport of slabs
// (`transport`, ernie/util/cloud_transport.h's lattice) and each volume's light map.
//
// References:
// - A. Bouthors, F. Neyret, N. Max, E. Bruneton and C. Crassin, "Interactive multiple anisotropic
//   scattering in clouds", ACM SIGGRAPH Symposium on Interactive 3D Graphics and Games 2008
//   (collectors found by fixed-point iteration over the lit surface, filtered at their size, from
//   depth maps from the light).
// - C. Dachsbacher and M. Stamminger, "Translucent shadow maps", Eurographics Workshop on
//   Rendering 2003 (the depth maps' MIP-maps, premultiplied by their coverage).

// The transport table's lattice (CloudTransportShape's): order sets, light cosines, thicknesses,
// depths, view cosines and azimuths.
static const uint kOrderSets = 8;
int transportDim(int k) {
    switch (k) {
    case 0: return 8;
    case 1: return 10;
    case 2: return 9;
    case 3: return 16;
    default: return 8;
    }
}
// A light map's header's float4s, and the iterations finding each collector.
static const uint kLightMapHeader = 3;
static const uint kCollectorIterations = 6;
// The most pieces of cloud along a ray whose collectors it finds (`collectorScattering`).
static const uint kCollectedPieces = 3;
// The fewest texels across a slab's filtered surface (`slabAt`).
static const float kSlabTexels = 8.0;

// Texels across instance `volume`'s light map at its finest (`VolumePass`'s, after its bounds).
uint lightMapSizeOf(VolumeGpu volume) {
    return uint(volume.boundsMax.w);
}

// A light map `size` texels across: its texels before level `level`, and its levels.
uint lightMapLevelOffset(uint size, uint level) {
    uint offset = kLightMapHeader;
    for (uint l = 0; l < level; ++l) {
        offset += size * size;
        size = max(size >> 1, 1u);
    }
    return offset;
}

uint lightMapLevels(uint size) {
    return firstbithigh(max(size, 1u)) + 1;
}

// The first distant light lighting instance `volume` (`volumeMain`'s sun), its slot, or `kNoSun`.
uint sunOf(VolumeGpu volume) {
    for (uint k = 0; k < pass.lightCount; ++k) {
        let light = pass.shadowLights[k];
        if (uint(light.position.w) == kDistantLight && light.shadow.y < 0.5 &&
            light.links.w == 0 && ((light.links.x >> volume.info.z) & 1u) != 0 &&
            !analyticBeams(light)) {
            return k;
        }
    }
    return kNoSun;
}

// A light map's frame: across it `e1` and `e2`, along it `towards` (to the light), about the
// volume's centre (relative to the frame's origin), `radius` either way, in `size` texels at its
// finest from float4 `base` on.
struct LightMapFrame {
    float3 e1;
    float3 e2;
    float3 towards;
    float3 centre;
    float radius;
    uint slot;
    uint size;
    uint base;
}

float3 worldOf(VolumeGpu volume, float3 local) {
    return float3(dot(volume.toWorld[0], float4(local, 1.0)),
                  dot(volume.toWorld[1], float4(local, 1.0)),
                  dot(volume.toWorld[2], float4(local, 1.0)));
}

float3 localOf(VolumeGpu volume, float3 world) {
    return float3(dot(volume.toLocal[0], float4(world, 1.0)),
                  dot(volume.toLocal[1], float4(world, 1.0)),
                  dot(volume.toLocal[2], float4(world, 1.0)));
}

LightMapFrame lightMapFrameOf(VolumeGpu volume) {
    LightMapFrame frame;
    frame.base = volume.info.w;
    let a = pass.lightMaps[frame.base];
    let b = pass.lightMaps[frame.base + 1];
    frame.e1 = a.xyz;
    frame.radius = a.w;
    frame.e2 = b.xyz;
    frame.slot = asuint(b.w);
    frame.towards = pass.lightMaps[frame.base + 2].xyz;
    frame.centre = worldOf(volume, 0.5 * (volume.boundsMin.xyz + volume.boundsMax.xyz));
    frame.size = lightMapSizeOf(volume);
    return frame;
}

// Each volume's light map, where the lighting caches are built again: for instance `id.z`'s
// texel `id.x`, across the light (its sun) through the volume, where its density first reaches
// the material's surface (`phase.z`) and where it last holds it (distances from the map's plane
// towards the light, `radius` past the centre) and its optical depth through, all times its
// coverage (1 with a surface, else 0), to filter (`lightMapMipsMain`).
[shader("compute")]
[numthreads(64, 1, 1)]
void lightMapMain(uint3 id: SV_DispatchThreadID) {
    if (pass.rebuild[8] == 0 || id.z >= pass.volumeCount) {
        return;
    }
    let volume = volumes[id.z];
    let size = lightMapSizeOf(volume);
    if (volume.info.w == kNoCache || id.x >= size * size) {
        return;
    }
    let base = volume.info.w;
    let slot = sunOf(volume);
    let towards = slot != kNoSun ? pass.shadowLights[slot].position.xyz : float3(0.0, 0.0, 1.0);
    let e1 = normalize(cross(towards, abs(towards.z) < 0.9 ? float3(0.0, 0.0, 1.0)
                                                           : float3(1.0, 0.0, 0.0)));
    let e2 = cross(towards, e1);
    // The bounds' sphere, in the world.
    let half = 0.5 * (volume.boundsMax.xyz - volume.boundsMin.xyz);
    let toWorld = float3x3(volume.toWorld[0].xyz, volume.toWorld[1].xyz, volume.toWorld[2].xyz);
    float radius = 0.0;
    for (int c = 0; c < 4; ++c) {
        let corner = half * float3((c & 1) != 0 ? -1.0 : 1.0, (c & 2) != 0 ? -1.0 : 1.0, 1.0);
        radius = max(radius, length(mul(toWorld, corner)));
    }
    if (id.x == 0) {
        pass.lightMaps[base] = float4(e1, radius);
        pass.lightMaps[base + 1] = float4(e2, asfloat(slot));
        pass.lightMaps[base + 2] = float4(towards, 0.0);
    }
    let centre = worldOf(volume, volume.boundsMin.xyz + half);
    let uv = ((float2(id.x % size, id.x / size) + 0.5) / float(size) * 2.0 - 1.0) * radius;
    let start = centre + uv.x * e1 + uv.y * e2 + radius * towards;
    let o = localOf(volume, start);
    let d = float3(dot(volume.toLocal[0].xyz, -towards), dot(volume.toLocal[1].xyz, -towards),
                   dot(volume.toLocal[2].xyz, -towards));
    let inverse = 1.0 / d;
    let t0 = (volume.boundsMin.xyz - o) * inverse;
    let t1 = (volume.boundsMax.xyz - o) * inverse;
    let near = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), max(min(t0.z, t1.z), 0.0));
    let exit = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), min(max(t0.z, t1.z), 2.0 * radius));
    float4 texel = float4(0.0);
    if (slot != kNoSun && near < exit && volume.density.grid.x != kNoGrid) {
        let surface = volumeMaterials[volume.info.x].phase.z;
        var accessor = accessorOf(volume.density);
        let indexOrigin = float3(dot(volume.density.toIndex[0], float4(o, 1.0)),
                                 dot(volume.density.toIndex[1], float4(o, 1.0)),
                                 dot(volume.density.toIndex[2], float4(o, 1.0)));
        let indexStep = float3(dot(volume.density.toIndex[0].xyz, d),
                               dot(volume.density.toIndex[1].xyz, d),
                               dot(volume.density.toIndex[2].xyz, d));
        let steps = clamp(uint(ceil((exit - near) * length(indexStep))), 1u, 4 * kMaxCacheSteps);
        let dt = (exit - near) / float(steps);
        float sum = 0.0;
        float first = -1.0;
        float last = 0.0;
        for (uint i = 0; i < steps; ++i) {
            let t = near + (float(i) + 0.5) * dt;
            let index = indexOrigin + indexStep * t;
            var next =
                stepAfterEmptyBricks(volume.density, index, indexOrigin, indexStep, near, dt, 0.5);
            if (next <= i) {
                next = stepAfterEmptyTile(volume.density, index, indexOrigin, indexStep, near, dt,
                                          0.5, accessor);
            }
            if (next > i) {
                i = next - 1;
                continue;
            }
            let rho = max(sampleField(volume.density, o + d * t, accessor), 0.0);
            sum += rho;
            if (rho >= surface) {
                first = first < 0.0 ? t : first;
                last = t;
            }
        }
        if (first >= 0.0) {
            texel = float4(first, last, sum * volume.params.x * dt, 1.0);
        }
    }
    pass.lightMaps[base + lightMapLevelOffset(size, 0) + id.x] = texel;
}

// Each light map's coarser levels, a group per volume: each texel the mean of the four below it.
[shader("compute")]
[numthreads(256, 1, 1)]
void lightMapMipsMain(uint3 id: SV_DispatchThreadID) {
    if (pass.rebuild[8] == 0 || id.z >= pass.volumeCount) {
        return;
    }
    let volume = volumes[id.z];
    if (volume.info.w == kNoCache) {
        return;
    }
    let base = volume.info.w;
    let finest = lightMapSizeOf(volume);
    var below = finest;
    for (uint level = 1; level < lightMapLevels(finest); ++level) {
        AllMemoryBarrierWithGroupSync();
        let size = max(below >> 1, 1u);
        let from = base + lightMapLevelOffset(finest, level - 1);
        let to = base + lightMapLevelOffset(finest, level);
        for (uint i = id.x; i < size * size; i += 256) {
            let x = 2 * (i % size);
            let y = 2 * (i / size);
            let x1 = min(x + 1, below - 1);
            let y1 = min(y + 1, below - 1);
            pass.lightMaps[to + i] =
                0.25 * (pass.lightMaps[from + y * below + x] + pass.lightMaps[from + y * below + x1] +
                        pass.lightMaps[from + y1 * below + x] +
                        pass.lightMaps[from + y1 * below + x1]);
        }
        below = size;
    }
}

// A light map at level `level`, at `uv` across it (world units from its centre), bilinear, as
// stored: times its coverage.
float4 lightMapTexel(LightMapFrame frame, uint level, float2 uv) {
    let size = max(frame.size >> level, 1u);
    let at = (uv / (2.0 * frame.radius) + 0.5) * float(size) - 0.5;
    let i0 = clamp(int2(floor(at)), int2(0), int2(size) - 1);
    let i1 = min(i0 + 1, int2(size) - 1);
    let f = saturate(at - float2(i0));
    let first = frame.base + lightMapLevelOffset(frame.size, level);
    let a = pass.lightMaps[first + i0.y * size + i0.x];
    let b = pass.lightMaps[first + i0.y * size + i1.x];
    let c = pass.lightMaps[first + i1.y * size + i0.x];
    let e = pass.lightMaps[first + i1.y * size + i1.x];
    return lerp(lerp(a, b, f.x), lerp(c, e, f.x), f.y);
}

// A light map's surface filtered at level `level` (fractional, its two levels blended), at `uv`
// across it: where it first and last reaches the surface and its optical depth through, over its
// coverage (w).
float4 lightMapAt(LightMapFrame frame, float level, float2 uv) {
    let low = uint(level);
    let high = min(low + 1, lightMapLevels(frame.size) - 1);
    let value = lerp(lightMapTexel(frame, low, uv), lightMapTexel(frame, high, uv),
                     level - float(low));
    return float4(value.xyz / max(value.w, 1e-6), value.w);
}

// The canonical transport of order set `set` (`CloudTransportTable`'s), multilinear in its
// lattice: the radiance per unit irradiance and the collector's centre and spread (optical depths)
// for the light's cosine `muL`, the slab's optical thickness `thickness`, the depth's fraction of
// it `depth`, and the view's cosine `muV` and azimuth `psi` from the light's.
float4 transportAt(uint set, float muL, float thickness, float depth, float muV, float psi) {
    let pi = 3.14159265;
    float at[5] = { muL * 8.0 - 1.0, log2(max(thickness, 1e-6) / 0.25),
                    acos(clamp(1.0 - 2.0 * depth, -1.0, 1.0)) / pi * 8.0,
                    (muV + 1.0) * 8.0 - 0.5, psi / pi * 8.0 - 0.5 };
    int low[5];
    float w[5];
    for (int k = 0; k < 5; ++k) {
        let c = clamp(at[k], 0.0, float(transportDim(k) - 1));
        low[k] = min(int(c), transportDim(k) - 2);
        w[k] = c - float(low[k]);
    }
    let cells = uint(transportDim(0) * transportDim(1) * transportDim(2) * transportDim(3) *
                     transportDim(4));
    float4 sum = float4(0.0);
    for (int corner = 0; corner < 32; ++corner) {
        float weight = 1.0;
        int cell = 0;
        for (int k = 0; k < 5; ++k) {
            let up = ((corner >> k) & 1) != 0;
            weight *= up ? w[k] : 1.0 - w[k];
            cell = cell * transportDim(k) + low[k] + (up ? 1 : 0);
        }
        if (weight > 0.0) {
            sum += weight * pass.transport[set * cells + uint(cell)];
        }
    }
    return sum;
}

// The slab tangent to a light map's surface filtered at a collector's size about it, as the
// canonical transport reads it from view point `p` looking along `toEye`'s opposite.
struct Slab {
    float muL;
    float thickness;
    float depth;
    float muV;
    float psi;
    // The light's way along it and across it towards the view's side, its extinction (per world
    // unit) and the view point's foot on it.
    float3 along;
    float3 across;
    float kappa;
    float3 foot;
}

// The slab about collector `c` of size `size` (world units), false where the light map has no
// surface there. `p`'s depth in it from its optical depth towards the light, `sunDepth` (the
// lighting cache's, through the cloud as it is: a part of it in another's shadow lies deep, but
// gaps between them add nothing, as the map's first surface would).
bool slabAt(LightMapFrame frame, float3 p, float3 toEye, float sunDepth, float3 c, float size,
            out Slab slab) {
    slab = (Slab)0;
    let texel = 2.0 * frame.radius / float(frame.size);
    // At least a few texels across: the finest level's slopes are its voxels' noise.
    let level = clamp(log2(max(size / texel, kSlabTexels)), 0.0,
                      float(lightMapLevels(frame.size) - 1));
    let step = texel * exp2(level);
    let r = c - frame.centre;
    let uv = float2(dot(r, frame.e1), dot(r, frame.e2));
    let surface = lightMapAt(frame, level, uv);
    if (surface.w < 0.05) {
        return false;
    }
    // The filtered surface's slope across the map, for its normal.
    let du = lightMapAt(frame, level, uv + float2(step, 0.0)).x -
             lightMapAt(frame, level, uv - float2(step, 0.0)).x;
    let dv = lightMapAt(frame, level, uv + float2(0.0, step)).x -
             lightMapAt(frame, level, uv - float2(0.0, step)).x;
    let outward = normalize(0.5 * (du * frame.e1 + dv * frame.e2) / step + frame.towards);
    let inward = -outward;
    let onSurface = frame.centre + uv.x * frame.e1 + uv.y * frame.e2 +
                    (frame.radius - surface.x) * frame.towards;
    let along = max(surface.y - surface.x, 0.5 * texel);
    slab.kappa = surface.z / along;
    if (slab.kappa <= 0.0) {
        return false;
    }
    slab.muL = max(dot(frame.towards, outward), 0.05);
    slab.thickness = surface.z * slab.muL;
    let depth = sunDepth * slab.muL;
    slab.depth = clamp(depth, 0.0, slab.thickness) / max(slab.thickness, 1e-6);
    slab.foot = p - inward * (depth / slab.kappa);
    slab.muV = dot(toEye, inward);
    // The eye sees `p` from outside the cloud, so on the slab's face towards it: the far face
    // where it looks from beyond the slab, the lit one where from the light's side (as computed
    // where it looks along the slab).
    slab.depth = slab.muV > 0.0 ? max(slab.depth, smoothstep(0.0, 0.25, slab.muV))
                                : min(slab.depth, 1.0 - smoothstep(0.0, 0.25, -slab.muV));
    var alongLight = -frame.towards - slab.muL * inward;
    let alongLength = length(alongLight);
    slab.along = alongLength > 1e-4 ? alongLight / alongLength
                                    : normalize(frame.e1 - dot(frame.e1, inward) * inward);
    let across = cross(inward, slab.along);
    let viewAcross = toEye - slab.muV * inward;
    let side = dot(viewAcross, across);
    slab.across = side < 0.0 ? -across : across;
    slab.psi = atan2(abs(side), dot(viewAcross, slab.along));
    return true;
}

// The light scattered twice or more towards unit `toEye` from where the view ray enters the
// cloud of instance `volume` at `p` (relative to the frame's origin), per unit irradiance from
// its light map's light: per order set, the collector found by fixed-point iteration from the
// whole lit surface down, the cloud about it as a slab tangent to its lit surface filtered at the
// collector's size. -1 where the light map holds no surface about `p`, for the octaves instead.
// `sunDepth` is `p`'s optical depth towards the light; `lit` the lit surface straight towards the
// light from `p`, for what shadows it.
float collectorScattering(VolumeGpu volume, LightMapFrame frame, float3 p, float3 toEye,
                          float sunDepth, uint slot, out float3 lit) {
    // `p`'s own piece of cloud: towards the light as far as a gap (the density below half the
    // surface's), its optical depth that far; what lies past the gap (another cloud) only
    // shadows it, letting through what a conservative slab of its depth does, diffusely, and
    // blurred across the light over ten mean free paths (with the lighting cache, `slot`), as
    // light scattered many times spreads past its shadow's edge.
    let local = localDepth(volume, p, frame.towards, sunDepth);
    float through = shadowThrough(sunDepth - local);
    if (slot != kNoSun) {
        let reach = 10.0 / max(volume.params.x, 1e-9);
        for (uint k = 0; k < 4; ++k) {
            let across = ((k & 1u) != 0 ? -reach : reach) * ((k & 2u) != 0 ? frame.e2 : frame.e1);
            let tap = cachedDepth(volume, slot, localOf(volume, p + across));
            through += shadowThrough(max(tap - local, 0.0));
        }
        through /= 5.0;
    }
    return collectorsOfPiece(volume, frame, p, toEye, local, lit) * through;
}

// What a conservative cloud of optical depth `depth` lets through: straight, or diffusely
// (`kDiffuseTransport`), the more.
float shadowThrough(float depth) {
    return max(exp(-depth), 1.0 / (1.0 + kDiffuseTransport * depth));
}

// The diffusion approximation's 3 / 4 (1 - g) for cloud droplets' phase function, its forward
// peak folded away (g about 0.85): a conservative slab of optical depth tau lets through
// 1 / (1 + it tau).
//
// Reference: J. H. Joseph, W. J. Wiscombe and J. A. Weinman, "The delta-Eddington approximation
// for radiative flux transfer", Journal of the Atmospheric Sciences 33(12), 1976.
static const float kDiffuseTransport = 0.11;

// The optical depth from `p` (relative to the frame's origin) along unit `towards` through
// instance `volume`'s density as far as its first gap (two steps below half its material's
// surface density), or `whole` (its optical depth through everything) without one near. The gap
// is soft, each step's emptiness ramping in below the surface's density, so the depth does not
// jump between neighbouring points whose paths graze a thin gap or miss it.
float localDepth(VolumeGpu volume, float3 p, float3 towards, float whole) {
    let surface = volumeMaterials[volume.info.x].phase.z;
    var accessor = accessorOf(volume.density);
    let o = localOf(volume, p);
    let d = float3(dot(volume.toLocal[0].xyz, towards), dot(volume.toLocal[1].xyz, towards),
                   dot(volume.toLocal[2].xyz, towards));
    let dt = 2.0 * volume.params.w;
    float depth = 0.0;
    float marched = 0.0;
    // How much of the path so far is still `p`'s piece, and the last step's emptiness.
    float own = 1.0;
    float empty = 0.0;
    for (uint i = 1; i <= 64 && own > 1e-3; ++i) {
        let rho = max(sampleField(volume.density, o + d * (float(i) * dt), accessor), 0.0);
        let emptiness = 1.0 - smoothstep(0.0, 0.5 * surface, rho);
        own *= 1.0 - emptiness * empty;
        empty = emptiness;
        let step = rho * volume.params.x * dt;
        depth += own * step;
        marched += step;
    }
    return min(depth + own * max(whole - marched, 0.0), whole);
}

// `collectorScattering` for `p`'s own piece of cloud, `sunDepth` its optical depth in it.
float collectorsOfPiece(VolumeGpu volume, LightMapFrame frame, float3 p, float3 toEye,
                        float sunDepth, out float3 lit) {
    let texel = 2.0 * frame.radius / float(frame.size);
    let rel = p - frame.centre;
    let at = float2(dot(rel, frame.e1), dot(rel, frame.e2));
    let straight = lightMapAt(frame, 0.0, at);
    lit = frame.centre + at.x * frame.e1 + at.y * frame.e2 +
          (frame.radius - straight.x) * frame.towards;
    // The first collector, the whole cloud's about `p` (alike for every set): as large as the
    // coarsest level the cloud still mostly covers there, as one volume may hold many clouds.
    uint footprint = 0;
    for (uint level = 1; level < lightMapLevels(frame.size); ++level) {
        if (lightMapTexel(frame, level, at).w < 0.5) {
            break;
        }
        footprint = level;
    }
    let whole = texel * exp2(float(footprint));
    Slab first;
    if (!slabAt(frame, p, toEye, sunDepth, p, whole, first)) {
        return -1.0;
    }
    float total = 0.0;
    for (uint set = 0; set < kOrderSets; ++set) {
        float3 c = p;
        float size = whole;
        Slab slab = first;
        // The radiance over the iterations after the first, as a fixed point between two
        // surfaces (a cloud in another's shadow) may cycle rather than settle.
        float radiance = 0.0;
        float sum = 0.0;
        float found = 0.0;
        for (uint n = 0; n < kCollectorIterations; ++n) {
            // Moved off the surface (past a cloud's edge in the light's view), settled where it
            // was: its radiance for the iterations left, so the mean does not jump with how far
            // the iteration got.
            if (n > 0 && !slabAt(frame, p, toEye, sunDepth, c, size, slab)) {
                sum += radiance * float(kCollectorIterations - n);
                found += float(kCollectorIterations - n);
                break;
            }
            let q = transportAt(set, slab.muL, slab.thickness, slab.depth, slab.muV, slab.psi);
            // Thinner slabs than the lattice's scatter twice and more as their thickness squared.
            let thin = saturate(slab.thickness / 0.25);
            radiance = q.x * thin * thin;
            if (n > 0) {
                sum += radiance;
                found += 1.0;
            }
            let target = slab.foot + (q.y * slab.along + q.z * slab.across) / slab.kappa;
            var move = target - c;
            let moved = length(move);
            if (moved > size) {
                move *= size / moved;
            }
            c += move;
            size = max(q.w / slab.kappa, texel);
        }
        total += found > 0.0 ? sum / found : radiance;
    }
    return total;
}

// Procedural cloud layers (a CloudLayersField, `VolumePass`'s): up to four layers of cumulus and
// the like over a weather map's coverage, carved from tileable Perlin-Worley shapes and Worley
// details (ernie/util/cloud_noise.h) at each sample, rather than read from a grid.
//
// References:
// - A. Schneider and N. Vos, "The Real-Time Volumetric Cloudscapes of Horizon: Zero Dawn",
//   SIGGRAPH 2015 Advances in Real-Time Rendering.
// - takram, three-geospatial's clouds (packages/clouds/src/shaders/clouds.glsl's sampleWeather and
//   sampleMedia), 2024, MIT, Copyright (c) 2024 Shota Matsuda: the coverage, shape and detail
//   remaps and density profiles here; their coverage after Skybolt's
//   (https://github.com/Prograda/Skybolt, Assets/Core/Shaders/Clouds.h).

// A field's grid type for cloud layers (`FieldGpu::grid.y`), and its block's float4s: x (coverage,
// turbulence displacement, weather tile, shape tile), (detail tile, turbulence tile, weather
// offset x, y), then per layer: bottoms, tops, density scales, shape amounts, detail amounts,
// weather exponents, shape biases, coverage filter widths, the density profile's a, b, c, d (a
// e^(b h) + c h + d over its height fraction h), the weather channels (asuint, 4 none) and
// whether it goes in the shadow map (`cloudShadowMain`, 1 or 0).
static const uint kCloudLayersGrid = 0xC10D5u;
static const uint kCloudLayerBlock = 16;
// The noise textures' sizes and where each starts in `pass.cloudNoise` (`CloudNoise`'s order).
static const uint kShapeSize = 128;
static const uint kDetailSize = 32;
static const uint kWeatherSize = 512;
static const uint kTurbulenceSize = 128;
static const uint kDetailStart = kShapeSize * kShapeSize * kShapeSize;
static const uint kWeatherStart = kDetailStart + kDetailSize * kDetailSize * kDetailSize;
static const uint kTurbulenceStart = kWeatherStart + kWeatherSize * kWeatherSize * 4;

bool isCloudLayers(FieldGpu field) {
    return field.grid.y == kCloudLayersGrid;
}

int3 wrapCell(int3 i, int size) {
    return ((i % size) + size) % size;
}

// A tileable 3D noise texture of `size`^3 values from `start`, trilinear at `uvw` (repeating each
// 1).
float noise3(uint start, uint size, float3 uvw) {
    let p = uvw * float(size) - 0.5;
    let b = floor(p);
    let f = p - b;
    let i0 = wrapCell(int3(b), int(size));
    let i1 = wrapCell(i0 + 1, int(size));
    float c[8];
    for (int k = 0; k < 8; ++k) {
        let x = (k & 1) != 0 ? i1.x : i0.x;
        let y = (k & 2) != 0 ? i1.y : i0.y;
        let z = (k & 4) != 0 ? i1.z : i0.z;
        c[k] = pass.cloudNoise[start + (uint(z) * size + uint(y)) * size + uint(x)];
    }
    return lerp(lerp(lerp(c[0], c[1], f.x), lerp(c[2], c[3], f.x), f.y),
                lerp(lerp(c[4], c[5], f.x), lerp(c[6], c[7], f.x), f.y), f.z);
}

// A tileable 2D noise texture of `size`^2 texels of `channels` values from `start`, bilinear at
// `uv` (repeating each 1), its first four channels.
float4 noise2(uint start, uint size, uint channels, float2 uv) {
    let p = uv * float(size) - 0.5;
    let b = floor(p);
    let f = p - b;
    let i0 = wrapCell(int3(int2(b), 0), int(size)).xy;
    let i1 = wrapCell(int3(i0 + 1, 0), int(size)).xy;
    float4 c[4];
    for (int k = 0; k < 4; ++k) {
        let x = (k & 1) != 0 ? i1.x : i0.x;
        let y = (k & 2) != 0 ? i1.y : i0.y;
        let at = start + (uint(y) * size + uint(x)) * channels;
        c[k] = float4(pass.cloudNoise[at], pass.cloudNoise[at + 1], pass.cloudNoise[at + 2],
                      channels > 3 ? pass.cloudNoise[at + 3] : 0.0);
    }
    return lerp(lerp(c[0], c[1], f.x), lerp(c[2], c[3], f.x), f.y);
}

float4 remapClamped(float4 x, float4 low, float4 high) {
    return saturate((x - low) / (high - low));
}

// The layers' coverage at field point `p` before their shapes carve it (0 outside them), and each
// layer's height fraction there.
float4 cloudCoverage(float4* block, float3 p, out float4 heightFraction) {
    let bottoms = block[2];
    let tops = block[3];
    heightFraction = float4(0.0);
    if (!any(p.z > bottoms && p.z < tops)) {
        return float4(0.0);
    }
    let general = block[0];
    let tiles = block[1];
    let coverage = general.x;
    // Each layer's height fraction, and its coverage from its channel of the weather map, rounded
    // towards its top and its bottom (Schneider and Vos's height gradient, a semicircle).
    let heightFraction = saturate((p.z - bottoms) / max(tops - bottoms, 1e-6));
    let uv = (p.xy + tiles.zw) / general.z;
    let weather = noise2(kWeatherStart, kWeatherSize, 4, uv);
    let channels = asuint(block[14]);
    float4 local4;
    for (uint l = 0; l < 4; ++l) {
        local4[l] = channels[l] < 4 ? weather[channels[l]] : 0.0;
    }
    let lw = pow(max(local4, 0.0), block[7]);
    let biased = pow(heightFraction, block[8]);
    let x = clamp(biased * 2.0 - 1.0, -1.0, 1.0);
    let heightScale = 1.0 - x * x;
    let factor = 1.0 - coverage * heightScale;
    let widths = block[9];
    return remapClamped(lerp(lw, float4(1.0), widths), factor, factor + widths) *
           float4(p.z > bottoms && p.z < tops);
}

// A cloud layers field's density at its volume's local point `local`, summed over its layers.
float cloudLayersDensity(FieldGpu field, float3 local) {
    return cloudLayersDensityOf(field, local, float4(1.0));
}

// `cloudLayersDensity` of the layers `mask` holds (1 each, else 0).
float cloudLayersDensityOf(FieldGpu field, float3 local, float4 mask) {
    let p = float3(dot(field.toIndex[0], float4(local, 1.0)),
                   dot(field.toIndex[1], float4(local, 1.0)),
                   dot(field.toIndex[2], float4(local, 1.0)));
    let block = pass.cloudLayers + field.grid.x * kCloudLayerBlock;
    let general = block[0];
    let tiles = block[1];
    float4 heightFraction;
    var density = cloudCoverage(block, p, heightFraction) * mask;
    if (all(density <= 0.0)) {
        return 0.0;
    }
    // Turbulence: curl noise displacing the shapes, more so low in a layer.
    let curl = noise2(kTurbulenceStart, kTurbulenceSize, 3, p.xy / tiles.y).xyz * 2.0 - 1.0;
    let turbulence = general.y * curl *
                     dot(density, remapClamped(heightFraction, float4(0.3), float4(0.0)));
    // The shape, then the detail eroding its edges: fluffy at the top, wispy at the bottom.
    let shape = noise3(0, kShapeSize, (p + turbulence) / general.w);
    density = remapClamped(density, (1.0 - shape) * block[5], float4(1.0));
    let detail = noise3(kDetailStart, kDetailSize, (p + turbulence) / tiles.x);
    let modifier = lerp(float4(pow(detail, 6.0)), float4(1.0 - detail),
                        remapClamped(heightFraction, float4(0.2), float4(0.4))) *
                   block[6];
    density = remapClamped(density * 2.0, modifier * 0.5, float4(1.0));
    // Each layer's density scale and profile over its height.
    let profile = block[10] * exp(block[11] * heightFraction) + block[12] * heightFraction +
                  block[13];
    density = saturate(density * block[4] * profile);
    return density.x + density.y + density.z + density.w;
}

// Empty-space skipping through cloud layers (`stepAfterEmptyTile`'s), from field point `index` on
// the ray `origin` + `step` t (steps `dt` from `start`, offset by `jitter`): between layers, the
// step where the ray next enters one; in one where the weather lets no cloud form, the step half a
// weather texel on across (no further than the layer's top or bottom); else 0.
//
// Reference: takram, three-geospatial's clouds (clouds.frag's march), 2024, MIT, Copyright (c)
//   2024 Shota Matsuda (the layers' height intervals and the weather's test before the shapes').
uint stepAfterClearSky(FieldGpu field, float3 index, float3 origin, float3 step, float start,
                       float dt, float jitter) {
    let block = pass.cloudLayers + field.grid.x * kCloudLayerBlock;
    let bottoms = block[2];
    let tops = block[3];
    let t = dot(index - origin, step) / max(dot(step, step), 1e-30);
    let z = index.z;
    float next = t;
    if (!any(z > bottoms && z < tops)) {
        float enter = 1e30;
        for (uint l = 0; l < 4; ++l) {
            if (bottoms[l] >= tops[l]) {
                continue;
            }
            if (step.z > 0.0 && z <= bottoms[l]) {
                enter = min(enter, (bottoms[l] - origin.z) / step.z);
            } else if (step.z < 0.0 && z >= tops[l]) {
                enter = min(enter, (tops[l] - origin.z) / step.z);
            }
        }
        next = enter;
    } else {
        float4 heightFraction;
        if (any(cloudCoverage(block, index, heightFraction) > 0.0)) {
            return 0;
        }
        let across = length(step.xy);
        var ahead = across > 1e-9 ? 0.5 * block[0].z / float(kWeatherSize) / across : 1e30;
        // No further than out of the layers it is in.
        for (uint l = 0; l < 4; ++l) {
            if (z > bottoms[l] && z < tops[l] && step.z != 0.0) {
                let leave = ((step.z > 0.0 ? tops[l] : bottoms[l]) - origin.z) / step.z;
                ahead = min(ahead, max(leave - t, 0.0));
            }
        }
        next = t + ahead;
    }
    return uint(clamp(ceil((next - start) / dt - jitter), 0.0, 1e9));
}

// Cloud layers' shadow maps: per volume of cloud layers, across its sun through it, each column's
// front (the distance from the map's plane into its clouds, weighted by the light still reaching
// there), its mean extinction where it holds any and its optical depth through as far as the light
// reaches (to a transmittance of 1e-4, a tail past it),
// for the optical depth towards the sun from any point in it without marching there. A header as
// a light map's (`lightMapMain`), its size in its third's w.
//
// Reference: takram, three-geospatial's clouds (shadow.frag and clouds.frag's sampleShadowOpticalDepth),
//   2024, MIT, Copyright (c) 2024 Shota Matsuda: their beer shadow maps' front depth, mean extinction
//   and largest optical depth; after Activision's "Beer shadow maps" (Call of Duty: Black Ops 3).

// Instance `volume`'s shadow map's first float4 (`kNoCache` none), in its bounds' w.
uint cloudShadowBase(VolumeGpu volume) {
    return asuint(volume.boundsMin.w);
}

// Whether instance `volume` has a shadow map, built for light slot `slot`.
bool hasCloudShadow(VolumeGpu volume, uint slot) {
    let base = cloudShadowBase(volume);
    return base != kNoCache && asuint(pass.cloudShadows[base + 1].w) == slot;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cloudShadowMain(uint3 id: SV_DispatchThreadID) {
    if (pass.rebuild[8] == 0 || id.z >= pass.volumeCount) {
        return;
    }
    let volume = volumes[id.z];
    let base = cloudShadowBase(volume);
    if (base == kNoCache) {
        return;
    }
    let size = lightMapSizeOf(volume);
    if (id.x >= size * size) {
        return;
    }
    let slot = sunOf(volume);
    let towards = slot != kNoSun ? pass.shadowLights[slot].position.xyz : float3(0.0, 0.0, 1.0);
    let e1 = normalize(cross(towards, abs(towards.z) < 0.9 ? float3(0.0, 0.0, 1.0)
                                                           : float3(1.0, 0.0, 0.0)));
    let e2 = cross(towards, e1);
    let half = 0.5 * (volume.boundsMax.xyz - volume.boundsMin.xyz);
    let toWorld = float3x3(volume.toWorld[0].xyz, volume.toWorld[1].xyz, volume.toWorld[2].xyz);
    float radius = 0.0;
    for (int c = 0; c < 4; ++c) {
        let corner = half * float3((c & 1) != 0 ? -1.0 : 1.0, (c & 2) != 0 ? -1.0 : 1.0, 1.0);
        radius = max(radius, length(mul(toWorld, corner)));
    }
    if (id.x == 0) {
        pass.cloudShadows[base] = float4(e1, radius);
        pass.cloudShadows[base + 1] = float4(e2, asfloat(slot));
        pass.cloudShadows[base + 2] = float4(towards, float(size));
    }
    let centre = worldOf(volume, volume.boundsMin.xyz + half);
    let uv = ((float2(id.x % size, id.x / size) + 0.5) / float(size) * 2.0 - 1.0) * radius;
    let start = centre + uv.x * e1 + uv.y * e2 + radius * towards;
    let o = localOf(volume, start);
    let d = float3(dot(volume.toLocal[0].xyz, -towards), dot(volume.toLocal[1].xyz, -towards),
                   dot(volume.toLocal[2].xyz, -towards));
    let inverse = 1.0 / d;
    let t0 = (volume.boundsMin.xyz - o) * inverse;
    let t1 = (volume.boundsMax.xyz - o) * inverse;
    let near = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), max(min(t0.z, t1.z), 0.0));
    let exit = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), min(max(t0.z, t1.z), 2.0 * radius));
    float4 texel = float4(0.0);
    if (slot != kNoSun && near < exit) {
        var accessor = accessorOf(volume.density);
        let indexOrigin = float3(dot(volume.density.toIndex[0], float4(o, 1.0)),
                                 dot(volume.density.toIndex[1], float4(o, 1.0)),
                                 dot(volume.density.toIndex[2], float4(o, 1.0)));
        let indexStep = float3(dot(volume.density.toIndex[0].xyz, d),
                               dot(volume.density.toIndex[1].xyz, d),
                               dot(volume.density.toIndex[2].xyz, d));
        // Two of the march's steps a step, at most 512 along the column.
        let dt = max(2.0 * volume.params.w, (exit - near) / 512.0);
        let steps = uint(ceil((exit - near) / dt));
        float depth = 0.0;
        float tail = 0.0;
        float transmittance = 1.0;
        float weighted = 0.0;
        float weights = 0.0;
        float extinctionSum = 0.0;
        float held = 0.0;
        for (uint i = 0; i < steps; ++i) {
            let t = near + (float(i) + 0.5) * dt;
            let index = indexOrigin + indexStep * t;
            let next = stepAfterEmptyTile(volume.density, index, indexOrigin, indexStep, near, dt,
                                          0.5, accessor);
            if (next > i) {
                i = next - 1;
                continue;
            }
            // The layers that go in it (`layerShadows`).
            let mask = pass.cloudLayers[volume.density.grid.x * kCloudLayerBlock + 15];
            let sigma =
                max(cloudLayersDensityOf(volume.density, o + d * t, mask), 0.0) * volume.params.x;
            if (sigma <= 0.0) {
                continue;
            }
            // The front: the distances where the light still reaches, weighted by how much.
            depth += sigma * dt;
            transmittance *= exp(-sigma * dt);
            weighted += t * transmittance;
            weights += transmittance;
            extinctionSum += sigma;
            held += 1.0;
            if (transmittance <= 1e-4) {
                // The light gone: what lies past it, about, as it falls off with the steps
                // taken (theirs).
                tail = min(2.0 * dt * exp(1.0 - held), 0.5 * dt);
                break;
            }
        }
        if (held > 0.0 && weights > 0.0) {
            texel = float4(weighted / weights, extinctionSum / held, depth + tail, 1.0);
        }
    }
    pass.cloudShadows[base + 3 + id.x] = texel;
}

// The optical depth from world point `world` (local `local`) towards the sun `towards` through
// cloud layers instance `volume`: a short march out to about 350 m (steps of 50, 100 and 200 m,
// jittered by `jitter`) for its fine shapes, then its shadow map's, from there, its mean
// extinction past the column's front, at most its depth as far as the light reaches.
float cloudSunDepth(VolumeGpu volume, float3 local, float3 world, float3 towards, float jitter) {
    let localTowards = float3(dot(volume.toLocal[0].xyz, towards),
                              dot(volume.toLocal[1].xyz, towards),
                              dot(volume.toLocal[2].xyz, towards));
    var accessor = accessorOf(volume.density);
    float depth = 0.0;
    float reached = 0.0;
    float step = 50.0;
    for (uint i = 0; i < 3; ++i) {
        let at = reached + step * (0.5 + 0.5 * (jitter - 0.5));
        depth += max(sampleField(volume.density, local + localTowards * at, accessor), 0.0) *
                 volume.params.x * step;
        reached += step;
        step *= 2.0;
    }
    let base = cloudShadowBase(volume);
    let a = pass.cloudShadows[base];
    let b = pass.cloudShadows[base + 1];
    let c = pass.cloudShadows[base + 2];
    let radius = a.w;
    let size = uint(c.w);
    let centre = worldOf(volume, 0.5 * (volume.boundsMin.xyz + volume.boundsMax.xyz));
    let q = world + towards * reached - centre;
    let at = (float2(dot(q, a.xyz), dot(q, b.xyz)) / (2.0 * radius) + 0.5) * float(size) - 0.5;
    let i0 = clamp(int2(floor(at)), int2(0), int2(size) - 1);
    let i1 = min(i0 + 1, int2(size) - 1);
    let f = saturate(at - float2(i0));
    let first = base + 3;
    let texel =
        lerp(lerp(pass.cloudShadows[first + i0.y * size + i0.x],
                  pass.cloudShadows[first + i0.y * size + i1.x], f.x),
             lerp(pass.cloudShadows[first + i1.y * size + i0.x],
                  pass.cloudShadows[first + i1.y * size + i1.x], f.x), f.y);
    // Where the column holds clouds (its coverage, w), its depth past their front.
    let along = radius - dot(q, c.xyz);
    let past = texel.w > 0.0 ? texel.xyz / texel.w : float3(0.0);
    return depth + texel.w * min(past.z, past.y * max(along - past.x, 0.0));
}
