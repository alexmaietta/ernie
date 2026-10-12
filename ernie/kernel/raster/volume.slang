// SPDX-License-Identifier: BSD-3-Clause
// Volumes (UsdVol): per pixel, its view ray marched through each Volume its view sees, up to the
// surface behind it or on to infinity, after shading. The radiance behind is dimmed by the
// volumes' transmittance and what they emit is added: L = L_behind T + sum of T(t) j(t) dt, with
// grey extinction (sigma_t, from the density field) and per-band emission j: the material's
// emission times its field, and what absorbs glowing at its temperature, sigma_a B_lambda(T)
// (Kirchhoff's law). Each step integrates its emission exactly for constant coefficients,
// (1 - exp(-sigma_t dt)) / sigma_t.
//
// The fields are NanoVDB grids in one buffer, which PNanoVDB walks in place; this module is
// compiled after pnanovdb_prelude.slang and PNanoVDB.h. A volume's fields each map its local
// frame to their grid's index space, where values sit at integer coordinates (trilinear between
// them).
//
// Bindings: 0 the shading uniforms (read at the byte offsets in `constants`, a block per view), 1
// depth and 2 ids of the G-buffer, 3 the spectral radiance (a layer per view and band), 4 the
// grids, 5 the volumes, 6 their materials, 7 each view's run of `viewVolumes` (offset, count), 8
// the volumes each view sees.
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

import projection;

static const uint kNoGrid = 0xFFFFFFFFu;
static const uint kMaxIntervals = 8;
static const uint kMaxGroups = 16;
// The transmittance below which a ray is opaque.
static const float kOpaque = 1e-3;

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
    // x: its material, y: its world; zw unused.
    uint4 info;
}

struct VolumeMaterialGpu {
    // Single-scattering albedo per band, four per float4.
    float4 albedo[16];
    // Emitted spectral radiance per world unit per unit field (W m^-2 sr^-1 nm^-1 per unit).
    float4 emission[16];
    // x: Henyey-Greenstein g; y: black-body scale; z: 1 when it emits; w unused.
    float4 params;
}

struct VolumeConstants {
    // x: the first view's block, y: the block stride in bytes, z: the most steps per volume, w:
    // the frame's seed.
    uint4 layout;
    // Byte offsets in a block: the inverse view-projection, a pushbroom's strip, the bands
    // (`range`) and the image size.
    uint4 offsets;
    // x: 1 when the background holds radiance (a sky); without one, this pass writes the
    // background's, no radiance but what volumes emit, and the sensor reads it as held. yzw unused.
    uint4 flags;
}

[[vk::binding(0, 0)]] ByteAddressBuffer shadingBlocks;
[[vk::binding(1, 0)]] Texture2DArray<float> gDepth;
[[vk::binding(2, 0)]] Texture2DArray<uint4> gIds;
[[vk::binding(3, 0)]] RWTexture2DArray<float> spectral;
[[vk::binding(4, 0)]] StructuredBuffer<uint> grids;
[[vk::binding(5, 0)]] StructuredBuffer<VolumeGpu> volumes;
[[vk::binding(6, 0)]] StructuredBuffer<VolumeMaterialGpu> volumeMaterials;
[[vk::binding(7, 0)]] StructuredBuffer<uint2> viewRuns;
[[vk::binding(8, 0)]] StructuredBuffer<uint> viewVolumes;
[[vk::push_constant]] ConstantBuffer<VolumeConstants> constants;

static const uint kBackground = 0xFFFFFFFF;

float4x4 loadMatrix(uint offset) {
    // glm's column-major order: four columns.
    let c0 = asfloat(shadingBlocks.Load4(offset));
    let c1 = asfloat(shadingBlocks.Load4(offset + 16));
    let c2 = asfloat(shadingBlocks.Load4(offset + 32));
    let c3 = asfloat(shadingBlocks.Load4(offset + 48));
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

[shader("compute")]
[numthreads(8, 8, 1)]
void volumeMain(uint3 id : SV_DispatchThreadID) {
    let block = (constants.layout.x + id.z) * constants.layout.y;
    let size = shadingBlocks.Load4(block + constants.offsets.w);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let range = asfloat(shadingBlocks.Load4(block + constants.offsets.z));
    let bands = uint(range.z);
    let groups = (bands + 3) / 4;
    let pixel = int4(id.xyz, 0);
    let background = gIds.Load(pixel).x == kBackground;
    // A background shading left unwritten: no radiance behind the volumes.
    let unwritten = background && constants.flags.x == 0;
    let run = viewRuns[constants.layout.x + id.z];
    if (run.y == 0) {
        if (unwritten) {
            for (uint b = 0; b < bands; ++b) {
                spectral[int3(id.xy, id.z * bands + b)] = 0.0;
            }
        }
        return;
    }
    let invViewProj = loadMatrix(block + constants.offsets.x);
    let strip = asfloat(shadingBlocks.Load4(block + constants.offsets.y));

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
                spectral[int3(id.xy, id.z * bands + b)] = 0.0;
            }
        }
        return;
    }

    // Per band: the emission field's and the black-body glow's, each integrated with the
    // transmittance up to it, the albedo applied at the end (both constant per volume, so per
    // run of samples).
    float4 emitted[kMaxGroups];
    for (uint g = 0; g < groups; ++g) {
        emitted[g] = float4(0.0);
    }
    float transmittance = 1.0;
    let jitter = hash(uint3(id.xy, constants.layout.w ^ (id.z * 7919u)));
    for (uint s = 0; s < count && transmittance > kOpaque; ++s) {
        let volume = volumes[which[s]];
        let material = volumeMaterials[volume.info.x];
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
        let kelvinBase = volume.params.y;
        let glows = material.params.y > 0.0 && (hasTemperature || kelvinBase > 0.0);
        var densityAccessor = accessorOf(volume.density);
        var temperatureAccessor = accessorOf(volume.temperature);
        var emissionAccessor = accessorOf(volume.emission);
        // Integrated with the transmittance: the emission field, and per band sigma_t B(T).
        float emissionSum = 0.0;
        float4 glow[kMaxGroups];
        for (uint g = 0; g < groups; ++g) {
            glow[g] = float4(0.0);
        }
        let start = transmittance;
        for (uint i = 0; i < steps && transmittance > kOpaque; ++i) {
            let local = o + d * (span.x + (float(i) + jitter) * dt);
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
                emissionSum += weight * field;
            }
            if (glows && sigma > 0.0) {
                let kelvin =
                    hasTemperature
                        ? sampleField(volume.temperature, local, temperatureAccessor) *
                              volume.params.z
                        : kelvinBase;
                if (kelvin > 0.0) {
                    for (uint g = 0; g < groups; ++g) {
                        let lambda = range.x + (float4(g * 4) + float4(0.5, 1.5, 2.5, 3.5)) * range.y;
                        glow[g] += weight * sigma * planck(lambda, kelvin);
                    }
                }
            }
            transmittance *= 1.0 - opacity;
        }
        if (start > transmittance || emissionSum > 0.0) {
            for (uint g = 0; g < groups; ++g) {
                emitted[g] += material.emission[g] * emissionSum +
                              (1.0 - material.albedo[g]) * material.params.y * glow[g];
            }
        }
    }

    for (uint b = 0; b < bands; ++b) {
        let layer = int3(id.xy, id.z * bands + b);
        let behind = unwritten ? 0.0 : spectral[layer];
        spectral[layer] = behind * transmittance + emitted[b / 4][b % 4];
    }
}
