// SPDX-License-Identifier: BSD-3-Clause
// Virtual shadow maps (`VirtualShadowMap`): a distant light's clipmap of 16384² virtual maps of
// 128² pages, the pages the frame needs backed by a pool of physical pages.
//
// Everything is in buffers read through their addresses (`VirtualShadow`), so casters and shading
// reach it from a uniform block without bindings of their own: the pool holds each physical page's
// 128² texels after the one before, as reversed-Z depth's float bits (0 far, so the nearest caster
// wins an atomic max); the page table and the request flags one uint per page, 128² per level,
// level after level. A page table entry is the physical page's index in bits 0-19 and bit 31 set
// when backed.
//
// `markMain` flags the pages the depth buffer's points fall in and `markVolumesMain` those along
// the camera's rays through volumes, `allocateMain` backs them from the pool in order and clears
// the flags, `clearMain` clears the backed pages; casters then write through `virtualShadowWrite`,
// and shading and volumes read through `virtualShadowDepth`.
//
// References:
// - Epic Games, "Virtual Shadow Maps in Unreal Engine" (documentation).
// - B. Karis, R. Stubbe and G. Wihlidal, "Nanite: A Deep Dive", SIGGRAPH 2021 Advances in
//   Real-Time Rendering course (virtual shadow maps: page marking, the pool, atomic writes).
// - A. Lefohn, S. Sengupta and J. D. Owens, "Resolution-Matched Shadow Maps", ACM TOG 26(4):20,
//   2007. https://doi.org/10.1145/1289603.1289611

module virtual_shadow;

public static const uint kVirtualPageTexels = 128;
public static const uint kVirtualPagesAcross = 128;
public static const uint kVirtualTexels = kVirtualPageTexels * kVirtualPagesAcross;
public static const uint kVirtualBacked = 0x80000000u;
public static const uint kVirtualPageMask = 0xfffffu;
// The coarser levels marked beside a receiver's own, which filters too wide for its level read.
public static const uint kVirtualFilterLevels = 4u;

// One clipmap level (`VirtualShadowLevel`).
public struct VirtualLevel {
    // Origin-relative world to clip space: x, y over the level, z reversed.
    public float4x4 viewProjection;
    // x: metres per texel, y: radius, z: the level (2^(level + 1) m), w: unused.
    public float4 info;
}

// A light's virtual shadow map (`VirtualShadowMap::Gpu`).
public struct VirtualShadow {
    public uint* physical;
    public uint* pageTable;
    public uint* requests;
    public uint* counter;
    public VirtualLevel* levels;
    public uint pages;
    public uint levelCount;
    public int firstLevel;
    // Added to log2 of a receiver's distance to pick its level.
    public float bias;
    // Each level's rectangle of backed pages, four uints: its first column and row, and its last
    // column and row bitwise inverted, so atomic minima from all ones gather all four
    // (`allocateMain`); casters clip to it (frame.slang's `shadowPosition`).
    public uint* rects;
    // The coarser levels marked beside a receiver's own, at most `kVirtualFilterLevels`: as many
    // as the light's widest penumbra needs (shading.slang's `virtualCoarserFor`), none for the
    // Sun's.
    public uint filterLevels;
}

// The virtual texel at clip-space `clip` of a level, as its casters rasterize it: x right, y
// down (`BeginRendering`'s viewport flips y).
public float2 virtualTexelAt(float2 clip) {
    return float2(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5) * float(kVirtualTexels);
}

// The map whose block is at `address` (low, high words), as a uniform block hands it over.
public VirtualShadow* virtualShadowAt(uint2 address) {
    // A bit cast of the two words, not 64-bit arithmetic, which would need shaderInt64.
    return bit_cast<VirtualShadow*>(address);
}

// Page `page` of level `levelIndex` in the page table and the request flags.
public uint virtualPageIndex(uint levelIndex, uint2 page) {
    return (levelIndex * kVirtualPagesAcross + page.y) * kVirtualPagesAcross + page.x;
}

// The pool's index of virtual texel `texel` of level `levelIndex`, if its page is backed.
public bool virtualPhysicalTexel(VirtualShadow* map, uint levelIndex, uint2 texel,
                                 out uint physical) {
    let entry = map.pageTable[virtualPageIndex(levelIndex, texel / kVirtualPageTexels)];
    let within = texel % kVirtualPageTexels;
    physical = (entry & kVirtualPageMask) * (kVirtualPageTexels * kVirtualPageTexels) +
               within.y * kVirtualPageTexels + within.x;
    return (entry & kVirtualBacked) != 0;
}

// A caster's fragment at viewport pixel `position.xy` of level `levelIndex` with reversed-Z depth
// `position.z`: kept where its page is backed and it is the nearest so far.
public void virtualShadowWrite(VirtualShadow* map, float4 position, uint levelIndex) {
    uint physical;
    if (virtualPhysicalTexel(map, levelIndex, uint2(position.xy), physical)) {
        InterlockedMax(map.physical[physical], asuint(saturate(position.z)));
    }
}

// The level whose texels suit a receiver `distance` metres from the camera, as
// `VirtualShadowMap::LevelFor`.
// No finer than the level whose radius reaches the receiver, so a negative bias (a long lens)
// cannot pick a level whose square, about the camera, misses it.
public int virtualLevelFor(float distance, float bias) {
    let l = log2(max(distance, 1e-6));
    return int(max(floor(l + bias), ceil(l) - 1.0));
}

// The stored depth over `world` (origin-relative) in level `levelIndex` or, where its page is not
// backed, the first coarser level whose page is, with the receiver's depth there and the level
// used; false where no level holds it.
public bool virtualShadowDepth(VirtualShadow* map, uint levelIndex, float3 world,
                               out float stored, out float receiver, out uint used) {
    stored = 0.0;
    receiver = 0.0;
    used = levelIndex;
    for (uint index = levelIndex; index < map.levelCount; ++index) {
        let clip = mul(map.levels[index].viewProjection, float4(world, 1.0));
        if (any(abs(clip.xy) > 1.0)) {
            continue;
        }
        let texel = uint2(min(virtualTexelAt(clip.xy), float(kVirtualTexels - 1)));
        uint physical;
        if (virtualPhysicalTexel(map, index, texel, physical)) {
            stored = asfloat(map.physical[physical]);
            receiver = clip.z;
            used = index;
            return true;
        }
    }
    return false;
}

// The step across the light from `world` to the centre of the texel holding it in level
// `levelIndex`, where the casters' depth was rasterized: a receiver's plane compared there rather
// than at `world` needs no bias for the texel's width (the receiver plane depth bias of Unreal's
// virtual shadow maps, `ComputeVirtualShadowMapOptimalSlopeBias`).
public float3 virtualTexelCentreStep(VirtualShadow* map, uint levelIndex, float3 world) {
    let m = map.levels[levelIndex].viewProjection;
    let clip = mul(m, float4(world, 1.0));
    let texel = floor(virtualTexelAt(clip.xy)) + 0.5;
    let centre = float2(texel.x / float(kVirtualTexels) * 2.0 - 1.0,
                        1.0 - texel.y / float(kVirtualTexels) * 2.0);
    // Clip x and y are the light's x and y axes over the level, rows 0 and 1, orthogonal.
    let x = m[0].xyz;
    let y = m[1].xyz;
    return (centre.x - clip.x) * x / dot(x, x) + (centre.y - clip.y) * y / dot(y, y);
}

struct VirtualConstants {
    // Clip space to origin-relative world (marking).
    float4x4 inverseViewProjection;
    // xyz: the camera, origin-relative; w: unused.
    float4 eye;
    // x, y: the depth buffer's size.
    uint4 sizes;
    // x, y: the address of the volumes' boxes (`markVolumesMain`), each its origin-relative low
    // then high corner; z: their count.
    uint4 volumes;
}

[[vk::binding(0, 0)]] ConstantBuffer<VirtualShadow> map;
[[vk::binding(1, 0)]] Texture2D<float> depth;
[[vk::push_constant]] ConstantBuffer<VirtualConstants> constants;

// The clip-space x and y of depth buffer pixel `pixel`'s centre, as shading reconstructs it: the
// views draw with a flipped viewport, row 0 at the top, y = +1.
float2 virtualPixelNdc(uint2 pixel) {
    let uv = (float2(pixel) + 0.5) / float2(constants.sizes.xy);
    return float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}

// Flags the page of virtual texel `texel` of level `levelIndex` and, within 5% of a page of its
// edges, those beside it, so filters near a page's edge find their neighbours backed.
void markAround(uint levelIndex, float2 texel) {
    let border = 0.05 * float(kVirtualPageTexels);
    let low = uint2(clamp(texel - border, 0.0, float(kVirtualTexels - 1))) / kVirtualPageTexels;
    let high = uint2(clamp(texel + border, 0.0, float(kVirtualTexels - 1))) / kVirtualPageTexels;
    for (uint y = low.y; y <= high.y; ++y) {
        for (uint x = low.x; x <= high.x; ++x) {
            map.requests[virtualPageIndex(levelIndex, uint2(x, y))] = 1u;
        }
    }
}

// Every pixel of the depth buffer: the page of its point in the level that suits its distance.
// Unreal marks every second pixel in x and y, which at grazing views leaves the pages between
// two rows' points unmarked far out on the ground.
[shader("compute")]
[numthreads(8, 8, 1)]
void markMain(uint3 id: SV_DispatchThreadID) {
    let count = map.levelCount;
    // The coarsest level's pages about the camera, for lookups past the marked ones.
    if (all(id == uint3(0))) {
        let centre = kVirtualPagesAcross / 2;
        for (uint y = centre - 1; y <= centre; ++y) {
            for (uint x = centre - 1; x <= centre; ++x) {
                map.requests[virtualPageIndex(count - 1, uint2(x, y))] = 1u;
            }
        }
    }
    let pixel = id.xy;
    if (any(pixel >= constants.sizes.xy)) {
        return;
    }
    let d = depth.Load(int3(int2(pixel), 0));
    if (d <= 0.0) {
        return;  // the sky
    }
    let ndc = virtualPixelNdc(pixel);
    let h = mul(constants.inverseViewProjection, float4(ndc, d, 1.0));
    let world = h.xyz / h.w;
    let level = virtualLevelFor(length(world - constants.eye.xyz), map.bias) - map.firstLevel;
    if (level >= int(count)) {
        return;  // past the last level: lit
    }
    // Its level and the next coarser ones, which wide filters read as a mip chain's.
    let first = uint(max(level, 0));
    for (uint index = first; index <= min(first + map.filterLevels, count - 1); ++index) {
        let clip = mul(map.levels[index].viewProjection, float4(world, 1.0));
        if (all(abs(clip.xy) <= 1.0)) {
            markAround(index, virtualTexelAt(clip.xy));
        }
    }
}

// Pixels apart of the rays `markVolumesMain` marks along, and the most steps along each.
static const uint kVolumeMarkStride = 8;
static const uint kVolumeMarkSteps = 1024;

// The pages volumes read their shadow from (volume.slang's `geometryShadow`) along the camera's
// rays through them, which the depth buffer's points do not mark: every eighth pixel's ray in x
// and y, through each volume's box (`constants.volumes`) up to the surface it meets, in steps of
// half a page of the level for each distance, so its points mark each page they cross; as
// Unreal's marking from the volumetric fog's froxels.
[shader("compute")]
[numthreads(8, 8, 1)]
void markVolumesMain(uint3 id: SV_DispatchThreadID) {
    let pixel = id.xy * kVolumeMarkStride;
    if (any(pixel >= constants.sizes.xy)) {
        return;
    }
    let ndc = virtualPixelNdc(pixel);
    let eye = constants.eye.xyz;
    let near = mul(constants.inverseViewProjection, float4(ndc, 1.0, 1.0));
    let direction = normalize(near.xyz / near.w - eye);
    // Unbounded past the sky.
    let d = depth.Load(int3(int2(pixel), 0));
    var surface = 3.0e38;
    if (d > 0.0) {
        let h = mul(constants.inverseViewProjection, float4(ndc, d, 1.0));
        surface = length(h.xyz / h.w - eye);
    }
    let inverse = 1.0 / select(abs(direction) > 1e-12, direction, float3(1e-12));
    let boxes = bit_cast<float4*>(constants.volumes.xy);
    for (uint b = 0; b < constants.volumes.z; ++b) {
        let t0 = (boxes[2 * b].xyz - eye) * inverse;
        let t1 = (boxes[2 * b + 1].xyz - eye) * inverse;
        let enter = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
        let end = min(min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z)), surface);
        var t = max(enter, 0.0);
        for (uint step = 0; step < kVolumeMarkSteps && t <= end; ++step) {
            let level = virtualLevelFor(t, map.bias) - map.firstLevel;
            if (level >= int(map.levelCount)) {
                break;  // past the last level: lit
            }
            let index = uint(max(level, 0));
            let clip = mul(map.levels[index].viewProjection, float4(eye + direction * t, 1.0));
            if (all(abs(clip.xy) <= 1.0)) {
                markAround(index, virtualTexelAt(clip.xy));
            }
            // The last step lands on the end.
            let next = t + 0.5 * float(kVirtualPageTexels) * map.levels[index].info.x;
            t = t < end && next > end ? end : next;
        }
    }
}

// One thread per page of every level: flagged pages take the next physical page, in order; the
// rest are unbacked. Pages past the pool stay unbacked (the counter still counts them). Backed
// pages widen their level's rectangle (`VirtualShadow::rects`).
[shader("compute")]
[numthreads(8, 8, 1)]
void allocateMain(uint3 id: SV_DispatchThreadID) {
    if (any(id.xy >= uint2(kVirtualPagesAcross)) || id.z >= map.levelCount) {
        return;
    }
    let page = virtualPageIndex(id.z, id.xy);
    let requested = map.requests[page] != 0;
    map.requests[page] = 0u;
    uint entry = 0;
    if (requested) {
        uint n;
        InterlockedAdd(map.counter[0], 1u, n);
        if (n < map.pages) {
            entry = n | kVirtualBacked;
            let rect = 4 * id.z;
            InterlockedMin(map.rects[rect], id.x);
            InterlockedMin(map.rects[rect + 1], id.y);
            InterlockedMin(map.rects[rect + 2], ~id.x);
            InterlockedMin(map.rects[rect + 3], ~id.y);
        }
    }
    map.pageTable[page] = entry;
}

// One thread per texel of each backed page: the far depth, 0.
[shader("compute")]
[numthreads(8, 8, 1)]
void clearMain(uint3 id: SV_DispatchThreadID) {
    let page = id.z;
    if (page >= min(map.counter[0], map.pages) || any(id.xy >= uint2(kVirtualPageTexels))) {
        return;
    }
    map.physical[page * (kVirtualPageTexels * kVirtualPageTexels) + id.y * kVirtualPageTexels +
                 id.x] = 0u;
}
