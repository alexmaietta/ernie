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
// `markMain` flags the pages the depth buffer's points fall in, `allocateMain` backs them from
// the pool in order and clears the flags, `clearMain` clears the backed pages; casters then write
// through `virtualShadowWrite`, and shading reads through `virtualShadowDepth`.
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
public int virtualLevelFor(float distance, float bias) {
    return int(floor(log2(max(distance, 1e-6)) + bias));
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
        let texel = uint2(min((clip.xy * 0.5 + 0.5) * float(kVirtualTexels),
                              float(kVirtualTexels - 1)));
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

struct VirtualConstants {
    // Clip space to origin-relative world (marking).
    float4x4 inverseViewProjection;
    // xyz: the camera, origin-relative; w: unused.
    float4 eye;
    // x, y: the depth buffer's size.
    uint4 sizes;
}

[[vk::binding(0, 0)]] ConstantBuffer<VirtualShadow> map;
[[vk::binding(1, 0)]] Texture2D<float> depth;
[[vk::push_constant]] ConstantBuffer<VirtualConstants> constants;

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

// Every second pixel of the depth buffer in x and y, as Unreal marks: the page of its point in
// the level that suits its distance.
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
    let pixel = id.xy * 2;
    if (any(pixel >= constants.sizes.xy)) {
        return;
    }
    let d = depth.Load(int3(int2(pixel), 0));
    if (d <= 0.0) {
        return;  // the sky
    }
    let ndc = (float2(pixel) + 0.5) / float2(constants.sizes.xy) * 2.0 - 1.0;
    let h = mul(constants.inverseViewProjection, float4(ndc, d, 1.0));
    let world = h.xyz / h.w;
    let level = virtualLevelFor(length(world - constants.eye.xyz), map.bias) - map.firstLevel;
    if (level >= int(count)) {
        return;  // past the last level: lit
    }
    let index = uint(max(level, 0));
    let clip = mul(map.levels[index].viewProjection, float4(world, 1.0));
    if (any(abs(clip.xy) > 1.0)) {
        return;
    }
    markAround(index, (clip.xy * 0.5 + 0.5) * float(kVirtualTexels));
}

// One thread per page of every level: flagged pages take the next physical page, in order; the
// rest are unbacked. Pages past the pool stay unbacked (the counter still counts them).
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
