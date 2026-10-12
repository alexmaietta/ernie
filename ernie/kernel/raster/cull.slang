// SPDX-License-Identifier: BSD-3-Clause
// GPU instance culling and level-of-detail selection: for each job, a run of one instanced mesh
// draw's instances (one world's, say) seen from one view or shadow view, tests every instance's
// bounding sphere against the view's frustum and, with occlusion culling, against the view's
// hierarchical depth (Hi-Z), picks for each kept one the coarsest level of detail of its mesh
// whose simplification error projects to at most the view's error in pixels, and writes the kept
// instances, in order, to one compacted id list per level, with one indirect draw per level that
// draws exactly them. The CPU never sees the result, so instance arrays written on the GPU (by
// CUDA) are culled without a readback.
//
// Modes (`constants.mode.x`):
// 0 frustum: keep what meets the frustum (shadow views, and views without occlusion culling);
// 1 occluders: keep what meets the frustum and the depth pyramid of the view's occluders does
//   not hide, after they are drawn;
// 2 two-pass, first: keep what meets the frustum and was visible in the view's previous frame,
//   to draw before the depth pyramid;
// 3 two-pass, second: test what meets the frustum against the depth pyramid, remember whether
//   it is visible for the next frame, and keep the visible ones mode 2 did not draw, writing the
//   job's second set of draws.
//
// One workgroup per job walks the job's instances 64 at a time; a prefix sum over each chunk's
// choices keeps each level's instances in their original order, so images do not depend on thread
// timing, and the workgroup writes the draws' instance counts last.
//
// Bindings: 0 draws' culling data, 1 jobs, 2 views (12 float4 each: 6 normalised frustum planes,
// the view-projection's 4 columns, then pixels per world unit at w = 1 and the error in pixels,
// then the depth pyramid's level-0 width and height and level count, 0 without one; a pushbroom's
// view holds its strip in the matrix's place, the strip's intrinsics' first two after the error
// and its line after the level count, its planes those of the strip's envelope), 3 instance
// ids of the draws (mesh.slang's binding 18), 4 positions, 5 orientations, 6 scales, 7 the
// compacted visible ids (a run of the job's instance count per level), 8 the indirect draws
// (VkDrawIndirectCommand, 4 uints each, one per level), 9 the levels of detail (first index
// relative to the mesh's, index count, error in mesh units as bits, unused), 10 the views' depth
// pyramids (one layer per view of the chunk), 11 each view's visibility of each instance in its
// previous frame, one bit per (view, instance id).
//
// References:
// - U. Haar and S. Aaltonen, "GPU-Driven Rendering Pipelines", SIGGRAPH 2015 Advances in
//   Real-Time Rendering in Games course (two-pass occlusion culling with a depth pyramid).
//   https://advances.realtimerendering.com/s2015/aaltonenhaar_siggraph2015_combined_final_footer_220dpi.pdf
// - N. Greene, M. Kass and G. Miller, "Hierarchical Z-Buffer Visibility", SIGGRAPH '93, 231-238,
//   1993. https://doi.org/10.1145/166117.166147
// - B. Shacklett et al., "An Extensible, Data-Oriented Architecture for High-Performance,
//   Many-World Simulation", ACM Transactions on Graphics 42(4), 2023 (Madrona's batch renderer
//   culls each view's instances in one workgroup). https://madrona-engine.github.io
// - V. Blanco, "Compute Shader Culling", Vulkan Guide. https://vkguide.dev/docs/gpudriven/compute_culling/
// - T. Akenine-Moller, E. Haines, N. Hoffman et al., "Real-Time Rendering", 4th ed., CRC Press,
//   2018, section 22.10.1 (sphere/plane test) and section 19.9 (screen-space error for levels of
//   detail).
// - A. Kapoulkine, "niagara", 2018-, a Vulkan renderer whose culling pass selects mesh levels of
//   detail and culls occlusion in two passes. https://github.com/zeux/niagara

// A pushbroom's strip projection (projection_pushbroom.slang), for views of its strip.
import projection;

struct CullItem {
    // World transform of the instancer.
    float4x4 model;
    // Bounding sphere of the prototype, its own transform applied: centre xyz, radius w.
    float4 sphere;
    // x: largest scale of `model`; y: flags (bit 0 orientations, bit 1 scales); z: largest
    // scale of the prototype's transform, for its levels' errors.
    float4 info;
    // x: first level of detail in `lods`, y: level count; z: the instancer's arrays' offset in
    // the double-buffered instance pools now, w: at the last frame (z when it stood still).
    uint4 lods;
    // `model` at the last frame, when it moved (flag kMoved).
    float4x4 previousModel;
}

struct CullJob {
    // Draw, first entry in `instanceIds`, number of instances, view (index of its 12 float4s).
    uint4 range;
    // First entry in `visibleIds` and first indirect draw of the job's draws, then of its second
    // set (two-pass occlusion culling's second pass).
    uint4 output;
}

struct CullConstants {
    // x: jobs in this dispatch, y: workgroups along x of the dispatch, z: first job, w: unused.
    uint4 size;
    // x: mode (see above), y: first view of the depth pyramids' layer 0, z: instance ids per view
    // in the visibility bits, w: unused.
    uint4 mode;
}

[[vk::binding(0, 0)]] StructuredBuffer<CullItem> items;
[[vk::binding(1, 0)]] StructuredBuffer<CullJob> jobs;
[[vk::binding(2, 0)]] StructuredBuffer<float4> views;
[[vk::binding(3, 0)]] StructuredBuffer<uint> instanceIds;
[[vk::binding(4, 0)]] StructuredBuffer<float> positions;
[[vk::binding(5, 0)]] StructuredBuffer<float> orientations;
[[vk::binding(6, 0)]] StructuredBuffer<float> scales;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> visibleIds;
[[vk::binding(8, 0)]] RWStructuredBuffer<uint> commands;
[[vk::binding(9, 0)]] StructuredBuffer<uint4> lods;
[[vk::binding(10, 0)]] Texture2DArray<float> hzb;
[[vk::binding(11, 0)]] RWStructuredBuffer<uint> visibility;
[[vk::push_constant]] ConstantBuffer<CullConstants> constants;

static const uint kGroup = 64;
static const uint kMaxLods = 8;
static const uint kViewStride = 12;
static const uint kHasOrientations = 1;
static const uint kHasScales = 2;
static const uint kMoved = 4;  // the instancer's transform moved since the last frame
// A draw without an instancer: its one instance is its transform's, with no instance arrays
// or ids to read.
static const uint kNoInstances = 8;

static const uint kFrustum = 0;
static const uint kOccluders = 1;
static const uint kTwoPassFirst = 2;
static const uint kTwoPassSecond = 3;

// Per thread of a chunk, its instance's level plus one, or 0 when not kept.
groupshared uint chosen[kGroup];
groupshared uint levelCounts[kMaxLods];

float3 load3(StructuredBuffer<float> buffer, uint i) {
    return float3(buffer[3 * i], buffer[3 * i + 1], buffer[3 * i + 2]);
}

float3 rotate(float4 q, float3 v) {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// An instance's bounding sphere in the world, and how much it scales its prototype at most.
struct Placed {
    float3 centre;
    float radius;
    float scale;
}

// Instance `instance` of `item` placed as mesh.slang places it, from its arrays at `base` in the
// double-buffered instance pools.
Placed placeAt(CullItem item, uint instance, uint base, float4x4 model) {
    let flags = asuint(item.info.y);
    let i = base + instance;
    var centre = item.sphere.xyz;
    var scale = item.info.x;
    if ((flags & kNoInstances) != 0) {
        Placed only;
        only.centre = mul(model, float4(centre, 1.0)).xyz;
        only.radius = item.sphere.w * scale;
        only.scale = scale;
        return only;
    }
    if ((flags & kHasScales) != 0) {
        let s = load3(scales, i);
        centre = centre * s;
        scale = scale * max(abs(s.x), max(abs(s.y), abs(s.z)));
    }
    if ((flags & kHasOrientations) != 0) {
        let q = float4(orientations[4 * i], orientations[4 * i + 1], orientations[4 * i + 2],
                       orientations[4 * i + 3]);
        centre = rotate(q / max(length(q), 1e-20), centre);
    }
    Placed placed;
    placed.centre = mul(model, float4(centre + load3(positions, i), 1.0)).xyz;
    placed.radius = item.sphere.w * scale;
    placed.scale = scale;
    return placed;
}

// Instance `instance` of `item` as its views see it: where it is, or for an instance that moved
// since the last frame (its arrays, `lods.z` and `lods.w` now and then, or the instancer's
// transform), a sphere around where it is and where it was, which the sub-frames of a
// motion-blurred frame see it between.
Placed place(CullItem item, uint instance) {
    var placed = placeAt(item, instance, item.lods.z, item.model);
    let moved = (asuint(item.info.y) & kMoved) != 0;
    if (item.lods.w != item.lods.z || moved) {
        let was =
            placeAt(item, instance, item.lods.w, moved ? item.previousModel : item.model);
        let radius = max(placed.radius, was.radius);
        placed.radius = radius + 0.5 * length(placed.centre - was.centre);
        placed.centre = 0.5 * (placed.centre + was.centre);
        placed.scale = max(placed.scale, was.scale);
    }
    return placed;
}

// Whether the view is a pushbroom's strip, its intrinsics' first two in the error's float4 and
// the line in the pyramid's.
bool isStrip(uint view) {
    return views[view + 10].z > 0.0;
}

float4 project(uint view, float3 p) {
    if (isStrip(view)) {
        // The strip's columns in the matrix's place (`PushbroomStrip::Packed`).
        let strip = transpose(
            float4x4(views[view + 6], views[view + 7], views[view + 8], views[view + 9]));
        let intrinsics = float4(views[view + 10].zw, views[view + 11].w, 0.0);
        return cameraProject(strip, intrinsics, p);
    }
    return views[view + 6] * p.x + views[view + 7] * p.y + views[view + 8] * p.z + views[view + 9];
}

bool inFrustum(Placed placed, uint view) {
    for (uint p = 0; p < 6; ++p) {
        let plane = views[view + p];
        if (dot(plane.xyz, placed.centre) + plane.w < -placed.radius) {
            return false;
        }
    }
    return true;
}

// Whether the view's depth pyramid (layer `layer`) hides the sphere: the nearest depth of the
// corners of its bounding box lies beyond the farthest depth below their screen rectangle, read
// from the level where the rectangle spans at most two texels each way.
bool occluded(Placed placed, uint view, uint layer) {
    let pyramid = views[view + 11];
    let mips = uint(pyramid.z);
    if (mips == 0) {
        return false;
    }
    float2 low = float2(1.0);
    float2 high = float2(0.0);
    float nearest = 1.0;
    for (uint corner = 0; corner < 8; ++corner) {
        let offset = float3((corner & 1) != 0 ? 1.0 : -1.0, (corner & 2) != 0 ? 1.0 : -1.0,
                            (corner & 4) != 0 ? 1.0 : -1.0);
        let clip = project(view, placed.centre + offset * placed.radius);
        if (clip.w <= 1e-6) {
            return false;  // reaches behind the camera: keep it
        }
        let ndc = clip.xyz / clip.w;
        // Image rows run down while NDC y runs up (sensor.slang's lens).
        let uv = float2(0.5 * ndc.x + 0.5, 0.5 - 0.5 * ndc.y);
        low = min(low, uv);
        high = max(high, uv);
        nearest = min(nearest, ndc.z);
    }
    let size = float2(pyramid.x, pyramid.y);
    if (isStrip(view)) {
        // A strip bends straight lines slightly, so its corners' rectangle widens by a tenth and
        // a pixel each way, and its nearest depth is also the sphere's nearest from its centre's
        // row: the view depth there less the radius, in the strip's [0, 1] depth.
        let margin = 0.1 * (high - low) + 1.0 / size;
        low -= margin;
        high += margin;
        let near = views[view + 6].w;
        let far = views[view + 7].w;
        let depth = project(view, placed.centre).w - placed.radius;
        if (depth <= near) {
            return false;
        }
        nearest = min(nearest, far * (depth - near) / ((far - near) * depth));
    }
    low = saturate(low);
    high = saturate(high);
    let extent = (high - low) * size;
    let level = min(uint(ceil(log2(max(max(extent.x, extent.y), 1.0)))), mips - 1);
    let levelSize = max(uint2(size) >> level, uint2(1));
    let first = min(uint2(low * float2(levelSize)), levelSize - 1);
    let last = min(uint2(high * float2(levelSize)), levelSize - 1);
    var farthest = 0.0;
    for (uint y = first.y; y <= min(last.y, first.y + 2); ++y) {
        for (uint x = first.x; x <= min(last.x, first.x + 2); ++x) {
            farthest = max(farthest, hzb.Load(int4(int(x), int(y), int(layer), int(level))));
        }
    }
    return nearest > farthest;
}

// The coarsest level whose error, at the sphere's nearest depth, spans at most the view's error
// in pixels; the full mesh when the view is inside the sphere or levels are off.
uint levelOf(CullItem item, Placed placed, uint view) {
    let w = project(view, placed.centre).w - placed.radius;
    let pixels = views[view + 10];
    if (pixels.y <= 0.0 || w <= 0.0) {
        return 0;
    }
    let perUnit = pixels.x / w * placed.scale * item.info.z;
    uint level = 0;
    for (uint l = 1; l < item.lods.y; ++l) {
        if (asfloat(lods[item.lods.x + l].z) * perUnit > pixels.y) {
            break;
        }
        level = l;
    }
    return level;
}

// Whether instance `id` was visible in view `slot`'s previous frame; with `update`, records
// `visible` for the next frame.
bool wasVisible(uint slot, uint id, bool update, bool visible) {
    let bit = slot * constants.mode.z + id;
    let mask = 1u << (bit & 31u);
    let before = (visibility[bit >> 5] & mask) != 0;
    if (update && visible != before) {
        if (visible) {
            InterlockedOr(visibility[bit >> 5], mask);
        } else {
            InterlockedAnd(visibility[bit >> 5], ~mask);
        }
    }
    return before;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cullMain(uint3 group : SV_GroupID, uint thread : SV_GroupIndex) {
    let index = group.y * constants.size.y + group.x;
    if (index >= constants.size.x) {
        return;
    }
    let mode = constants.mode.x;
    let job = jobs[constants.size.z + index];
    let item = items[job.range.x];
    let view = kViewStride * job.range.w;
    let layer = job.range.w - constants.mode.y;
    let levels = max(item.lods.y, 1u);
    // Two-pass's second pass writes the job's second set of draws.
    let second = mode == kTwoPassSecond;
    let firstId = second ? job.output.z : job.output.x;
    let firstDraw = second ? job.output.w : job.output.y;
    if (thread < kMaxLods) {
        levelCounts[thread] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint start = 0; start < job.range.z; start += kGroup) {
        let k = start + thread;
        uint id = 0;
        uint pick = 0;
        if (k < job.range.z) {
            id = (asuint(item.info.y) & kNoInstances) != 0 ? 0u : instanceIds[job.range.y + k];
            let placed = place(item, id);
            var keep = inFrustum(placed, view);
            if (mode == kOccluders) {
                keep = keep && !occluded(placed, view, layer);
            } else if (mode == kTwoPassFirst) {
                keep = keep && wasVisible(job.range.w, id, false, false);
            } else if (mode == kTwoPassSecond) {
                let visible = keep && !occluded(placed, view, layer);
                let before = wasVisible(job.range.w, id, true, visible);
                keep = visible && !before;
            }
            pick = keep ? levelOf(item, placed, view) + 1 : 0;
        }
        chosen[thread] = pick;
        GroupMemoryBarrierWithGroupSync();
        // This thread's place among the chunk's instances of its level, so each level keeps
        // their order.
        if (pick != 0) {
            uint before = 0;
            for (uint j = 0; j < thread; ++j) {
                before += chosen[j] == pick ? 1u : 0u;
            }
            let level = pick - 1;
            visibleIds[firstId + level * job.range.z + levelCounts[level] + before] = id;
        }
        GroupMemoryBarrierWithGroupSync();
        if (thread < levels) {
            for (uint j = 0; j < kGroup; ++j) {
                levelCounts[thread] += chosen[j] == thread + 1 ? 1u : 0u;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (thread < levels) {
        let command = 4 * (firstDraw + thread);
        commands[command] = item.lods.y == 0 ? 0u : lods[item.lods.x + thread].y;
        commands[command + 1] = levelCounts[thread];
        commands[command + 2] = 0;
        commands[command + 3] = 0;
    }
}
