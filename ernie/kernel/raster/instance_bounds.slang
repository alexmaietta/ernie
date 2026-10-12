// SPDX-License-Identifier: BSD-3-Clause
// Bounds of positions on the GPU (`MeshDrawable::RecordBounds`), for instancers and meshes written
// from the compute device without an extent: their lowest and highest positions and largest scale,
// read back on the host frames later, rather than each frame's positions copied to the host. A mesh
// whose topology changes within a capacity bounds only the points it uses (`activeCounts`). Every
// job of a frame in two dispatches: one empties their words, one measures them all, a row of
// workgroups per job.
//
// Per job, 8 words of `bounds`: x, y, z of the lowest position, of the highest, the largest scale,
// as floats ordered as unsigned integers (`ordered`), so atomic min and max compare them.
//
// Bindings: 0 instance positions, 1 instance scales, 2 bounds, 3 meshes' points and triangle
// indices in use, 4 mesh points, 5 the jobs.

struct BoundsJob {
    uint first;      // the first position in its pool
    uint count;      // positions held
    uint flags;      // kScaled when there are scales, kPoints when they are a mesh's points
    uint out;        // its first word of `bounds`
    uint countSlot;  // a mesh's slot in `activeCounts` plus 1, 0 for every position
    uint3 unused;
}

struct BoundsConstants {
    uint jobs;    // jobs this frame
    uint groups;  // workgroups along x in each job's row
    uint2 unused;
}

static const uint kScaled = 1;
static const uint kPoints = 2;

[[vk::binding(0, 0)]] StructuredBuffer<float> positions;
[[vk::binding(1, 0)]] StructuredBuffer<float> scales;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> bounds;
[[vk::binding(3, 0)]] StructuredBuffer<int> activeCounts;
[[vk::binding(4, 0)]] StructuredBuffer<float> points;
[[vk::binding(5, 0)]] StructuredBuffer<BoundsJob> jobs;
[[vk::push_constant]] ConstantBuffer<BoundsConstants> constants;

// `f` as an unsigned integer in the same order as the floats.
uint ordered(float f) {
    let bits = asuint(f);
    return (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;
}

// Every job's words emptied: lows at the largest value, highs and scale at the smallest.
[shader("compute")]
[numthreads(64, 1, 1)]
void boundsClearMain(uint3 id : SV_DispatchThreadID) {
    if (id.x < 8 * constants.jobs) {
        bounds[id.x] = id.x % 8 < 3 ? 0xffffffffu : 0u;
    }
}

// Job `group.y`'s positions, each thread every one so many along (the row's threads), reduced
// across the subgroup before one lane meets the atomics.
[shader("compute")]
[numthreads(64, 1, 1)]
void boundsMain(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID) {
    let job = jobs[group.y];
    let count = job.countSlot != 0
                    ? min(job.count, uint(max(activeCounts[2 * (job.countSlot - 1)], 0)))
                    : job.count;
    let stride = constants.groups * 64;
    var low = float3(3.4e38);
    var high = float3(-3.4e38);
    var largest = 0.0;
    var any = false;
    for (uint k = id.x; k < count; k += stride) {
        let i = job.first + k;
        let p = (job.flags & kPoints) != 0
                    ? float3(points[3 * i], points[3 * i + 1], points[3 * i + 2])
                    : float3(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]);
        low = min(low, p);
        high = max(high, p);
        var scale = 1.0;
        if ((job.flags & kScaled) != 0) {
            let s = abs(float3(scales[3 * i], scales[3 * i + 1], scales[3 * i + 2]));
            scale = max(s.x, max(s.y, s.z));
        }
        largest = max(largest, scale);
        any = true;
    }
    low = WaveActiveMin(low);
    high = WaveActiveMax(high);
    largest = WaveActiveMax(largest);
    if (WaveIsFirstLane() && WaveActiveAnyTrue(any)) {
        InterlockedMin(bounds[job.out], ordered(low.x));
        InterlockedMin(bounds[job.out + 1], ordered(low.y));
        InterlockedMin(bounds[job.out + 2], ordered(low.z));
        InterlockedMax(bounds[job.out + 3], ordered(high.x));
        InterlockedMax(bounds[job.out + 4], ordered(high.y));
        InterlockedMax(bounds[job.out + 5], ordered(high.z));
        InterlockedMax(bounds[job.out + 6], ordered(largest));
    }
}
