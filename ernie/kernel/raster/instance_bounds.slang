// SPDX-License-Identifier: BSD-3-Clause
// Bounds of instancers' positions on the GPU (`MeshDrawable::RecordBounds`), for instancers written
// from the compute device without an extent: their lowest and highest positions and largest scale,
// read back on the host frames later, rather than each frame's positions copied to the host.
//
// Per job, 8 words of `bounds`: x, y, z of the lowest position, of the highest, the largest scale,
// as floats ordered as unsigned integers (`ordered`), so atomic min and max compare them.
//
// Bindings: 0 instance positions, 1 instance scales, 2 bounds.

struct BoundsJob {
    uint first;   // the instancer's first instance in the pools
    uint count;
    uint scaled;  // 1 when the instancer has scales
    uint out;     // its first word of `bounds`
}

[[vk::binding(0, 0)]] StructuredBuffer<float> positions;
[[vk::binding(1, 0)]] StructuredBuffer<float> scales;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> bounds;
[[vk::push_constant]] ConstantBuffer<BoundsJob> job;

// `f` as an unsigned integer in the same order as the floats.
uint ordered(float f) {
    let bits = asuint(f);
    return (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;
}

// The job's words emptied: lows at the largest value, highs and scale at the smallest.
[shader("compute")]
[numthreads(8, 1, 1)]
void boundsClearMain(uint3 id : SV_DispatchThreadID) {
    bounds[job.out + id.x] = id.x < 3 ? 0xffffffffu : 0u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void boundsMain(uint3 id : SV_DispatchThreadID) {
    let active = id.x < job.count;
    let i = job.first + min(id.x, max(job.count, 1) - 1);
    let p = float3(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]);
    var scale = 1.0;
    if (job.scaled != 0) {
        let s = abs(float3(scales[3 * i], scales[3 * i + 1], scales[3 * i + 2]));
        scale = max(s.x, max(s.y, s.z));
    }
    // Reduced across the subgroup first, so one lane in each meets the atomics.
    let low = WaveActiveMin(active ? p : float3(3.4e38));
    let high = WaveActiveMax(active ? p : float3(-3.4e38));
    let largest = WaveActiveMax(active ? scale : 0.0);
    if (WaveIsFirstLane() && WaveActiveAnyTrue(active)) {
        InterlockedMin(bounds[job.out], ordered(low.x));
        InterlockedMin(bounds[job.out + 1], ordered(low.y));
        InterlockedMin(bounds[job.out + 2], ordered(low.z));
        InterlockedMax(bounds[job.out + 3], ordered(high.x));
        InterlockedMax(bounds[job.out + 4], ordered(high.y));
        InterlockedMax(bounds[job.out + 5], ordered(high.z));
        InterlockedMax(bounds[job.out + 6], ordered(largest));
    }
}
