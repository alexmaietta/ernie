// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;

#include "ernie/scene/records/bounds.h"
#include "ernie/scene/tables/kernels/points.h"

using namespace ernie;

uint FindJob(PointsParams p, uint item) {
    uint low = 0;
    uint high = p.jobCount - 1;
    while (low < high) {
        const uint middle = (low + high + 1) / 2;
        if (p.jobs[middle].first <= item) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    return low;
}

float3 PointCenter(PointsRecord record, uint point) {
    return Ptr<Float3>(record.points)[point].Vector();
}

groupshared uint s_extents[6];
groupshared uint s_job;

void BeginExtents(PointsParams p, uint thread, uint item) {
    if (thread == 0) {
        s_job = FindJob(p, min(item, p.total - 1));
        for (uint axis = 0; axis < 3; ++axis) {
            s_extents[axis] = ~0u;
            s_extents[3 + axis] = 0;
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

void Extend(PointsParams p, uint index, float3 lower, float3 upper) {
    for (uint axis = 0; axis < 3; ++axis) {
        const uint low = OrderedFloat(lower[axis]);
        const uint high = OrderedFloat(upper[axis]);
        if (index == s_job) {
            InterlockedMin(s_extents[axis], low);
            InterlockedMax(s_extents[3 + axis], high);
        } else {
            InterlockedMin(p.extents[index * 3 + axis], low);
            InterlockedMax(p.extents[(p.jobCount + index) * 3 + axis], high);
        }
    }
}

void EndExtents(PointsParams p, uint thread, uint item) {
    GroupMemoryBarrierWithGroupSync();
    if (thread < 6 && item - thread < p.total) {
        const uint axis = thread % 3;
        if (thread < 3) {
            InterlockedMin(p.extents[s_job * 3 + axis], s_extents[thread]);
        } else {
            InterlockedMax(p.extents[(p.jobCount + s_job) * 3 + axis], s_extents[thread]);
        }
    }
}

[shader("compute")]
[numthreads(kPointsGroup, 1, 1)]
void ReducePointExtents(uniform PointsParams p, uint thread: SV_GroupIndex) {
    const uint item = LinearThreadIndex();
    BeginExtents(p, thread, item);
    if (item < p.total) {
        const uint index = FindJob(p, item);
        const PointsRecord record = p.jobs[index].record;
        const uint point = item - p.jobs[index].first;
        if (point < record.pointCount) {
            const float3 center = PointCenter(record, point);
            Extend(p, index, center, center);
        }
    }
    EndExtents(p, thread, item);
}

uint SpreadBits(uint x) {
    x &= 0x3FFu;
    x = (x | (x << 16)) & 0x030000FFu;
    x = (x | (x << 8)) & 0x0300F00Fu;
    x = (x | (x << 4)) & 0x030C30C3u;
    x = (x | (x << 2)) & 0x09249249u;
    return x;
}

[shader("compute")]
[numthreads(kPointsGroup, 1, 1)]
void WritePointKeys(uniform PointsParams p) {
    const uint item = LinearThreadIndex();
    if (item >= p.total) {
        return;
    }
    const uint index = FindJob(p, item);
    const PointsJob job = p.jobs[index];
    const uint point = item - job.first;
    uint code = ~0u;
    if (point < job.record.pointCount) {
        const float3 lower = job.bounds[0].Vector();
        const float3 extent = max(job.bounds[1].Vector() - lower, float3(1e-30f));
        const float3 at = saturate((PointCenter(job.record, point) - lower) / extent);
        const uint3 cell = min(uint3(at * 1024), uint3(1023));
        code = SpreadBits(cell.x) | (SpreadBits(cell.y) << 1) | (SpreadBits(cell.z) << 2);
    }
    p.keys[item * 2] = code;
    p.keys[item * 2 + 1] = index;
    p.values[item] = point;
}

[shader("compute")]
[numthreads(kPointsGroup, 1, 1)]
void StorePointOrder(uniform PointsParams p) {
    const uint item = LinearThreadIndex();
    if (item >= p.total) {
        return;
    }
    const uint index = FindJob(p, item);
    const PointsJob job = p.jobs[index];
    Ptr<uint>(job.record.order)[item - job.first] = p.values[item];
}

[shader("compute")]
[numthreads(kPointsGroup, 1, 1)]
void WritePointBoxes(uniform PointsParams p, uint thread: SV_GroupIndex) {
    const uint item = LinearThreadIndex();
    BeginExtents(p, thread, item);
    if (item < p.total) {
        const uint index = FindJob(p, item);
        const PointsJob job = p.jobs[index];
        const PointsRecord record = job.record;
        const uint slot = item - job.first;
        const uint perBox = max(record.perBox, 1u);
        const uint start = slot * perBox;
        BvhBox box;
        if (start < record.pointCount) {
            float3 lower = float3(kInfinity);
            float3 upper = float3(-kInfinity);
            const uint end = min(start + perBox, record.pointCount);
            for (uint at = start; at < end; ++at) {
                const uint point = perBox == 1 ? at : Ptr<uint>(record.order)[at];
                const float3 center = PointCenter(record, point);
                const float radius = PointRadius(record, point);
                if (perBox > 1) {
                    Ptr<float4>(record.packed)[at] = float4(center, radius);
                }
                lower = min(lower, center - radius);
                upper = max(upper, center + radius);
            }
            for (uint axis = 0; axis < 3; ++axis) {
                box.lower[axis] = lower[axis];
                box.upper[axis] = upper[axis];
            }
            Extend(p, index, lower, upper);
        } else {
            for (uint axis = 0; axis < 3; ++axis) {
                box.lower[axis] = asfloat(0x7FC00000u);
                box.upper[axis] = asfloat(0x7FC00000u);
            }
        }
        job.boxes[slot] = box;
    }
    EndExtents(p, thread, item);
}

[shader("compute")]
[numthreads(kPointsGroup, 1, 1)]
void FinishPointBounds(uniform PointsParams p) {
    const uint index = LinearThreadIndex();
    if (index >= p.jobCount) {
        return;
    }
    float3 lower;
    float3 upper;
    for (uint axis = 0; axis < 3; ++axis) {
        lower[axis] = FloatFromOrdered(p.extents[index * 3 + axis]);
        upper[axis] = FloatFromOrdered(p.extents[(p.jobCount + index) * 3 + axis]);
    }
    const PointsJob job = p.jobs[index];
    job.bounds[0] = Float3(lower);
    job.bounds[1] = Float3(upper);
}
