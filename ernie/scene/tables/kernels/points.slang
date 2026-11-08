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

groupshared uint s_extents[6];
groupshared uint s_job;

[shader("compute")]
[numthreads(kPointsGroup, 1, 1)]
void WritePointBoxes(uniform PointsParams p, uint thread: SV_GroupIndex) {
    const uint item = LinearThreadIndex();
    if (thread == 0) {
        s_job = FindJob(p, min(item, p.total - 1));
        for (uint axis = 0; axis < 3; ++axis) {
            s_extents[axis] = ~0u;
            s_extents[3 + axis] = 0;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (item < p.total) {
        const uint index = FindJob(p, item);
        const PointsJob job = p.jobs[index];
        const uint point = item - job.first;
        const float3 center = Ptr<Float3>(job.record.points)[point].Vector();
        const float radius = PointRadius(job.record, point);
        const float3 lower = center - radius;
        const float3 upper = center + radius;
        BvhBox box;
        for (uint axis = 0; axis < 3; ++axis) {
            box.lower[axis] = lower[axis];
            box.upper[axis] = upper[axis];
        }
        job.boxes[point] = box;
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
