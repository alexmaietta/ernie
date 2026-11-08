// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;

#include "ernie/scene/tables/kernels/subsets.h"

using namespace ernie;

uint FindJob(SubsetParams p, uint index) {
    uint low = 0;
    uint high = p.jobCount - 1;
    while (low < high) {
        const uint middle = (low + high + 1) / 2;
        if (p.jobs[middle].firstIndex <= index) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    return low;
}

uint FindOutput(SubsetParams p, uint entry) {
    uint low = 0;
    uint high = p.outputCount - 1;
    while (low < high) {
        const uint middle = (low + high + 1) / 2;
        if (p.outputs[middle].firstEntry <= entry) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    return low;
}

[shader("compute")]
[numthreads(kSubsetGroup, 1, 1)]
void ClaimFaces(uniform SubsetParams p) {
    const uint index = LinearThreadIndex();
    if (index >= p.indexCount) {
        return;
    }
    const uint rank = FindJob(p, index);
    const SubsetJob job = p.jobs[rank];
    const int face = job.indices[index - job.firstIndex];
    if (face >= 0 && uint(face) < job.faceCount) {
        InterlockedMin(p.ranks[job.firstEntry + uint(face)], rank);
    }
}

[shader("compute")]
[numthreads(kSubsetGroup, 1, 1)]
void WriteFaces(uniform SubsetParams p) {
    const uint entry = LinearThreadIndex();
    if (entry >= p.entryCount) {
        return;
    }
    const SubsetOutput output = p.outputs[FindOutput(p, entry)];
    const uint rank = p.ranks[entry];
    uint value = kNoPrim;
    if (rank < p.jobCount) {
        const SubsetJob job = p.jobs[rank];
        bool binds;
        value = SubsetMap(job.map) == SubsetMap::Materials ? BoundMaterial(p.scene, job.subset, binds)
                                                            : job.subset;
    }
    output.values[entry - output.firstEntry] = value;
}
