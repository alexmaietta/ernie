// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.device.kernels.device;

#include "ernie/bvh/kernels/instances.h"
#include "ernie/bvh/kernels/refit.h"

using namespace ernie;

uint FindOffset(uint* offsets, uint count, uint i) {
    uint low = 0;
    uint high = count - 1;
    while (low < high) {
        const uint middle = (low + high + 1) / 2;
        if (offsets[middle] <= i) {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    return low;
}

uint Coherent(uint* at) {
    uint value;
    InterlockedAdd(*at, 0u, value);
    return value;
}

float* Vertex(LbvhGeometry mesh, uint local, uint k) {
    const uint vertex = mesh.indices == nullptr ? local * 3 + k : mesh.indices[local * 3 + k];
    return mesh.data + uint64_t(vertex) * mesh.stride;
}

BvhCorners Corners(LbvhGeometry mesh, uint local) {
    float* a = Vertex(mesh, local, 0);
    float* b = Vertex(mesh, local, 1);
    float* c = Vertex(mesh, local, 2);
    BvhCorners corners;
    for (uint axis = 0; axis < 3; ++axis) {
        corners.a[axis] = a[axis];
        corners.b[axis] = b[axis];
        corners.c[axis] = c[axis];
    }
    return corners;
}

BvhBox InputBox(LbvhGeometry boxes, uint local) {
    float* at = boxes.data + uint64_t(local) * boxes.stride;
    if ((asuint(at[0]) & 0x7FFFFFFFu) > 0x7F800000u) {
        return BvhEmptyBox();
    }
    BvhBox box;
    for (uint axis = 0; axis < 3; ++axis) {
        box.lower[axis] = at[axis];
        box.upper[axis] = at[3 + axis];
    }
    return box;
}
