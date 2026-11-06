// SPDX-License-Identifier: BSD-3-Clause

extern "C" {
__constant__ unsigned long long ernie_dispatch_args;
}

__device__ __forceinline__ bool ernie_outside_dispatch() {
    if (ernie_dispatch_args == 0) {
        return false;
    }
    const unsigned* counts = reinterpret_cast<const unsigned*>(ernie_dispatch_args);
    return blockIdx.x >= counts[0] || blockIdx.y >= counts[1] || blockIdx.z >= counts[2];
}

__device__ __forceinline__ unsigned ernie_linear_group_index() {
    unsigned x = gridDim.x;
    unsigned y = gridDim.y;
    if (ernie_dispatch_args != 0) {
        const unsigned* counts = reinterpret_cast<const unsigned*>(ernie_dispatch_args);
        x = counts[0];
        y = counts[1];
    }
    return (blockIdx.z * y + blockIdx.y) * x + blockIdx.x;
}
