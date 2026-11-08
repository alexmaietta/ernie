// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/cuda/device.h"

#include <algorithm>

#include <fmt/format.h>

#include "ernie/device/compute/compute.h"

namespace ernie::device::cuda {
namespace {

int Attribute(const Api& cuda, CUdevice device, CUdevice_attribute attribute) {
    int value = 0;
    cuda.Check(cuda.cuDeviceGetAttribute(&value, attribute, device), "cuDeviceGetAttribute");
    return value;
}

AdapterInfo Describe(const Api& cuda, int ordinal) {
    CUdevice device = 0;
    cuda.Check(cuda.cuDeviceGet(&device, ordinal), "cuDeviceGet");

    char name[256] = {};
    cuda.Check(cuda.cuDeviceGetName(name, int(sizeof(name)), device), "cuDeviceGetName");
    CUuuid uuid{};
    cuda.Check(cuda.cuDeviceGetUuid(&uuid, device), "cuDeviceGetUuid");
    size_t memory = 0;
    cuda.Check(cuda.cuDeviceTotalMem(&memory, device), "cuDeviceTotalMem");

    compute::ComputeDevice reported = {
        .backend = Backend::Cuda,
        .ordinal = uint32_t(ordinal),
        .name = name,
        .memory = memory,
        .integrated = Attribute(cuda, device, CU_DEVICE_ATTRIBUTE_INTEGRATED) != 0,
        .warpSize = uint32_t(Attribute(cuda, device, CU_DEVICE_ATTRIBUTE_WARP_SIZE)),
        .maxThreads = uint32_t(Attribute(cuda, device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)),
        .textures = true};
    std::copy(std::begin(uuid.bytes), std::end(uuid.bytes), reported.uuid.begin());
    AdapterInfo adapter = compute::DescribeCompute(reported);

    const Version capability{Attribute(cuda, device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
                             Attribute(cuda, device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)};
    adapter.architecture = fmt::format("sm_{}{}", capability.major, capability.minor);
    adapter.apiVersion = cuda.driverVersion;
    adapter.driverVersion = cuda.driverVersion;

    if (capability < kMinimumComputeCapability) {
        adapter.unsupported =
            fmt::format("compute capability is {}; Ernie needs {} or later", capability.ToString(),
                        kMinimumComputeCapability.ToString());
    }
    return adapter;
}

}

std::vector<AdapterInfo> ListAdapters() {
    const Api& cuda = GetApi();
    int count = 0;
    cuda.Check(cuda.cuDeviceGetCount(&count), "cuDeviceGetCount");
    std::vector<AdapterInfo> adapters;
    adapters.reserve(count);
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        adapters.push_back(Describe(cuda, ordinal));
    }
    return adapters;
}

}
