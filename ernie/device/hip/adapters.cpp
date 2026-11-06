// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/hip/device.h"

#include <algorithm>
#include <array>
#include <cstring>

#include "ernie/device/compute/compute.h"

namespace ernie::device::hip {
namespace {

int Attribute(const Api& hip, int device, hipDeviceAttribute_t attribute) {
    int value = 0;
    hip.Check(hip.hipDeviceGetAttribute(&value, attribute, device), "hipDeviceGetAttribute");
    return value;
}

std::string ArchitectureName(const Api& hip, int device) {
    alignas(8) std::array<char, kDevicePropertiesR0600Bytes> properties{};
    hip.Check(hip.hipGetDevicePropertiesR0600(properties.data(), device),
              "hipGetDevicePropertiesR0600");
    const char* name = properties.data() + kDevicePropertiesR0600ArchNameOffset;
    const std::string_view full(name, strnlen(name, 256));
    return std::string(full.substr(0, full.find(':')));
}

AdapterInfo Describe(const Api& hip, int device) {
    char name[256] = {};
    hip.Check(hip.hipDeviceGetName(name, int(sizeof(name)), device), "hipDeviceGetName");
    hipUUID uuid{};
    hip.Check(hip.hipDeviceGetUuid(&uuid, device), "hipDeviceGetUuid");
    size_t memory = 0;
    hip.Check(hip.hipDeviceTotalMem(&memory, device), "hipDeviceTotalMem");
    const std::string architecture = ArchitectureName(hip, device);

    compute::ComputeDevice reported = {
        .backend = Backend::Hip,
        .ordinal = uint32_t(device),
        .name = name,
        .memory = memory,
        .integrated = Attribute(hip, device, hipDeviceAttributeIntegrated) != 0,
        .warpSize = uint32_t(Attribute(hip, device, hipDeviceAttributeWarpSize)),
        .maxThreads = uint32_t(Attribute(hip, device, hipDeviceAttributeMaxThreadsPerBlock)),
        .textures = Attribute(hip, device, hipDeviceAttributeImageSupport) != 0 &&
                    !architecture.starts_with("gfx94") && !architecture.starts_with("gfx95")};
    std::copy(std::begin(uuid.bytes), std::end(uuid.bytes), reported.uuid.begin());
    AdapterInfo adapter = compute::DescribeCompute(reported);

    adapter.architecture = architecture;
    adapter.apiVersion = hip.runtimeVersion;
    adapter.driverVersion = hip.driverVersion;
    return adapter;
}

}

std::vector<AdapterInfo> ListAdapters() {
    const Api& hip = GetApi();
    int count = 0;
    hip.Check(hip.hipGetDeviceCount(&count), "hipGetDeviceCount");
    std::vector<AdapterInfo> adapters;
    adapters.reserve(count);
    for (int device = 0; device < count; ++device) {
        adapters.push_back(Describe(hip, device));
    }
    return adapters;
}

}
