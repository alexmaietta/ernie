// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/vulkan/device.h"

#include <algorithm>
#include <string>

#include "ernie/util/library.h"
#include "ernie/util/text.h"
#include "ernie/util/version.h"

namespace ernie::device::vulkan {
namespace {

constexpr uint32_t kVendorNvidia = 0x10de;
constexpr uint32_t kVendorIntel = 0x8086;

Version DecodeDriverVersion(uint32_t vendor, uint32_t packed) {
    if (vendor == kVendorNvidia) {
        return {int(packed >> 22), int(packed >> 14 & 0xff), int(packed >> 6 & 0xff)};
    }
#if defined(_WIN32)
    if (vendor == kVendorIntel) {
        return {int(packed >> 14), int(packed & 0x3fff)};
    }
#else
    static_cast<void>(kVendorIntel);
#endif
    return DecodeVersion(packed);
}

std::string DriverName(VkDriverId id, const char* name) {
    switch (id) {
    case VK_DRIVER_ID_NVIDIA_PROPRIETARY:
        return "nvidia";
    case VK_DRIVER_ID_AMD_PROPRIETARY:
        return "amd";
    case VK_DRIVER_ID_AMD_OPEN_SOURCE:
        return "amdvlk";
    case VK_DRIVER_ID_MESA_RADV:
        return "radv";
    case VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS:
        return "intel";
    case VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA:
        return "anv";
    case VK_DRIVER_ID_MESA_NVK:
        return "nvk";
    case VK_DRIVER_ID_MESA_LLVMPIPE:
        return "llvmpipe";
    case VK_DRIVER_ID_MOLTENVK:
        return "moltenvk";
    case VK_DRIVER_ID_MESA_HONEYKRISP:
        return "honeykrisp";
    case VK_DRIVER_ID_MESA_DOZEN:
        return "dozen";
    case VK_DRIVER_ID_MESA_VENUS:
        return "venus";
    case VK_DRIVER_ID_GOOGLE_SWIFTSHADER:
        return "swiftshader";
    default:
        return Lowercase(name);
    }
}

AdapterType ToAdapterType(VkPhysicalDeviceType type) {
    switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        return AdapterType::Discrete;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        return AdapterType::Integrated;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        return AdapterType::Virtual;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
        return AdapterType::Cpu;
    default:
        return AdapterType::Other;
    }
}

void DescribeMemory(VkPhysicalDevice device, AdapterInfo& adapter) {
    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(device, &memory);
    uint64_t local = 0;
    for (uint32_t heap = 0; heap < memory.memoryHeapCount; ++heap) {
        if ((memory.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
            local = std::max(local, memory.memoryHeaps[heap].size);
        }
    }
    // NOLINTNEXTLINE(bugprone-signed-bitwise)
    constexpr VkMemoryPropertyFlags kMapped = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint64_t mapped = 0;
    for (uint32_t type = 0; type < memory.memoryTypeCount; ++type) {
        if ((memory.memoryTypes[type].propertyFlags & kMapped) == kMapped) {
            mapped = std::max(mapped, memory.memoryHeaps[memory.memoryTypes[type].heapIndex].size);
        }
    }
    adapter.memoryBytes = local;
    adapter.features.unifiedMemory =
        adapter.type == AdapterType::Integrated || adapter.type == AdapterType::Cpu;
    adapter.features.hostMappedMemory =
        adapter.features.unifiedMemory || (local > 0 && mapped >= local / 10 * 9);
}

void DescribeFeatures(AdapterInfo& adapter, const Extensions& extensions,
                      const FeatureChain& supported,
                      const VkPhysicalDeviceVulkan12Properties& vulkan12,
                      const VkPhysicalDeviceVulkan13Properties& vulkan13) {
    Features& features = adapter.features;
    const bool rayQueries = std::ranges::all_of(
        kRayQueryExtensions, [&](const char* name) { return extensions.Has(name); });
    features.hardwareTraversal = rayQueries && supported.rayQuery.rayQuery &&
                                 supported.accelerationStructure.accelerationStructure;
#if defined(_WIN32)
    features.externalMemory = extensions.Has("VK_KHR_external_memory_win32");
#else
    features.externalMemory = extensions.Has(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif

    const VkPhysicalDeviceVulkan12Features& indexing = supported.vulkan12;
    features.hardwareTextures = indexing.runtimeDescriptorArray &&
                                indexing.descriptorBindingPartiallyBound &&
                                indexing.descriptorBindingSampledImageUpdateAfterBind &&
                                indexing.shaderSampledImageArrayNonUniformIndexing;
    adapter.limits.maxTextures =
        features.hardwareTextures
            ? std::min({kMaxTextures, vulkan12.maxPerStageDescriptorUpdateAfterBindSampledImages,
                        vulkan12.maxDescriptorSetUpdateAfterBindSampledImages})
            : 0;

    features.blockCompression = supported.base.features.textureCompressionBC;
    features.anisotropicFiltering = supported.base.features.samplerAnisotropy;
    features.float64 = supported.base.features.shaderFloat64;
    features.atomicFloat32Add = supported.atomicFloat.shaderBufferFloat32AtomicAdd;
    features.atomicInt64 = supported.vulkan12.shaderBufferInt64Atomics;
    features.shaderClock = supported.clock.shaderDeviceClock;
    features.kernelStatistics = supported.executables.pipelineExecutableInfo;
    features.waveSizeMin = vulkan13.minSubgroupSize;
    features.waveSizeMax = vulkan13.maxSubgroupSize;
}

std::vector<std::string> MissingFeatures(VkPhysicalDevice device, const FeatureChain& supported) {
    std::vector<std::string> missing;
    if (!supported.vulkan12.bufferDeviceAddress) {
        missing.emplace_back("buffer device addresses");
    }
    if (!supported.vulkan12.timelineSemaphore) {
        missing.emplace_back("timeline semaphores");
    }
    if (!supported.vulkan13.synchronization2) {
        missing.emplace_back("synchronization2");
    }
    if (!supported.vulkan13.subgroupSizeControl || !supported.vulkan13.computeFullSubgroups) {
        missing.emplace_back("full subgroups");
    }
    if (!supported.base.features.shaderInt64) {
        missing.emplace_back("64-bit integers in shaders");
    }
    if (!ComputeQueueFamily(device)) {
        missing.emplace_back("a compute queue");
    }
    return missing;
}

AdapterInfo Describe(VkPhysicalDevice device, uint32_t ordinal) {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(device, &properties);

    AdapterInfo adapter;
    adapter.backend = Backend::Vulkan;
    adapter.ordinal = ordinal;
    adapter.name = properties.deviceName;
    adapter.type = ToAdapterType(properties.deviceType);
    adapter.apiVersion = DecodeVersion(properties.apiVersion);
    adapter.driverVersion = DecodeDriverVersion(properties.vendorID, properties.driverVersion);
    adapter.limits.maxGroupSize = properties.limits.maxComputeWorkGroupInvocations;
    adapter.limits.maxParamBytes = properties.limits.maxPushConstantsSize;
    adapter.limits.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    DescribeMemory(device, adapter);
    if (adapter.apiVersion < kMinimumInstanceVersion) {
        adapter.unsupported = TooOld("its Vulkan", adapter.apiVersion, kMinimumInstanceVersion);
        return adapter;
    }

    VkPhysicalDeviceVulkan13Properties vulkan13{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    VkPhysicalDeviceVulkan12Properties vulkan12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES, &vulkan13};
    VkPhysicalDeviceVulkan11Properties vulkan11{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES, &vulkan12};
    VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                            &vulkan11};
    vkGetPhysicalDeviceProperties2(device, &properties2);
    std::copy(std::begin(vulkan11.deviceUUID), std::end(vulkan11.deviceUUID), adapter.uuid.begin());
    adapter.architecture = DriverName(vulkan12.driverID, vulkan12.driverName);
    adapter.limits.maxBufferBytes = vulkan13.maxBufferSize;
    adapter.limits.maxReservedBytes = adapter.limits.maxBufferBytes;

    const Extensions extensions(device);
    FeatureChain supported;
    supported.Link([&](const char* name) { return extensions.Has(name); });
    vkGetPhysicalDeviceFeatures2(device, &supported.base);
    DescribeFeatures(adapter, extensions, supported, vulkan12, vulkan13);
    adapter.features.presentation =
        Extensions().Has(VK_KHR_SURFACE_EXTENSION_NAME) &&
        extensions.Has(VK_KHR_SWAPCHAIN_EXTENSION_NAME) && supported.vulkan13.dynamicRendering &&
        ComputeQueueFamily(device, VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT).has_value();

    const std::vector<std::string> missing = MissingFeatures(device, supported);
    if (!missing.empty()) {
        adapter.unsupported = "it lacks " + Join(missing);
    }
    return adapter;
}

}

std::vector<AdapterInfo> ListAdapters() {
    GetApi();
    const std::vector<VkPhysicalDevice> devices = PhysicalDevices(Instance(false));
    std::vector<AdapterInfo> adapters;
    adapters.reserve(devices.size());
    for (uint32_t ordinal = 0; ordinal < devices.size(); ++ordinal) {
        adapters.push_back(Describe(devices[ordinal], ordinal));
    }
    return adapters;
}

}
