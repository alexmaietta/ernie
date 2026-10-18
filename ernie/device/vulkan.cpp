// SPDX-License-Identifier: BSD-3-Clause
/// @file vulkan.cpp
/// @brief Vulkan results as device errors, probing physical devices against Ernie's requirements
/// and listing them.
#include "ernie/device/vulkan.h"

#include "ernie/device/instance.h"
#include "ernie/util/algorithm.h"

#include <algorithm>
#include <cstdio>

namespace ernie {
namespace {

#if defined(_WIN32)
constexpr const char* kExternalMemory = VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME;
constexpr const char* kExternalSemaphore = VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME;
#else
constexpr const char* kExternalMemory = VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
constexpr const char* kExternalSemaphore = VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME;
#endif

/// @brief An optional extension and the capability it is part of.
struct Extension {
    const char* name;
    bool DeviceCapabilities::* capability;
};

/// Every optional extension Ernie enables; Vulkan 1.3 core covers all it requires.
const std::array kExtensions{
    Extension{VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &DeviceCapabilities::rayQuery},
    Extension{VK_KHR_RAY_QUERY_EXTENSION_NAME, &DeviceCapabilities::rayQuery},
    Extension{VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, &DeviceCapabilities::rayQuery},
    Extension{kExternalMemory, &DeviceCapabilities::externalMemory},
    Extension{kExternalSemaphore, &DeviceCapabilities::externalMemory},
    Extension{VK_EXT_MEMORY_BUDGET_EXTENSION_NAME, &DeviceCapabilities::memoryBudget},
    Extension{VK_EXT_DEVICE_FAULT_EXTENSION_NAME, &DeviceCapabilities::deviceFault},
};

/// @brief A feature bit Ernie requires: its name, for the error naming it, and where it lives.
struct RequiredBit {
    const char* name;
    VkBool32* (*field)(FeatureChain& chain);
};

#define ERNIE_REQUIRE(part, bit)                                                                   \
    RequiredBit {                                                                                  \
        #bit, [](FeatureChain& f) { return &f.part.bit; }                                          \
    }

const std::array kRequirements{
    ERNIE_REQUIRE(v12, bufferDeviceAddress),
    ERNIE_REQUIRE(v12, timelineSemaphore),
    ERNIE_REQUIRE(v12, scalarBlockLayout),
    ERNIE_REQUIRE(v12, descriptorIndexing),
    ERNIE_REQUIRE(v12, runtimeDescriptorArray),
    ERNIE_REQUIRE(v12, descriptorBindingPartiallyBound),
    ERNIE_REQUIRE(v12, descriptorBindingSampledImageUpdateAfterBind),
    ERNIE_REQUIRE(v12, descriptorBindingStorageImageUpdateAfterBind),
    ERNIE_REQUIRE(v12, shaderSampledImageArrayNonUniformIndexing),
    ERNIE_REQUIRE(v13, synchronization2),
    ERNIE_REQUIRE(v13, maintenance4),
    ERNIE_REQUIRE(v13, dynamicRendering),
};

#undef ERNIE_REQUIRE

VulkanDeviceType ToDeviceType(VkPhysicalDeviceType type) {
    switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        return VulkanDeviceType::IntegratedGpu;
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        return VulkanDeviceType::DiscreteGpu;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        return VulkanDeviceType::VirtualGpu;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
        return VulkanDeviceType::Cpu;
    default:
        return VulkanDeviceType::Other;
    }
}

std::string Hex(const std::uint8_t* bytes, std::size_t count) {
    std::string text;
    for (std::size_t i = 0; i < count; ++i) {
        std::array<char, 3> digits{};
        std::snprintf(digits.data(), digits.size(), "%02x", bytes[i]);
        text += digits.data();
    }
    return text;
}

}  // namespace

void CheckVk(VkResult result, std::string_view what) {
    if (result >= VK_SUCCESS) {
        return;
    }
    const DeviceErrorKind kind =
        result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY
            ? DeviceErrorKind::OutOfMemory
        : result == VK_ERROR_DEVICE_LOST ? DeviceErrorKind::DeviceLost
        : result == VK_ERROR_FEATURE_NOT_PRESENT || result == VK_ERROR_EXTENSION_NOT_PRESENT ||
                result == VK_ERROR_INCOMPATIBLE_DRIVER
            ? DeviceErrorKind::Unsupported
            : DeviceErrorKind::Failed;
    throw DeviceError(kind, std::string(what) + " failed (VkResult " +
                                std::to_string(static_cast<int>(result)) + ")");
}

FeatureChain& FeatureChain::operator=(const FeatureChain& other) {
    core = other.core;
    v11 = other.v11;
    v12 = other.v12;
    v13 = other.v13;
    accelerationStructure = other.accelerationStructure;
    rayQuery = other.rayQuery;
    fault = other.fault;
    optional_ = other.optional_;
    Relink();
    return *this;
}

void FeatureChain::Link(bool accelerationStructures, bool rayQueries, bool faults) {
    optional_ = {accelerationStructures, rayQueries, faults};
    Relink();
}

void FeatureChain::Relink() {
    core.pNext = &v11;
    v11.pNext = &v12;
    v12.pNext = &v13;
    void** next = &v13.pNext;
    const auto add = [&](auto& features, bool included) {
        if (included) {
            *next = &features;
            next = &features.pNext;
        }
    };
    add(accelerationStructure, optional_[0]);
    add(rayQuery, optional_[1]);
    add(fault, optional_[2]);
    *next = nullptr;
}

PhysicalDevice Probe(VkPhysicalDevice handle) {
    PhysicalDevice device;
    device.handle = handle;
    std::uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(handle, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateDeviceExtensionProperties(handle, nullptr, &count, extensions.data());
    device.extensions = Map(
        extensions, [](const VkExtensionProperties& e) { return std::string(e.extensionName); });
    const auto has = [&](std::string_view name) {
        return std::ranges::find(device.extensions, name) != device.extensions.end();
    };

    VkPhysicalDeviceIDProperties ids{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                           .pNext = &ids};
    vkGetPhysicalDeviceProperties2(handle, &properties);
    // Optional structs are read only when their extension is there.
    FeatureChain& features = device.features;
    features.Link(has(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME),
                  has(VK_KHR_RAY_QUERY_EXTENSION_NAME), has(VK_EXT_DEVICE_FAULT_EXTENSION_NAME));
    vkGetPhysicalDeviceFeatures2(handle, &features.core);

    const VkPhysicalDeviceProperties& p = properties.properties;
    device.info = {.name = p.deviceName,
                   .type = ToDeviceType(p.deviceType),
                   .apiMajor = VK_API_VERSION_MAJOR(p.apiVersion),
                   .apiMinor = VK_API_VERSION_MINOR(p.apiVersion),
                   .uuid = Hex(ids.deviceUUID, VK_UUID_SIZE)};
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(handle, &memory);
    for (std::uint32_t i = 0; i < memory.memoryHeapCount; ++i) {
        if ((memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
            device.deviceLocalBytes += memory.memoryHeaps[i].size;
        }
    }

    if (p.apiVersion < VK_API_VERSION_1_3) {
        device.missing = "Vulkan 1.3";
    }
    for (const RequiredBit& requirement : kRequirements) {
        if (device.missing.empty() && *requirement.field(features) != VK_TRUE) {
            device.missing = requirement.name;
        }
    }
    // A capability needs every one of its extensions and, for ray queries, the feature bits.
    DeviceCapabilities& capabilities = device.info.capabilities;
    capabilities = {
        .rayQuery = true, .externalMemory = true, .memoryBudget = true, .deviceFault = true};
    for (const Extension& extension : kExtensions) {
        capabilities.*extension.capability &= has(extension.name);
    }
    capabilities.rayQuery &= features.accelerationStructure.accelerationStructure == VK_TRUE &&
                             features.rayQuery.rayQuery == VK_TRUE;
    capabilities.deviceFault &= features.fault.deviceFault == VK_TRUE;
    capabilities.shaderInt64 = features.core.features.shaderInt64 == VK_TRUE;
    capabilities.shaderFloat16 = features.v12.shaderFloat16 == VK_TRUE;
    capabilities.sparseResidency = features.core.features.sparseResidencyImage2D == VK_TRUE;
    device.info.supported = device.missing.empty();
    return device;
}

std::vector<const char*> PhysicalDevice::EnabledExtensions() const {
    std::vector<const char*> names;
    for (const Extension& extension : kExtensions) {
        if (info.capabilities.*extension.capability) {
            names.push_back(extension.name);
        }
    }
    // Drivers layered over Metal require their portability subset enabled whenever they offer it.
    if (std::ranges::find(extensions, "VK_KHR_portability_subset") != extensions.end()) {
        names.push_back("VK_KHR_portability_subset");
    }
    return names;
}

FeatureChain PhysicalDevice::EnabledFeatures() const {
    const DeviceCapabilities& capabilities = info.capabilities;
    FeatureChain supported = features;
    FeatureChain enabled;
    for (const RequiredBit& requirement : kRequirements) {
        *requirement.field(enabled) = *requirement.field(supported);
    }
    const auto bit = [](bool on) { return on ? VK_TRUE : VK_FALSE; };
    enabled.core.features.shaderInt64 = bit(capabilities.shaderInt64);
    enabled.v12.shaderFloat16 = bit(capabilities.shaderFloat16);
    enabled.accelerationStructure.accelerationStructure = bit(capabilities.rayQuery);
    enabled.rayQuery.rayQuery = bit(capabilities.rayQuery);
    enabled.fault.deviceFault = bit(capabilities.deviceFault);
    enabled.Link(capabilities.rayQuery, capabilities.rayQuery, capabilities.deviceFault);
    return enabled;
}

std::uint64_t PhysicalDevice::Score() const {
    if (!missing.empty()) {
        return 0;
    }
    std::uint64_t kind = 1;
    switch (info.type) {
    case VulkanDeviceType::DiscreteGpu:
        kind = 4;
        break;
    case VulkanDeviceType::IntegratedGpu:
        kind = 3;
        break;
    case VulkanDeviceType::VirtualGpu:
        kind = 2;
        break;
    default:
        break;
    }
    // The kind decides; memory, in MiB, breaks ties.
    return (kind << 48) + (deviceLocalBytes >> 20);
}

std::vector<VulkanDeviceInfo> ListVulkanDevices() {
    const Instance instance;
    return Map(instance.PhysicalDevices(),
               [](VkPhysicalDevice device) { return Probe(device).info; });
}

const char* ToString(VulkanDeviceType type) {
    switch (type) {
    case VulkanDeviceType::IntegratedGpu:
        return "integrated gpu";
    case VulkanDeviceType::DiscreteGpu:
        return "discrete gpu";
    case VulkanDeviceType::VirtualGpu:
        return "virtual gpu";
    case VulkanDeviceType::Cpu:
        return "cpu";
    default:
        return "other";
    }
}

}  // namespace ernie
