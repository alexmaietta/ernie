// SPDX-License-Identifier: BSD-3-Clause
/// @file vulkan.h
/// @brief Vulkan and its allocator as the device layer sees them: results checked into device
/// errors, and physical devices probed against Ernie's requirements, each extension declared once
/// with its feature bits. Internal: no Vulkan type leaves `device/`.
#pragma once

#include "ernie/device/device.h"
#include "ernie/util/log.h"

#include <vk_mem_alloc.h>
#include <volk.h>

#include <array>
#include <concepts>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// The usage every buffer takes, so shaders reach it by address and copies need no extra flags.
constexpr VkBufferUsageFlags kBufferUsage =
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

/// @throws DeviceError naming `what` unless `result` succeeded.
void CheckVk(VkResult result, std::string_view what);
/// @brief As above, describing what failed only when it did, for calls on hot paths.
template <std::invocable Describe> void CheckVk(VkResult result, Describe&& describe) {
    if (result < VK_SUCCESS) {
        CheckVk(result, std::string(describe()));
    }
}

/// @brief Runs teardown work, logging what it throws: destructors must not throw, and on device
/// loss every wait does.
template <typename Teardown> void Quietly(Teardown&& teardown) noexcept {
    try {
        teardown();
    } catch (const std::exception& error) {
        log::Info(std::string("Ignored in teardown: ") + error.what());
    }
}

/// @brief Records a global memory barrier: `src` work and writes before `dst` work and accesses.
/// The frame graph places barriers between passes; this is for work recorded by hand.
void Barrier(VkCommandBuffer commands, VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
             VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess);

/// @brief The feature structs Ernie reads and enables, chained.
struct FeatureChain {
    VkPhysicalDeviceFeatures2 core{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan11Features v11{.sType =
                                             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features v12{.sType =
                                             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features v13{.sType =
                                             VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructure{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR rayQuery{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    VkPhysicalDeviceFaultFeaturesEXT fault{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT};

    FeatureChain() {
        Relink();
    }
    FeatureChain(const FeatureChain& other) : FeatureChain() {
        *this = other;
    }
    FeatureChain& operator=(const FeatureChain& other);

    /// Chains the core structs and the optional ones named, in place: `core` heads the chain.
    /// Optional structs whose extensions are absent must stay out of it.
    void Link(bool accelerationStructures, bool rayQueries, bool faults);

  private:
    void Relink();
    std::array<bool, 3> optional_{};
};

/// @brief A physical device as Ernie judges it.
struct PhysicalDevice {
    VkPhysicalDevice handle = VK_NULL_HANDLE;
    VulkanDeviceInfo info;
    /// The device-local heaps' bytes, for choosing between devices.
    std::uint64_t deviceLocalBytes = 0;
    /// The first requirement it fails; empty when it meets them all.
    std::string missing;
    /// What it supports, read back.
    FeatureChain features;
    std::vector<std::string> extensions;

    /// @return The extensions to enable: the optional ones of its capabilities (Vulkan 1.3 core
    /// covers every requirement).
    std::vector<const char*> EnabledExtensions() const;
    /// @return The features to enable: the required ones and its capabilities', nothing else.
    FeatureChain EnabledFeatures() const;
    /// @return Higher for a better device: one meeting the requirements, discrete over integrated
    /// over CPU, then more memory.
    std::uint64_t Score() const;
};

/// @return What `handle` is and supports.
PhysicalDevice Probe(VkPhysicalDevice handle);

}  // namespace ernie
