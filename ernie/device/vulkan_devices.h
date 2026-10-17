// SPDX-License-Identifier: BSD-3-Clause
/// @file vulkan_devices.h
/// @brief The Vulkan devices available to Ernie.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ernie {

/// @brief What kind of device a Vulkan physical device is.
enum class VulkanDeviceType : std::uint8_t {
    Other,
    IntegratedGpu,
    DiscreteGpu,
    VirtualGpu,
    Cpu,
};

/// @brief A Vulkan physical device and the features Ernie cares about.
struct VulkanDeviceInfo {
    std::string name;
    VulkanDeviceType type = VulkanDeviceType::Other;
    std::uint32_t apiMajor = 0;  ///< Vulkan version the driver supports, major part.
    std::uint32_t apiMinor = 0;  ///< Vulkan version the driver supports, minor part.
    bool rayQuery = false;       ///< Supports VK_KHR_ray_query for hardware ray tracing.
};

/// @brief Lists the Vulkan devices of the installed drivers.
/// @return Every physical device, in the loader's order.
/// @throws std::runtime_error When no Vulkan loader is installed or the instance cannot be created.
std::vector<VulkanDeviceInfo> ListVulkanDevices();

/// @brief Names a device type for messages.
/// @param type The device type.
/// @return A short lowercase name, such as "cpu".
const char* ToString(VulkanDeviceType type);

}  // namespace ernie
