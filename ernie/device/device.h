// SPDX-License-Identifier: BSD-3-Clause
/// @file device.h
/// @brief The devices Ernie can run on, what they can do, and device errors, in Ernie's types:
/// what the layers above `device/` see.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ernie {

/// @brief What kind of device a physical device is.
enum class VulkanDeviceType : std::uint8_t {
    Other,
    IntegratedGpu,
    DiscreteGpu,
    VirtualGpu,
    Cpu,
};

/// @brief What a device can do beyond what Ernie requires; passes test these before relying on
/// them.
struct DeviceCapabilities {
    /// Ray queries over hardware acceleration structures; the software BVH otherwise.
    bool rayQuery = false;
    /// Memory shared with CUDA, HIP and other contexts.
    bool externalMemory = false;
    /// Timeline semaphores shared the same way, so work is ordered without the host.
    bool externalTimeline = false;
    bool shaderInt64 = false;
    bool shaderFloat16 = false;
    /// The driver's memory budget per heap.
    bool memoryBudget = false;
    /// What went wrong on device loss.
    bool deviceFault = false;
    /// The device's and host's clocks read together, so GPU zones line up with CPU zones.
    bool calibratedTimestamps = false;
    bool sparseResidency = false;
};

/// @brief A device and what it can do.
struct VulkanDeviceInfo {
    std::string name;
    VulkanDeviceType type = VulkanDeviceType::Other;
    std::uint32_t apiMajor = 0;  ///< Vulkan version the driver supports, major part.
    std::uint32_t apiMinor = 0;  ///< Vulkan version the driver supports, minor part.
    /// The device's UUID in hex: stable across runs, and how CUDA and HIP name the same device.
    std::string uuid;
    /// Meets Ernie's requirements (Vulkan 1.3 and the features it builds on).
    bool supported = false;
    DeviceCapabilities capabilities;
};

/// @brief What went wrong on a device, in Ernie's terms.
enum class DeviceErrorKind : std::uint8_t { OutOfMemory, DeviceLost, Unsupported, Failed };

/// @brief A device call that failed: which call, on what, and how.
class DeviceError : public std::runtime_error {
  public:
    DeviceError(DeviceErrorKind kind, const std::string& message)
        : std::runtime_error(message), kind_(kind) {}
    DeviceErrorKind Kind() const {
        return kind_;
    }

  private:
    DeviceErrorKind kind_;
};

/// @brief Lists the Vulkan devices of the installed drivers.
/// @return Every physical device, in the loader's order.
/// @throws DeviceError When no Vulkan loader is installed or the instance cannot be created.
std::vector<VulkanDeviceInfo> ListVulkanDevices();

/// @return A short lowercase name of a device type, such as "cpu".
const char* ToString(VulkanDeviceType type);

}  // namespace ernie
