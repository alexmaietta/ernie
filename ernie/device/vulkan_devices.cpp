// SPDX-License-Identifier: BSD-3-Clause
/// @file vulkan_devices.cpp
/// @brief Lists Vulkan devices through a short-lived instance.
#include "ernie/device/vulkan_devices.h"

#include <volk.h>

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string_view>

namespace ernie {
namespace {

bool HasExtension(const std::vector<VkExtensionProperties>& extensions, std::string_view name) {
    return std::ranges::any_of(
        extensions, [&](const VkExtensionProperties& e) { return name == e.extensionName; });
}

std::vector<VkExtensionProperties> InstanceExtensions() {
    std::uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data());
    return extensions;
}

std::vector<VkExtensionProperties> DeviceExtensions(VkPhysicalDevice device) {
    std::uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data());
    return extensions;
}

/// Destroys the instance when the listing is done, even on errors.
class ScopedInstance {
  public:
    ScopedInstance() {
        if (volkInitialize() != VK_SUCCESS) {
            throw std::runtime_error("No Vulkan loader is installed");
        }
        VkApplicationInfo application{.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                      .pApplicationName = "ernie",
                                      .apiVersion = VK_API_VERSION_1_3};
        VkInstanceCreateInfo info{.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                  .pApplicationInfo = &application};
        // Drivers layered over Metal (MoltenVK, Mesa on macOS) are listed only with portability
        // enumeration.
        const char* portability = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
        if (HasExtension(InstanceExtensions(), portability)) {
            info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
            info.enabledExtensionCount = 1;
            info.ppEnabledExtensionNames = &portability;
        }
        if (vkCreateInstance(&info, nullptr, &instance_) != VK_SUCCESS) {
            throw std::runtime_error("Cannot create a Vulkan 1.3 instance");
        }
        volkLoadInstanceOnly(instance_);
    }
    ~ScopedInstance() {
        vkDestroyInstance(instance_, nullptr);
    }
    ScopedInstance(const ScopedInstance&) = delete;
    ScopedInstance& operator=(const ScopedInstance&) = delete;

    VkInstance Get() const {
        return instance_;
    }

  private:
    VkInstance instance_ = VK_NULL_HANDLE;
};

VulkanDeviceType DeviceTypeOf(VkPhysicalDeviceType type) {
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

VulkanDeviceInfo InfoOf(VkPhysicalDevice device) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(device, &properties);
    return {.name = properties.deviceName,
            .type = DeviceTypeOf(properties.deviceType),
            .apiMajor = VK_API_VERSION_MAJOR(properties.apiVersion),
            .apiMinor = VK_API_VERSION_MINOR(properties.apiVersion),
            .rayQuery = HasExtension(DeviceExtensions(device), VK_KHR_RAY_QUERY_EXTENSION_NAME)};
}

}  // namespace

std::vector<VulkanDeviceInfo> ListVulkanDevices() {
    ScopedInstance instance;
    std::uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance.Get(), &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance.Get(), &count, devices.data());

    std::vector<VulkanDeviceInfo> infos;
    infos.reserve(devices.size());
    std::ranges::transform(devices, std::back_inserter(infos), InfoOf);
    return infos;
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
