// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <vector>

#include <volk.h>

#include "ernie/util/version.h"

namespace ernie::device::vulkan {

constexpr Version kMinimumInstanceVersion{1, 3};

inline constexpr const char* kSurfaceExtensions[] = {
    VK_KHR_SURFACE_EXTENSION_NAME, "VK_KHR_xlib_surface",  "VK_KHR_xcb_surface",
    "VK_KHR_wayland_surface",      "VK_KHR_win32_surface", "VK_EXT_metal_surface"};

constexpr Version DecodeVersion(uint32_t packed) {
    return {int(packed >> 22 & 0x7f), int(packed >> 12 & 0x3ff), int(packed & 0xfff)};
}

struct Api {
    Version instanceVersion;
};

template <class T, class Count = uint32_t, class Query> std::vector<T> Enumerate(Query query) {
    Count count = 0;
    query(&count, static_cast<T*>(nullptr));
    std::vector<T> items(count);
    query(&count, items.data());
    items.resize(count);
    return items;
}

bool Available();

const Api& GetApi();

const char* ResultName(VkResult result);

void Check(VkResult result, const char* what);

class Extensions {
  public:
    explicit Extensions(VkPhysicalDevice device = VK_NULL_HANDLE);
    bool Has(const char* name) const;

  private:
    std::vector<VkExtensionProperties> m_extensions;
};

VkInstance Instance(bool validation);

struct Creator;

class OwnedInstance {
  public:
    OwnedInstance() = default;

    OwnedInstance(VkInstance instance, VkDebugUtilsMessengerEXT messenger)
        : m_instance(instance), m_messenger(messenger) {}

    OwnedInstance(OwnedInstance&& other) noexcept;
    OwnedInstance& operator=(OwnedInstance&& other) noexcept;
    ~OwnedInstance();

    VkInstance Get() const {
        return m_instance;
    }

  private:
    void Destroy() noexcept;

    VkInstance m_instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_messenger = VK_NULL_HANDLE;
};

OwnedInstance CreateInstance(bool validation, const Creator& creator);

bool ValidationEnabled();

uint64_t ValidationMessageCount();

std::vector<VkPhysicalDevice> PhysicalDevices(VkInstance instance);

void LoadDeviceFunctions(VolkDeviceTable& functions, VkDevice device);

}
