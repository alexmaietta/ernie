// SPDX-License-Identifier: BSD-3-Clause
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/adapter.h"
#include "ernie/device/device.h"
#include "ernie/device/vulkan/native.h"
#include "ernie/util/error.h"
#include "support/buffers.h"

namespace ernie::device {
namespace {

struct Made {
    PFN_vkGetInstanceProcAddr get = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t instances = 0;
    uint32_t devices = 0;
    std::array<uint8_t, 16> wanted{};
};

template <class Function> Function Load(const Made& made, const char* name) {
    return reinterpret_cast<Function>(made.get(made.instance, name));
}

std::optional<AdapterInfo> LastVulkanAdapter() {
    std::optional<AdapterInfo> last;
    for (const AdapterInfo& adapter : EnumerateAdapters()) {
        if (adapter.backend == Backend::Vulkan && adapter.unsupported.empty()) {
            last = adapter;
        }
    }
    return last;
}

std::shared_ptr<vulkan::Creator> CountingCreator(const std::shared_ptr<Made>& made) {
    auto creator = std::make_shared<vulkan::Creator>();
    creator->createInstance = [made](PFN_vkGetInstanceProcAddr get,
                                     const VkInstanceCreateInfo& info) {
        made->get = get;
        const auto create =
            reinterpret_cast<PFN_vkCreateInstance>(get(VK_NULL_HANDLE, "vkCreateInstance"));
        EXPECT_EQ(create(&info, nullptr, &made->instance), VK_SUCCESS);
        ++made->instances;
        return made->instance;
    };
    creator->createDevice = [made](PFN_vkGetInstanceProcAddr, VkPhysicalDevice adapter,
                                   const VkDeviceCreateInfo& info) {
        const auto create = Load<PFN_vkCreateDevice>(*made, "vkCreateDevice");
        EXPECT_EQ(create(adapter, &info, nullptr, &made->device), VK_SUCCESS);
        ++made->devices;
        return made->device;
    };
    return creator;
}

TEST(CreatorTest, MakesTheInstanceAndDeviceOnTheAdapterItPicks) {
    const std::optional<AdapterInfo> adapter = LastVulkanAdapter();
    if (!adapter) {
        GTEST_SKIP() << "no supported Vulkan adapter";
    }
    const auto made = std::make_shared<Made>();
    made->wanted = adapter->uuid;
    const std::shared_ptr<vulkan::Creator> creator = CountingCreator(made);
    creator->pickAdapter = [made](VkInstance instance) {
        const auto enumerate =
            Load<PFN_vkEnumeratePhysicalDevices>(*made, "vkEnumeratePhysicalDevices");
        const auto describe =
            Load<PFN_vkGetPhysicalDeviceProperties2>(*made, "vkGetPhysicalDeviceProperties2");
        uint32_t count = 0;
        enumerate(instance, &count, nullptr);
        std::vector<VkPhysicalDevice> devices(count);
        enumerate(instance, &count, devices.data());
        for (VkPhysicalDevice device : devices) {
            VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                                   &id};
            describe(device, &properties);
            if (std::memcmp(id.deviceUUID, made->wanted.data(), VK_UUID_SIZE) == 0) {
                return device;
            }
        }
        return VkPhysicalDevice(VK_NULL_HANDLE);
    };

    {
        const Ref<Device> device = Device::Create({.creator = creator});
        EXPECT_EQ(device->Adapter().uuid, adapter->uuid);
        EXPECT_EQ(made->instances, 1u);
        EXPECT_EQ(made->devices, 1u);
        const vulkan::Native native = vulkan::NativeHandles(*device);
        EXPECT_EQ(native.instance, made->instance);
        EXPECT_EQ(native.device, made->device);

        const std::vector<uint32_t> values = {1, 2, 3, 4};
        const Ref<Buffer> buffer = test::Upload(*device, values);
        EXPECT_EQ(test::Download<uint32_t>(*device, *buffer, values.size()), values);
    }
}

TEST(CreatorTest, UsesTheChosenAdapterWithoutAPick) {
    const std::optional<AdapterInfo> adapter = LastVulkanAdapter();
    if (!adapter) {
        GTEST_SKIP() << "no supported Vulkan adapter";
    }
    const auto made = std::make_shared<Made>();
    const Ref<Device> device =
        Device::Create({.adapter = adapter->alias, .creator = CountingCreator(made)});
    EXPECT_EQ(device->Adapter().uuid, adapter->uuid);
    EXPECT_EQ(vulkan::NativeHandles(*device).instance, made->instance);
    EXPECT_EQ(made->devices, 1u);
}

TEST(CreatorTest, NeedsTheVulkanBackendAndBothCallbacks) {
    const auto made = std::make_shared<Made>();
    EXPECT_THROW(Device::Create({.backend = Backend::Cuda, .creator = CountingCreator(made)}),
                 Error);
    auto partial = std::make_shared<vulkan::Creator>();
    partial->createInstance = CountingCreator(made)->createInstance;
    if (LastVulkanAdapter()) {
        EXPECT_THROW(Device::Create({.creator = partial}), Error);
    }
    EXPECT_EQ(made->instances, 0u);
}

}
}
