// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <functional>

#include <vulkan/vulkan_core.h>

#ifndef VK_API_VERSION_1_3
#error "Ernie's Vulkan interop needs the Vulkan 1.3 headers or later"
#endif

#include "ernie/device/device.h"
#include "ernie/device/operation.h"
#include "ernie/device/tensor.h"
#include "ernie/export.h"
#include "ernie/util/span.h"

namespace ernie::device::vulkan {

enum class QueueRole : uint8_t { Render, Display };

struct Creator {
    std::function<VkInstance(PFN_vkGetInstanceProcAddr, const VkInstanceCreateInfo& info)>
        createInstance;
    std::function<VkPhysicalDevice(VkInstance instance)> pickAdapter;
    std::function<VkDevice(PFN_vkGetInstanceProcAddr, VkPhysicalDevice adapter,
                           const VkDeviceCreateInfo& info)>
        createDevice;
};

struct Native {
    PFN_vkGetInstanceProcAddr getInstanceProcAddr = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t apiVersion = 0;
    uint32_t queueFamily = 0;
    uint32_t queueIndex = 0;
    bool displaySharesRender = true;
};

struct NativeSubmit {
    Span<const VkCommandBuffer> commands;
    Span<const Operation> after;
    Span<const VkSemaphoreSubmitInfo> waits;
    Span<const VkSemaphoreSubmitInfo> signals;
};

ERNIE_API Native NativeHandles(Device& device);

ERNIE_API VkBuffer NativeBuffer(const Tensor& tensor);

ERNIE_API Operation Submit(Device& device, const NativeSubmit& submit,
                           QueueRole role = QueueRole::Render);

ERNIE_API VkResult Present(Device& device, const VkPresentInfoKHR& present);

ERNIE_API void Exclusive(Device& device, const std::function<void(VkQueue queue)>& use,
                         QueueRole role = QueueRole::Display);

}
