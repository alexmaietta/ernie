// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/vulkan/native.h"

#include "ernie/device/buffer.h"
#include "ernie/device/queue.h"
#include "ernie/device/vulkan/device.h"
#include "ernie/util/error.h"

namespace ernie::device::vulkan {
namespace {

VulkanDevice& Checked(Device& device) {
    if (device.Adapter().backend != Backend::Vulkan) {
        throw UnsupportedError("{} is a {} device, not a Vulkan one", device.Adapter().alias,
                               BackendName(device.Adapter().backend));
    }
    return static_cast<VulkanDevice&>(device);
}

Queue& RoleQueue(VulkanDevice& vulkan, QueueRole role) {
    return role == QueueRole::Display ? vulkan.DisplayQueue() : vulkan.GetQueue();
}

}

Native NativeHandles(Device& device) {
    VulkanDevice& vulkan = Checked(device);
    return {.getInstanceProcAddr = vkGetInstanceProcAddr,
            .instance = vulkan.InstanceHandle(),
            .physicalDevice = vulkan.PhysicalDevice(),
            .device = vulkan.Handle(),
            .apiVersion = VK_API_VERSION_1_3,
            .queueFamily = vulkan.QueueFamily(),
            .displaySharesRender = vulkan.SharesDisplayQueue()};
}

VkBuffer NativeBuffer(const Tensor& tensor) {
    if (tensor.Empty()) {
        throw Error("an empty tensor has no Vulkan buffer");
    }
    Buffer& buffer = tensor.GetBuffer();
    Checked(buffer.GetDevice());
    return static_cast<VulkanBuffer&>(buffer).Handle();
}

Operation Submit(Device& device, const NativeSubmit& submit, QueueRole role) {
    Queue& queue = RoleQueue(Checked(device), role);
    for (const Operation& operation : submit.after) {
        queue.Follow(operation);
    }
    return SubmitNative(queue, submit);
}

VkResult Present(Device& device, const VkPresentInfoKHR& present) {
    Queue& queue = Checked(device).DisplayQueue();
    if (!device.Presents()) {
        throw UnsupportedError("{} wasn't created with presentation, so it presents nothing",
                               device.Adapter().alias);
    }
    return PresentNative(queue, present);
}

void Exclusive(Device& device, const std::function<void(VkQueue queue)>& use, QueueRole role) {
    Queue& queue = RoleQueue(Checked(device), role);
    queue.Exclusive([&] { use(NativeQueue(queue)); });
}

}
