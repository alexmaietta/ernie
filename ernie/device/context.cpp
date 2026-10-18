// SPDX-License-Identifier: BSD-3-Clause
/// @file context.cpp
/// @brief Choosing a device, creating it with its queues and allocator, and retiring objects.
#include "ernie/device/context.h"

#include "ernie/util/algorithm.h"
#include "ernie/util/on_failure.h"
#include "ernie/util/text.h"

#include <algorithm>
#include <array>
#include <optional>

namespace ernie {
namespace {

/// The releases this thread is running, which a `WaitIdle` inside one must not wait for.
thread_local std::size_t releasingHere = 0;

PhysicalDevice Choose(const Instance& instance, std::string_view wanted) {
    std::vector<PhysicalDevice> devices = Map(instance.PhysicalDevices(), Probe);
    std::erase_if(devices, [&](const PhysicalDevice& device) {
        return !wanted.empty() && device.info.uuid != wanted &&
               ToLower(device.info.name).find(ToLower(wanted)) == std::string::npos;
    });
    const auto best = std::ranges::max_element(devices, {}, &PhysicalDevice::Score);
    if (best == devices.end()) {
        throw DeviceError(DeviceErrorKind::Unsupported,
                          wanted.empty()
                              ? "No Vulkan device"
                              : "No Vulkan device matches '" + std::string(wanted) + "'");
    }
    if (!best->missing.empty()) {
        throw DeviceError(DeviceErrorKind::Unsupported,
                          best->info.name + " lacks " + best->missing + ", which Ernie requires");
    }
    return *best;
}

/// @brief The families the queues come from: main (graphics and compute), compute and transfer.
struct Families {
    std::uint32_t main = 0;
    std::uint32_t compute = 0;
    std::uint32_t transfer = 0;
};

/// Prefers a compute-only family for compute and a transfer-only one for uploads, where present.
Families FindFamilies(VkPhysicalDevice device) {
    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());
    const auto find = [&](VkQueueFlags required,
                          VkQueueFlags excluded) -> std::optional<std::uint32_t> {
        for (std::uint32_t i = 0; i < count; ++i) {
            if ((families[i].queueFlags & required) == required &&
                (families[i].queueFlags & excluded) == 0 && families[i].queueCount > 0) {
                return i;
            }
        }
        return std::nullopt;
    };
    const std::optional<std::uint32_t> main = find(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT, 0);
    if (!main) {
        throw DeviceError(DeviceErrorKind::Unsupported,
                          "No queue family does graphics and compute");
    }
    Families result{*main, *main, *main};
    result.compute = find(VK_QUEUE_COMPUTE_BIT, VK_QUEUE_GRAPHICS_BIT).value_or(*main);
    result.transfer = find(VK_QUEUE_TRANSFER_BIT, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)
                          .value_or(result.compute);
    return result;
}

}  // namespace

DeviceContext::DeviceContext(const DeviceContextOptions& options)
    : instance_({.validation = options.validation, .extensions = options.instanceExtensions}),
      physical_(Choose(instance_, options.device)) {
    VkPhysicalDeviceMaintenance4Properties maintenance4{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                           .pNext = &maintenance4};
    vkGetPhysicalDeviceProperties2(physical_.handle, &properties);
    limits_ = properties.properties.limits;
    maxBufferSize_ =
        std::min<std::uint64_t>(maintenance4.maxBufferSize, limits_.maxStorageBufferRange);

    const Families families = FindFamilies(physical_.handle);
    std::vector<std::uint32_t> distinct{families.main};
    for (const std::uint32_t family : {families.compute, families.transfer}) {
        if (std::ranges::find(distinct, family) == distinct.end()) {
            distinct.push_back(family);
        }
    }
    const float priority = 1.0f;
    const std::vector<VkDeviceQueueCreateInfo> queueInfos =
        Map(distinct, [&](std::uint32_t family) {
            return VkDeviceQueueCreateInfo{.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                           .queueFamilyIndex = family,
                                           .queueCount = 1,
                                           .pQueuePriorities = &priority};
        });
    const FeatureChain features = physical_.EnabledFeatures();
    const std::vector<const char*> extensions = physical_.EnabledExtensions();
    const VkDeviceCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &features.core,
        .queueCreateInfoCount = static_cast<std::uint32_t>(queueInfos.size()),
        .pQueueCreateInfos = queueInfos.data(),
        .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data()};
    CheckVk(vkCreateDevice(physical_.handle, &info, nullptr, &device_),
            "Creating the device on " + physical_.info.name);
    const OnFailure freeDevice([&] { vkDestroyDevice(device_, nullptr); });

    const VmaVulkanFunctions functions{.vkGetInstanceProcAddr = vkGetInstanceProcAddr,
                                       .vkGetDeviceProcAddr = vkGetDeviceProcAddr};
    VmaAllocatorCreateInfo allocator{.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
                                     .physicalDevice = physical_.handle,
                                     .device = device_,
                                     .pVulkanFunctions = &functions,
                                     .instance = instance_.Handle(),
                                     .vulkanApiVersion = VK_API_VERSION_1_3};
    if (physical_.info.capabilities.memoryBudget) {
        allocator.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    }
    CheckVk(vmaCreateAllocator(&allocator, &allocator_), "Creating the memory allocator");
    const OnFailure freeAllocator([&] {
        queues_.clear();
        vmaDestroyAllocator(allocator_);
    });

    const std::array names{"main", "compute", "transfer"};
    const std::array wanted{families.main, families.compute, families.transfer};
    std::array<Queue*, 3> chosen{};
    for (std::size_t i = 0; i < wanted.size(); ++i) {
        const auto existing = std::ranges::find(queues_, wanted[i], &Queue::Family);
        if (existing != queues_.end()) {
            chosen[i] = existing->get();
            continue;
        }
        queues_.push_back(std::make_unique<Queue>(device_, wanted[i], names[i],
                                                  physical_.info.capabilities.externalTimeline));
        chosen[i] = queues_.back().get();
        Name(VK_OBJECT_TYPE_QUEUE, reinterpret_cast<std::uint64_t>(chosen[i]->Handle()), names[i]);
    }
    main_ = chosen[0];
    compute_ = chosen[1];
    transfer_ = chosen[2];
}

DeviceContext::~DeviceContext() {
    Quietly([this] { WaitIdle(); });
    queues_.clear();
    vmaDestroyAllocator(allocator_);
    vkDestroyDevice(device_, nullptr);
}

std::vector<Queue*> DeviceContext::Queues() const {
    return Map(queues_, [](const std::unique_ptr<Queue>& queue) { return queue.get(); });
}

void DeviceContext::Retire(std::shared_ptr<const void> object) {
    if (object != nullptr) {
        Retire([held = std::move(object)]() mutable { held.reset(); });
    }
}

void DeviceContext::Retire(std::function<void()> release) {
    std::vector<Ticket> tickets = Map(queues_, [](const auto& queue) { return queue->Last(); });
    const std::lock_guard lock(retiredMutex_);
    retired_.push_back({std::move(release), std::move(tickets)});
}

void DeviceContext::Collect() {
    std::vector<Retired> done;
    {
        const std::lock_guard lock(retiredMutex_);
        const auto finished = std::ranges::partition(retired_, [](const Retired& retired) {
            return !std::ranges::all_of(retired.tickets,
                                        [](const Ticket& ticket) { return ticket.Done(); });
        });
        done.assign(std::make_move_iterator(finished.begin()),
                    std::make_move_iterator(finished.end()));
        retired_.erase(finished.begin(), finished.end());
        releasing_ += done.size();
    }
    // Released without the lock: a release may retire more.
    for (Retired& retired : done) {
        ++releasingHere;
        retired.release();
        --releasingHere;
        const std::lock_guard lock(retiredMutex_);
        --releasing_;
    }
    released_.notify_all();
}

void DeviceContext::WaitIdle() {
    for (const auto& queue : queues_) {
        queue->Last().Wait();
    }
    Collect();
    // Releases another thread took are done too, so their owners may go.
    std::unique_lock lock(retiredMutex_);
    released_.wait(lock, [this] { return releasing_ <= releasingHere; });
}

void DeviceContext::Name(VkObjectType type, std::uint64_t handle, std::string_view name) const {
    instance_.Name(device_, type, handle, name);
}

}  // namespace ernie
