// SPDX-License-Identifier: BSD-3-Clause
/// @file context.h
/// @brief A device and everything that serves it: its queues, its allocator, object names and the
/// retirement of objects the GPU may still use.
#pragma once

#include "ernie/device/instance.h"
#include "ernie/device/queue.h"
#include "ernie/device/vulkan.h"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

struct DeviceContextOptions {
    /// The Khronos validation layer with synchronization validation, when installed.
    bool validation = false;
    /// The device to use: its UUID in hex, or part of its name; empty for the best scoring one.
    std::string device;
    std::vector<std::string> instanceExtensions;
};

/// @brief One device. Nothing is global: two contexts in a process, on one device or two, keep
/// their queues, memory and messages apart.
class DeviceContext {
  public:
    /// @throws DeviceError if no device meets Ernie's requirements, or none matches `device`.
    explicit DeviceContext(const DeviceContextOptions& options = {});
    ~DeviceContext();
    DeviceContext(const DeviceContext&) = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;

    const VulkanDeviceInfo& Info() const {
        return physical_.info;
    }
    const DeviceCapabilities& Capabilities() const {
        return physical_.info.capabilities;
    }
    const VkPhysicalDeviceLimits& Limits() const {
        return limits_;
    }
    /// @return The largest buffer a shader can address whole: arenas are no larger.
    std::uint64_t MaxBufferSize() const {
        return maxBufferSize_;
    }

    /// The queue for graphics and compute work.
    Queue& Main() const {
        return *main_;
    }
    /// The queue for asynchronous compute: a compute-only family's, else the main queue.
    Queue& Compute() const {
        return *compute_;
    }
    /// The queue for uploads: a transfer-only family's, else the compute queue.
    Queue& Transfer() const {
        return *transfer_;
    }
    /// @return Each distinct queue once.
    std::vector<Queue*> Queues() const;

    /// @brief Keeps `object` alive until every queue's work submitted so far is done.
    void Retire(std::shared_ptr<const void> object);
    /// @brief Runs `release` once every queue's work submitted so far is done: how memory the GPU
    /// may still read is given back.
    void Retire(std::function<void()> release);
    /// @brief Frees retired objects whose work is done.
    void Collect();
    /// @brief Waits for every queue, then frees everything retired.
    void WaitIdle();

    /// @brief Names a Vulkan object for validation messages, captures and crash reports.
    void Name(VkObjectType type, std::uint64_t handle, std::string_view name) const;
    /// @return The validation layer's messages since the last call; empty without validation.
    std::vector<std::string> TakeValidationMessages() {
        return instance_.TakeMessages();
    }

    VkInstance Instance() const {
        return instance_.Handle();
    }
    VkPhysicalDevice Physical() const {
        return physical_.handle;
    }
    VkDevice Device() const {
        return device_;
    }
    VmaAllocator Allocator() const {
        return allocator_;
    }

  private:
    struct Retired {
        std::function<void()> release;
        std::vector<Ticket> tickets;
    };

    ernie::Instance instance_;
    PhysicalDevice physical_;
    VkPhysicalDeviceLimits limits_{};
    std::uint64_t maxBufferSize_ = 0;
    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    std::vector<std::unique_ptr<Queue>> queues_;
    Queue* main_ = nullptr;
    Queue* compute_ = nullptr;
    Queue* transfer_ = nullptr;
    std::mutex retiredMutex_;
    std::vector<Retired> retired_;
};

}  // namespace ernie
