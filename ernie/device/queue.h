// SPDX-License-Identifier: BSD-3-Clause
/// @file queue.h
/// @brief A device queue whose submissions are numbered on one timeline semaphore, and the tickets
/// that name them.
#pragma once

#include "ernie/device/vulkan.h"

#include <cstdint>
#include <functional>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace ernie {

class Queue;

/// @brief A point in one queue's submissions, done once the queue's timeline reaches `value`.
/// Tickets name their queue, and through it their device, so they stay meaningful across queues
/// and devices.
struct Ticket {
    const Queue* queue = nullptr;
    std::uint64_t value = 0;
    bool operator==(const Ticket&) const = default;
};

/// @brief A queue of one family. Submissions take a recording function and return a ticket;
/// command buffers are recycled once their tickets complete. One thread submits at a time (its
/// command pool is not thread-safe); any thread may test or wait on tickets.
class Queue {
  public:
    Queue(VkDevice device, std::uint32_t family, std::string name);
    ~Queue();
    Queue(const Queue&) = delete;
    Queue& operator=(const Queue&) = delete;

    /// @brief Records commands and submits them once `waits` are done.
    /// @return The submission's ticket.
    /// @throws DeviceError if a wait is another device's, or submission fails.
    Ticket Submit(const std::function<void(VkCommandBuffer)>& record,
                  std::span<const Ticket> waits = {});
    /// @return Whether the ticket's work is done; a default ticket always is.
    bool Done(const Ticket& ticket) const;
    /// @brief Blocks until the ticket's work is done.
    /// @throws DeviceError on device loss.
    void Wait(const Ticket& ticket) const;
    /// @return The ticket of the latest submission: waiting on it waits for all of them.
    Ticket Last() const;

    VkQueue Handle() const {
        return queue_;
    }
    std::uint32_t Family() const {
        return family_;
    }
    VkDevice Device() const {
        return device_;
    }
    const std::string& Name() const {
        return name_;
    }

  private:
    /// Moves finished command buffers to the free list.
    void Recycle();

    VkDevice device_;
    std::uint32_t family_;
    std::string name_;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkSemaphore timeline_ = VK_NULL_HANDLE;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    mutable std::mutex mutex_;
    std::uint64_t submitted_ = 0;
    std::vector<std::pair<VkCommandBuffer, std::uint64_t>> inFlight_;
    std::vector<VkCommandBuffer> free_;
};

}  // namespace ernie
