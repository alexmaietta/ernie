// SPDX-License-Identifier: BSD-3-Clause
/// @file queue.h
/// @brief Timeline semaphores and the tickets that name points on them, and device queues whose
/// submissions are numbered on one.
#pragma once

#include "ernie/device/external.h"
#include "ernie/device/vulkan.h"

#include <cstdint>
#include <functional>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace ernie {

class Timeline;

/// @brief A point on a timeline, reached once the timeline's value reaches `value`. Tickets name
/// their timeline, and through it its queue or source and its device, so they stay meaningful
/// across queues, devices and APIs. A default ticket is always done.
struct Ticket {
    const Timeline* timeline = nullptr;
    std::uint64_t value = 0;
    bool operator==(const Ticket&) const = default;

    bool Done() const;
    /// @brief Blocks until the ticket is reached.
    /// @throws DeviceError on device loss.
    void Wait() const;
};

/// @brief A timeline semaphore: a queue's, signalled by its submissions, or one shared with
/// another API or context, which signals it there.
class Timeline {
  public:
    /// @param exportable Whether `Export` may share it; the device needs
    /// `DeviceCapabilities::externalTimeline`.
    Timeline(VkDevice device, std::string name, bool exportable = false);
    /// @brief Imports a timeline exported by another API or context, taking `handle` where the
    /// platform does.
    Timeline(VkDevice device, std::string name, ExternalHandle handle);
    ~Timeline();
    Timeline(const Timeline&) = delete;
    Timeline& operator=(const Timeline&) = delete;

    /// @return The value reached so far.
    std::uint64_t Reached() const;
    /// @brief Sets the value from the host, as a producer outside any queue does.
    void Signal(std::uint64_t value) const;
    /// @return A handle another API or context imports the timeline from.
    /// @throws DeviceError unless it was created exportable.
    ExternalHandle Export() const;

    Ticket At(std::uint64_t value) const {
        return {this, value};
    }
    VkSemaphore Handle() const {
        return semaphore_;
    }
    VkDevice Device() const {
        return device_;
    }
    const std::string& Name() const {
        return name_;
    }

  private:
    VkDevice device_;
    std::string name_;
    bool exportable_;
    VkSemaphore semaphore_ = VK_NULL_HANDLE;
};

/// @brief A queue of one family. Submissions take a recording function and return a ticket on the
/// queue's timeline; command buffers are recycled once their tickets complete. One thread submits
/// at a time (its command pool is not thread-safe); any thread may test or wait on tickets.
class Queue {
  public:
    /// @param exportable Whether its timeline may be shared, for other APIs to wait on its work.
    Queue(VkDevice device, std::uint32_t family, std::string name, bool exportable = false);
    ~Queue();
    Queue(const Queue&) = delete;
    Queue& operator=(const Queue&) = delete;

    /// @brief Records commands and submits them once `waits` are done.
    /// @return The submission's ticket.
    /// @throws DeviceError if a wait is another device's, or submission fails.
    Ticket Submit(const std::function<void(VkCommandBuffer)>& record,
                  std::span<const Ticket> waits = {});
    /// @return The ticket of the latest submission: waiting on it waits for all of them.
    Ticket Last() const;

    const ernie::Timeline& Timeline() const {
        return timeline_;
    }
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
    ernie::Timeline timeline_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    mutable std::mutex mutex_;
    std::uint64_t submitted_ = 0;
    std::vector<std::pair<VkCommandBuffer, std::uint64_t>> inFlight_;
    std::vector<VkCommandBuffer> free_;
};

}  // namespace ernie
