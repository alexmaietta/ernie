// SPDX-License-Identifier: BSD-3-Clause
/// @file queue.cpp
/// @brief Submitting to a queue with synchronization2 and a timeline semaphore.
#include "ernie/device/queue.h"

#include "ernie/util/algorithm.h"

#include <limits>

namespace ernie {

Queue::Queue(VkDevice device, std::uint32_t family, std::string name)
    : device_(device), family_(family), name_(std::move(name)) {
    vkGetDeviceQueue(device_, family_, 0, &queue_);
    const VkSemaphoreTypeCreateInfo timeline{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                             .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
    const VkSemaphoreCreateInfo semaphore{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                                          .pNext = &timeline};
    CheckVk(vkCreateSemaphore(device_, &semaphore, nullptr, &timeline_),
            "Creating " + name_ + "'s timeline");
    const VkCommandPoolCreateInfo pool{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                       .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                       .queueFamilyIndex = family_};
    CheckVk(vkCreateCommandPool(device_, &pool, nullptr, &pool_),
            "Creating " + name_ + "'s command pool");
}

Queue::~Queue() {
    Wait(Last());
    vkDestroyCommandPool(device_, pool_, nullptr);
    vkDestroySemaphore(device_, timeline_, nullptr);
}

Ticket Queue::Submit(const std::function<void(VkCommandBuffer)>& record,
                     std::span<const Ticket> waits) {
    std::vector<VkSemaphoreSubmitInfo> waitInfos;
    for (const Ticket& wait : waits) {
        if (wait.queue == nullptr || wait.queue->Done(wait)) {
            continue;
        }
        if (wait.queue->device_ != device_) {
            throw DeviceError(DeviceErrorKind::Failed,
                              name_ + " cannot wait on another device's work");
        }
        waitInfos.push_back({.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                             .semaphore = wait.queue->timeline_,
                             .value = wait.value,
                             .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT});
    }

    std::unique_lock lock(mutex_);
    Recycle();
    VkCommandBuffer commands = VK_NULL_HANDLE;
    if (free_.empty()) {
        const VkCommandBufferAllocateInfo allocate{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = pool_,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1};
        CheckVk(vkAllocateCommandBuffers(device_, &allocate, &commands),
                "Allocating " + name_ + "'s command buffer");
    } else {
        commands = free_.back();
        free_.pop_back();
    }
    lock.unlock();
    // Recorded without the lock, so recording may ask this queue for tickets.
    CheckVk(vkResetCommandBuffer(commands, 0), "Resetting " + name_ + "'s command buffer");
    const VkCommandBufferBeginInfo begin{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    try {
        CheckVk(vkBeginCommandBuffer(commands, &begin), "Recording on " + name_);
        record(commands);
        CheckVk(vkEndCommandBuffer(commands), "Recording on " + name_);
    } catch (...) {
        // Nothing was submitted: the command buffer goes back, to be reset on its next use.
        lock.lock();
        free_.push_back(commands);
        throw;
    }

    lock.lock();
    const std::uint64_t value = submitted_ + 1;
    const VkCommandBufferSubmitInfo commandInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = commands};
    const VkSemaphoreSubmitInfo signal{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                       .semaphore = timeline_,
                                       .value = value,
                                       .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    const VkSubmitInfo2 submit{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                               .waitSemaphoreInfoCount =
                                   static_cast<std::uint32_t>(waitInfos.size()),
                               .pWaitSemaphoreInfos = waitInfos.data(),
                               .commandBufferInfoCount = 1,
                               .pCommandBufferInfos = &commandInfo,
                               .signalSemaphoreInfoCount = 1,
                               .pSignalSemaphoreInfos = &signal};
    CheckVk(vkQueueSubmit2(queue_, 1, &submit, VK_NULL_HANDLE), "Submitting to " + name_);
    submitted_ = value;
    inFlight_.emplace_back(commands, value);
    return {this, value};
}

bool Queue::Done(const Ticket& ticket) const {
    if (ticket.queue == nullptr) {
        return true;
    }
    std::uint64_t reached = 0;
    CheckVk(vkGetSemaphoreCounterValue(ticket.queue->device_, ticket.queue->timeline_, &reached),
            "Reading " + ticket.queue->name_ + "'s timeline");
    return reached >= ticket.value;
}

void Queue::Wait(const Ticket& ticket) const {
    if (ticket.queue == nullptr) {
        return;
    }
    const VkSemaphoreWaitInfo wait{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                   .semaphoreCount = 1,
                                   .pSemaphores = &ticket.queue->timeline_,
                                   .pValues = &ticket.value};
    CheckVk(
        vkWaitSemaphores(ticket.queue->device_, &wait, std::numeric_limits<std::uint64_t>::max()),
        "Waiting on " + ticket.queue->name_);
}

Ticket Queue::Last() const {
    const std::lock_guard lock(mutex_);
    return submitted_ == 0 ? Ticket() : Ticket{this, submitted_};
}

void Queue::Recycle() {
    std::uint64_t reached = 0;
    CheckVk(vkGetSemaphoreCounterValue(device_, timeline_, &reached),
            "Reading " + name_ + "'s timeline");
    std::erase_if(inFlight_, [&](const auto& entry) {
        if (entry.second > reached) {
            return false;
        }
        free_.push_back(entry.first);
        return true;
    });
}

}  // namespace ernie
