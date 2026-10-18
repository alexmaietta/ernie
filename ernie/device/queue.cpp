// SPDX-License-Identifier: BSD-3-Clause
/// @file queue.cpp
/// @brief Timeline semaphores, and submitting to a queue with synchronization2.
#include "ernie/device/queue.h"

#include "ernie/util/algorithm.h"
#include "ernie/util/on_failure.h"

#include <array>
#include <limits>

namespace ernie {
namespace {

VkSemaphore CreateTimeline(VkDevice device, const std::string& name, bool exportable) {
    const VkExportSemaphoreCreateInfo exported{
        .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
        .handleTypes = static_cast<VkExternalSemaphoreHandleTypeFlags>(kExternalSemaphoreType)};
    const VkSemaphoreTypeCreateInfo timeline{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                             .pNext = exportable ? &exported : nullptr,
                                             .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
    const VkSemaphoreCreateInfo info{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                                     .pNext = &timeline};
    VkSemaphore semaphore = VK_NULL_HANDLE;
    CheckVk(vkCreateSemaphore(device, &info, nullptr, &semaphore), "Creating " + name);
    return semaphore;
}

}  // namespace

bool Ticket::Done() const {
    return timeline == nullptr || timeline->Reached() >= value;
}

void Ticket::Wait() const {
    if (timeline == nullptr) {
        return;
    }
    ERNIE_ZONE("Wait for the GPU");
    const VkSemaphore semaphore = timeline->Handle();
    const VkSemaphoreWaitInfo wait{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                   .semaphoreCount = 1,
                                   .pSemaphores = &semaphore,
                                   .pValues = &value};
    CheckVk(vkWaitSemaphores(timeline->Device(), &wait, std::numeric_limits<std::uint64_t>::max()),
            [&] { return "Waiting on " + timeline->Name(); });
}

Timeline::Timeline(VkDevice device, std::string name, bool exportable)
    : device_(device), name_(std::move(name)), exportable_(exportable),
      semaphore_(CreateTimeline(device_, name_, exportable)) {}

Timeline::Timeline(VkDevice device, std::string name, ExternalHandle handle)
    : device_(device), name_(std::move(name)), exportable_(false),
      semaphore_(CreateTimeline(device_, name_, false)) {
    const OnFailure destroy([this] { vkDestroySemaphore(device_, semaphore_, nullptr); });
    ImportSemaphore(device_, semaphore_, handle, name_);
}

Timeline::~Timeline() {
    vkDestroySemaphore(device_, semaphore_, nullptr);
}

std::uint64_t Timeline::Reached() const {
    std::uint64_t reached = 0;
    CheckVk(vkGetSemaphoreCounterValue(device_, semaphore_, &reached),
            [&] { return "Reading " + name_; });
    return reached;
}

void Timeline::Signal(std::uint64_t value) const {
    const VkSemaphoreSignalInfo signal{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = semaphore_, .value = value};
    CheckVk(vkSignalSemaphore(device_, &signal), [&] { return "Signalling " + name_; });
}

ExternalHandle Timeline::Export() const {
    if (!exportable_) {
        throw DeviceError(DeviceErrorKind::Unsupported, name_ + " was not created exportable");
    }
    return ExportSemaphore(device_, semaphore_, name_);
}

Queue::Queue(VkDevice device, std::uint32_t family, std::string name, bool exportable)
    : device_(device), family_(family), name_(std::move(name)),
      timeline_(device, name_ + "'s timeline", exportable) {
    vkGetDeviceQueue(device_, family_, 0, &queue_);
    const VkCommandPoolCreateInfo pool{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                       .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                       .queueFamilyIndex = family_};
    CheckVk(vkCreateCommandPool(device_, &pool, nullptr, &pool_),
            "Creating " + name_ + "'s command pool");
}

Queue::~Queue() {
    Quietly([this] { Last().Wait(); });
    vkDestroyCommandPool(device_, pool_, nullptr);
}

Ticket Queue::Submit(const std::function<void(VkCommandBuffer)>& record,
                     std::span<const Ticket> waits, BinarySemaphores binary) {
    // One recording at a time: the command pool is not thread-safe.
    const std::lock_guard recording(recording_);
    std::vector<VkSemaphoreSubmitInfo> waitInfos;
    for (const Ticket& wait : waits) {
        if (wait.timeline == nullptr) {
            continue;
        }
        // Checked first, so the error does not depend on how quickly the other device runs.
        if (wait.timeline->Device() != device_) {
            throw DeviceError(DeviceErrorKind::Failed,
                              name_ + " cannot wait on another device's " + wait.timeline->Name());
        }
        if (wait.Done()) {
            continue;
        }
        waitInfos.push_back({.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                             .semaphore = wait.timeline->Handle(),
                             .value = wait.value,
                             .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT});
    }
    if (binary.wait != VK_NULL_HANDLE) {
        waitInfos.push_back({.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                             .semaphore = binary.wait,
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
    CheckVk(vkResetCommandBuffer(commands, 0),
            [&] { return "Resetting " + name_ + "'s command buffer"; });
    const VkCommandBufferBeginInfo begin{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    try {
        CheckVk(vkBeginCommandBuffer(commands, &begin), [&] { return "Recording on " + name_; });
        record(commands);
        CheckVk(vkEndCommandBuffer(commands), [&] { return "Recording on " + name_; });
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
    const std::array signals{
        VkSemaphoreSubmitInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                              .semaphore = timeline_.Handle(),
                              .value = value,
                              .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
        VkSemaphoreSubmitInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                              .semaphore = binary.signal,
                              .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT}};
    const VkSubmitInfo2 submit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .waitSemaphoreInfoCount = static_cast<std::uint32_t>(waitInfos.size()),
        .pWaitSemaphoreInfos = waitInfos.data(),
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &commandInfo,
        .signalSemaphoreInfoCount = binary.signal != VK_NULL_HANDLE ? 2u : 1u,
        .pSignalSemaphoreInfos = signals.data()};
    const VkResult submitted = vkQueueSubmit2(queue_, 1, &submit, VK_NULL_HANDLE);
    if (submitted < VK_SUCCESS) {
        free_.push_back(commands);
    }
    CheckVk(submitted, [&] { return "Submitting to " + name_; });
    submitted_ = value;
    inFlight_.emplace_back(commands, value);
    return timeline_.At(value);
}

Ticket Queue::Last() const {
    const std::lock_guard lock(mutex_);
    return submitted_ == 0 ? Ticket() : timeline_.At(submitted_);
}

VkResult Queue::Present(const VkPresentInfoKHR& info) {
    // Vulkan leaves a queue's host access for its users to synchronize.
    const std::lock_guard lock(mutex_);
    return vkQueuePresentKHR(queue_, &info);
}

void Queue::Borrow(const std::function<void(VkQueue)>& use) {
    const std::lock_guard lock(mutex_);
    use(queue_);
}

void Queue::Recycle() {
    const std::uint64_t reached = timeline_.Reached();
    std::erase_if(inFlight_, [&](const auto& entry) {
        if (entry.second > reached) {
            return false;
        }
        free_.push_back(entry.first);
        return true;
    });
}

}  // namespace ernie
