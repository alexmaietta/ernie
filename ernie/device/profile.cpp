// SPDX-License-Identifier: BSD-3-Clause
/// @file profile.cpp
/// @brief Ernie's GPU timer and Tracy's host-calibrated Vulkan context, and their zones around
/// commands.
#include "ernie/device/profile.h"

#include "ernie/device/context.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#if ERNIE_TRACY
#include <tracy/TracyVulkan.hpp>

#include <new>
#endif

namespace ernie {
namespace {

/// Zones in flight at once, two queries each.
constexpr std::uint32_t kZones = 4096;
/// Frames after which a zone still without timestamps is taken as never submitted.
constexpr std::uint64_t kStaleFrames = 64;
/// How often the device's clock is read again with the host's, against drift.
constexpr std::uint64_t kRecalibrateNs = 1'000'000'000;

#if defined(__linux__)
/// The host clock the steady clock reads, so timestamps convert to `profiler::Now`.
constexpr VkTimeDomainEXT kSteadyDomain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT;
#endif

/// @brief The zones this thread has open, for their nesting depths.
struct OpenZones {
    std::array<VkCommandBuffer, 32> commands{};
    std::uint32_t count = 0;
};
thread_local OpenZones openZones;

/// @return Whether the device can read its clock together with the steady clock's.
bool CalibratesSteadyClock(VkPhysicalDevice physical) {
#if defined(__linux__)
    std::uint32_t count = 0;
    vkGetPhysicalDeviceCalibrateableTimeDomainsEXT(physical, &count, nullptr);
    std::vector<VkTimeDomainEXT> domains(count);
    vkGetPhysicalDeviceCalibrateableTimeDomainsEXT(physical, &count, domains.data());
    return std::ranges::find(domains, kSteadyDomain) != domains.end() &&
           std::ranges::find(domains, VK_TIME_DOMAIN_DEVICE_EXT) != domains.end();
#else
    static_cast<void>(physical);
    return false;
#endif
}

/// @return The timestamp bits every queue's family keeps; 0 if one keeps none.
std::uint32_t TimestampBits(const DeviceContext& context) {
    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(context.Physical(), &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(context.Physical(), &count, families.data());
    std::uint32_t bits = 64;
    for (const Queue* queue : context.Queues()) {
        bits = std::min(bits, families[queue->Family()].timestampValidBits);
    }
    return bits;
}

}  // namespace

/// @brief Timestamp queries around zones, read back into `profiler::AddGpuZone` on the steady
/// clock: calibrated where the device reads its clock with the steady clock's, else aligned once
/// to when the first zone was recorded.
class GpuProfiler::Timer {
  public:
    Timer(const DeviceContext& context, std::uint32_t bits)
        : device_(context.Device()), period_(context.Limits().timestampPeriod),
          mask_(bits >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << bits) - 1),
          calibrates_(CalibratesSteadyClock(context.Physical())) {
        const VkQueryPoolCreateInfo info{.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                         .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                         .queryCount = 2 * kZones};
        CheckVk(vkCreateQueryPool(device_, &info, nullptr, &pool_), "Creating the GPU timer");
        context.Name(VK_OBJECT_TYPE_QUERY_POOL, reinterpret_cast<std::uint64_t>(pool_),
                     "GPU timer");
        if (calibrates_) {
            Calibrate();
        }
    }
    ~Timer() {
        vkDestroyQueryPool(device_, pool_, nullptr);
    }
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;

    std::uint32_t Begin(VkCommandBuffer commands, const char* name) {
        OpenZones& open = openZones;
        if (open.count == open.commands.size()) {
            return kUntimed;
        }
        const std::uint32_t depth = static_cast<std::uint32_t>(
            std::count(open.commands.begin(), open.commands.begin() + open.count, commands));
        std::uint32_t zone = 0;
        {
            const std::lock_guard lock(mutex_);
            if (tail_ - head_ == kZones) {
                return kUntimed;
            }
            zone = static_cast<std::uint32_t>(tail_++ % kZones);
            slots_[zone] = {.name = profiler::Intern(name),
                            .frame = profiler::CurrentFrame(),
                            .recorded = profiler::Now(),
                            .depth = depth,
                            .state = State::Open};
        }
        open.commands[open.count++] = commands;
        vkResetQueryPool(device_, pool_, 2 * zone, 2);
        vkCmdWriteTimestamp2(commands, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pool_, 2 * zone);
        return zone;
    }

    void End(VkCommandBuffer commands, std::uint32_t zone) {
        vkCmdWriteTimestamp2(commands, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pool_, 2 * zone + 1);
        --openZones.count;
        const std::lock_guard lock(mutex_);
        slots_[zone].state = State::Closed;
    }

    void Collect() {
        const std::lock_guard lock(mutex_);
        if (calibrates_ && profiler::Now() - calibratedAt_ > kRecalibrateNs) {
            Calibrate();
        }
        const std::uint64_t frame = profiler::CurrentFrame();
        for (std::uint64_t i = head_; i < tail_; ++i) {
            const auto zone = static_cast<std::uint32_t>(i % kZones);
            Slot& slot = slots_[zone];
            if (slot.state != State::Closed) {
                continue;
            }
            // Each timestamp followed by its availability.
            std::array<std::uint64_t, 4> results{};
            const VkResult result = vkGetQueryPoolResults(
                device_, pool_, 2 * zone, 2, sizeof(results), results.data(),
                2 * sizeof(std::uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
            if (result >= 0 && results[1] != 0 && results[3] != 0) {
                if (!aligned_) {
                    deviceTicks_ = results[0] & mask_;
                    hostNs_ = slot.recorded;
                    aligned_ = true;
                }
                const std::uint64_t start = ToHost(results[0]);
                profiler::AddGpuZone(slot.frame, {.name = slot.name,
                                                  .start = start,
                                                  .end = std::max(start, ToHost(results[2])),
                                                  .depth = slot.depth});
                slot.state = State::Read;
            } else if (slot.frame + kStaleFrames < frame) {
                slot.state = State::Read;
            }
        }
        while (head_ < tail_ && slots_[head_ % kZones].state == State::Read) {
            slots_[head_++ % kZones].state = State::Free;
        }
    }

  private:
    enum class State : std::uint8_t { Free, Open, Closed, Read };
    struct Slot {
        const char* name = "";
        std::uint64_t frame = 0;
        /// When it was recorded, on the steady clock.
        std::uint64_t recorded = 0;
        std::uint32_t depth = 0;
        State state = State::Free;
    };

    /// Reads the device's clock with the steady clock's.
    void Calibrate() {
#if defined(__linux__)
        const std::array<VkCalibratedTimestampInfoEXT, 2> domains{
            VkCalibratedTimestampInfoEXT{.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT,
                                         .timeDomain = VK_TIME_DOMAIN_DEVICE_EXT},
            VkCalibratedTimestampInfoEXT{.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT,
                                         .timeDomain = kSteadyDomain}};
        std::array<std::uint64_t, 2> timestamps{};
        std::uint64_t deviation = 0;
        if (vkGetCalibratedTimestampsEXT(device_, 2, domains.data(), timestamps.data(),
                                         &deviation) == VK_SUCCESS) {
            deviceTicks_ = timestamps[0] & mask_;
            hostNs_ = timestamps[1];
            aligned_ = true;
        }
#endif
        calibratedAt_ = profiler::Now();
    }

    /// @return Device ticks on the steady clock, in nanoseconds.
    std::uint64_t ToHost(std::uint64_t ticks) const {
        // Ticks wrap at the family's valid bits; those before the reference count back.
        const std::uint64_t ahead = (ticks - deviceTicks_) & mask_;
        const std::uint64_t behind = (deviceTicks_ - ticks) & mask_;
        const double ns = ahead <= mask_ / 2 ? static_cast<double>(ahead) * period_
                                             : -static_cast<double>(behind) * period_;
        return static_cast<std::uint64_t>(static_cast<double>(hostNs_) + ns);
    }

    VkDevice device_;
    VkQueryPool pool_ = VK_NULL_HANDLE;
    double period_;
    std::uint64_t mask_;
    bool calibrates_;
    std::mutex mutex_;
    std::array<Slot, kZones> slots_{};
    /// Monotonic slot counts: the oldest unread and the next to record.
    std::uint64_t head_ = 0;
    std::uint64_t tail_ = 0;
    /// A device tick and the steady clock's nanosecond at the same moment.
    bool aligned_ = false;
    std::uint64_t deviceTicks_ = 0;
    std::uint64_t hostNs_ = 0;
    std::uint64_t calibratedAt_ = 0;
};

GpuProfiler::GpuProfiler(void* tracy, std::unique_ptr<Timer> timer)
    : tracy_(tracy), timer_(std::move(timer)) {}

std::unique_ptr<GpuProfiler> GpuProfiler::Create(const DeviceContext& context) {
    if (!context.Capabilities().calibratedTimestamps) {
        return nullptr;
    }
    const std::uint32_t bits = TimestampBits(context);
    std::unique_ptr<Timer> timer = bits > 0 ? std::make_unique<Timer>(context, bits) : nullptr;
    void* tracy = nullptr;
#if ERNIE_TRACY
    tracy::VkCtx* vk = tracy::CreateVkContext(
        context.Physical(), context.Device(), vkResetQueryPool,
        vkGetPhysicalDeviceCalibrateableTimeDomainsEXT, vkGetCalibratedTimestampsEXT);
    const std::string& name = context.Info().name;
    vk->Name(name.c_str(), static_cast<std::uint16_t>(name.size()));
    tracy = vk;
#endif
    if (tracy == nullptr && timer == nullptr) {
        return nullptr;
    }
    return std::unique_ptr<GpuProfiler>(new GpuProfiler(tracy, std::move(timer)));
}

GpuProfiler::~GpuProfiler() {
#if ERNIE_TRACY
    tracy::DestroyVkContext(static_cast<tracy::VkCtx*>(tracy_));
#endif
}

void GpuProfiler::Collect() {
    // One thread collects at a time; a busy one is collecting already.
    const std::unique_lock lock(collecting_, std::try_to_lock);
    if (!lock.owns_lock()) {
        return;
    }
    if (timer_) {
        timer_->Collect();
    }
#if ERNIE_TRACY
    static_cast<tracy::VkCtx*>(tracy_)->Collect(VK_NULL_HANDLE);
#endif
}

std::uint32_t GpuProfiler::Begin(VkCommandBuffer commands, const char* name) {
    return timer_ ? timer_->Begin(commands, name) : kUntimed;
}

void GpuProfiler::End(VkCommandBuffer commands, std::uint32_t zone) {
    timer_->End(commands, zone);
}

GpuZone::GpuZone(const DeviceContext& context, VkCommandBuffer commands, const char* name)
    : context_(context), commands_(commands) {
    context.BeginLabel(commands, name);
    GpuProfiler* profiler = context.Profiler();
    if (profiler == nullptr) {
        return;
    }
#if ERNIE_TRACY
    static_assert(sizeof(tracy::VkCtxScope) <= sizeof(zone_) &&
                  alignof(tracy::VkCtxScope) <= alignof(std::max_align_t));
    // A transient zone: its name is copied, so it may be built at run time.
    static constexpr char kFunction[] = "GpuZone";
    new (zone_) tracy::VkCtxScope(static_cast<tracy::VkCtx*>(profiler->Tracy()), __LINE__, __FILE__,
                                  std::strlen(__FILE__), kFunction, sizeof(kFunction) - 1, name,
                                  std::strlen(name), commands, true);
    tracy_ = true;
#endif
    if (profiler::GpuEnabled()) {
        timed_ = profiler->Begin(commands, name);
    }
}

GpuZone::~GpuZone() {
    if (timed_ != GpuProfiler::kUntimed) {
        context_.Profiler()->End(commands_, timed_);
    }
#if ERNIE_TRACY
    if (tracy_) {
        std::launder(reinterpret_cast<tracy::VkCtxScope*>(zone_))->~VkCtxScope();
    }
#endif
    context_.EndLabel(commands_);
}

}  // namespace ernie
