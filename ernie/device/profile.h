// SPDX-License-Identifier: BSD-3-Clause
/// @file profile.h
/// @brief GPU profiling zones: a debug label and timestamps around commands. While
/// `profiler::GpuEnabled()`, the timestamps reach Ernie's profiler (`profiler::Frames()`), in the
/// frame that recorded them; with -DERNIE_TRACY=ON they show in Tracy too. No timestamps are
/// recorded on devices without `DeviceCapabilities::calibratedTimestamps`.
///
///     // The rest of the scope's commands; the name may be built at run time.
///     ERNIE_GPU_ZONE(context, commands, pass.name.c_str());
///
/// References:
/// - B. Taudul, "Tracy Profiler", its Vulkan GPU zones (TracyVulkan.hpp), host calibrated.
///   https://github.com/wolfpld/tracy
#pragma once

#include "ernie/device/vulkan.h"
#include "ernie/util/profiler.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

namespace ernie {

class DeviceContext;

/// @brief A device's GPU timeline: timestamp queries reset from the host and clocks read with the
/// host's, collected as the context collects retired work.
class GpuProfiler {
  public:
    /// A zone that records no timestamps.
    static constexpr std::uint32_t kUntimed = UINT32_MAX;

    /// @return The profiler, or null on devices that cannot calibrate their clock.
    static std::unique_ptr<GpuProfiler> Create(const DeviceContext& context);
    ~GpuProfiler();
    GpuProfiler(const GpuProfiler&) = delete;
    GpuProfiler& operator=(const GpuProfiler&) = delete;

    /// @brief Reads the timestamps of finished zones into Ernie's profiler and Tracy, unless
    /// another thread is reading them.
    void Collect();
    /// @return Tracy's Vulkan context; null in builds without Tracy.
    void* Tracy() const {
        return tracy_;
    }

    /// @brief Writes a zone's start timestamp for Ernie's profiler.
    /// @param commands The commands the zone is in.
    /// @param name     The zone's name; copied.
    /// @return The zone, or `kUntimed` when every query is in use.
    std::uint32_t Begin(VkCommandBuffer commands, const char* name);
    /// @brief Writes the end timestamp of a zone `Begin` returned.
    /// @param commands The commands the zone is in.
    /// @param zone     The zone.
    void End(VkCommandBuffer commands, std::uint32_t zone);

  private:
    class Timer;
    GpuProfiler(void* tracy, std::unique_ptr<Timer> timer);
    void* tracy_;
    std::unique_ptr<Timer> timer_;
    std::mutex collecting_;
};

/// @brief A debug label and timestamps around the rest of the scope's commands.
class GpuZone {
  public:
    GpuZone(const DeviceContext& context, VkCommandBuffer commands, const char* name);
    ~GpuZone();
    GpuZone(const GpuZone&) = delete;
    GpuZone& operator=(const GpuZone&) = delete;

  private:
    const DeviceContext& context_;
    VkCommandBuffer commands_;
    /// Tracy's zone, built in place when recording; unused in builds without Tracy.
    [[maybe_unused]] alignas(std::max_align_t) std::byte zone_[32];
    [[maybe_unused]] bool tracy_ = false;
    std::uint32_t timed_ = GpuProfiler::kUntimed;
};

}  // namespace ernie

#define ERNIE_GPU_ZONE(context, commands, name)                                                    \
    const ::ernie::GpuZone ERNIE_PROFILE_CONCAT(ernieGpuZone, __LINE__)((context), (commands),     \
                                                                        (name))
