// SPDX-License-Identifier: BSD-3-Clause
/// @file profile.h
/// @brief GPU profiling zones: timestamps around commands, shown in Tracy beside the CPU zones
/// with -DERNIE_TRACY=ON. Nothing is recorded in other builds, or on devices without
/// `DeviceCapabilities::calibratedTimestamps`.
///
///     // The rest of the scope's commands; the name may be built at run time.
///     ERNIE_GPU_ZONE(context, commands, pass.name);
///
/// References:
/// - B. Taudul, "Tracy Profiler", its Vulkan GPU zones (TracyVulkan.hpp), host calibrated.
///   https://github.com/wolfpld/tracy
#pragma once

#include "ernie/device/vulkan.h"
#include "ernie/util/profiler.h"

#include <memory>
#include <string_view>

namespace ernie {

class DeviceContext;

/// @brief A device's GPU timeline for Tracy: timestamp queries reset from the host and clocks
/// read with the host's, collected as the context collects retired work.
class GpuProfiler {
  public:
    /// @return The profiler, or null in builds without Tracy and on devices that cannot
    /// calibrate their clock.
    static std::unique_ptr<GpuProfiler> Create(const DeviceContext& context);
    ~GpuProfiler();
    GpuProfiler(const GpuProfiler&) = delete;
    GpuProfiler& operator=(const GpuProfiler&) = delete;

    /// @brief Reads the timestamps of finished zones.
    void Collect();
    /// @return Tracy's Vulkan context.
    void* Tracy() const {
        return tracy_;
    }

  private:
    explicit GpuProfiler(void* tracy) : tracy_(tracy) {}
    void* tracy_;
};

/// @brief Timestamps around the rest of the scope's commands, as one GPU zone.
class GpuZone {
  public:
    GpuZone(const DeviceContext& context, VkCommandBuffer commands, std::string_view name);
    ~GpuZone();
    GpuZone(const GpuZone&) = delete;
    GpuZone& operator=(const GpuZone&) = delete;

  private:
    /// Tracy's zone, when recording.
    std::unique_ptr<void, void (*)(void*)> zone_;
};

}  // namespace ernie

#define ERNIE_GPU_ZONE(context, commands, name)                                                    \
    const ::ernie::GpuZone ERNIE_PROFILE_CONCAT(ernieGpuZone, __LINE__)((context), (commands),     \
                                                                        (name))
