#pragma once

#include <volk.h>

#include <cstdint>
#include <mutex>
#include <vector>

namespace ernie {

/// Times GPU zones (`ERNIE_GPU_ZONE`) for Ernie's profiler (ernie/util/profiler.h) with timestamp
/// queries: a zone writes one timestamp where it starts and one where it ends in its command
/// buffer, and `Collect` hands finished zones to the profiler without waiting for the device,
/// then frees their queries by host reset (Vulkan 1.2's hostQueryReset). Zones are dropped while
/// the profiler is not recording, or when the queries in flight run out.
///
/// References:
/// - Khronos, Vulkan 1.3 specification, section 18.5 "Timestamp Queries".
class GpuTimer {
  public:
    /// Timing on `device`'s queue family with `timestampBits` valid bits (0: none) and
    /// `periodNs` nanoseconds per tick.
    GpuTimer(VkDevice device, std::uint32_t timestampBits, float periodNs);
    ~GpuTimer();
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    /// Writes a zone's start into `commands`; returns its handle for `End`, or -1 if not timed.
    std::int64_t Begin(VkCommandBuffer commands, const char* name);
    void End(VkCommandBuffer commands, std::int64_t zone);

    /// Hands the zones the device has finished to the profiler.
    void Collect();

  private:
    struct Pending {
        const char* name;
        std::uint32_t query;  ///< The first of its two queries.
        std::uint64_t frame;
        std::uint32_t depth;
        bool ended = false;
    };

    static constexpr std::uint32_t kQueries = 4096;

    VkDevice device_;
    VkQueryPool pool_ = VK_NULL_HANDLE;
    std::uint64_t mask_ = 0;
    double periodNs_ = 1.0;
    std::mutex mutex_;
    std::vector<Pending> pending_;
    std::vector<std::uint32_t> free_;  ///< The first queries of the pairs free to hand out.
    std::uint32_t depth_ = 0;          ///< Zones open in the recording, for nesting.
};

}  // namespace ernie
