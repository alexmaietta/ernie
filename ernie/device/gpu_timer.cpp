// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/gpu_timer.h"

#include "ernie/util/profiler.h"

#include <algorithm>
#include <array>

namespace ernie {

GpuTimer::GpuTimer(VkDevice device, std::uint32_t timestampBits,
                   float periodNs)
    : device_(device), periodNs_(periodNs) {
    if (timestampBits == 0) {
        return;
    }
    mask_ = timestampBits >= 64 ? ~0ull : (1ull << timestampBits) - 1;
    VkQueryPoolCreateInfo info{
        VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = kQueries;
    if (vkCreateQueryPool(device_, &info, nullptr, &pool_) !=
        VK_SUCCESS) {
        pool_ = VK_NULL_HANDLE;
        return;
    }
    vkResetQueryPool(device_, pool_, 0, kQueries);
    for (std::uint32_t query = kQueries; query >= 2; query -= 2) {
        free_.push_back(query - 2);
    }
}

GpuTimer::~GpuTimer() {
    if (pool_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device_, pool_, nullptr);
    }
}

std::int64_t GpuTimer::Begin(VkCommandBuffer commands,
                             const char* name) {
    if (pool_ == VK_NULL_HANDLE || !profiler::GpuEnabled()) {
        return -1;
    }
    std::lock_guard lock(mutex_);
    if (free_.empty()) {
        return -1;
    }
    const std::uint32_t query = free_.back();
    free_.pop_back();
    vkCmdWriteTimestamp2(
        commands, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, pool_, query);
    pending_.push_back(
        {name, query, profiler::CurrentFrame(), depth_++});
    return query;
}

void GpuTimer::End(VkCommandBuffer commands, std::int64_t zone) {
    if (zone < 0) {
        return;
    }
    const auto query = static_cast<std::uint32_t>(zone);
    vkCmdWriteTimestamp2(commands,
                         VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                         pool_, query + 1);
    std::lock_guard lock(mutex_);
    depth_ = depth_ > 0 ? depth_ - 1 : 0;
    for (Pending& pending : pending_) {
        if (pending.query == query) {
            pending.ended = true;
        }
    }
}

void GpuTimer::Collect() {
    if (pool_ == VK_NULL_HANDLE) {
        return;
    }
    std::lock_guard lock(mutex_);
    std::vector<Pending> waiting;
    for (const Pending& pending : pending_) {
        std::array<std::uint64_t, 4>
            values{};  // two (timestamp, availability) pairs
        const VkResult result =
            pending.ended
                ? vkGetQueryPoolResults(
                      device_, pool_, pending.query, 2,
                      sizeof(values), values.data(),
                      2 * sizeof(std::uint64_t),
                      VK_QUERY_RESULT_64_BIT |
                          VK_QUERY_RESULT_WITH_AVAILABILITY_BIT)
                : VK_NOT_READY;
        if (result != VK_SUCCESS || values[1] == 0 ||
            values[3] == 0) {
            waiting.push_back(pending);
            continue;
        }
        const std::uint64_t start = values[0] & mask_;
        const std::uint64_t end = std::max(values[2] & mask_, start);
        profiler::Zone zone;
        zone.name = pending.name;
        zone.start = static_cast<std::uint64_t>(
            static_cast<double>(start) * periodNs_);
        zone.end = static_cast<std::uint64_t>(
            static_cast<double>(end) * periodNs_);
        zone.depth = pending.depth;
        profiler::AddGpuZone(pending.frame, zone);
        retiring_.push_back({pending.query, 0});
    }
    pending_ = std::move(waiting);
    // Reset and handed out again only some collects later, once the
    // fence of the submission that wrote them has surely been waited
    // on: a host reset must follow the submission's completion, which
    // its results being available alone does not show (the
    // validation layer would see the next write as unreset).
    std::vector<Retiring> kept;
    for (Retiring& retiring : retiring_) {
        if (++retiring.collects <= kRetireCollects) {
            kept.push_back(retiring);
            continue;
        }
        vkResetQueryPool(device_, pool_, retiring.query, 2);
        free_.push_back(retiring.query);
    }
    retiring_ = std::move(kept);
}

}  // namespace ernie
