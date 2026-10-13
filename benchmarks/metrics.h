// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// What samples record besides time, vendor-neutral: the process's CPU
// time and memory from the OS, device memory from Vulkan's memory
// budget, and GPU time from Ernie's timestamp-query profiler.
//
// References:
// - Khronos, VK_EXT_memory_budget,
//   https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_memory_budget.html

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ernie {
class Renderer;
class VulkanContext;
}  // namespace ernie

namespace ernie::bench {

/// The process's CPU time and resident memory.
struct ProcessSnapshot {
    double cpuSeconds = 0.0;
    double rssMib = 0.0;
};

ProcessSnapshot ProcessNow();

/// The device-local heaps' use by this process and the budget the
/// driver allows it, MiB; none without VK_EXT_memory_budget.
struct DeviceMemory {
    double usedMib = 0.0;
    double budgetMib = 0.0;
    double heapMib = 0.0;  ///< The heaps' total size.
};

std::optional<DeviceMemory>
DeviceMemoryNow(const VulkanContext& context);

/// The GPU time of a span of the profiler's frames: the mean of their
/// outermost zones, and each outermost zone's mean and those of the
/// passes within it ("outer/pass"), over the frames whose zones
/// arrived.
struct GpuTimes {
    double frameMs = 0.0;
    std::size_t frames = 0;
    std::vector<std::pair<std::string, double>> passes;
};

/// Records GPU zones (and only those) from now on, keeping at least
/// `frames` frames.
void StartGpuTimes(std::size_t frames);
void StopGpuTimes();

/// The profiler's newest frame index, to mark where a sample starts.
std::uint64_t CurrentFrame();

/// Starts the profiler's next frame, before a step, while GPU times
/// are recorded (a headless renderer starts none itself).
void NextFrame();

/// The GPU times of frames `first` to `last` inclusive, of those the
/// profiler still holds.
GpuTimes GpuTimesOf(std::uint64_t first, std::uint64_t last);

/// The machine, OS, GPU and build the results came from.
nlohmann::json
Environment(Renderer& renderer,
            const std::vector<std::string>& commandLine);

}  // namespace ernie::bench
