// SPDX-License-Identifier: BSD-3-Clause
#include "metrics.h"

#include "ernie/device/vulkan_context.h"
#include "ernie/session/renderer.h"
#include "ernie/util/profile_report.h"
#include "ernie/util/profiler.h"

#include <volk.h>

#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <thread>

#ifndef ERNIE_BENCH_COMMIT
#define ERNIE_BENCH_COMMIT "unknown"
#endif
#ifndef ERNIE_BENCH_BUILD_TYPE
#define ERNIE_BENCH_BUILD_TYPE "unknown"
#endif

namespace ernie::bench {

namespace {

constexpr double kMib = 1024.0 * 1024.0;

bool SupportsMemoryBudget(VkPhysicalDevice device) {
    std::uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count,
                                         nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count,
                                         extensions.data());
    return std::any_of(
        extensions.begin(), extensions.end(),
        [](const VkExtensionProperties& extension) {
            return std::strcmp(extension.extensionName,
                               VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) ==
                   0;
        });
}

std::string CpuModel() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.starts_with("model name")) {
            const std::size_t colon = line.find(':');
            return colon == std::string::npos
                       ? line
                       : line.substr(colon + 2);
        }
    }
    return "unknown";
}

}  // namespace

ProcessSnapshot ProcessNow() {
    ProcessSnapshot snapshot;
    timespec cpu{};
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu) == 0) {
        snapshot.cpuSeconds = static_cast<double>(cpu.tv_sec) +
                              static_cast<double>(cpu.tv_nsec) * 1e-9;
    }
    // statm: total and resident pages.
    std::ifstream statm("/proc/self/statm");
    long total = 0;
    long resident = 0;
    if (statm >> total >> resident) {
        snapshot.rssMib = static_cast<double>(resident) *
                          static_cast<double>(sysconf(_SC_PAGESIZE)) /
                          kMib;
    }
    return snapshot;
}

std::optional<DeviceMemory>
DeviceMemoryNow(const VulkanContext& context) {
    static const bool supported =
        SupportsMemoryBudget(context.PhysicalDevice());
    if (!supported) {
        return std::nullopt;
    }
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{
        .sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 properties{
        .sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
        .pNext = &budget};
    vkGetPhysicalDeviceMemoryProperties2(context.PhysicalDevice(),
                                         &properties);
    DeviceMemory memory;
    const VkPhysicalDeviceMemoryProperties& heaps =
        properties.memoryProperties;
    for (std::uint32_t h = 0; h < heaps.memoryHeapCount; ++h) {
        if ((heaps.memoryHeaps[h].flags &
             VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) {
            continue;
        }
        memory.usedMib +=
            static_cast<double>(budget.heapUsage[h]) / kMib;
        memory.budgetMib +=
            static_cast<double>(budget.heapBudget[h]) / kMib;
        memory.heapMib +=
            static_cast<double>(heaps.memoryHeaps[h].size) / kMib;
    }
    return memory;
}

void StartGpuTimes(std::size_t frames) {
    profiler::SetHistory(std::max(frames, profiler::History()));
    profiler::SetCpuEnabled(false);
    profiler::SetGpuEnabled(true);
}

void StopGpuTimes() {
    profiler::SetGpuEnabled(false);
    profiler::Clear();
}

std::uint64_t CurrentFrame() {
    return profiler::CurrentFrame();
}

void NextFrame() {
    if (profiler::GpuEnabled()) {
        profiler::BeginFrame();
    }
}

GpuTimes GpuTimesOf(std::uint64_t first, std::uint64_t last) {
    std::vector<profiler::Frame> frames;
    for (profiler::Frame& frame : profiler::Frames()) {
        if (frame.index >= first && frame.index <= last &&
            !frame.gpu.empty()) {
            frames.push_back(std::move(frame));
        }
    }
    GpuTimes times;
    if (frames.empty()) {
        return times;
    }
    const profiler::Report report =
        profiler::Summarize(frames, profiler::ThreadNames());
    times.frameMs = report.gpuMs;
    times.frames = report.gpuFrames;
    // The outermost zones and the passes within them.
    for (const profiler::ReportNode& zone : report.gpu) {
        times.passes.emplace_back(zone.name, zone.ms);
        for (const profiler::ReportNode& pass : zone.children) {
            times.passes.emplace_back(zone.name + "/" + pass.name,
                                      pass.ms);
        }
    }
    return times;
}

nlohmann::json
Environment(Renderer& renderer,
            const std::vector<std::string>& commandLine) {
    utsname name{};
    uname(&name);
    char host[256] = {};
    gethostname(host, sizeof(host) - 1);
    const VulkanContext& context = renderer.Context();
    std::string uuid;
    for (const std::uint8_t byte : context.DeviceUuid()) {
        char hex[3];
        std::snprintf(hex, sizeof(hex), "%02x", byte);
        uuid += hex;
    }
    nlohmann::json gpu = {
        {"name", renderer.DeviceName()},
        {"driver", context.DriverDescription()},
        {"vulkan", context.VulkanVersion()},
        {"uuid", uuid},
        {"compute_interop", renderer.Delegate().Compute() != nullptr},
    };
    if (const std::optional<DeviceMemory> memory =
            DeviceMemoryNow(context)) {
        gpu["device_local_mib"] = memory->heapMib;
    }
    return {
        {"host", host},
        {"operating_system",
         {{"system", name.sysname},
          {"release", name.release},
          {"version", name.version}}},
        {"hardware",
         {{"machine", name.machine},
          {"cpu", CpuModel()},
          {"cpu_count", std::thread::hardware_concurrency()},
          {"memory_mib",
           static_cast<double>(sysconf(_SC_PHYS_PAGES)) *
               static_cast<double>(sysconf(_SC_PAGESIZE)) / kMib}}},
        {"gpu", gpu},
        {"build",
         {{"commit", ERNIE_BENCH_COMMIT},
          {"build_type", ERNIE_BENCH_BUILD_TYPE},
          {"compiler", __VERSION__}}},
        {"command_line", commandLine},
    };
}

}  // namespace ernie::bench
