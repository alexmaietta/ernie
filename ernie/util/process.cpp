// SPDX-License-Identifier: BSD-3-Clause
/// @file process.cpp
/// @brief Resident memory from /proc on Linux, the task's info on macOS and the process's memory
/// counters on Windows.
#include "ernie/util/process.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// After windows.h, which it needs.
#include <psapi.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>

#include <fstream>
#endif

namespace ernie {

std::optional<std::uint64_t> ReadResidentBytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        return counters.WorkingSetSize;
    }
#elif defined(__APPLE__)
    mach_task_basic_info info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info),
                  &count) == KERN_SUCCESS) {
        return info.resident_size;
    }
#elif defined(__linux__)
    // Total and resident pages.
    std::ifstream statm("/proc/self/statm");
    std::uint64_t total = 0;
    std::uint64_t resident = 0;
    if (statm >> total >> resident) {
        return resident * static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
    }
#endif
    return std::nullopt;
}

}  // namespace ernie
