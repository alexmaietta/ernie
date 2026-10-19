// SPDX-License-Identifier: BSD-3-Clause
/// @file process.cpp
/// @brief Resident memory from /proc on Linux, the task's info on macOS and the process's memory
/// counters on Windows; programs found on the PATH and run without a shell.
#include "ernie/util/process.h"

#include <cstdlib>
#include <system_error>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// After windows.h, which they need.
#include <process.h>
#include <psapi.h>
#else
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;
#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <fstream>
#endif
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

std::filesystem::path FindProgram(std::string_view name) {
    const char* path = std::getenv("PATH");
    if (path == nullptr) {
        return {};
    }
#if defined(_WIN32)
    constexpr char kSeparator = ';';
    const std::string file = std::string(name) + ".exe";
#else
    constexpr char kSeparator = ':';
    const std::string file(name);
#endif
    const std::string_view directories(path);
    for (std::size_t begin = 0; begin <= directories.size();) {
        std::size_t end = directories.find(kSeparator, begin);
        if (end == std::string_view::npos) {
            end = directories.size();
        }
        if (end > begin) {
            const std::filesystem::path candidate =
                std::filesystem::path(directories.substr(begin, end - begin)) / file;
            std::error_code error;
            if (std::filesystem::is_regular_file(candidate, error)) {
                return candidate;
            }
        }
        begin = end + 1;
    }
    return {};
}

int RunProcess(const std::vector<std::string>& command) {
    if (command.empty()) {
        return -1;
    }
#if defined(_WIN32)
    // The C runtime joins the arguments with spaces: each is quoted.
    std::vector<std::string> quoted;
    for (const std::string& argument : command) {
        std::string q = "\"";
        for (const char c : argument) {
            if (c == '"') {
                q += '\\';
            }
            q += c;
        }
        quoted.push_back(q + "\"");
    }
    std::vector<const char*> argv;
    for (const std::string& argument : quoted) {
        argv.push_back(argument.c_str());
    }
    argv.push_back(nullptr);
    const intptr_t status = _spawnv(_P_WAIT, command.front().c_str(), argv.data());
    return status < 0 ? -1 : static_cast<int>(status);
#else
    std::vector<char*> argv;
    for (const std::string& argument : command) {
        argv.push_back(const_cast<char*>(argument.c_str()));
    }
    argv.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    pid_t pid = 0;
    const int spawned = posix_spawn(&pid, argv.front(), &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    if (spawned != 0) {
        return -1;
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return -1;
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

}  // namespace ernie
