// SPDX-License-Identifier: BSD-3-Clause
/// @file log.cpp
/// @brief Verbosity and console messages.
#include "ernie/core/log.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace ernie::log {

namespace {

/// -1 until first read from the environment.
std::atomic<int> level{-1};
std::mutex output;

bool Set(const char* variable) {
    const char* value = std::getenv(variable);
    return value != nullptr && *value != '\0' && std::string_view(value) != "0";
}

void Print(Verbosity at, std::string_view message) {
    if (GetVerbosity() < at) {
        return;
    }
    // Whole lines, from any thread.
    const std::lock_guard lock(output);
    std::fwrite(message.data(), 1, message.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

}  // namespace

Verbosity GetVerbosity() {
    int current = level.load(std::memory_order_relaxed);
    if (current < 0) {
        const Verbosity initial = Set("ERNIE_QUIET")     ? Verbosity::Quiet
                                  : Set("ERNIE_VERBOSE") ? Verbosity::Verbose
                                                         : Verbosity::Normal;
        current = static_cast<int>(initial);
        int unset = -1;
        // Unless set meanwhile.
        if (!level.compare_exchange_strong(unset, current)) {
            current = unset;
        }
    }
    return static_cast<Verbosity>(current);
}

void SetVerbosity(Verbosity verbosity) {
    level.store(static_cast<int>(verbosity), std::memory_order_relaxed);
}

void Info(std::string_view message) {
    Print(Verbosity::Normal, message);
}

void Detail(std::string_view message) {
    Print(Verbosity::Verbose, message);
}

}  // namespace ernie::log
