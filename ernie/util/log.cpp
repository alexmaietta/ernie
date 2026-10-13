// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/log.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

#ifndef ERNIE_VERSION
#define ERNIE_VERSION "unknown"
#endif
#ifndef ERNIE_BUILD_TYPE
#define ERNIE_BUILD_TYPE "unknown"
#endif

namespace ernie::log {

namespace {

/// -1 until first read from the environment.
std::atomic<int> level{-1};
std::mutex output;

bool Set(const char* variable) {
    const char* value = std::getenv(variable);
    return value != nullptr && *value != '\0' &&
           std::string_view(value) != "0";
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
        const Verbosity initial =
            Set("ERNIE_QUIET")     ? Verbosity::Quiet
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
    level.store(static_cast<int>(verbosity),
                std::memory_order_relaxed);
}

void Info(std::string_view message) {
    Print(Verbosity::Normal, message);
}

void Detail(std::string_view message) {
    Print(Verbosity::Verbose, message);
}

std::string_view Version() {
    return ERNIE_VERSION;
}

std::string_view BuildDate() {
    return __DATE__ " " __TIME__;
}

std::string_view BuildType() {
    // Empty when a single-configuration build sets none.
    constexpr std::string_view type = ERNIE_BUILD_TYPE;
    return type.empty() ? "unknown" : type;
}

std::string_view Compiler() {
#if defined(__clang__)
    return "Clang " __clang_version__;
#elif defined(__GNUC__)
    return "GCC " __VERSION__;
#elif defined(_MSC_VER)
    static const std::string compiler =
        "MSVC " + std::to_string(_MSC_FULL_VER);
    return compiler;
#else
    return "unknown";
#endif
}

}  // namespace ernie::log
