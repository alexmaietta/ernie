// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <utility>

#ifdef ERNIE_FMT
#include <fmt/format.h>
#endif

#include "ernie/export.h"

namespace ernie {

enum class LogLevel : int { Debug, Info, Warning, Error, Off };

using LogSink = std::function<void(LogLevel level, std::string_view message)>;

ERNIE_API LogLevel GetLogLevel();

ERNIE_API void SetLogLevel(LogLevel level);

ERNIE_API void SetLogSink(LogSink sink);

ERNIE_API void LogMessage(LogLevel level, std::string_view message);

#ifdef ERNIE_FMT
template <class... Args>
void Log(LogLevel level, fmt::format_string<Args...> format, Args&&... args) {
    if (level >= GetLogLevel()) {
        LogMessage(level, fmt::format(format, std::forward<Args>(args)...));
    }
}

inline std::string Counted(uint64_t count, std::string_view noun) {
    return fmt::format("{} {}{}", count, noun, count == 1 ? "" : "s");
}
#endif

}
