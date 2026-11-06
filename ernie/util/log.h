// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <functional>
#include <string_view>
#include <utility>

#if __has_include(<version>)
#include <version>
#endif
#ifdef __cpp_lib_format
#include <format>
#endif

#include "ernie/export.h"

namespace ernie {

enum class LogLevel : int { Debug, Info, Warning, Error, Off };

using LogSink = std::function<void(LogLevel level, std::string_view message)>;

ERNIE_API LogLevel GetLogLevel();

ERNIE_API void SetLogLevel(LogLevel level);

ERNIE_API void SetLogSink(LogSink sink);

ERNIE_API void LogMessage(LogLevel level, std::string_view message);

#ifdef __cpp_lib_format
template <class... Args>
void Log(LogLevel level, std::format_string<Args...> format, Args&&... args) {
    if (level >= GetLogLevel()) {
        LogMessage(level, std::format(format, std::forward<Args>(args)...));
    }
}
#endif

}
