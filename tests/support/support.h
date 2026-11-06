// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/util/log.h"

namespace ernie::test {

class ScopedEnvironment {
  public:
    ScopedEnvironment(const char* name, std::optional<std::string> value) : m_name(name) {
        if (const char* previous = std::getenv(name)) {
            m_previous = previous;
        }
        Set(value);
    }

    ~ScopedEnvironment() {
        Set(m_previous);
    }

    ScopedEnvironment(const ScopedEnvironment&) = delete;
    ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;

  private:
    void Set(const std::optional<std::string>& value) const {
#ifdef _WIN32
        _putenv_s(m_name, value ? value->c_str() : "");
#else
        if (value) {
            setenv(m_name, value->c_str(), 1);
        } else {
            unsetenv(m_name);
        }
#endif
    }

    const char* m_name;
    std::optional<std::string> m_previous;
};

class TemporaryDirectory {
  public:
    TemporaryDirectory() {
        static std::atomic<int> counter{0};
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        m_path = std::filesystem::temp_directory_path() /
                 std::format("ernie-test-{:x}-{}", now, counter++);
        std::filesystem::create_directories(m_path);
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(m_path, error);
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

    const std::filesystem::path& Path() const {
        return m_path;
    }

  private:
    std::filesystem::path m_path;
};

class CapturedLog {
  public:
    explicit CapturedLog(LogLevel level = LogLevel::Debug) : m_level(GetLogLevel()) {
        SetLogLevel(level);
        SetLogSink([this](LogLevel at, std::string_view message) {
            m_messages.emplace_back(at, std::string(message));
        });
    }

    ~CapturedLog() {
        SetLogSink(nullptr);
        SetLogLevel(m_level);
    }

    CapturedLog(const CapturedLog&) = delete;
    CapturedLog& operator=(const CapturedLog&) = delete;

    const std::vector<std::pair<LogLevel, std::string>>& Messages() const {
        return m_messages;
    }

  private:
    LogLevel m_level;
    std::vector<std::pair<LogLevel, std::string>> m_messages;
};

inline void UseFreshProcessForDeathTests() {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
}

}
