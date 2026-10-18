// SPDX-License-Identifier: BSD-3-Clause
/// @file message_log.h
/// @brief The viewer's messages, as its Log panel lists them.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <utility>

namespace ernie::app {

enum class Severity : std::uint8_t { Info, Warning, Error };

struct Message {
    /// Seconds since the log was made.
    double time = 0.0;
    Severity severity = Severity::Info;
    std::string text;
};

/// @brief The newest messages, oldest first, up to a limit.
class MessageLog {
  public:
    explicit MessageLog(std::size_t limit = 10'000) : limit_(limit) {}

    void Add(Severity severity, std::string text);
    void Info(std::string text) {
        Add(Severity::Info, std::move(text));
    }
    void Warning(std::string text) {
        Add(Severity::Warning, std::move(text));
    }
    void Error(std::string text) {
        Add(Severity::Error, std::move(text));
    }
    void Clear() {
        messages_.clear();
    }
    const std::deque<Message>& Messages() const {
        return messages_;
    }

  private:
    std::size_t limit_;
    std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
    std::deque<Message> messages_;
};

}  // namespace ernie::app
