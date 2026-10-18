// SPDX-License-Identifier: BSD-3-Clause
/// @file message_log.cpp
/// @brief Keeping the newest messages.
#include "ernie/app/message_log.h"

namespace ernie::app {

void MessageLog::Add(Severity severity, std::string text) {
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start_;
    messages_.push_back({elapsed.count(), severity, std::move(text)});
    while (messages_.size() > limit_) {
        messages_.pop_front();
    }
}

}  // namespace ernie::app
