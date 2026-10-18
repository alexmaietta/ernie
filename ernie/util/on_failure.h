// SPDX-License-Identifier: BSD-3-Clause
/// @file on_failure.h
/// @brief Undoing a step when what follows it throws, so constructors that make several handles
/// leak none when one fails.
#pragma once

#include <exception>
#include <utility>

namespace ernie {

/// @brief Runs `undo` when destroyed by an exception's unwinding, and not otherwise.
template <typename Undo> class OnFailure {
  public:
    explicit OnFailure(Undo undo)
        : undo_(std::move(undo)), exceptions_(std::uncaught_exceptions()) {}
    ~OnFailure() {
        if (std::uncaught_exceptions() > exceptions_) {
            undo_();
        }
    }
    OnFailure(const OnFailure&) = delete;
    OnFailure& operator=(const OnFailure&) = delete;

  private:
    Undo undo_;
    int exceptions_;
};

}  // namespace ernie
