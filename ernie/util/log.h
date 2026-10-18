// SPDX-License-Identifier: BSD-3-Clause
/// @file log.h
/// @brief What Ernie prints about itself on standard output.
/// A banner on GPU start-up and a line per shader compile; more when
/// verbose, nothing when quiet. `ERNIE_QUIET` or `ERNIE_VERBOSE` (set
/// to anything but "" or "0") picks the starting level.
///
/// References:
/// - NVIDIA, Warp `warp.config` (quiet, verbose) and its kernel cache
///   messages.
#pragma once

#include "ernie/export.h"

#include <string_view>

namespace ernie::log {

/// @brief How much Ernie prints.
enum class Verbosity {
    /// Nothing.
    Quiet,
    /// The banner and slow work: shader compiles.
    Normal,
    /// Also what caches answer and other detail.
    Verbose,
};

/// @return The current verbosity.
ERNIE_API Verbosity GetVerbosity();
/// @brief Sets the verbosity; before the banner, it applies to it.
/// @param verbosity The new level.
ERNIE_API void SetVerbosity(Verbosity verbosity);

/// @brief Prints `message` and a newline unless quiet.
/// @param message The text.
ERNIE_API void Info(std::string_view message);
/// @brief Prints `message` and a newline when verbose.
/// @param message The text.
ERNIE_API void Detail(std::string_view message);

}  // namespace ernie::log
