// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <string_view>

/// What Ernie prints about itself on standard output, as Warp does
/// (`wp.config.quiet`, `wp.config.verbose`): a banner once it starts
/// on a GPU, with its version, device and cache folders, and a line
/// for each shader it compiles, more with verbosity (shaders and
/// textures its caches answer) and nothing when quiet.
///
/// The level starts from the environment: `ERNIE_QUIET` set (to
/// anything but "" or "0") is `Quiet`, else `ERNIE_VERBOSE` set is
/// `Verbose`, else `Normal`; `SetVerbosity` (Python's
/// `ernie.config.quiet` and `ernie.config.verbose`) changes it,
/// before the banner to silence that too.
///
/// Reference: NVIDIA, Warp `warp.config` (quiet, verbose) and its
/// kernel cache messages.
namespace ernie::log {

enum class Verbosity {
    Quiet,    ///< Nothing.
    Normal,   ///< The banner and slow work: shader compiles.
    Verbose,  ///< Also what caches answer and other detail.
};

Verbosity GetVerbosity();
void SetVerbosity(Verbosity verbosity);

/// Prints `message` and a newline unless quiet.
void Info(std::string_view message);
/// Prints `message` and a newline when verbose.
void Detail(std::string_view message);

/// Ernie's version, such as "0.4.0".
std::string_view Version();

/// When Ernie's core was compiled ("Oct 13 2026 09:12:45"), the build
/// configuration ("Release", "Debug", ...; "unknown" without one) and
/// the compiler ("GCC 14.2.0", "Clang 19.1.0"): the About window's
/// and bug reports'.
std::string_view BuildDate();
std::string_view BuildType();
std::string_view Compiler();

}  // namespace ernie::log
