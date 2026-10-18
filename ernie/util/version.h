// SPDX-License-Identifier: BSD-3-Clause
/// @file version.h
/// @brief Ernie's version and how this copy was built.
#pragma once

#include "ernie/export.h"

#include <string_view>

namespace ernie {

/// @brief Returns Ernie's version.
/// @return The version this library was built as, such as "0.2.0" or "0.2.0.dev12".
ERNIE_API std::string_view Version();

/// @brief Returns when the library was compiled.
/// @return The compile date and time, such as "Oct 13 2026 09:12:45".
ERNIE_API std::string_view BuildDate();

/// @brief Returns the build configuration.
/// @return "Release", "RelWithDebInfo" and so on, or "unknown".
ERNIE_API std::string_view BuildType();

/// @brief Returns the compiler that built the library.
/// @return Its name and version, such as "GCC 14.2.0".
ERNIE_API std::string_view Compiler();

}  // namespace ernie
