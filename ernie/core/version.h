// SPDX-License-Identifier: BSD-3-Clause
/// @file version.h
/// @brief Ernie's version.
#pragma once

#include <string_view>

namespace ernie {

/// @brief Returns Ernie's version, such as "0.2.0" or "0.2.0.dev12".
/// @return The version this library was built as.
std::string_view Version();

}  // namespace ernie
