// SPDX-License-Identifier: BSD-3-Clause
/// @file version.cpp
/// @brief Ernie's version, set by the build.
#include "ernie/core/version.h"

namespace ernie {

std::string_view Version() {
    return ERNIE_VERSION_STRING;
}

}  // namespace ernie
