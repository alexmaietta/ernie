// SPDX-License-Identifier: BSD-3-Clause
/// @file version.cpp
/// @brief Ernie's version and build information, set by the build.
#include "ernie/util/version.h"

#include <string>

namespace ernie {

std::string_view Version() {
    return ERNIE_VERSION_STRING;
}

std::string_view BuildDate() {
    return __DATE__ " " __TIME__;
}

std::string_view BuildType() {
    // Empty when a single-configuration build sets none.
    constexpr std::string_view type = ERNIE_BUILD_TYPE;
    return type.empty() ? "unknown" : type;
}

std::string_view Compiler() {
#if defined(__clang__)
    return "Clang " __clang_version__;
#elif defined(__GNUC__)
    return "GCC " __VERSION__;
#elif defined(_MSC_VER)
    static const std::string compiler = "MSVC " + std::to_string(_MSC_FULL_VER);
    return compiler;
#else
    return "unknown";
#endif
}

}  // namespace ernie
