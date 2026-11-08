// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <string>
#include <string_view>
#include <tuple>

#include "ernie/export.h"

namespace ernie {

struct Version {
    int major = 0;
    int minor = 0;
    int patch = 0;

    friend constexpr bool operator==(const Version& a, const Version& b) {
        return std::tie(a.major, a.minor, a.patch) == std::tie(b.major, b.minor, b.patch);
    }

    friend constexpr bool operator!=(const Version& a, const Version& b) {
        return !(a == b);
    }

    friend constexpr bool operator<(const Version& a, const Version& b) {
        return std::tie(a.major, a.minor, a.patch) < std::tie(b.major, b.minor, b.patch);
    }

    friend constexpr bool operator>(const Version& a, const Version& b) {
        return b < a;
    }

    friend constexpr bool operator<=(const Version& a, const Version& b) {
        return !(b < a);
    }

    friend constexpr bool operator>=(const Version& a, const Version& b) {
        return !(a < b);
    }

    ERNIE_API std::string ToString() const;
};

struct ERNIE_API BuildInfo {
    Version version;
    std::string_view commit;
    bool modified = false;
    std::string_view commitDate;
    std::string_view buildDate;
    std::string_view fullVersion;
};

ERNIE_API const BuildInfo& GetBuildInfo();

}
