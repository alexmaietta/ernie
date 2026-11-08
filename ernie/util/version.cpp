// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/version.h"

#include <fmt/format.h>

#include "build_info.h"

namespace ernie {

std::string Version::ToString() const {
    if (patch != 0) {
        return fmt::format("{}.{}.{}", major, minor, patch);
    }
    return fmt::format("{}.{}", major, minor);
}

const BuildInfo& GetBuildInfo() {
    static const BuildInfo info{{ERNIE_VERSION_MAJOR, ERNIE_VERSION_MINOR, ERNIE_VERSION_PATCH},
                                ERNIE_GIT_SHA,
                                ERNIE_GIT_DIRTY,
                                ERNIE_COMMIT_DATE,
                                ERNIE_BUILD_DATE,
                                ERNIE_VERSION_FULL};
    return info;
}

}
