// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/version.h"

#include <format>

#include "build_info.h"

namespace ernie {

std::string Version::ToString() const {
    if (patch != 0) {
        return std::format("{}.{}.{}", major, minor, patch);
    }
    return std::format("{}.{}", major, minor);
}

const BuildInfo& GetBuildInfo() {
    static const BuildInfo info{{ERNIE_VERSION_MAJOR, ERNIE_VERSION_MINOR, ERNIE_VERSION_PATCH},
                                ERNIE_GIT_SHA,
                                ERNIE_GIT_DIRTY,
                                ERNIE_COMMIT_DATE,
                                ERNIE_BUILD_DATE};
    return info;
}

}
