# SPDX-License-Identifier: BSD-3-Clause
set(ERNIE_DEP_VERSION_googletest v1.18.0 CACHE STRING "Tag of GoogleTest to fetch")
mark_as_advanced(ERNIE_DEP_VERSION_googletest)

set(BUILD_GMOCK OFF)
set(INSTALL_GTEST OFF)
set(gtest_force_shared_crt ON)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG ${ERNIE_DEP_VERSION_googletest}
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(googletest)
