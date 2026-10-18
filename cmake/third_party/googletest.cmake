# SPDX-License-Identifier: BSD-3-Clause
set(ERNIE_DEP_VERSION_googletest v1.18.0 CACHE STRING "Tag of GoogleTest to fetch")
mark_as_advanced(ERNIE_DEP_VERSION_googletest)

set(BUILD_GMOCK OFF)
set(INSTALL_GTEST OFF)
set(gtest_force_shared_crt ON)

ernie_fetch(
    googletest
    URL
        https://github.com/google/googletest/archive/refs/tags/${ERNIE_DEP_VERSION_googletest}.tar.gz
)
