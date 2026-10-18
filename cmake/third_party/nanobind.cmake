# SPDX-License-Identifier: BSD-3-Clause
set(ERNIE_DEP_VERSION_nanobind
    v3.1.0
    CACHE STRING
    "Tag of nanobind to fetch when it is not installed"
)
mark_as_advanced(ERNIE_DEP_VERSION_nanobind)

find_package(
    Python
    3.12
    REQUIRED
    COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT}
)

# The nanobind installed for the target Python, else a fetched copy.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
find_package(nanobind CONFIG QUIET)
if(NOT nanobind_FOUND)
    ernie_fetch(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG ${ERNIE_DEP_VERSION_nanobind}
        GIT_SHALLOW TRUE
        GIT_SUBMODULES ext/robin_map
    )
endif()
