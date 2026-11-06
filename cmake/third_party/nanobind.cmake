# SPDX-License-Identifier: BSD-3-Clause
find_package(
    Python
    3.10
    REQUIRED
    COMPONENTS
        Interpreter
        Development.Module
        ${SKBUILD_SABI_COMPONENT}
)
execute_process(
    COMMAND
        "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
find_package(nanobind CONFIG QUIET)
if(NOT nanobind_FOUND)
    message(
        FATAL_ERROR
        "The Python bindings need nanobind: ${Python_EXECUTABLE} -m pip install nanobind"
    )
endif()
