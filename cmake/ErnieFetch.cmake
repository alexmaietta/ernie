# SPDX-License-Identifier: BSD-3-Clause
include_guard(GLOBAL)
include(FetchContent)

set(ERNIE_DEPS_DIR
    "${PROJECT_SOURCE_DIR}/.deps"
    CACHE PATH
    "Where fetched dependency sources are kept"
)

function(ernie_fetch name)
    # The FetchContent_Declare arguments recipes use, declared so others are rejected and gersemi
    # formats calls by keyword.
    cmake_parse_arguments(
        ARG
        ""
        "URL;URL_HASH;GIT_REPOSITORY;GIT_TAG;GIT_SHALLOW;SOURCE_SUBDIR"
        "GIT_SUBMODULES"
        ${ARGN}
    )
    if(ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "ernie_fetch(${name}): unexpected ${ARG_UNPARSED_ARGUMENTS}")
    endif()
    string(TOLOWER "${name}" lower)
    string(TOUPPER "${name}" upper)
    set(source "${ERNIE_DEPS_DIR}/${lower}-src")
    set(stamp "${source}/.ernie-fetched")
    if(EXISTS "${stamp}")
        file(READ "${stamp}" fetched)
        if(fetched STREQUAL "${ARGN}")
            set(FETCHCONTENT_SOURCE_DIR_${upper} "${source}")
        else()
            file(REMOVE_RECURSE "${source}")
        endif()
    endif()
    FetchContent_Declare(${name} ${ARGN} SOURCE_DIR "${source}")
    FetchContent_MakeAvailable(${name})
    file(WRITE "${stamp}" "${ARGN}")
    set(${lower}_SOURCE_DIR "${${lower}_SOURCE_DIR}" PARENT_SCOPE)
    set(${lower}_BINARY_DIR "${${lower}_BINARY_DIR}" PARENT_SCOPE)
endfunction()

# Builds third-party targets without warnings: their code is not Ernie's to fix.
function(ernie_quiet_third_party)
    foreach(target ${ARGN})
        get_target_property(type ${target} TYPE)
        if(NOT type STREQUAL "INTERFACE_LIBRARY")
            target_compile_options(${target} PRIVATE "$<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>")
        endif()
    endforeach()
endfunction()
