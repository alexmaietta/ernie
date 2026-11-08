# SPDX-License-Identifier: BSD-3-Clause
function(git result)
    execute_process(
        COMMAND
            git -C "${SOURCE}" ${ARGN}
        OUTPUT_VARIABLE output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE failed
        ERROR_QUIET
    )
    if(failed)
        set(output "")
    endif()
    set(${result} "${output}" PARENT_SCOPE)
endfunction()

git(sha
    rev-parse
    --short=12
    HEAD
)
git(commit_date
    log
    -1
    --format=%cs
)
git(changes
    status
    --porcelain
    --untracked-files=no
)
if(sha STREQUAL "")
    set(sha "unknown")
endif()
if(changes STREQUAL "")
    set(dirty false)
else()
    set(dirty true)
endif()
string(
    REPLACE "."
    ";"
    parts
    "${VERSION}"
)
list(GET parts 0 major)
list(GET parts 1 minor)
list(GET parts 2 patch)
set(key "// ${VERSION}${PRERELEASE} ${sha} ${dirty}")
if(EXISTS "${OUTPUT}")
    file(READ "${OUTPUT}" old)
    string(
        FIND "${old}"
        "${key}\n"
        found
    )
    if(found EQUAL 0)
        return()
    endif()
endif()
string(TIMESTAMP built "%Y-%m-%dT%H:%M:%SZ" UTC)
file(
    WRITE "${OUTPUT}"
    "${key}\n#define ERNIE_VERSION_MAJOR ${major}\n#define ERNIE_VERSION_MINOR ${minor}\n"
    "#define ERNIE_VERSION_PATCH ${patch}\n#define ERNIE_VERSION_FULL \"${VERSION}${PRERELEASE}\"\n"
    "#define ERNIE_GIT_SHA \"${sha}\"\n#define ERNIE_GIT_DIRTY ${dirty}\n"
    "#define ERNIE_COMMIT_DATE \"${commit_date}\"\n#define ERNIE_BUILD_DATE \"${built}\"\n"
)
