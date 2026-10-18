# SPDX-License-Identifier: BSD-3-Clause
# stb has no releases, so a commit is pinned.
set(ERNIE_DEP_VERSION_stb
    2c980bb59875b0d32144a71867fbdebb2f77cd20
    CACHE STRING
    "Commit of stb to fetch"
)
mark_as_advanced(ERNIE_DEP_VERSION_stb)

ernie_fetch(
    stb
    URL https://github.com/nothings/stb/archive/${ERNIE_DEP_VERSION_stb}.tar.gz
    SOURCE_SUBDIR no-build
)

# stb is header-only: its implementations of stb_image (PNG, JPEG) and stb_image_write are compiled
# once, from a source written into the build tree.
set(ernie_stb_source "${CMAKE_BINARY_DIR}/stb/stb.cpp")
file(
    CONFIGURE
    OUTPUT "${ernie_stb_source}"
    CONTENT
        "#define STB_IMAGE_IMPLEMENTATION\n#define STB_IMAGE_WRITE_IMPLEMENTATION\n#include <stb_image.h>\n#include <stb_image_write.h>\n"
)
add_library(ernie_stb STATIC "${ernie_stb_source}")
target_include_directories(ernie_stb SYSTEM PUBLIC "${stb_SOURCE_DIR}")
target_compile_definitions(ernie_stb PUBLIC STBI_ONLY_PNG STBI_ONLY_JPEG)
set_target_properties(ernie_stb PROPERTIES POSITION_INDEPENDENT_CODE ON)
