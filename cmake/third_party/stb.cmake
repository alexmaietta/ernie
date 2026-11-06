# SPDX-License-Identifier: BSD-3-Clause
include_guard(GLOBAL)
FetchContent_Declare(
    stb
    URL
        https://github.com/nothings/stb/archive/003501f6d7a8668ce8b6d875e34dd97ad0d5d5b5.tar.gz
    URL_HASH SHA256=a6c31950b3bdf05eb92258768ba2a11bd44449490574b1aac67028452288c40d
)
FetchContent_MakeAvailable(stb)

file(
    CONFIGURE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/stb.cpp"
    CONTENT
        "#define STB_IMAGE_IMPLEMENTATION\n#define STB_IMAGE_WRITE_IMPLEMENTATION\n#define STB_TRUETYPE_IMPLEMENTATION\n#include <stb_image.h>\n#include <stb_image_write.h>\n#include <stb_truetype.h>\n"
)
add_library(ernie_stb STATIC "${CMAKE_CURRENT_BINARY_DIR}/stb.cpp")
target_include_directories(ernie_stb SYSTEM PUBLIC "${stb_SOURCE_DIR}")
target_compile_definitions(
    ernie_stb
    PUBLIC
        STBI_ONLY_PNG
        STBI_ONLY_JPEG
)
set_target_properties(
    ernie_stb
    PROPERTIES
        POSITION_INDEPENDENT_CODE
            ON
)
target_compile_options(ernie_stb PRIVATE "$<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>")
