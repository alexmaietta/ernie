# SPDX-License-Identifier: BSD-3-Clause
FetchContent_Declare(
    mikktspace
    URL
        https://github.com/mmikk/MikkTSpace/archive/3e895b49d05ea07e4c2133156cfa94369e19e409.tar.gz
    URL_HASH SHA256=aeba65ddee85a679133d510d71d00b108f603752f90b15b9ecf7777033f6e351
)
FetchContent_MakeAvailable(mikktspace)

add_library(ernie_mikktspace STATIC "${mikktspace_SOURCE_DIR}/mikktspace.c")
target_include_directories(ernie_mikktspace SYSTEM PUBLIC "${mikktspace_SOURCE_DIR}")
set_target_properties(
    ernie_mikktspace
    PROPERTIES
        POSITION_INDEPENDENT_CODE
            ON
)
target_compile_options(ernie_mikktspace PRIVATE "$<IF:$<C_COMPILER_ID:MSVC>,/w,-w>")
