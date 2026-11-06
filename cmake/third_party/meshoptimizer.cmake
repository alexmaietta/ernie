# SPDX-License-Identifier: BSD-3-Clause
FetchContent_Declare(
    meshoptimizer
    URL
        https://github.com/zeux/meshoptimizer/archive/refs/tags/v1.3.tar.gz
    URL_HASH SHA256=ca4f47d451201593904b9e3067fa54b65db94bc34ab5173864950862b748a145
    SOURCE_SUBDIR none
)
FetchContent_MakeAvailable(meshoptimizer)

file(GLOB ernie_meshoptimizer_sources "${meshoptimizer_SOURCE_DIR}/src/*.cpp")
add_library(ernie_meshoptimizer STATIC ${ernie_meshoptimizer_sources})
target_include_directories(ernie_meshoptimizer SYSTEM PUBLIC "${meshoptimizer_SOURCE_DIR}/src")
set_target_properties(
    ernie_meshoptimizer
    PROPERTIES
        POSITION_INDEPENDENT_CODE
            ON
)
target_compile_options(ernie_meshoptimizer PRIVATE "$<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>")
