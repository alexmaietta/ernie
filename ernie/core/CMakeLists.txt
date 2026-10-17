# SPDX-License-Identifier: BSD-3-Clause
add_library(ernie_core STATIC version.cpp)
target_include_directories(ernie_core PUBLIC "${PROJECT_SOURCE_DIR}")
target_compile_definitions(ernie_core PRIVATE ERNIE_VERSION_STRING="${ERNIE_VERSION}")
target_link_libraries(ernie_core PRIVATE ernie_build_options)
