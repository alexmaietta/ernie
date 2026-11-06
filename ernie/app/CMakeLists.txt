# SPDX-License-Identifier: BSD-3-Clause
ernie_object_library(ernie_app EXPORTS ernie_app_EXPORTS cli.cpp)
target_link_libraries(
    ernie_app
    PUBLIC
        ernie
        ${CMAKE_DL_LIBS}
)
target_compile_definitions(ernie_app PRIVATE ERNIE_SOVERSION="${PROJECT_VERSION_MAJOR}")

add_executable(ernie_cli main.cpp)
ernie_warnings(ernie_cli)
set_target_properties(
    ernie_cli
    PROPERTIES
        OUTPUT_NAME
            ernie
)
target_link_libraries(ernie_cli PRIVATE ernie_app)

ernie_check_includes(
    ernie/app
    ernie/app/cli.cpp
    ernie/app/cli.h
    ernie/app/main.cpp
)

if(ERNIE_BUILD_VIEWER)
    include("${PROJECT_SOURCE_DIR}/cmake/third_party/json.cmake")
    add_subdirectory(viewer)
endif()
