# SPDX-License-Identifier: BSD-3-Clause
set(ERNIE_IMGUI_TARGET
    ""
    CACHE STRING
    "A shared library target of Dear ImGui's docking branch to use instead of Ernie's own"
)

if(ERNIE_IMGUI_TARGET)
    if(NOT TARGET "${ERNIE_IMGUI_TARGET}")
        message(FATAL_ERROR "ERNIE_IMGUI_TARGET names ${ERNIE_IMGUI_TARGET}, which is no target")
    endif()
    get_target_property(imgui_type "${ERNIE_IMGUI_TARGET}" TYPE)
    if(NOT imgui_type STREQUAL "SHARED_LIBRARY")
        message(
            FATAL_ERROR
            "ERNIE_IMGUI_TARGET must be a shared library, for Ernie and its callers to share one "
            "ImGui context; ${ERNIE_IMGUI_TARGET} is a ${imgui_type}"
        )
    endif()
    get_target_property(imgui_includes "${ERNIE_IMGUI_TARGET}" INTERFACE_INCLUDE_DIRECTORIES)
    string(
        REGEX REPLACE "\\$<BUILD_INTERFACE:([^>]*)>"
        "\\1"
        imgui_includes
        "${imgui_includes}"
    )
    find_path(
        ERNIE_IMGUI_SOURCE_DIR
        backends/imgui_impl_vulkan.cpp
        PATHS
            ${imgui_includes}
        NO_DEFAULT_PATH
    )
    if(NOT ERNIE_IMGUI_SOURCE_DIR)
        message(
            FATAL_ERROR
            "No ImGui sources with backends/ beside ${ERNIE_IMGUI_TARGET}'s headers; set "
            "ERNIE_IMGUI_SOURCE_DIR to them"
        )
    endif()
    set(imgui "${ERNIE_IMGUI_SOURCE_DIR}")
    add_library(ernie_imgui INTERFACE)
    target_link_libraries(ernie_imgui INTERFACE "${ERNIE_IMGUI_TARGET}")
    target_include_directories(ernie_imgui SYSTEM INTERFACE "${imgui}/misc/cpp")
    set(ernie_imgui_stdlib "${imgui}/misc/cpp/imgui_stdlib.cpp")
else()
    FetchContent_Declare(
        dear_imgui
        URL
            https://github.com/ocornut/imgui/archive/refs/tags/v1.92.9b-docking.tar.gz
        URL_HASH SHA256=90ded916bd57db2e0e171b6b098940a47c6f5042725dcdc67fb19940ca8bfdcc
    )
    FetchContent_MakeAvailable(dear_imgui)

    set(imgui "${dear_imgui_SOURCE_DIR}")
    add_library(
        ernie_imgui
        OBJECT
        "${imgui}/imgui.cpp"
        "${imgui}/imgui_demo.cpp"
        "${imgui}/imgui_draw.cpp"
        "${imgui}/imgui_tables.cpp"
        "${imgui}/imgui_widgets.cpp"
        "${imgui}/misc/cpp/imgui_stdlib.cpp"
    )
    target_include_directories(
        ernie_imgui
        SYSTEM
        PUBLIC
            "${imgui}"
            "${imgui}/misc/cpp"
    )
    target_include_directories(
        ernie_imgui
        PUBLIC
            "${PROJECT_SOURCE_DIR}"
            "${ERNIE_GENERATED_INCLUDE_DIR}"
    )
    target_compile_definitions(
        ernie_imgui
        PUBLIC
            IMGUI_DISABLE_OBSOLETE_FUNCTIONS
            "$<BUILD_INTERFACE:IMGUI_USER_CONFIG=\"ernie/app/viewer/imgui_config.h\">"
        PRIVATE
            ernie_viewer_EXPORTS
    )
    target_compile_options(ernie_imgui PRIVATE "$<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>")
    set(ernie_imgui_stdlib "")
endif()

add_library(
    ernie_imgui_backends
    STATIC
    "${imgui}/backends/imgui_impl_glfw.cpp"
    "${imgui}/backends/imgui_impl_vulkan.cpp"
    ${ernie_imgui_stdlib}
)
target_include_directories(ernie_imgui_backends SYSTEM PUBLIC "${imgui}/backends")
target_compile_definitions(
    ernie_imgui_backends
    PUBLIC
        IMGUI_IMPL_VULKAN_USE_VOLK
    PRIVATE
        GLFW_INCLUDE_NONE
        ernie_viewer_EXPORTS
)
target_link_libraries(
    ernie_imgui_backends
    PUBLIC
        ernie_imgui
        volk::volk
        glfw
)
target_compile_options(ernie_imgui_backends PRIVATE "$<IF:$<CXX_COMPILER_ID:MSVC>,/w,-w>")
