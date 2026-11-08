# SPDX-License-Identifier: BSD-3-Clause
include(CMakePackageConfigHelpers)

set(ernie_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/Ernie")
set(ernie_third_party_dir "${CMAKE_INSTALL_INCLUDEDIR}/ernie/third_party")
set(ernie_private_libdir "${CMAKE_INSTALL_LIBDIR}/ernie")
set(ernie_viewer OFF)
if(TARGET ernie_viewer)
    set(ernie_viewer ON)
endif()

install(
    TARGETS
        ernie_cli
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT runtime
)
install(
    TARGETS
        ernie
    EXPORT ErnieTargets
    RUNTIME
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT runtime
    LIBRARY
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        COMPONENT runtime
        NAMELINK_COMPONENT dev
    ARCHIVE
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        COMPONENT dev
)
if(ernie_viewer)
    install(
        TARGETS
            ernie_viewer
        EXPORT ErnieViewerTargets
        RUNTIME
            DESTINATION "${CMAKE_INSTALL_BINDIR}"
            COMPONENT viewer
        LIBRARY
            DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            COMPONENT viewer
            NAMELINK_COMPONENT viewer_dev
        ARCHIVE
            DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            COMPONENT viewer_dev
    )
endif()
if(TARGET ernie_xr)
    install(
        TARGETS
            ernie_xr
        EXPORT ErnieXrTargets
        RUNTIME
            DESTINATION "${CMAKE_INSTALL_BINDIR}"
            COMPONENT xr
        LIBRARY
            DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            COMPONENT xr
            NAMELINK_COMPONENT xr_dev
        ARCHIVE
            DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            COMPONENT xr_dev
    )
    install(
        DIRECTORY
            "${PROJECT_SOURCE_DIR}/ernie/xr"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/ernie"
        COMPONENT xr_dev
    )
    install(
        FILES
            "${ERNIE_GENERATED_INCLUDE_DIR}/ernie/xr_export.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/ernie"
        COMPONENT xr_dev
    )
    install(
        EXPORT ErnieXrTargets
        NAMESPACE Ernie::
        DESTINATION "${ernie_cmake_dir}"
        COMPONENT xr_dev
    )
endif()
if(WIN32)
    install(FILES ${ERNIE_SLANG_RUNTIME} DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT runtime)
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION "${CMAKE_INSTALL_BINDIR}")
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_COMPONENT runtime)
    include(InstallRequiredSystemLibraries)
    foreach(target IN ITEMS ernie ernie_viewer ernie_cli ernie_xr)
        if(TARGET ${target})
            set_target_properties(
                ${target}
                PROPERTIES
                    DEBUG_POSTFIX
                        d
            )
        endif()
    endforeach()
else()
    if(APPLE)
        set(ernie_origin "@loader_path")
    else()
        set(ernie_origin "$ORIGIN")
    endif()
    install(PROGRAMS ${ERNIE_SLANG_RUNTIME} DESTINATION "${ernie_private_libdir}" COMPONENT runtime)
    set_target_properties(
        ernie
        PROPERTIES
            INSTALL_RPATH
                "${ernie_origin}/ernie;${ernie_origin}"
    )
    if(TARGET ernie_xr)
        set_target_properties(
            ernie_xr
            PROPERTIES
                INSTALL_RPATH
                    "${ernie_origin}"
        )
    endif()
    if(ernie_viewer)
        set_target_properties(
            ernie_viewer
            PROPERTIES
                INSTALL_RPATH
                    "${ernie_origin};${ernie_origin}/ernie"
        )
    endif()
    if(TARGET ernie_cli)
        file(
            RELATIVE_PATH
            ernie_cli_to_libdir
            "/prefix/${CMAKE_INSTALL_BINDIR}"
            "/prefix/${CMAKE_INSTALL_LIBDIR}"
        )
        set_target_properties(
            ernie_cli
            PROPERTIES
                INSTALL_RPATH
                    "${ernie_origin}/${ernie_cli_to_libdir};${ernie_origin}/${ernie_cli_to_libdir}/ernie"
        )
    endif()
endif()

get_property(ernie_public_files GLOBAL PROPERTY ERNIE_PUBLIC_FILES)
foreach(file IN LISTS ernie_public_files)
    get_filename_component(directory "${file}" DIRECTORY)
    install(
        FILES
            "${PROJECT_SOURCE_DIR}/${file}"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${directory}"
        COMPONENT dev
    )
endforeach()
install(
    FILES
        "${ERNIE_GENERATED_INCLUDE_DIR}/ernie/export.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/ernie"
    COMPONENT dev
)
install(
    FILES
        "${ERNIE_GENERATED_INCLUDE_DIR}/ernie/scene/defines.h"
        "${ERNIE_GENERATED_INCLUDE_DIR}/ernie/scene/typed.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/ernie/scene"
    COMPONENT dev
)
install(
    DIRECTORY
        "${ERNIE_GENERATED_INCLUDE_DIR}/ernie/scene/kernels/schema/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/ernie/scene/kernels/schema"
    COMPONENT dev
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "columns.h"
        EXCLUDE
)
set(ernie_dlpack_bundled OFF)
set(ernie_pc_dlpack "")
if(NOT ERNIE_DLPACK_TARGET)
    set(ernie_dlpack_bundled ON)
    set(ernie_pc_dlpack "-I\${includedir}/ernie/third_party/dlpack")
    install(
        FILES
            "${dlpack_SOURCE_DIR}/include/dlpack/dlpack.h"
        DESTINATION "${ernie_third_party_dir}/dlpack/dlpack"
        COMPONENT dev
    )
endif()
set(ernie_pc_imgui "")
if(ernie_viewer)
    install(
        FILES
            "${PROJECT_SOURCE_DIR}/ernie/app/viewer/viewer.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/ernie/app/viewer"
        COMPONENT viewer_dev
    )
    install(
        FILES
            "${PROJECT_SOURCE_DIR}/ernie/app/viewer/imgui_config.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/ernie/app/viewer"
        COMPONENT viewer_dev
    )
    install(
        FILES
            "${ERNIE_GENERATED_INCLUDE_DIR}/ernie/viewer_export.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/ernie"
        COMPONENT viewer_dev
    )
    FetchContent_GetProperties(dear_imgui)
    FetchContent_GetProperties(glfw)
    FetchContent_GetProperties(lucide_license)
    if(NOT ERNIE_IMGUI_TARGET)
        file(
            WRITE "${PROJECT_BINARY_DIR}/imgui/imconfig.h"
            "// Dear ImGui's configuration as installed with Ernie: its API is the viewer library's.\n"
            "#pragma once\n\n#define IMGUI_DISABLE_OBSOLETE_FUNCTIONS\n"
            "#include \"ernie/viewer_export.h\"\n#define IMGUI_API ERNIE_VIEWER_API\n"
        )
        install(
            FILES
                "${dear_imgui_SOURCE_DIR}/imgui.h"
                "${PROJECT_BINARY_DIR}/imgui/imconfig.h"
                "${dear_imgui_SOURCE_DIR}/imgui_internal.h"
                "${dear_imgui_SOURCE_DIR}/imstb_rectpack.h"
                "${dear_imgui_SOURCE_DIR}/imstb_textedit.h"
                "${dear_imgui_SOURCE_DIR}/imstb_truetype.h"
                "${dear_imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.h"
            DESTINATION "${ernie_third_party_dir}/imgui"
            COMPONENT viewer_dev
        )
        set(ernie_pc_imgui "-I\${includedir}/ernie/third_party/imgui")
    endif()
endif()

install(EXPORT ErnieTargets NAMESPACE Ernie:: DESTINATION "${ernie_cmake_dir}" COMPONENT dev)
if(ernie_viewer)
    install(
        EXPORT ErnieViewerTargets
        NAMESPACE Ernie::
        DESTINATION "${ernie_cmake_dir}"
        COMPONENT viewer_dev
    )
endif()
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/ErnieConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/ErnieConfig.cmake"
    INSTALL_DESTINATION "${ernie_cmake_dir}"
)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/ErnieConfigVersion.cmake"
    COMPATIBILITY SameMinorVersion
)
install(
    FILES
        "${PROJECT_BINARY_DIR}/ErnieConfig.cmake"
        "${PROJECT_BINARY_DIR}/ErnieConfigVersion.cmake"
    DESTINATION "${ernie_cmake_dir}"
    COMPONENT dev
)
file(
    RELATIVE_PATH
    ernie_pc_to_prefix
    "/prefix/${CMAKE_INSTALL_LIBDIR}/pkgconfig"
    "/prefix"
)
set(ernie_pc_name ernie)
set(ernie_pc_library ernie)
set(ernie_pc_description "${PROJECT_DESCRIPTION}")
set(ernie_pc_cflags "")
set(ernie_pc_requires "")
configure_file("${PROJECT_SOURCE_DIR}/cmake/ernie.pc.in" "${PROJECT_BINARY_DIR}/ernie.pc" @ONLY)
install(
    FILES
        "${PROJECT_BINARY_DIR}/ernie.pc"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
    COMPONENT dev
)
if(ernie_viewer)
    set(ernie_pc_name ernie_viewer)
    set(ernie_pc_library ernie_viewer)
    set(ernie_pc_description "The Ernie viewer")
    set(ernie_pc_cflags "${ernie_pc_imgui}")
    set(ernie_pc_requires ernie)
    configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/ernie.pc.in"
        "${PROJECT_BINARY_DIR}/ernie_viewer.pc"
        @ONLY
    )
    install(
        FILES
            "${PROJECT_BINARY_DIR}/ernie_viewer.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
        COMPONENT viewer_dev
    )
endif()

set(ernie_notices
    "slang|${slang_SOURCE_DIR}/LICENSE"
    "vulkan_headers|${vulkan_headers_SOURCE_DIR}/LICENSE.md"
    "volk|${volk_SOURCE_DIR}/LICENSE.md"
    "vma|${vma_SOURCE_DIR}/LICENSE.txt"
    "nvtx|${nvtx_SOURCE_DIR}/LICENSE.txt"
    "stb|${stb_SOURCE_DIR}/LICENSE"
    "openexr|${openexr_SOURCE_DIR}/LICENSE.md"
    "imath|${imath_SOURCE_DIR}/LICENSE.md"
    "libdeflate|${openexr_SOURCE_DIR}/external/deflate/COPYING"
    "zstd|${openexr_SOURCE_DIR}/external/zstd/LICENSE"
    "openjph|${openexr_SOURCE_DIR}/external/OpenJPH/LICENSE"
    "mikktspace|${mikktspace_SOURCE_DIR}/mikktspace.h"
    "tinyobjloader|${tinyobjloader_SOURCE_DIR}/LICENSE"
    "fmt|${fmt_SOURCE_DIR}/LICENSE"
)
if(ernie_dlpack_bundled)
    list(APPEND ernie_notices "dlpack|${dlpack_SOURCE_DIR}/LICENSE")
endif()
set(ernie_viewer_notices)
if(ernie_viewer)
    list(
        APPEND ernie_viewer_notices
        "lucide|${lucide_license_SOURCE_DIR}/LICENSE"
        "glfw|${glfw_SOURCE_DIR}/LICENSE.md"
    )
    if(NOT ERNIE_IMGUI_TARGET)
        list(APPEND ernie_viewer_notices "dear_imgui|${dear_imgui_SOURCE_DIR}/LICENSE.txt")
    endif()
    if(TARGET ernie_xr)
        FetchContent_GetProperties(openxr)
        list(APPEND ernie_viewer_notices "openxr_sdk|${openxr_SOURCE_DIR}/LICENSE")
    endif()
endif()

install(
    FILES
        "${PROJECT_SOURCE_DIR}/LICENSE"
    DESTINATION "${CMAKE_INSTALL_DOCDIR}"
    COMPONENT runtime
)
file(READ "${PROJECT_SOURCE_DIR}/LICENSE" ernie_license)
string(
    REGEX REPLACE "\n"
    "\n "
    ernie_license
    " ${ernie_license}"
)
set(ernie_copyright
    "Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/\n"
    "Upstream-Name: Ernie\nSource: ${PROJECT_HOMEPAGE_URL}\n\n"
    "Files: *\nCopyright: Alex Maietta\nLicense: BSD-3-Clause\n${ernie_license}\n"
)
foreach(notice IN LISTS ernie_notices ernie_viewer_notices)
    string(
        REPLACE "|"
        ";"
        parts
        "${notice}"
    )
    list(GET parts 0 name)
    list(GET parts 1 file)
    if(NOT EXISTS "${file}")
        message(FATAL_ERROR "No license for ${name} at ${file}")
    endif()
    get_filename_component(file_name "${file}" NAME)
    if(notice IN_LIST ernie_viewer_notices)
        set(component viewer)
        set(python_component python_viewer)
        set(python_target _ernie_viewer)
    else()
        set(component runtime)
        set(python_component python)
        set(python_target _ernie)
    endif()
    install(
        FILES
            "${file}"
        DESTINATION "${CMAKE_INSTALL_DOCDIR}/third_party/${name}"
        COMPONENT ${component}
    )
    if(TARGET ${python_target})
        install(FILES "${file}" DESTINATION "ernie/licenses/${name}" COMPONENT ${python_component})
    endif()
    list(
        APPEND ernie_copyright
        "\nFiles: bundled/${name}/*\nCopyright: see the license\n"
        "License: see /usr/${CMAKE_INSTALL_DOCDIR}/third_party/${name}/${file_name}\n"
    )
endforeach()
list(JOIN ernie_copyright "" ernie_copyright)
file(WRITE "${PROJECT_BINARY_DIR}/copyright" "${ernie_copyright}")
install(
    FILES
        "${PROJECT_BINARY_DIR}/copyright"
    DESTINATION "${CMAKE_INSTALL_DOCDIR}"
    COMPONENT runtime
)
