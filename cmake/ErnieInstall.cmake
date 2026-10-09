# Installing Ernie (ERNIE_INSTALL): one shared library, libernie, holding every module and their
# third-party dependencies; the headers, with the third-party headers they include; Slang's runtime
# in a private directory; the viewer; the licenses of everything bundled; and a CMake package, so
# that `find_package(Ernie)` gives `Ernie::ernie`, as the build tree does.
#
# Components: `runtime` (the libraries), `dev` (headers, CMake package, the library's namelink),
# `viewer` (the `ernie` executable). See ErniePackaging.cmake for the packages built from them.
include_guard(GLOBAL)

include(FetchContent)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(ERNIE_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/Ernie")
# Slang's runtime, found through RPATHs so it never clashes with another Slang on the system.
set(ERNIE_INSTALL_PRIVATE_LIBDIR "${CMAKE_INSTALL_LIBDIR}/ernie")
# Third-party headers that Ernie's headers include, apart from system ones like /usr/include/glm.
set(ERNIE_INSTALL_EXTERNAL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/ernie/external")
set(ERNIE_INSTALL_DOCDIR "${CMAKE_INSTALL_DATAROOTDIR}/doc/ernie")

if(APPLE)
    set(ernie_origin "@loader_path")
else()
    set(ernie_origin "$ORIGIN")
endif()

# --- libernie ---------------------------------------------------------------------------------

set(ernie_modules
    ernie_util
    ernie_geometry
    ernie_scene
    ernie_device
    ernie_kernel
    ernie_session
    ernie_raster
    ernie_backends)
if(TARGET ernie_app)
    list(APPEND ernie_modules ernie_app)
endif()
list(JOIN ernie_modules "," ernie_modules_joined)

# A library needs a source of its own; the modules provide everything else.
set(ernie_shared_source "${PROJECT_BINARY_DIR}/generated/ernie/ernie_shared.cpp")
file(
    CONFIGURE
    OUTPUT
    "${ernie_shared_source}"
    CONTENT
    "// libernie: Ernie's module libraries, whole, linked into one shared library.\n")

add_library(ernie_shared SHARED "${ernie_shared_source}")
target_link_libraries(ernie_shared PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${ernie_modules_joined}>")
set_target_properties(
    ernie_shared
    PROPERTIES OUTPUT_NAME ernie
               EXPORT_NAME ernie
               VERSION ${PROJECT_VERSION}
               # Before 1.0 every minor version may break the ABI.
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
               LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
               INSTALL_RPATH "${ernie_origin}/ernie")

# Only Ernie's own symbols (and Dear ImGui's and ImPlot's, for viewer panels) are exported; the
# third-party libraries inside stay hidden so they cannot clash with an application's own copies.
# macOS's two-level namespace keeps them apart without a list.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(ernie_exports "${PROJECT_SOURCE_DIR}/cmake/ernie_exports.map")
    target_link_options(ernie_shared PRIVATE "LINKER:--version-script=${ernie_exports}"
                        "LINKER:--no-undefined")
    set_target_properties(ernie_shared PROPERTIES LINK_DEPENDS "${ernie_exports}")
endif()

# What code built against the installed headers needs: their directories, C++20, and the
# definitions that change what the third-party headers declare.
target_include_directories(ernie_shared INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(ernie_shared SYSTEM
                           INTERFACE $<INSTALL_INTERFACE:${ERNIE_INSTALL_EXTERNAL_INCLUDEDIR}>)
target_compile_features(ernie_shared INTERFACE cxx_std_20)
set(ernie_interface_definitions)
foreach(dependency ernie_glm ernie_volk ernie_vma dear_imgui)
    if(TARGET ${dependency})
        get_target_property(definitions ${dependency} INTERFACE_COMPILE_DEFINITIONS)
        if(definitions)
            list(APPEND ernie_interface_definitions ${definitions})
        endif()
    endif()
endforeach()
target_compile_definitions(ernie_shared INTERFACE ${ernie_interface_definitions})

install(
    TARGETS ernie_shared
    EXPORT ErnieTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime NAMELINK_COMPONENT dev
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT runtime)
# Executable, as shared libraries are installed: RPM only lists what an executable library provides.
install(
    PROGRAMS ${ERNIE_SLANG_RUNTIME}
    DESTINATION ${ERNIE_INSTALL_PRIVATE_LIBDIR}
    COMPONENT runtime)

# --- The viewer -------------------------------------------------------------------------------

if(TARGET ernie_viewer)
    file(RELATIVE_PATH ernie_bin_to_lib "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}"
         "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
    set_target_properties(ernie_viewer PROPERTIES INSTALL_RPATH "${ernie_origin}/${ernie_bin_to_lib}")
    install(TARGETS ernie_viewer RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT viewer)
endif()

# --- Headers ----------------------------------------------------------------------------------

set(ernie_header_modules
    util
    geometry
    scene
    device
    session
    raster)
if(TARGET ernie_app)
    list(APPEND ernie_header_modules app)
endif()
foreach(module ${ernie_header_modules})
    install(
        DIRECTORY "${PROJECT_SOURCE_DIR}/ernie/${module}/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/ernie/${module}"
        COMPONENT dev
        FILES_MATCHING
        PATTERN "*.h")
endforeach()

foreach(dependency glm vulkan_headers volk vma dear_imgui implot)
    FetchContent_GetProperties(${dependency})
endforeach()
install(
    DIRECTORY "${glm_SOURCE_DIR}/glm"
    DESTINATION ${ERNIE_INSTALL_EXTERNAL_INCLUDEDIR}
    COMPONENT dev
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.inl")
install(
    DIRECTORY "${vulkan_headers_SOURCE_DIR}/include/vulkan" "${vulkan_headers_SOURCE_DIR}/include/vk_video"
    DESTINATION ${ERNIE_INSTALL_EXTERNAL_INCLUDEDIR}
    COMPONENT dev
    FILES_MATCHING
    PATTERN "*.h")
install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/third_party/dlpack/include/dlpack"
    DESTINATION ${ERNIE_INSTALL_EXTERNAL_INCLUDEDIR}
    COMPONENT dev)
install(
    FILES "${volk_SOURCE_DIR}/volk.h" "${vma_SOURCE_DIR}/include/vk_mem_alloc.h"
    DESTINATION ${ERNIE_INSTALL_EXTERNAL_INCLUDEDIR}
    COMPONENT dev)
if(TARGET ernie_app)
    install(
        FILES "${dear_imgui_SOURCE_DIR}/imgui.h" "${dear_imgui_SOURCE_DIR}/imconfig.h"
              "${implot_SOURCE_DIR}/implot.h"
        DESTINATION ${ERNIE_INSTALL_EXTERNAL_INCLUDEDIR}
        COMPONENT dev)
endif()

# --- Licenses ---------------------------------------------------------------------------------

include("${PROJECT_SOURCE_DIR}/cmake/ErnieLicenses.cmake")
ernie_install_licenses(${ERNIE_INSTALL_DOCDIR} runtime)

# --- CMake package ----------------------------------------------------------------------------

install(
    EXPORT ErnieTargets
    NAMESPACE Ernie::
    DESTINATION ${ERNIE_INSTALL_CMAKEDIR}
    COMPONENT dev)
configure_package_config_file("${PROJECT_SOURCE_DIR}/cmake/ErnieConfig.cmake.in"
                              "${PROJECT_BINARY_DIR}/ErnieConfig.cmake"
                              INSTALL_DESTINATION ${ERNIE_INSTALL_CMAKEDIR})
# Before 1.0 a minor version may break the API.
write_basic_package_version_file("${PROJECT_BINARY_DIR}/ErnieConfigVersion.cmake"
                                 COMPATIBILITY SameMinorVersion)
install(
    FILES "${PROJECT_BINARY_DIR}/ErnieConfig.cmake" "${PROJECT_BINARY_DIR}/ErnieConfigVersion.cmake"
    DESTINATION ${ERNIE_INSTALL_CMAKEDIR}
    COMPONENT dev)
