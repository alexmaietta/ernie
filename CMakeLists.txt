cmake_minimum_required(VERSION 3.22)
# Files extracted from URL downloads (Slang, fonts, JSON) get the time of extraction, so a new
# download rebuilds what depends on them.
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

project(
    Ernie
    VERSION 0.1.0
    DESCRIPTION
        "Efficient Renderer for Natural and Interplanetary Environments: a spectral Vulkan renderer"
    LANGUAGES C CXX)

option(ERNIE_BUILD_TESTS "Build Ernie's tests" ${PROJECT_IS_TOP_LEVEL})
option(ERNIE_BUILD_PYTHON "Build Ernie's Python module" OFF)
option(ERNIE_BUILD_DEMOS "Build Ernie's C++ demos" ${PROJECT_IS_TOP_LEVEL})
option(ERNIE_BUILD_APP
       "Build Ernie's app, the GUI viewer (GLFW, Dear ImGui, ImPlot, ImGuizmo, fonts)" OFF)
option(ERNIE_TRACY "Instrument Ernie for the Tracy profiler (CPU and Vulkan GPU zones, on demand)"
       OFF)
option(ERNIE_CCACHE "Compile through ccache when it is installed" ON)
# libernie and the install rules need CMake 3.24's whole-archive linking; Windows needs export
# annotations first.
if(PROJECT_IS_TOP_LEVEL
   AND NOT WIN32
   AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(ernie_install_default ON)
else()
    set(ernie_install_default OFF)
endif()
option(ERNIE_INSTALL "Build libernie, the shared library, and Ernie's install rules and packages"
       ${ernie_install_default})
if(ERNIE_INSTALL AND CMAKE_VERSION VERSION_LESS 3.24)
    message(FATAL_ERROR "ERNIE_INSTALL needs CMake 3.24 or later")
endif()
if(ERNIE_INSTALL AND WIN32)
    message(FATAL_ERROR "ERNIE_INSTALL is not supported on Windows yet")
endif()

set(ERNIE_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}"
    CACHE INTERNAL "Ernie source directory")

# An optimised build with debug info unless a build type is given: Ernie's CPU side (frame setup,
# culling lists, scene sync) is many times slower without optimisation.
get_property(ernie_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(PROJECT_IS_TOP_LEVEL
   AND NOT ernie_multi_config
   AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE
        RelWithDebInfo
        CACHE STRING "Build type" FORCE)
endif()

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
endif()

# A compiler cache, so reconfigures, branch switches and clean builds reuse earlier compiles; a
# launcher given on the command line wins.
if(PROJECT_IS_TOP_LEVEL
   AND ERNIE_CCACHE
   AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(ERNIE_CCACHE_PROGRAM ccache)
    if(ERNIE_CCACHE_PROGRAM)
        set(CMAKE_C_COMPILER_LAUNCHER "${ERNIE_CCACHE_PROGRAM}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${ERNIE_CCACHE_PROGRAM}")
    endif()
endif()

# The Python module and libernie link Ernie's static libraries into shared objects.
if(ERNIE_BUILD_PYTHON OR ERNIE_INSTALL)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

include(cmake/third_party/vulkan_headers.cmake)
include(cmake/third_party/volk.cmake)
include(cmake/third_party/vma.cmake)
include(cmake/third_party/slang.cmake)
if(APPLE)
    include(cmake/third_party/moltenvk.cmake)
endif()
include(cmake/third_party/glm.cmake)
include(cmake/third_party/dlpack.cmake)
include(cmake/third_party/stb.cmake)
include(cmake/third_party/openexr.cmake)
include(cmake/third_party/meshoptimizer.cmake)
include(cmake/third_party/tracy.cmake)
include(cmake/third_party/json.cmake)

# Warning and coverage flags: TerrainGen's when built inside it, Ernie's own otherwise.
add_library(ernie_build_options INTERFACE)
if(TARGET terraingen_warnings)
    target_link_libraries(ernie_build_options INTERFACE terraingen_warnings terraingen_coverage)
elseif(MSVC)
    target_compile_options(ernie_build_options INTERFACE /W4)
else()
    target_compile_options(
        ernie_build_options
        INTERFACE -Wall
                  -Wextra
                  -Wpedantic
                  -Wshadow
                  -Wconversion
                  -Wsign-conversion
                  -Wdouble-promotion)
endif()

add_subdirectory(ernie)

if(ERNIE_INSTALL)
    include(cmake/ErnieInstall.cmake)
    include(cmake/ErniePackaging.cmake)
endif()

if(ERNIE_BUILD_DEMOS)
    add_subdirectory(demos)
endif()

if(ERNIE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(ernie/test)
endif()
