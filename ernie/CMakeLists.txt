# SPDX-License-Identifier: BSD-3-Clause
# Ernie's modules, one library each, every module depending only on the ones before it:
#   util      images, spectra, raster views, tensors; no Vulkan
#   geometry  frusta, the LOD quadtree, mesh data and primitive shapes
#   scene     the scene, its schema and tokens, cameras, sensors and materials
#   device    Vulkan device, memory, pipelines, Slang compiler, presentation, CUDA interop
#   kernel    the Slang shaders, embedded into the binaries
#   session   the backend-agnostic render delegate, its outputs and the scene's resolution
#   raster    the rasterizer backend
#   app       the interactive viewer (ERNIE_BUILD_APP)
# Headers sit next to their sources and are included from the repository root as
# "ernie/<module>/<file>.h".

# ernie_add_library(<target> SOURCES ... [PUBLIC ...] [PRIVATE ...]): a static module library.
function(ernie_add_library target)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" ""
                          "SOURCES;PUBLIC;PRIVATE")
    add_library(${target} STATIC ${ARG_SOURCES})
    target_include_directories(
        ${target} PUBLIC "${PROJECT_SOURCE_DIR}"
                         "${PROJECT_BINARY_DIR}/generated")
    target_compile_features(${target} PUBLIC cxx_std_20)
    target_link_libraries(
        ${target}
        PUBLIC ${ARG_PUBLIC}
        PRIVATE ${ARG_PRIVATE} ernie_build_options)
    string(REPLACE "ernie_" "" name "${target}")
    add_library(Ernie::${name} ALIAS ${target})
endfunction()

add_subdirectory(util)
add_subdirectory(geometry)
add_subdirectory(scene)
add_subdirectory(device)
add_subdirectory(kernel)
add_subdirectory(session)
add_subdirectory(raster)

# The backends `CreateRenderDelegate` can make, and `Renderer`, which creates one on a GPU.
ernie_add_library(ernie_backends SOURCES session/backends.cpp
                  session/renderer.cpp PUBLIC ernie_raster)

add_library(ernie INTERFACE)
target_link_libraries(ernie INTERFACE ernie_backends)
add_library(Ernie::ernie ALIAS ernie)

# The USD importer plugin, which ernie_scene loads when a USD file is first imported.
if(ERNIE_USD)
    add_subdirectory(usd)
    target_compile_definitions(
        ernie_scene
        PRIVATE ERNIE_USD_PLUGIN_BUILD_PATH="$<TARGET_FILE:ernie_usd>"
    )
endif()

# The interactive viewer, on top of everything; optional so that headless and Python-only builds
# need no windowing or UI libraries.
if(ERNIE_BUILD_APP)
    include("${PROJECT_SOURCE_DIR}/cmake/third_party/glfw.cmake")
    include(
        "${PROJECT_SOURCE_DIR}/cmake/third_party/dear_imgui.cmake")
    include("${PROJECT_SOURCE_DIR}/cmake/third_party/implot.cmake")
    include("${PROJECT_SOURCE_DIR}/cmake/third_party/imguizmo.cmake")
    include("${PROJECT_SOURCE_DIR}/cmake/third_party/gui_fonts.cmake")
    add_subdirectory(app)
endif()

if(ERNIE_BUILD_PYTHON)
    include("${PROJECT_SOURCE_DIR}/cmake/third_party/nanobind.cmake")
    add_subdirectory(python)
endif()
