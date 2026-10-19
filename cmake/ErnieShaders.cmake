# SPDX-License-Identifier: BSD-3-Clause
# Slang shaders kept in .slang files and embedded in the library, so editors highlight and
# resolve them and the program needs no files beside it.
#
#     ernie_add_shaders(ernie_shaders math/affine.slang rt/ray_tracer.slang)
#     add_dependencies(ernie_render ernie_shaders)
#     // In C++: #include "ernie/shaders/rt/ray_tracer.h" for ernie::shaders::kRayTracer.

# Adds `target`, which writes each file's header beside its path under the build's include
# directory. Module names are the files' names, so each must be unique.
function(ernie_add_shaders target)
    set(headers "")
    set(names "")
    foreach(file IN LISTS ARGN)
        get_filename_component(source "${file}" ABSOLUTE)
        get_filename_component(name "${file}" NAME_WLE)
        if(name IN_LIST names)
            message(FATAL_ERROR "Two shader modules are named ${name}")
        endif()
        list(APPEND names "${name}")
        file(RELATIVE_PATH relative "${PROJECT_SOURCE_DIR}" "${source}")
        string(REGEX REPLACE "\\.slang$" ".h" relative "${relative}")
        set(header "${PROJECT_BINARY_DIR}/include/${relative}")
        add_custom_command(
            OUTPUT "${header}"
            COMMAND
                "${CMAKE_COMMAND}" "-DINPUT=${source}" "-DOUTPUT=${header}" "-DNAME=${name}"
                "-DROOT=${PROJECT_SOURCE_DIR}" -P "${PROJECT_SOURCE_DIR}/cmake/EmbedShader.cmake"
            DEPENDS "${source}" "${PROJECT_SOURCE_DIR}/cmake/EmbedShader.cmake"
            COMMENT "Embedding ${relative}"
            VERBATIM
        )
        list(APPEND headers "${header}")
    endforeach()
    add_custom_target(${target} DEPENDS ${headers} SOURCES ${ARGN})
endfunction()
