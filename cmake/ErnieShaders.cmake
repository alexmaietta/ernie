# SPDX-License-Identifier: BSD-3-Clause
# Slang shaders kept in .slang files and embedded in the library, so editors highlight and
# resolve them and the program needs no files beside it.
#
#     ernie_add_shaders(ernie_render shaders/affine.slang shaders/world.slang)
#     // In C++: #include "ernie/render/shaders/affine.h" for ernie::shaders::kAffine.

# Builds each file's header beside its path under the build's include directory, before the
# target compiles.
function(ernie_add_shaders target)
    set(headers "")
    foreach(file IN LISTS ARGN)
        get_filename_component(source "${file}" ABSOLUTE)
        get_filename_component(name "${file}" NAME_WE)
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
    target_sources(${target} PRIVATE ${headers})
endfunction()
