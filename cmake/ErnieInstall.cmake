# SPDX-License-Identifier: BSD-3-Clause
include(CMakePackageConfigHelpers)

set(ERNIE_THIRD_PARTY_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/ernie/third_party")
set(ernie_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/Ernie")
set(ernie_doc_dir "${CMAKE_INSTALL_DOCDIR}")

install(
    TARGETS ernie
    EXPORT ErnieTargets
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT runtime
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT runtime NAMELINK_COMPONENT dev
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT dev
)

# Slang's runtime, private to Ernie in lib/ernie and found through RPATHs, so it never clashes with
# another Slang on the system; on Windows beside the DLLs, where programs look. Installed as
# programs, as shared libraries are: RPM lists only what an executable library provides.
if(APPLE)
    set(ernie_origin "@loader_path")
else()
    set(ernie_origin "$ORIGIN")
endif()
set(ernie_private_libdir "${CMAKE_INSTALL_LIBDIR}/ernie")
set_target_properties(ernie PROPERTIES INSTALL_RPATH "${ernie_origin}/ernie")
if(TARGET ernie_app)
    file(
        RELATIVE_PATH
        ernie_bin_to_private
        "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}"
        "${CMAKE_INSTALL_PREFIX}/${ernie_private_libdir}"
    )
    set_target_properties(
        ernie_app
        PROPERTIES INSTALL_RPATH "${ernie_origin}/${ernie_bin_to_private}"
    )
endif()
if(WIN32)
    install(FILES ${ERNIE_SLANG_RUNTIME} DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT runtime)
else()
    install(PROGRAMS ${ERNIE_SLANG_RUNTIME} DESTINATION "${ernie_private_libdir}" COMPONENT runtime)
endif()

set(ernie_public_headers
    util/algorithm.h
    util/bytes.h
    util/glm.h
    util/hash_index.h
    util/log.h
    util/file.h
    util/profiler.h
    util/tensor.h
    util/version.h
    io/ies.h
    io/image.h
    io/raster.h
    io/texture.h
    geometry/bounds.h
    physics/astronomy.h
    scene/convert.h
    scene/extent_cache.h
    scene/holder_index.h
    scene/path.h
    scene/scene.h
    scene/schema.h
    scene/storage.h
    scene/time_samples.h
    scene/tokens.h
    scene/value.h
    scene/view.h
    scene/xform.h
    scene/schema/camera.h
    scene/schema/collection.h
    scene/schema/geom.h
    scene/schema/instancer.h
    scene/schema/lux.h
    scene/schema/mesh.h
    scene/schema/planet.h
    scene/schema/render.h
    scene/schema/shade.h
    scene/schema/vol.h
    render/renderer.h
)
foreach(header ${ernie_public_headers})
    get_filename_component(directory "${header}" DIRECTORY)
    install(
        FILES "${PROJECT_SOURCE_DIR}/ernie/${header}"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/ernie/${directory}"
        COMPONENT dev
    )
endforeach()
install(
    FILES "${PROJECT_BINARY_DIR}/include/ernie/export.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/ernie"
    COMPONENT dev
)
install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/third_party/glm/glm"
    DESTINATION "${ERNIE_THIRD_PARTY_INCLUDEDIR}/glm"
    COMPONENT dev
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.inl"
    PATTERN "*.h"
)
install(
    FILES "${PROJECT_SOURCE_DIR}/third_party/dlpack/include/dlpack/dlpack.h"
    DESTINATION "${ERNIE_THIRD_PARTY_INCLUDEDIR}/dlpack/dlpack"
    COMPONENT dev
)

install(EXPORT ErnieTargets NAMESPACE Ernie:: DESTINATION "${ernie_cmake_dir}" COMPONENT dev)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/ErnieConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/ErnieConfig.cmake"
    INSTALL_DESTINATION "${ernie_cmake_dir}"
    PATH_VARS ERNIE_THIRD_PARTY_INCLUDEDIR
)
# 0.x releases keep compatibility within a minor series.
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/ErnieConfigVersion.cmake"
    COMPATIBILITY SameMinorVersion
)
install(
    FILES "${PROJECT_BINARY_DIR}/ErnieConfig.cmake" "${PROJECT_BINARY_DIR}/ErnieConfigVersion.cmake"
    DESTINATION "${ernie_cmake_dir}"
    COMPONENT dev
)
# Relocatable: paths are relative to the .pc file, so an unpacked archive works anywhere.
file(RELATIVE_PATH ernie_pc_to_prefix "/prefix/${CMAKE_INSTALL_LIBDIR}/pkgconfig" "/prefix")
configure_file("${PROJECT_SOURCE_DIR}/cmake/ernie.pc.in" "${PROJECT_BINARY_DIR}/ernie.pc" @ONLY)
install(
    FILES "${PROJECT_BINARY_DIR}/ernie.pc"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
    COMPONENT dev
)

# Ernie's licence and those of what the library and its headers carry.
install(FILES "${PROJECT_SOURCE_DIR}/LICENSE" DESTINATION "${ernie_doc_dir}" COMPONENT runtime)
set(ernie_notices
    "basisu|${PROJECT_SOURCE_DIR}/third_party/basisu/LICENSE"
    "basisu|${PROJECT_SOURCE_DIR}/third_party/basisu/NOTICE"
    "bc7enc_rdo|${PROJECT_SOURCE_DIR}/third_party/bc7enc/LICENSE"
    "dlpack|${PROJECT_SOURCE_DIR}/third_party/dlpack/LICENSE"
    "glm|${PROJECT_SOURCE_DIR}/third_party/glm/copying.txt"
    "imath|${imath_SOURCE_DIR}/LICENSE.md"
    "libdeflate|${openexr_SOURCE_DIR}/external/deflate/COPYING"
    "nanovdb|${PROJECT_SOURCE_DIR}/third_party/nanovdb/LICENSE"
    "nlohmann_json|${PROJECT_SOURCE_DIR}/third_party/json/LICENSE.MIT"
    "openexr|${openexr_SOURCE_DIR}/LICENSE.md"
    "openjph|${openexr_SOURCE_DIR}/external/OpenJPH/LICENSE"
    "slang|${slang_SOURCE_DIR}/LICENSE"
    "stb|${PROJECT_SOURCE_DIR}/third_party/stb/LICENSE"
    "vma|${PROJECT_SOURCE_DIR}/third_party/vma/LICENSE.txt"
    "volk|${PROJECT_SOURCE_DIR}/third_party/volk/LICENSE.md"
    "vulkan_headers|${PROJECT_SOURCE_DIR}/third_party/vulkan_headers/LICENSE.md"
    "zstd|${openexr_SOURCE_DIR}/external/zstd/LICENSE"
)
# A Debian machine-readable copyright file: Ernie's licence, and where each bundled one is.
file(READ "${PROJECT_SOURCE_DIR}/LICENSE" ernie_license)
string(REGEX REPLACE "\n" "\n " ernie_license " ${ernie_license}")
set(ernie_copyright
    "Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/\n"
    "Upstream-Name: Ernie\nSource: ${PROJECT_HOMEPAGE_URL}\n\n"
    "Files: *\nCopyright: Alex Maietta\nLicense: BSD-3-Clause\n${ernie_license}\n"
)
foreach(notice ${ernie_notices})
    string(REPLACE "|" ";" parts "${notice}")
    list(GET parts 0 name)
    list(GET parts 1 file)
    get_filename_component(file_name "${file}" NAME)
    install(FILES "${file}" DESTINATION "${ernie_doc_dir}/third_party/${name}" COMPONENT runtime)
    list(
        APPEND ernie_copyright
        "\nFiles: bundled/${name}/*\nCopyright: see the licence\n"
        "License: see /usr/${ernie_doc_dir}/third_party/${name}/${file_name}\n"
    )
endforeach()
list(JOIN ernie_copyright "" ernie_copyright)
file(WRITE "${PROJECT_BINARY_DIR}/copyright" "${ernie_copyright}")
install(FILES "${PROJECT_BINARY_DIR}/copyright" DESTINATION "${ernie_doc_dir}" COMPONENT runtime)
