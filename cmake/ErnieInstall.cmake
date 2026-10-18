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

set(ernie_public_headers
    util/glm.h
    util/log.h
    util/file.h
    util/profiler.h
    util/tensor.h
    util/version.h
    io/ies.h
    io/image.h
    io/raster.h
    io/texture.h
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
    DIRECTORY "${glm_SOURCE_DIR}/glm"
    DESTINATION "${ERNIE_THIRD_PARTY_INCLUDEDIR}/glm"
    COMPONENT dev
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.inl"
    PATTERN "*.h"
)
install(
    FILES "${dlpack_SOURCE_DIR}/include/dlpack/dlpack.h"
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
    "bc7enc_rdo|${bc7enc_SOURCE_DIR}/LICENSE"
    "dlpack|${dlpack_SOURCE_DIR}/LICENSE"
    "glm|${glm_SOURCE_DIR}/copying.txt"
    "imath|${imath_SOURCE_DIR}/LICENSE.md"
    "libdeflate|${openexr_SOURCE_DIR}/external/deflate/COPYING"
    "nanovdb|${PROJECT_SOURCE_DIR}/third_party/nanovdb/LICENSE"
    "nlohmann_json|${nlohmann_json_SOURCE_DIR}/LICENSE.MIT"
    "openexr|${openexr_SOURCE_DIR}/LICENSE.md"
    "openjph|${openexr_SOURCE_DIR}/external/OpenJPH/LICENSE"
    "stb|${stb_SOURCE_DIR}/LICENSE"
    "volk|${volk_SOURCE_DIR}/LICENSE.md"
    "vulkan_headers|${vulkan_headers_SOURCE_DIR}/LICENSE.md"
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
