# SPDX-License-Identifier: BSD-3-Clause
if(DEFINED CPACK_TEMPORARY_INSTALL_DIRECTORY)
    if(NOT CPACK_GENERATOR STREQUAL "DEB")
        return()
    endif()

    set(prefix "${CPACK_TEMPORARY_INSTALL_DIRECTORY}")
    set(usr "${CPACK_PACKAGING_INSTALL_PREFIX}")
    set(runtime "${CPACK_DEBIAN_ERNIE_PACKAGE_NAME}")
    file(READ "${CPACK_ERNIE_COPYRIGHT}" copyright)
    string(
        REPLACE "${usr}/${CPACK_ERNIE_DOCDIR}/"
        "${usr}/share/doc/${runtime}/"
        copyright
        "${copyright}"
    )

    foreach(group IN LISTS CPACK_ERNIE_GROUPS)
        string(TOUPPER "${group}" upper)
        set(docs "${prefix}/${group}${usr}/share/doc/${CPACK_DEBIAN_${upper}_PACKAGE_NAME}")
        set(installed "${prefix}/${group}${usr}/${CPACK_ERNIE_DOCDIR}")
        if(EXISTS "${installed}")
            file(RENAME "${installed}" "${docs}")
        else()
            file(MAKE_DIRECTORY "${docs}")
        endif()
        file(WRITE "${docs}/copyright" "${copyright}")
    endforeach()
    return()
endif()

set(CPACK_PACKAGE_NAME ernie)
set(ERNIE_PACKAGE_VERSION "${PROJECT_VERSION}" CACHE STRING "The version of the packages")
set(CPACK_PACKAGE_VERSION "${ERNIE_PACKAGE_VERSION}")
set(CPACK_PACKAGE_VENDOR "Alex Maietta")
set(CPACK_PACKAGE_CONTACT "Alex Maietta <maiettaalex@gmail.com>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PROJECT_DESCRIPTION}")
set(CPACK_PACKAGE_HOMEPAGE_URL "${PROJECT_HOMEPAGE_URL}")
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_DIRECTORY "${PROJECT_BINARY_DIR}/packages")
set(ernie_arch "${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    set(ernie_arch x86_64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    set(ernie_arch aarch64)
endif()
set(ernie_system "${CMAKE_SYSTEM_NAME}")
if(APPLE)
    set(ernie_system macOS)
endif()
set(CPACK_SYSTEM_NAME "${ernie_system}-${ernie_arch}")
if(WIN32)
    set(CPACK_GENERATOR ZIP)
elseif(APPLE)
    set(CPACK_GENERATOR TGZ)
else()
    set(CPACK_GENERATOR
        TGZ
        DEB
        RPM
    )
endif()

set(CPACK_COMPONENTS_ALL
    runtime
    dev
)
set(CPACK_COMPONENT_RUNTIME_GROUP ernie)
set(CPACK_COMPONENT_DEV_GROUP dev)
if(TARGET ernie_viewer)
    list(
        APPEND CPACK_COMPONENTS_ALL
        viewer
        viewer_dev
    )
    set(CPACK_COMPONENT_VIEWER_GROUP viewer)
    set(CPACK_COMPONENT_VIEWER_DEV_GROUP viewer_dev)
endif()
if(TARGET ernie_xr)
    list(
        APPEND CPACK_COMPONENTS_ALL
        xr
        xr_dev
    )
    set(CPACK_COMPONENT_XR_GROUP viewer)
    set(CPACK_COMPONENT_XR_DEV_GROUP viewer_dev)
endif()
set(CPACK_ERNIE_GROUPS
    ernie
    dev
)
if(TARGET ernie_viewer)
    list(
        APPEND CPACK_ERNIE_GROUPS
        viewer
        viewer_dev
    )
endif()
set(CPACK_COMPONENTS_GROUPING ONE_PER_GROUP)
set(CPACK_ARCHIVE_COMPONENT_INSTALL OFF)
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_RPM_COMPONENT_INSTALL ON)

set(CPACK_PRE_BUILD_SCRIPTS "${CMAKE_CURRENT_LIST_FILE}")
set(CPACK_ERNIE_COPYRIGHT "${PROJECT_BINARY_DIR}/copyright")
set(CPACK_ERNIE_DOCDIR "${CMAKE_INSTALL_DOCDIR}")

set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
if(ernie_arch STREQUAL "x86_64")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE amd64)
elseif(ernie_arch STREQUAL "aarch64")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE arm64)
endif()
set(CPACK_DEBIAN_ERNIE_PACKAGE_NAME ernie)
set(CPACK_DEBIAN_ERNIE_PACKAGE_DEPENDS "libc6 (>= 2.34), libstdc++6 (>= 11), libvulkan1")
set(CPACK_DEBIAN_ERNIE_PACKAGE_RECOMMENDS "mesa-vulkan-drivers")
set(CPACK_DEBIAN_ERNIE_PACKAGE_SECTION graphics)
set(CPACK_DEBIAN_DEV_PACKAGE_NAME ernie-dev)
set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS "ernie (= ${ERNIE_PACKAGE_VERSION})")
set(CPACK_DEBIAN_DEV_PACKAGE_SECTION libdevel)
set(CPACK_DEBIAN_VIEWER_PACKAGE_NAME ernie-viewer)
set(CPACK_DEBIAN_VIEWER_PACKAGE_DEPENDS "ernie (= ${ERNIE_PACKAGE_VERSION})")
set(CPACK_DEBIAN_VIEWER_PACKAGE_RECOMMENDS "libx11-6 | libwayland-client0")
set(CPACK_DEBIAN_VIEWER_PACKAGE_SECTION graphics)
set(CPACK_DEBIAN_VIEWER_DEV_PACKAGE_NAME ernie-viewer-dev)
set(CPACK_DEBIAN_VIEWER_DEV_PACKAGE_DEPENDS
    "ernie-dev (= ${ERNIE_PACKAGE_VERSION}), ernie-viewer (= ${ERNIE_PACKAGE_VERSION})"
)
set(CPACK_DEBIAN_VIEWER_DEV_PACKAGE_SECTION libdevel)

set(CPACK_RPM_FILE_NAME RPM-DEFAULT)
set(CPACK_RPM_PACKAGE_LICENSE "BSD-3-Clause")
set(CPACK_RPM_ERNIE_PACKAGE_NAME ernie)
set(CPACK_RPM_ERNIE_PACKAGE_REQUIRES "vulkan-loader")
set(CPACK_RPM_ERNIE_PACKAGE_GROUP "Applications/Graphics")
set(CPACK_RPM_DEV_PACKAGE_NAME ernie-devel)
set(CPACK_RPM_DEV_PACKAGE_REQUIRES "ernie = ${ERNIE_PACKAGE_VERSION}")
set(CPACK_RPM_DEV_PACKAGE_GROUP "Development/Libraries")
set(CPACK_RPM_VIEWER_PACKAGE_NAME ernie-viewer)
set(CPACK_RPM_VIEWER_PACKAGE_REQUIRES "ernie = ${ERNIE_PACKAGE_VERSION}")
set(CPACK_RPM_VIEWER_PACKAGE_GROUP "Applications/Graphics")
set(CPACK_RPM_VIEWER_DEV_PACKAGE_NAME ernie-viewer-devel)
set(CPACK_RPM_VIEWER_DEV_PACKAGE_REQUIRES
    "ernie-devel = ${ERNIE_PACKAGE_VERSION}, ernie-viewer = ${ERNIE_PACKAGE_VERSION}"
)
set(CPACK_RPM_VIEWER_DEV_PACKAGE_GROUP "Development/Libraries")

include(CPack)
