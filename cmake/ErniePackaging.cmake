# SPDX-License-Identifier: BSD-3-Clause
# DEB, RPM and TGZ packages of the ernie program, through CPack.

set(CPACK_PACKAGE_NAME ernie)
set(CPACK_PACKAGE_VERSION "${ERNIE_VERSION}")
set(CPACK_PACKAGE_VENDOR "Alex Maietta")
set(CPACK_PACKAGE_CONTACT "Alex Maietta <maiettaalex@gmail.com>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PROJECT_DESCRIPTION}")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/alexmaietta/ernie")
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_DIRECTORY "${PROJECT_BINARY_DIR}/packages")
# Archives named for their architecture, so x86_64 and aarch64 tarballs can sit side by side.
set(CPACK_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_COMPONENTS_ALL app)
set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)

# The program loads the Vulkan loader at run time, so it is a dependency the linker cannot see.
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
# Set here, as CPack otherwise asks dpkg, which the manylinux build image lacks.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE arm64)
endif()
# Built on manylinux_2_34 (glibc 2.34, GCC 11's libstdc++), so these are the floors.
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.34), libstdc++6 (>= 11), libvulkan1")
set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "mesa-vulkan-drivers")
set(CPACK_DEBIAN_PACKAGE_SECTION graphics)

set(CPACK_RPM_FILE_NAME RPM-DEFAULT)
set(CPACK_RPM_PACKAGE_LICENSE "BSD-3-Clause")
set(CPACK_RPM_PACKAGE_REQUIRES "vulkan-loader")
set(CPACK_RPM_PACKAGE_GROUP "Applications/Graphics")

include(CPack)
