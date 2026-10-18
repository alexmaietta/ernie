# SPDX-License-Identifier: BSD-3-Clause
# DEB and RPM packages per install component, and one TGZ of everything, through CPack:
#   runtime  libernie0 / ernie-libs    libernie, share/ernie data, licences and notices
#   app      ernie / ernie             the ernie program, depending on the runtime package
#   dev      libernie-dev / ernie-devel headers, the CMake package and pkg-config file

set(CPACK_PACKAGE_NAME ernie)
set(CPACK_PACKAGE_VERSION "${ERNIE_VERSION}")
set(CPACK_PACKAGE_VENDOR "Alex Maietta")
set(CPACK_PACKAGE_CONTACT "Alex Maietta <maiettaalex@gmail.com>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PROJECT_DESCRIPTION}")
set(CPACK_PACKAGE_HOMEPAGE_URL "${PROJECT_HOMEPAGE_URL}")
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_DIRECTORY "${PROJECT_BINARY_DIR}/packages")
# Archives named for their architecture, so x86_64 and aarch64 tarballs can sit side by side.
set(CPACK_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

set(CPACK_COMPONENTS_ALL runtime app dev)
set(CPACK_COMPONENT_APP_DEPENDS runtime)
set(CPACK_COMPONENT_DEV_DEPENDS runtime)
set(CPACK_COMPONENTS_GROUPING IGNORE)
set(CPACK_ARCHIVE_COMPONENT_INSTALL OFF)
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_RPM_COMPONENT_INSTALL ON)

set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS ON)
# Set here, as CPack otherwise asks dpkg, which the manylinux build image lacks.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE arm64)
endif()
set(CPACK_DEBIAN_RUNTIME_PACKAGE_NAME libernie0)
set(CPACK_DEBIAN_APP_PACKAGE_NAME ernie)
set(CPACK_DEBIAN_DEV_PACKAGE_NAME libernie-dev)
# Built on manylinux_2_34 (glibc 2.34, GCC 11's libstdc++), so these are the floors. The library
# loads the Vulkan loader at run time, a dependency the linker cannot see.
set(CPACK_DEBIAN_RUNTIME_PACKAGE_DEPENDS "libc6 (>= 2.34), libstdc++6 (>= 11), libvulkan1")
set(CPACK_DEBIAN_RUNTIME_PACKAGE_RECOMMENDS "mesa-vulkan-drivers")
set(CPACK_DEBIAN_RUNTIME_PACKAGE_SECTION libs)
set(CPACK_DEBIAN_APP_PACKAGE_DEPENDS "libc6 (>= 2.34), libstdc++6 (>= 11), libvulkan1")
set(CPACK_DEBIAN_APP_PACKAGE_SECTION graphics)
set(CPACK_DEBIAN_DEV_PACKAGE_SECTION libdevel)

set(CPACK_RPM_FILE_NAME RPM-DEFAULT)
set(CPACK_RPM_PACKAGE_LICENSE "BSD-3-Clause")
set(CPACK_RPM_RUNTIME_PACKAGE_NAME ernie-libs)
set(CPACK_RPM_APP_PACKAGE_NAME ernie)
set(CPACK_RPM_DEV_PACKAGE_NAME ernie-devel)
set(CPACK_RPM_RUNTIME_PACKAGE_REQUIRES "vulkan-loader")
set(CPACK_RPM_APP_PACKAGE_REQUIRES "vulkan-loader")
set(CPACK_RPM_RUNTIME_PACKAGE_GROUP "System Environment/Libraries")
set(CPACK_RPM_APP_PACKAGE_GROUP "Applications/Graphics")
set(CPACK_RPM_DEV_PACKAGE_GROUP "Development/Libraries")

include(CPack)
