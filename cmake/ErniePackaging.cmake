# SPDX-License-Identifier: BSD-3-Clause
# Ernie's packages (CPack): one `ernie` package with everything in ErnieInstall.cmake's components
# (libernie and Slang's runtime, the viewer, the headers and the CMake package), as a TGZ to unpack
# anywhere, a DEB and an RPM.
#
# Configure with the `package-deb` or `package-rpm` preset (CMakePresets.json), which build Release
# with GLFW linked in, then run `cpack --preset <same name>`. Vulkan and the windowing libraries
# are loaded at run time (volk, GLFW), so the packages declare them by hand.
include_guard(GLOBAL)

set(CPACK_PACKAGE_NAME ernie)
set(CPACK_PACKAGE_VENDOR "Ernie")
set(CPACK_PACKAGE_CONTACT "Alex Maietta <maiettaalex@gmail.com>")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/alexmaietta/ernie")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "Spectral Vulkan renderer for natural and interplanetary environments"
)
set(CPACK_PACKAGE_DESCRIPTION
    "Ernie is a physically based, spectral, real-time renderer for planetary scenes: terrain, star \
fields, sunlight and camera sensors described in physical units, rendered in batches on any \
Vulkan 1.3 GPU. This package has the library, the viewer, and the headers and CMake package for \
building against it.")
# Between releases, "0.1.0~dev12": Debian and RPM sort "~" before the release, as PEP 440 does
# ".dev12".
string(REPLACE "." "~" ernie_package_suffix "${ERNIE_VERSION_SUFFIX}")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}${ernie_package_suffix}")
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_CHECKSUM SHA256)
set(CPACK_STRIP_FILES ON)
set(CPACK_PACKAGE_DIRECTORY "${PROJECT_BINARY_DIR}/packages")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(ernie_deb_architecture amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ernie_deb_architecture arm64)
else()
    set(ernie_deb_architecture ${CMAKE_SYSTEM_PROCESSOR})
endif()
set(CPACK_PACKAGE_FILE_NAME
    "ernie-${PROJECT_VERSION}${ERNIE_VERSION_SUFFIX}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}"
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CPACK_GENERATOR TGZ DEB RPM)
else()
    set(CPACK_GENERATOR TGZ)
endif()

# Ernie's components only (not the wheel's `python` component, nor rules that dependencies install
# without one), all in one package.
set(CPACK_COMPONENTS_ALL runtime dev)
if(TARGET ernie_viewer)
    list(APPEND CPACK_COMPONENTS_ALL viewer)
endif()
# A program per C++ demo (demos/CMakeLists.txt), which is added after this file.
if(ERNIE_BUILD_DEMOS AND TARGET ernie_app)
    list(APPEND CPACK_COMPONENTS_ALL demos)
endif()
set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_RPM_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)
set(CPACK_PROJECT_CONFIG_FILE
    "${PROJECT_SOURCE_DIR}/cmake/ErnieCPackOptions.cmake")

# What GLFW loads for the viewer's windows: X11, and Wayland when the session has it (falling back
# to X11 without). Only the viewer and applications that open windows need them.
set(ernie_deb_window_libraries
    "libx11-6, libxcursor1, libxrandr2, libxinerama1, libxi6, libxext6, libxkbcommon0, libwayland-client0, libwayland-cursor0"
)

# DEB

set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEBIAN_PACKAGE_NAME ernie)
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE ${ernie_deb_architecture})
set(CPACK_DEBIAN_PACKAGE_SECTION graphics)
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libvulkan1")
# The C and C++ runtime dependencies, from the binaries, when dpkg-shlibdeps is available; libernie
# and Slang's runtime are in the package itself.
find_program(ERNIE_DPKG_SHLIBDEPS dpkg-shlibdeps)
mark_as_advanced(ERNIE_DPKG_SHLIBDEPS)
if(ERNIE_DPKG_SHLIBDEPS)
    set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
    set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS_PRIVATE_DIRS
        "${PROJECT_BINARY_DIR}/lib" "${ERNIE_SLANG_RUNTIME_DIR}")
else()
    # Built where dpkg-shlibdeps is missing (the manylinux container): the build host's glibc, and
    # the libstdc++ of GCC 11 that Slang's prebuilt library needs (GLIBCXX_3.4.29).
    execute_process(
        COMMAND getconf GNU_LIBC_VERSION
        OUTPUT_VARIABLE ernie_glibc
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "[0-9]+\\.[0-9]+" ernie_glibc "${ernie_glibc}")
    string(
        APPEND
        CPACK_DEBIAN_PACKAGE_DEPENDS
        ", libc6 (>= ${ernie_glibc}), libstdc++6 (>= 11), libgcc-s1 (>= 3.4)"
    )
    # The system's libcurl, which the demos link (curl.cmake): renamed
    # libcurl4t64 in Debian 13 and Ubuntu 24.04.
    if(TARGET ernie_curl AND CURL_FOUND)
        string(APPEND CPACK_DEBIAN_PACKAGE_DEPENDS
               ", libcurl4t64 | libcurl4")
    endif()
endif()
# Recommended rather than required, so headless servers can leave the window libraries out
# (`apt install --no-install-recommends`).
set(CPACK_DEBIAN_PACKAGE_RECOMMENDS
    "mesa-vulkan-drivers | vulkan-icd, ${ernie_deb_window_libraries}")

# RPM

set(CPACK_RPM_FILE_NAME RPM-DEFAULT)
set(CPACK_RPM_PACKAGE_NAME ernie)
set(CPACK_RPM_PACKAGE_LICENSE "BSD-3-Clause")
set(CPACK_RPM_PACKAGE_GROUP "Applications/Graphics")
# CPack's RPMs have no weak "Recommends", so the X11 client libraries (small, and no X server) are
# required; Wayland's and Mesa's drivers are suggested.
set(CPACK_RPM_PACKAGE_REQUIRES
    "vulkan-loader, libX11, libXcursor, libXrandr, libXinerama, libXi, libXext, libxkbcommon"
)
set(CPACK_RPM_PACKAGE_SUGGESTS
    "mesa-vulkan-drivers, libwayland-client, libwayland-cursor")
# Directories that the system's own packages own.
set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake"
    "${CMAKE_INSTALL_PREFIX}/share/doc")

include(CPack)

cpack_add_component(
    runtime
    DISPLAY_NAME "Ernie library"
    DESCRIPTION "libernie, with Slang's runtime"
    REQUIRED)
cpack_add_component(
    dev
    DISPLAY_NAME "Ernie development files"
    DESCRIPTION "Headers and the CMake package (find_package(Ernie))"
    DEPENDS runtime)
cpack_add_component(
    viewer
    DISPLAY_NAME "Ernie viewer"
    DESCRIPTION "The `ernie` interactive viewer"
    DEPENDS runtime)
