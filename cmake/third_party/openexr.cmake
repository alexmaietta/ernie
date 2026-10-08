set(ERNIE_DEP_VERSION_openexr
    v3.5.2
    CACHE STRING "Tag of OpenEXR to fetch.")
mark_as_advanced(ERNIE_DEP_VERSION_openexr)

# OpenEXR (BSD-3-Clause), the reference library, for reading every EXR compression, including
# the lossy DWAA and DWAB that texture libraries such as Poly Haven use. It fetches its own Imath
# and libdeflate; only the libraries are built.
include(FetchContent)

set(OPENEXR_BUILD_TOOLS OFF)
set(OPENEXR_BUILD_EXAMPLES OFF)
set(OPENEXR_BUILD_DOCS OFF)
set(OPENEXR_BUILD_PYTHON OFF)
set(OPENEXR_INSTALL OFF)
set(OPENEXR_INSTALL_TOOLS OFF)
set(OPENEXR_INSTALL_PKG_CONFIG OFF)
set(OPENEXR_TEST_LIBRARIES OFF)
set(OPENEXR_TEST_TOOLS OFF)
set(OPENEXR_TEST_PYTHON OFF)
set(OPENEXR_FORCE_INTERNAL_IMATH ON)
set(OPENEXR_FORCE_INTERNAL_DEFLATE ON)
set(IMATH_INSTALL OFF)
set(BUILD_SHARED_LIBS OFF)

FetchContent_Declare(
    openexr
    GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/openexr.git
    GIT_TAG ${ERNIE_DEP_VERSION_openexr}
    GIT_SHALLOW TRUE)
# Its own tests stay out of Ernie's ctest.
set(ERNIE_SAVED_BUILD_TESTING ${BUILD_TESTING})
set(BUILD_TESTING OFF)
FetchContent_MakeAvailable(openexr)
set(BUILD_TESTING ${ERNIE_SAVED_BUILD_TESTING})

add_library(ernie_openexr INTERFACE)
target_link_libraries(ernie_openexr INTERFACE OpenEXR::OpenEXR)
