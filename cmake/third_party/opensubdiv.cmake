set(ERNIE_DEP_VERSION_opensubdiv
    v3_7_0
    CACHE STRING "Tag of OpenSubdiv to fetch.")
mark_as_advanced(ERNIE_DEP_VERSION_opensubdiv)

# OpenSubdiv (Apache 2.0, Pixar), for subdivision surfaces: only its core libraries (sdc, vtr, far,
# bfr), compiled into a library of Ernie's own, without OpenSubdiv's project and its GPU backends.
include(FetchContent)

FetchContent_Declare(
    opensubdiv
    GIT_REPOSITORY https://github.com/PixarAnimationStudios/OpenSubdiv.git
    GIT_TAG ${ERNIE_DEP_VERSION_opensubdiv}
    GIT_SHALLOW TRUE
    # Fetched, not added: no CMakeLists.txt there.
    SOURCE_SUBDIR ernie-sources-only)
FetchContent_MakeAvailable(opensubdiv)

file(
    GLOB
    ernie_opensubdiv_sources
    "${opensubdiv_SOURCE_DIR}/opensubdiv/version.cpp"
    "${opensubdiv_SOURCE_DIR}/opensubdiv/sdc/*.cpp"
    "${opensubdiv_SOURCE_DIR}/opensubdiv/vtr/*.cpp"
    "${opensubdiv_SOURCE_DIR}/opensubdiv/far/*.cpp"
    "${opensubdiv_SOURCE_DIR}/opensubdiv/bfr/*.cpp")
add_library(ernie_opensubdiv STATIC ${ernie_opensubdiv_sources})
target_include_directories(ernie_opensubdiv SYSTEM
                           PUBLIC "${opensubdiv_SOURCE_DIR}")
target_compile_features(ernie_opensubdiv PUBLIC cxx_std_14)
# Its schemes use M_PI, which MSVC's <cmath> defines only on request.
target_compile_definitions(ernie_opensubdiv PRIVATE $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES>)
set_target_properties(
    ernie_opensubdiv
    PROPERTIES POSITION_INDEPENDENT_CODE ON
               COMPILE_WARNING_AS_ERROR OFF
               CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN ON)
