set(ERNIE_DEP_VERSION_materialx
    v1.39.5
    CACHE STRING "Tag of MaterialX to fetch.")
mark_as_advanced(ERNIE_DEP_VERSION_materialx)

# MaterialX (Apache-2.0), for MaterialX materials: its documents, its standard data libraries
# (stdlib, pbrlib, bxdf) and its Slang shader generator (MaterialXGenSlang, first in 1.39.5),
# which Ernie's material code generation subclasses. Built in statically, in a namespace of its
# own (MaterialX_ernie_v1_39_5) with its symbols kept inside Ernie's libraries
# (cmake/ernie_exports.map), so that an application's own MaterialX (OpenUSD's, say) never meets
# Ernie's. Only the core, format and generation libraries are built.
#
# References:
#   https://github.com/AcademySoftwareFoundation/MaterialX
include(FetchContent)

set(MATERIALX_BUILD_SHARED_LIBS OFF)
set(MATERIALX_BUILD_MONOLITHIC OFF)
set(MATERIALX_BUILD_PYTHON OFF)
set(MATERIALX_BUILD_JS OFF)
set(MATERIALX_BUILD_VIEWER OFF)
set(MATERIALX_BUILD_GRAPH_EDITOR OFF)
set(MATERIALX_BUILD_DOCS OFF)
set(MATERIALX_BUILD_TESTS OFF)
set(MATERIALX_BUILD_BENCHMARK_TESTS OFF)
set(MATERIALX_BUILD_RENDER OFF)
set(MATERIALX_BUILD_RENDER_PLATFORMS OFF)
set(MATERIALX_BUILD_GEN_GLSL OFF)
set(MATERIALX_BUILD_GEN_OSL OFF)
set(MATERIALX_BUILD_GEN_MDL OFF)
set(MATERIALX_BUILD_GEN_MSL OFF)
set(MATERIALX_BUILD_GEN_SLANG ON)
set(MATERIALX_BUILD_USE_CCACHE OFF)
set(MATERIALX_INSTALL_PYTHON OFF)
set(MATERIALX_INSTALL_RESOURCES OFF)
set(MATERIALX_NAMESPACE_SUFFIX
    "ernie"
    CACHE STRING "" FORCE)
set(MATERIALX_LIBNAME_SUFFIX
    "_ernie"
    CACHE STRING "" FORCE)

FetchContent_Declare(
    materialx
    GIT_REPOSITORY https://github.com/AcademySoftwareFoundation/MaterialX.git
    GIT_TAG ${ERNIE_DEP_VERSION_materialx}
    GIT_SHALLOW TRUE
    # Its headers warn under Ernie's flags.
    SYSTEM)
set(ERNIE_SAVED_BUILD_TESTING ${BUILD_TESTING})
set(BUILD_TESTING OFF)
set(ERNIE_SAVED_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
FetchContent_MakeAvailable(materialx)
set(BUILD_TESTING ${ERNIE_SAVED_BUILD_TESTING})
set(BUILD_SHARED_LIBS ${ERNIE_SAVED_BUILD_SHARED_LIBS})

add_library(ernie_materialx INTERFACE)
target_link_libraries(ernie_materialx INTERFACE MaterialXCore MaterialXFormat MaterialXGenShader
                                                MaterialXGenHw MaterialXGenSlang)
# MaterialX's data libraries (stdlib, pbrlib, bxdf, lights and their Slang implementations),
# which generation reads at run time: the fetched source's in a build, installed beside Ernie.
set(ERNIE_MATERIALX_LIBRARIES "${materialx_SOURCE_DIR}/libraries")
