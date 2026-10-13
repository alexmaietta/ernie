# OpenVDB (Apache-2.0), for reading OpenVDB files (.vdb): their grids are read with it and turned
# into NanoVDB grids with NanoVDB's tools (from the same source), which the renderer reads as it
# reads a .nvdb (ernie/util/openvdb.cpp). Built from source once, with oneTBB (Apache-2.0), Blosc
# (BSD-3-Clause) and zlib (Zlib), every library static, into an install directory beside the
# fetched dependencies (FETCHCONTENT_BASE_DIR, which CI caches), named for the versions and
# compiler; configuring never waits for it. Their symbols stay inside Ernie's libraries
# (cmake/ernie_exports.map, the Python module's export list), so an application's own OpenVDB,
# oneTBB or zlib never meets Ernie's.
#
# References:
#   https://www.openvdb.org/documentation/doxygen/build.html

option(ERNIE_OPENVDB "Read OpenVDB files (.vdb), with OpenVDB built in statically" ON)

set(ERNIE_DEP_VERSION_openvdb
    13.1.0
    CACHE STRING "OpenVDB release to build.")
set(ERNIE_DEP_HASH_openvdb
    "SHA256=21659ef2330a06805519dd8d4369375f181a0dfee205b0180da69b4edd3329ae"
    CACHE STRING "Hash of the OpenVDB release's source archive.")
set(ERNIE_DEP_VERSION_onetbb
    2023.1.0
    CACHE STRING "oneTBB release to build for OpenVDB.")
set(ERNIE_DEP_HASH_onetbb
    "SHA256=191288b52e1e6b17198000b64d77d194bb65e791be46ebc606e9b091781e2070"
    CACHE STRING "Hash of the oneTBB release's source archive.")
set(ERNIE_DEP_VERSION_blosc
    1.21.6
    CACHE STRING "Blosc release to build for OpenVDB.")
set(ERNIE_DEP_HASH_blosc
    "SHA256=9fcd60301aae28f97f1301b735f966cc19e7c49b6b4321b839b4579a0c156f38"
    CACHE STRING "Hash of the Blosc release's source archive.")
set(ERNIE_DEP_VERSION_zlib
    1.3.2
    CACHE STRING "zlib release to build for OpenVDB.")
set(ERNIE_DEP_HASH_zlib
    "SHA256=b99a0b86c0ba9360ec7e78c4f1e43b1cbdf1e6936c8fa0f6835c0cd694a495a1"
    CACHE STRING "Hash of the zlib release's source archive.")
set(ERNIE_OPENVDB_BUILD_JOBS
    4
    CACHE STRING "Parallel jobs building OpenVDB and its dependencies")
mark_as_advanced(
    ERNIE_DEP_VERSION_openvdb
    ERNIE_DEP_HASH_openvdb
    ERNIE_DEP_VERSION_onetbb
    ERNIE_DEP_HASH_onetbb
    ERNIE_DEP_VERSION_blosc
    ERNIE_DEP_HASH_blosc
    ERNIE_DEP_VERSION_zlib
    ERNIE_DEP_HASH_zlib)

add_library(ernie_openvdb INTERFACE)
if(NOT ERNIE_OPENVDB)
    return()
endif()

# Each archive as name|url|hash, built in this order into one prefix.
set(ernie_openvdb_sources
    "zlib|https://github.com/madler/zlib/archive/refs/tags/v${ERNIE_DEP_VERSION_zlib}.tar.gz|${ERNIE_DEP_HASH_zlib}"
    "blosc|https://github.com/Blosc/c-blosc/archive/refs/tags/v${ERNIE_DEP_VERSION_blosc}.tar.gz|${ERNIE_DEP_HASH_blosc}"
    "onetbb|https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v${ERNIE_DEP_VERSION_onetbb}.tar.gz|${ERNIE_DEP_HASH_onetbb}"
    "openvdb|https://github.com/AcademySoftwareFoundation/openvdb/archive/refs/tags/v${ERNIE_DEP_VERSION_openvdb}.tar.gz|${ERNIE_DEP_HASH_openvdb}"
)
if(NOT FETCHCONTENT_BASE_DIR)
    set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/_deps")
endif()
string(SHA1 ernie_openvdb_key
            "${ernie_openvdb_sources};${CMAKE_CXX_COMPILER_ID};${CMAKE_CXX_COMPILER_VERSION}")
string(SUBSTRING "${ernie_openvdb_key}" 0 10 ernie_openvdb_key)
set(ernie_openvdb_dir "${FETCHCONTENT_BASE_DIR}/openvdb-${ERNIE_DEP_VERSION_openvdb}-${ernie_openvdb_key}")

if(WIN32)
    set(ernie_openvdb_libraries
        "${ernie_openvdb_dir}/lib/libopenvdb.lib" "${ernie_openvdb_dir}/lib/tbb12.lib"
        "${ernie_openvdb_dir}/lib/libblosc.lib" "${ernie_openvdb_dir}/lib/zs.lib")
else()
    set(ernie_openvdb_libraries
        "${ernie_openvdb_dir}/lib/libopenvdb.a" "${ernie_openvdb_dir}/lib/libtbb.a"
        "${ernie_openvdb_dir}/lib/libblosc.a" "${ernie_openvdb_dir}/lib/libz.a")
endif()

if(EXISTS "${ernie_openvdb_dir}/.built")
    # Built already (by an earlier build, or restored from a cache).
    add_custom_target(ernie_openvdb_build)
else()
    string(REPLACE ";" "^" ernie_openvdb_sources_joined "${ernie_openvdb_sources}")
    # Under a lock (openvdb_build.cmake): builds started at once wait for each other.
    add_custom_command(
        OUTPUT ${ernie_openvdb_libraries}
        COMMAND
            ${CMAKE_COMMAND} "-DSOURCES=${ernie_openvdb_sources_joined}"
            "-DINSTALL=${ernie_openvdb_dir}" "-DWORK=${FETCHCONTENT_BASE_DIR}/openvdb-work"
            "-DJOBS=${ERNIE_OPENVDB_BUILD_JOBS}" "-DGENERATOR=${CMAKE_GENERATOR}"
            "-DC_COMPILER=${CMAKE_C_COMPILER}" "-DCXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DMAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}" -P
            "${CMAKE_CURRENT_LIST_DIR}/openvdb_build.cmake"
        COMMENT "Building OpenVDB, oneTBB, Blosc and zlib (once)"
        VERBATIM)
    add_custom_target(ernie_openvdb_build DEPENDS ${ernie_openvdb_libraries})
endif()
add_dependencies(ernie_openvdb ernie_openvdb_build)

file(MAKE_DIRECTORY "${ernie_openvdb_dir}/include")
target_include_directories(ernie_openvdb SYSTEM INTERFACE "${ernie_openvdb_dir}/include")
target_link_libraries(ernie_openvdb INTERFACE ${ernie_openvdb_libraries} ${CMAKE_DL_LIBS})
find_package(Threads REQUIRED)
target_link_libraries(ernie_openvdb INTERFACE Threads::Threads)
# OpenVDB and oneTBB as static libraries; NanoVDB's tools with OpenVDB's grids.
target_compile_definitions(
    ernie_openvdb INTERFACE OPENVDB_STATICLIB NANOVDB_USE_OPENVDB ERNIE_HAS_OPENVDB
                            $<$<BOOL:${WIN32}>:__TBB_NO_IMPLICIT_LINKAGE>)
