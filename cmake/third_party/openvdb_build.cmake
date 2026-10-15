# Builds zlib, Blosc, oneTBB and OpenVDB (with NanoVDB's headers) as static libraries into INSTALL
# once (cmake -P, from openvdb.cmake), under a lock on the install directory: builds of the same
# tree started at once (two targets, two sessions on one build directory) wait for each other,
# and the later ones find it built.
#
# Takes SOURCES (name|url|hash entries, separated by "^"), INSTALL, WORK, JOBS, GENERATOR,
# C_COMPILER, CXX_COMPILER and MAKE_PROGRAM.

file(MAKE_DIRECTORY "${INSTALL}")
file(LOCK "${INSTALL}.lock" GUARD PROCESS TIMEOUT 7200 RESULT_VARIABLE lock_result)
if(NOT lock_result EQUAL 0)
    message(FATAL_ERROR "Cannot lock ${INSTALL}.lock: ${lock_result}")
endif()
if(EXISTS "${INSTALL}/.built")
    message(STATUS "OpenVDB is built in ${INSTALL}")
    return()
endif()

# Every library static, position independent and with hidden symbols, in Release, found in the
# install directory first (each its <Package>_ROOT). PATH stays searched: the tools live there
# (ar in manylinux's gcc-toolset, rc and mt in an MSVC prompt).
set(common
    -G "${GENERATOR}"
    "-DCMAKE_C_COMPILER=${C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}"
    "-DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}"
    -DCMAKE_BUILD_TYPE=Release
    "-DCMAKE_INSTALL_PREFIX=${INSTALL}"
    -DCMAKE_INSTALL_LIBDIR=lib
    "-DCMAKE_PREFIX_PATH=${INSTALL}"
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DBUILD_SHARED_LIBS=OFF
    -DCMAKE_C_VISIBILITY_PRESET=hidden
    -DCMAKE_CXX_VISIBILITY_PRESET=hidden
    -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5)
set(options_zlib -DZLIB_BUILD_SHARED=OFF -DZLIB_BUILD_STATIC=ON -DZLIB_BUILD_TESTING=OFF)
# Blosc compresses with its own LZ4 and Zstd and with the zlib above (not a copy of its own).
set(options_blosc
    -DBUILD_SHARED=OFF
    -DBUILD_STATIC=ON
    -DBUILD_TESTS=OFF
    -DBUILD_FUZZERS=OFF
    -DBUILD_BENCHMARKS=OFF
    -DDEACTIVATE_SNAPPY=ON
    -DPREFER_EXTERNAL_ZLIB=ON
    "-DZLIB_ROOT=${INSTALL}"
    -DZLIB_USE_STATIC_LIBS=ON)
set(options_onetbb
    -DTBB_TEST=OFF
    -DTBB_EXAMPLES=OFF
    -DTBB_STRICT=OFF
    -DTBBMALLOC_BUILD=OFF
    -DTBB_ENABLE_IPO=OFF
    -DTBB_DISABLE_HWLOC_AUTOMATIC_SEARCH=ON)
# The core library alone, without Boost (delayed loading) or explicit instantiation (which
# slows its build), and NanoVDB's headers without its tools.
set(options_openvdb
    -DOPENVDB_BUILD_CORE=ON
    -DOPENVDB_CORE_SHARED=OFF
    -DOPENVDB_CORE_STATIC=ON
    -DOPENVDB_BUILD_BINARIES=OFF
    -DOPENVDB_BUILD_VDB_PRINT=OFF
    -DOPENVDB_BUILD_PYTHON_MODULE=OFF
    -DOPENVDB_BUILD_UNITTESTS=OFF
    -DOPENVDB_BUILD_DOCS=OFF
    -DOPENVDB_USE_DELAYED_LOADING=OFF
    -DUSE_EXPLICIT_INSTANTIATION=OFF
    -DOPENVDB_BUILD_NANOVDB=ON
    -DNANOVDB_BUILD_TOOLS=OFF
    -DNANOVDB_USE_OPENVDB=OFF
    -DOPENVDB_ENABLE_UNINSTALL=OFF
    -DOPENVDB_FUTURE_DEPRECATION=OFF
    -DUSE_PKGCONFIG=OFF
    -DUSE_CCACHE=OFF
    "-DTBB_ROOT=${INSTALL}"
    "-DBLOSC_ROOT=${INSTALL}"
    "-DZLIB_ROOT=${INSTALL}"
    -DTBB_USE_STATIC_LIBS=ON
    -DBLOSC_USE_STATIC_LIBS=ON
    -DZLIB_USE_STATIC_LIBS=ON)

# Stops with the end of `log`, so that CI shows why a step failed.
function(fail_with_log what log)
    file(READ "${log}" text)
    string(LENGTH "${text}" length)
    if(length GREATER 8000)
        math(EXPR start "${length} - 8000")
        string(SUBSTRING "${text}" ${start} -1 text)
    endif()
    message(FATAL_ERROR "${what} failed; the end of ${log}:\n${text}")
endfunction()

string(REPLACE "^" ";" sources "${SOURCES}")
foreach(source IN LISTS sources)
    string(REPLACE "|" ";" fields "${source}")
    list(GET fields 0 name)
    list(GET fields 1 url)
    list(GET fields 2 hash)
    set(archive "${WORK}/download/${name}.tar.gz")
    file(DOWNLOAD "${url}" "${archive}" EXPECTED_HASH ${hash} STATUS status)
    list(GET status 0 code)
    if(NOT code EQUAL 0)
        message(FATAL_ERROR "Cannot download ${url}: ${status}")
    endif()
    set(source_dir "${WORK}/src/${name}")
    file(REMOVE_RECURSE "${source_dir}" "${WORK}/build/${name}")
    file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${source_dir}")
    # The archive's single top-level directory.
    file(GLOB top LIST_DIRECTORIES true "${source_dir}/*")
    message(STATUS "Building ${name}")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -S "${top}" -B "${WORK}/build/${name}" ${common}
                ${options_${name}}
        OUTPUT_FILE "${WORK}/${name}-configure.log"
        ERROR_FILE "${WORK}/${name}-configure.log"
        RESULT_VARIABLE result)
    if(NOT result EQUAL 0)
        fail_with_log("Configuring ${name}" "${WORK}/${name}-configure.log")
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" --build "${WORK}/build/${name}" --config Release --parallel
                ${JOBS} --target install
        OUTPUT_FILE "${WORK}/${name}-build.log"
        ERROR_FILE "${WORK}/${name}-build.log"
        RESULT_VARIABLE result)
    if(NOT result EQUAL 0)
        fail_with_log("Building ${name}" "${WORK}/${name}-build.log")
    endif()
endforeach()
file(TOUCH "${INSTALL}/.built")
