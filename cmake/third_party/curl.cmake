# Included before the packaging (CMakeLists.txt) and by the demos.
include_guard(GLOBAL)

set(ERNIE_DEP_VERSION_curl
    curl-8_18_0
    CACHE STRING "Tag of curl to fetch when the system has no libcurl.")
mark_as_advanced(ERNIE_DEP_VERSION_curl)

# libcurl (curl license, MIT-style), for the C++ demos to fetch their data (LOLA's terrain) into
# the cache as the Python demos do. The system's when it has one (Linux distributions and macOS
# ship it), else built from source with HTTP alone, over Windows's own TLS (Schannel) or OpenSSL.
# Off, the demos read only what the cache already holds.
option(ERNIE_DEMO_DOWNLOADS "Let the C++ demos fetch their data with libcurl" ON)

if(ERNIE_DEMO_DOWNLOADS)
    find_package(CURL QUIET)
    if(NOT CURL_FOUND)
        include(FetchContent)
        set(BUILD_CURL_EXE OFF)
        set(BUILD_LIBCURL_DOCS OFF)
        set(BUILD_MISC_DOCS OFF)
        set(ENABLE_CURL_MANUAL OFF)
        set(BUILD_EXAMPLES OFF)
        set(CURL_DISABLE_INSTALL ON)
        set(HTTP_ONLY ON)
        set(CURL_USE_LIBPSL OFF)
        set(CURL_USE_LIBSSH2 OFF)
        set(USE_LIBIDN2 OFF)
        set(USE_NGHTTP2 OFF)
        set(CURL_BROTLI OFF)
        set(CURL_ZSTD OFF)
        if(WIN32)
            set(CURL_USE_SCHANNEL ON)
        else()
            set(CURL_USE_OPENSSL ON)
        endif()
        set(ERNIE_SAVED_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
        set(ERNIE_SAVED_BUILD_TESTING ${BUILD_TESTING})
        set(BUILD_SHARED_LIBS OFF)
        set(BUILD_STATIC_LIBS ON)
        set(BUILD_TESTING OFF)
        FetchContent_Declare(
            curl
            GIT_REPOSITORY https://github.com/curl/curl.git
            GIT_TAG ${ERNIE_DEP_VERSION_curl}
            GIT_SHALLOW TRUE)
        FetchContent_MakeAvailable(curl)
        set(BUILD_SHARED_LIBS ${ERNIE_SAVED_BUILD_SHARED_LIBS})
        set(BUILD_TESTING ${ERNIE_SAVED_BUILD_TESTING})
    endif()
    add_library(ernie_curl INTERFACE)
    target_link_libraries(ernie_curl INTERFACE CURL::libcurl)
    target_compile_definitions(ernie_curl INTERFACE ERNIE_DEMO_DOWNLOADS)
endif()
