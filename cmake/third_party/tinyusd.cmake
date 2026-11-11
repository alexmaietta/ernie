# SPDX-License-Identifier: BSD-3-Clause
set(TINYUSD_USE_PREBUILT
    ON
    CACHE BOOL
    "Download a prebuilt tinyusd release instead of building with cargo"
)
set(TINYUSD_STATIC ON CACHE BOOL "Link tinyusd as a static library")
set(TINYUSD_BUILD_TESTS OFF CACHE BOOL "Build tinyusd tests")
set(TINYUSD_BUILD_EXAMPLES OFF CACHE BOOL "Build tinyusd examples")
set(TINYUSD_INSTALL OFF CACHE BOOL "Generate tinyusd install rules")
FetchContent_Declare(
    tinyusd
    URL
        https://github.com/alexmaietta/tinyusd/archive/refs/tags/v0.3.0.tar.gz
    URL_HASH SHA256=6725c94c73ba24bb0bcdc3104c4e6d9c2b1460fd8a6e5daf918a9a6217c3f39d
)
FetchContent_MakeAvailable(tinyusd)

if(APPLE AND TARGET tinyusd::tinyusd)
    get_target_property(tinyusd_aliased tinyusd::tinyusd ALIASED_TARGET)
    if(NOT tinyusd_aliased)
        set(tinyusd_aliased tinyusd::tinyusd)
    endif()
    get_target_property(tinyusd_links ${tinyusd_aliased} INTERFACE_LINK_LIBRARIES)
    if(tinyusd_links)
        list(REMOVE_ITEM tinyusd_links System)
        set_target_properties(
            ${tinyusd_aliased}
            PROPERTIES
                INTERFACE_LINK_LIBRARIES
                    "${tinyusd_links}"
        )
    endif()
endif()
