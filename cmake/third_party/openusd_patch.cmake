# SPDX-License-Identifier: BSD-3-Clause
# Patches the headers of an OpenUSD Ernie built (cmake -P, before the plugin compiles), once each:
# GCC before 14 rejects a template-id naming a constructor in C++20 (CWG 2237), which
# retainedDataSource.h's HdRetainedTypedSampledDataSource<bool> specialization does.
#
# Takes USD_INSTALL.

set(header "${USD_INSTALL}/include/pxr/imaging/hd/retainedDataSource.h")
if(EXISTS "${header}")
    file(READ "${header}" text)
    string(REPLACE "HdRetainedTypedSampledDataSource<bool>(const bool& value)"
                   "HdRetainedTypedSampledDataSource(const bool& value)" patched "${text}")
    if(NOT patched STREQUAL text)
        file(WRITE "${header}" "${patched}")
    endif()
endif()
