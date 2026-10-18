# SPDX-License-Identifier: BSD-3-Clause
set(ERNIE_DEP_VERSION_json v3.12.0 CACHE STRING "Release of nlohmann/json to fetch")
mark_as_advanced(ERNIE_DEP_VERSION_json)

# The release's single-include archive, without the repository's tests and history.
ernie_fetch(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/${ERNIE_DEP_VERSION_json}/include.zip
)

add_library(ernie_json INTERFACE)
target_include_directories(ernie_json SYSTEM INTERFACE "${nlohmann_json_SOURCE_DIR}/single_include")
