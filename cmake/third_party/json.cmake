# SPDX-License-Identifier: BSD-3-Clause
FetchContent_Declare(
    nlohmann_json
    URL
        https://github.com/nlohmann/json/releases/download/v3.12.0/include.zip
    URL_HASH SHA256=b8cb0ef2dd7f57f18933997c9934bb1fa962594f701cd5a8d3c2c80541559372
    SOURCE_SUBDIR none
)
FetchContent_MakeAvailable(nlohmann_json)

add_library(ernie_json INTERFACE)
target_include_directories(ernie_json SYSTEM INTERFACE "${nlohmann_json_SOURCE_DIR}/single_include")
