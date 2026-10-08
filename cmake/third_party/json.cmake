set(ERNIE_DEP_VERSION_json
    v3.12.0
    CACHE STRING "Release of nlohmann/json to fetch.")
mark_as_advanced(ERNIE_DEP_VERSION_json)

# nlohmann/json (MIT), header only, for reading JSON such as cameras' image support data. The
# release's single-include archive, without the repository's tests and history.
include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/${ERNIE_DEP_VERSION_json}/include.zip)
FetchContent_MakeAvailable(nlohmann_json)

add_library(ernie_json INTERFACE)
target_include_directories(ernie_json SYSTEM INTERFACE "${nlohmann_json_SOURCE_DIR}/single_include")
