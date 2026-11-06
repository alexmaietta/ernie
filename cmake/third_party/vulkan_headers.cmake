# SPDX-License-Identifier: BSD-3-Clause
FetchContent_Declare(
    vulkan_headers
    URL
        https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/vulkan-sdk-1.4.363.0.tar.gz
    URL_HASH SHA256=4a078be12bef21cfebc09d878b77a63cff9d68f899254a0b00d0e37ef73e7f7e
)
FetchContent_MakeAvailable(vulkan_headers)
