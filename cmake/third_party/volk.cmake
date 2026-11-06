# SPDX-License-Identifier: BSD-3-Clause
set(VULKAN_HEADERS_INSTALL_DIR "${vulkan_headers_SOURCE_DIR}" CACHE PATH "" FORCE)
FetchContent_Declare(
    volk
    URL
        https://github.com/zeux/volk/archive/refs/tags/vulkan-sdk-1.4.363.0.tar.gz
    URL_HASH SHA256=1547d8d74395d4048fb3f4a6313da56db626e89b55db238d0d7e8944c8a645f3
)
FetchContent_MakeAvailable(volk)
