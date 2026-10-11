set(ERNIE_DEP_VERSION_volk
    vulkan-sdk-1.4.363.0
    CACHE STRING "Tag of volk to fetch.")
mark_as_advanced(ERNIE_DEP_VERSION_volk)

include(FetchContent)

# volk's own build is not used; it is one C file compiled here against the fetched headers.
FetchContent_Declare(
    volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG ${ERNIE_DEP_VERSION_volk}
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR no-build)
FetchContent_MakeAvailable(volk)

# Loads the Vulkan loader (or MoltenVK) at run time, so nothing links against libvulkan.
add_library(ernie_volk STATIC "${volk_SOURCE_DIR}/volk.c")
target_include_directories(ernie_volk SYSTEM PUBLIC "${volk_SOURCE_DIR}")
target_compile_definitions(ernie_volk PUBLIC VK_NO_PROTOTYPES)
if(WIN32)
    # Vulkan's Win32 types and entry points (VK_KHR_external_memory_win32 and
    # VK_KHR_external_semaphore_win32, for CUDA interop), which bring in <windows.h>: lean, and
    # without its min and max macros.
    target_compile_definitions(ernie_volk PUBLIC VK_USE_PLATFORM_WIN32_KHR WIN32_LEAN_AND_MEAN
                                                 NOMINMAX)
endif()
target_link_libraries(ernie_volk PUBLIC ernie_vulkan_headers ${CMAKE_DL_LIBS})
set_target_properties(ernie_volk PROPERTIES POSITION_INDEPENDENT_CODE ON)
