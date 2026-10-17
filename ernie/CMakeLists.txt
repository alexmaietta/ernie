# SPDX-License-Identifier: BSD-3-Clause

# One target per layer; a layer links only the layers below it.
add_subdirectory(core)
add_subdirectory(physics)
add_subdirectory(io)
add_subdirectory(scene)
add_subdirectory(device)
add_subdirectory(graph)
add_subdirectory(render)

add_library(ernie INTERFACE)
add_library(Ernie::ernie ALIAS ernie)
target_link_libraries(ernie INTERFACE ernie_render)

if(ERNIE_BUILD_APP)
    add_subdirectory(app)
endif()
if(ERNIE_BUILD_PYTHON)
    add_subdirectory(python)
endif()
