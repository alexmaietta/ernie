# SPDX-License-Identifier: BSD-3-Clause
add_executable(
    ernie_benchmarks
    main.cpp
    app/viewer_frame.cpp
    device/pose_step.cpp
    render/batch_step.cpp
    render/hide_seek.cpp
    render/shared_prims.cpp
    render/transform_graph.cpp
    scene/extents.cpp
    scene/storage.cpp
)
target_link_libraries(
    ernie_benchmarks
    PRIVATE ernie_viewer ernie_static ernie_volk ernie_vma ernie_build_options
)
