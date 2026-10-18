# SPDX-License-Identifier: BSD-3-Clause
add_executable(ernie_benchmarks main.cpp device/pose_step.cpp scene/extents.cpp scene/storage.cpp)
target_link_libraries(
    ernie_benchmarks
    PRIVATE ernie_static ernie_volk ernie_vma ernie_build_options
)
