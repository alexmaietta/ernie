# SPDX-License-Identifier: BSD-3-Clause
set(SIMULATOR_SRCS
    types.hpp
    sim.hpp
    sim.inl
    sim.cpp
)

add_library(ernie_bench_madrona_cpu_impl STATIC ${SIMULATOR_SRCS})

target_link_libraries(
    ernie_bench_madrona_cpu_impl
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_common
        madrona_rendering_system
)

add_library(
    ernie_bench_madrona_mgr
    SHARED
    mgr.hpp
    mgr.cpp
)

target_link_libraries(
    ernie_bench_madrona_mgr
    PRIVATE
        ernie_bench_madrona_cpu_impl
        madrona_mw_cpu
        madrona_common
        madrona_importer
        madrona_render
        madrona_render_asset_processor
)

madrona_build_compile_defns(
    OUT_TARGET
        ernie_bench_madrona_gpu_srcs
    SOURCES_DEFN
        GPU_SIM_SRC_LIST
    FLAGS_DEFN
        GPU_SIM_COMPILE_FLAGS
    SRCS
        ${SIMULATOR_SRCS}
)

target_link_libraries(
    ernie_bench_madrona_mgr
    PRIVATE
        ernie_bench_madrona_gpu_srcs
        madrona_mw_gpu
        madrona_cuda
)

madrona_python_module(_madrona_bench
    bindings.cpp
)

target_link_libraries(_madrona_bench PRIVATE ernie_bench_madrona_mgr)
