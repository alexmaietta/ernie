# The shaders, embedded as strings: ernie/kernel/embedded_shaders.h declares them and a .cpp
# generated at build time defines them, so a shader edit recompiles one file and does not re-run
# CMake. Each entry is the shader's path and the name it is embedded under.
set(ernie_shaders
    closure/brdf.slang=kBrdfSlang
    closure/hapke.slang=kBrdfHapkeSlang
    closure/lambert.slang=kBrdfLambertSlang
    closure/lommel_seeliger.slang=kBrdfLommelSeeligerSlang
    closure/lunar_lambert.slang=kBrdfLunarLambertSlang
    closure/oren_nayar.slang=kBrdfOrenNayarSlang
    film/filter_bayer.slang=kFilterBayerSlang
    film/filter_blooming.slang=kFilterBloomingSlang
    film/filter_color_matrix.slang=kFilterColorMatrixSlang
    film/filter_compand.slang=kFilterCompandSlang
    film/filter_defocus.slang=kFilterDefocusSlang
    film/filter_demosaic.slang=kFilterDemosaicSlang
    film/filter_lag.slang=kFilterLagSlang
    film/filter_psf.slang=kFilterPsfSlang
    film/filter_spectral_transmission.slang=kFilterSpectralTransmissionSlang
    film/filter_temporal_average.slang=kFilterTemporalAverageSlang
    film/filter_tone_curve.slang=kFilterToneCurveSlang
    film/filter_white_balance.slang=kFilterWhiteBalanceSlang
    film/sensor.slang=kSensorSlang
    raster/cull.slang=kCullSlang
    raster/depth_range.slang=kDepthRangeSlang
    raster/hzb.slang=kHzbSlang
    raster/frame.slang=kFrameSlang
    raster/ground.slang=kGroundSlang
    raster/mesh.slang=kMeshSlang
    raster/meter.slang=kMeterSlang
    raster/overlay.slang=kOverlaySlang
    raster/present.slang=kPresentSlang
    raster/shading.slang=kShadingSlang
    raster/surface.slang=kSurfaceSlang
    raster/terrain.slang=kTerrainSlang
    raster/terrain_color.slang=kTerrainColorSlang
    raster/terrain_shadow.slang=kTerrainShadowSlang
    raster/views.slang=kViewsSlang)
set(shader_files)
foreach(entry ${ernie_shaders})
    string(REGEX REPLACE "=.*" "" shader "${entry}")
    list(APPEND shader_files "${CMAKE_CURRENT_SOURCE_DIR}/${shader}")
endforeach()

set(embedded "${PROJECT_BINARY_DIR}/generated/ernie/kernel/embedded_shaders.cpp")
list(JOIN ernie_shaders "," shaders_arg)
add_custom_command(
    OUTPUT "${embedded}"
    COMMAND
        "${CMAKE_COMMAND}" -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} -DOUTPUT=${embedded}
        "-DSHADERS=${shaders_arg}" -P "${PROJECT_SOURCE_DIR}/cmake/EmbedShaders.cmake"
    DEPENDS ${shader_files} "${PROJECT_SOURCE_DIR}/cmake/EmbedShaders.cmake"
    COMMENT "Embedding Ernie's shaders"
    VERBATIM)

ernie_add_library(ernie_kernel SOURCES "${embedded}")
