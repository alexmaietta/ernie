# SPDX-License-Identifier: BSD-3-Clause
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
    closure/preview_surface.slang=kBrdfPreviewSurfaceSlang
    film/cube_faces.slang=kCubeFacesSlang
    film/filter_bayer.slang=kFilterBayerSlang
    film/filter_blooming.slang=kFilterBloomingSlang
    film/filter_color_matrix.slang=kFilterColorMatrixSlang
    film/filter_compand.slang=kFilterCompandSlang
    film/filter_defocus.slang=kFilterDefocusSlang
    film/filter_compand_table.slang=kFilterCompandTableSlang
    film/filter_dark_current.slang=kFilterDarkCurrentSlang
    film/filter_defect_pixels.slang=kFilterDefectPixelsSlang
    film/filter_demosaic.slang=kFilterDemosaicSlang
    film/filter_detector_pattern.slang=kFilterDetectorPatternSlang
    film/filter_flare.slang=kFilterFlareSlang
    film/filter_glare.slang=kFilterGlareSlang
    film/filter_jpeg.slang=kFilterJpegSlang
    film/filter_lag.slang=kFilterLagSlang
    film/filter_measured_qe.slang=kFilterMeasuredQeSlang
    film/filter_psf.slang=kFilterPsfSlang
    film/filter_readout_taps.slang=kFilterReadoutTapsSlang
    film/filter_smear.slang=kFilterSmearSlang
    film/filter_spectral_transmission.slang=kFilterSpectralTransmissionSlang
    film/filter_stray_light.slang=kFilterStrayLightSlang
    film/filter_tdi.slang=kFilterTdiSlang
    film/filter_temporal_average.slang=kFilterTemporalAverageSlang
    film/filter_tone_curve.slang=kFilterToneCurveSlang
    film/filter_white_balance.slang=kFilterWhiteBalanceSlang
    film/lens.slang=kLensSlang
    film/pyramid.slang=kPyramidSlang
    film/sensor.slang=kSensorSlang
    raster/cull.slang=kCullSlang
    raster/depth_range.slang=kDepthRangeSlang
    raster/hzb.slang=kHzbSlang
    raster/instance_compose.slang=kInstanceComposeSlang
    raster/instance_anchors.slang=kInstanceAnchorsSlang
    raster/light_views.slang=kLightViewsSlang
    raster/minimal_resolve.slang=kMinimalResolveSlang
    raster/frame.slang=kFrameSlang
    raster/ground.slang=kGroundSlang
    raster/mesh.slang=kMeshSlang
    raster/mesh_normals.slang=kMeshNormalsSlang
    raster/meter.slang=kMeterSlang
    raster/motion.slang=kMotionSlang
    raster/motion_vectors.slang=kMotionVectorsSlang
    raster/overlay.slang=kOverlaySlang
    raster/planet.slang=kPlanetSlang
    raster/planet_color.slang=kPlanetColorSlang
    raster/planet_shadow.slang=kPlanetShadowSlang
    raster/points.slang=kPointsSlang
    raster/present.slang=kPresentSlang
    raster/projection_pinhole.slang=kProjectionPinholeSlang
    raster/projection_pushbroom.slang=kProjectionPushbroomSlang
    raster/shading.slang=kShadingSlang
    raster/surface.slang=kSurfaceSlang
    raster/terrain.slang=kTerrainSlang
    raster/terrain_color.slang=kTerrainColorSlang
    raster/terrain_shadow.slang=kTerrainShadowSlang
    raster/terrain_update.slang=kTerrainUpdateSlang
    raster/terrain_horizon.slang=kTerrainHorizonSlang
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
    COMMAND "${CMAKE_COMMAND}" -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} -DOUTPUT=${embedded}
            "-DSHADERS=${shaders_arg}" -P "${PROJECT_SOURCE_DIR}/cmake/EmbedShaders.cmake"
    DEPENDS ${shader_files} "${PROJECT_SOURCE_DIR}/cmake/EmbedShaders.cmake"
    COMMENT "Embedding Ernie's shaders"
    VERBATIM)

ernie_add_library(ernie_kernel SOURCES "${embedded}")
