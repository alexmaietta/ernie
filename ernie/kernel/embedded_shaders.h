// SPDX-License-Identifier: BSD-3-Clause
#pragma once

/// The Slang shaders, embedded at build time
/// (ernie/kernel/CMakeLists.txt) as null-terminated source strings.
namespace ernie::shaders {

/// Shared reflectance-model module (`brdf`) and the built-in models.
extern const char kBrdfSlang[];
extern const char kBrdfLambertSlang[];
extern const char kBrdfLommelSeeligerSlang[];
extern const char kBrdfLunarLambertSlang[];
extern const char kBrdfOrenNayarSlang[];
extern const char kBrdfHapkeSlang[];
extern const char kBrdfPreviewSurfaceSlang[];

/// Per-frame resources and helpers every raster pass shares.
extern const char kFrameSlang[];
/// The camera projections, each loaded as module `projection`: frame
/// cameras' and pushbrooms'.
extern const char kProjectionPinholeSlang[];
extern const char kProjectionPushbroomSlang[];

/// Spectral shading, sensor filters and the present pass.
extern const char kShadingSlang[];
extern const char kSensorSlang[];
extern const char kPresentSlang[];
/// The sensor's lens projections (pinhole, Brown-Conrady, fisheye).
extern const char kLensSlang[];
/// The cube faces a wide camera's views are rendered on.
extern const char kCubeFacesSlang[];
/// Image pyramids for spatial sensor filters that read far around a
/// pixel.
extern const char kPyramidSlang[];

/// The standard sensor filters (`StandardSensorFilters`).
extern const char kFilterBayerSlang[];
extern const char kFilterBloomingSlang[];
extern const char kFilterColorMatrixSlang[];
extern const char kFilterCompandSlang[];
extern const char kFilterDefocusSlang[];
extern const char kFilterCompandTableSlang[];
extern const char kFilterCompandLookupSlang[];
extern const char kFilterDarkCurrentSlang[];
extern const char kFilterDefectPixelsSlang[];
extern const char kFilterDemosaicSlang[];
extern const char kFilterJpegSlang[];
extern const char kFilterSmearSlang[];
extern const char kFilterRadiationSlang[];
extern const char kFilterCommsErrorsSlang[];
extern const char kFilterGaussianPsfSlang[];
extern const char kFilterDetectorPatternSlang[];
extern const char kFilterFlareSlang[];
extern const char kFilterGlareSlang[];
extern const char kFilterLagSlang[];
extern const char kFilterMeasuredQeSlang[];
extern const char kFilterPsfSlang[];
extern const char kFilterReadoutTapsSlang[];
extern const char kFilterReadoutTableSlang[];
extern const char kFilterTdiSlang[];
extern const char kFilterTemporalAverageSlang[];
extern const char kFilterSpectralTransmissionSlang[];
extern const char kFilterStrayLightSlang[];
extern const char kFilterToneCurveSlang[];
extern const char kFilterWhiteBalanceSlang[];

/// Debug drawing over the display image.
extern const char kOverlaySlang[];
/// Overlay batches' items sorted by world, for batched views.
extern const char kOverlaySortSlang[];

/// The infinite ground plane's pass.
extern const char kGroundSlang[];
/// Point clouds as ray-cast spheres.
extern const char kPointsSlang[];

/// Hierarchical depth (Hi-Z) pyramid for occlusion culling.
extern const char kHzbSlang[];
extern const char kViewsSlang[];
extern const char kVirtualShadowSlang[];
extern const char kVirtualShadowCasterSlang[];

/// The visible depth range of a view, for sample distribution shadow
/// maps.
extern const char kDepthRangeSlang[];

/// A view's log-luminance histogram, for exposure metering.
extern const char kMeterSlang[];

/// Motion blur and rolling shutter: the sub-frames' radiance summed
/// per row's exposure.
extern const char kMotionSlang[];
/// The MotionVector var's camera pass: each pixel's motion from the
/// camera's alone.
extern const char kMotionVectorsSlang[];

/// GPU instance culling for mesh draws.
extern const char kAtmosphereSlang[];
extern const char kAtmosphereLutsSlang[];
extern const char kCullSlang[];

/// Mesh and PointInstancer passes.
extern const char kMeshSlang[];
/// Vertex normals of meshes whose points move on the GPU.
extern const char kMeshNormalsSlang[];
/// Instance poses composed on the GPU from transform ops and parents.
extern const char kInstanceComposeSlang[];
/// Bounds of instancers' positions written from the compute device.
extern const char kInstanceBoundsSlang[];
/// Indirect draws of meshes' triangles in use.
extern const char kActiveDrawsSlang[];
/// Views and point clouds posed under prims posed on the GPU.
extern const char kInstanceAnchorsSlang[];
/// Lights posed on the GPU: their light table entries, shadow views
/// and those views' cull views.
extern const char kLightViewsSlang[];
/// Minimal shading's outputs, from its attachments into linear
/// buffers.
extern const char kMinimalResolveSlang[];

/// Terrain modules: shared resources and grid vertex, the surface
/// material, and the passes.
extern const char kTerrainSlang[];
extern const char kSurfaceSlang[];
extern const char kTerrainShadowSlang[];
extern const char kTerrainColorSlang[];
/// Gradients and LOD leaf ranges of heightfields written on the GPU.
extern const char kTerrainUpdateSlang[];
/// Horizon maps of heights, for dome occlusion over terrain.
extern const char kTerrainHorizonSlang[];

/// Planet modules: shared resources and cube-sphere grid vertex, and
/// the passes.
extern const char kPlanetSlang[];
extern const char kPlanetColorSlang[];
extern const char kPlanetShadowSlang[];

/// The bindless heap's arrays (`BindlessHeap`), module `bindless`.
extern const char kBindlessSlang[];

/// The volume march (`VolumePass`), and PNanoVDB's reader
/// (third_party/nanovdb) with what precedes it, which the march is
/// compiled after.
extern const char kVolumeSlang[];
extern const char kPNanoVdbPreludeSlang[];
extern const char kPNanoVdbHeader[];

}  // namespace ernie::shaders
