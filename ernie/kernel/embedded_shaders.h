#pragma once

/// The Slang shaders, embedded at build time (ernie/kernel/CMakeLists.txt) as null-terminated
/// source strings.
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
/// The camera projections, each loaded as module `projection`: frame cameras' and pushbrooms'.
extern const char kProjectionPinholeSlang[];
extern const char kProjectionPushbroomSlang[];

/// Spectral shading, sensor filters and the present pass.
extern const char kShadingSlang[];
extern const char kSensorSlang[];
extern const char kPresentSlang[];
/// The sensor's lens projections (pinhole, Brown-Conrady, fisheye).
extern const char kLensSlang[];
/// Image pyramids for spatial sensor filters that read far around a pixel.
extern const char kPyramidSlang[];

/// The standard sensor filters (`StandardSensorFilters`).
extern const char kFilterBayerSlang[];
extern const char kFilterBloomingSlang[];
extern const char kFilterColorMatrixSlang[];
extern const char kFilterCompandSlang[];
extern const char kFilterDefocusSlang[];
extern const char kFilterCompandTableSlang[];
extern const char kFilterDarkCurrentSlang[];
extern const char kFilterDefectPixelsSlang[];
extern const char kFilterDemosaicSlang[];
extern const char kFilterJpegSlang[];
extern const char kFilterSmearSlang[];
extern const char kFilterDetectorPatternSlang[];
extern const char kFilterGlareSlang[];
extern const char kFilterLagSlang[];
extern const char kFilterMeasuredQeSlang[];
extern const char kFilterPsfSlang[];
extern const char kFilterReadoutTapsSlang[];
extern const char kFilterTdiSlang[];
extern const char kFilterTemporalAverageSlang[];
extern const char kFilterSpectralTransmissionSlang[];
extern const char kFilterToneCurveSlang[];
extern const char kFilterWhiteBalanceSlang[];

/// Debug drawing over the display image.
extern const char kOverlaySlang[];

/// The infinite ground plane's pass.
extern const char kGroundSlang[];
/// Point clouds as ray-cast spheres.
extern const char kPointsSlang[];


/// Hierarchical depth (Hi-Z) pyramid for occlusion culling.
extern const char kHzbSlang[];
extern const char kViewsSlang[];

/// The visible depth range of a view, for sample distribution shadow maps.
extern const char kDepthRangeSlang[];

/// A view's log-luminance histogram, for exposure metering.
extern const char kMeterSlang[];

/// Motion blur and rolling shutter: the sub-frames' radiance summed per row's exposure.
extern const char kMotionSlang[];
/// The MotionVector var's camera pass: each pixel's motion from the camera's alone.
extern const char kMotionVectorsSlang[];

/// GPU instance culling for mesh draws.
extern const char kCullSlang[];

/// Mesh and PointInstancer passes.
extern const char kMeshSlang[];
/// Vertex normals of meshes whose points move on the GPU.
extern const char kMeshNormalsSlang[];

/// Terrain modules: shared resources and grid vertex, the surface material, and the passes.
extern const char kTerrainSlang[];
extern const char kSurfaceSlang[];
extern const char kTerrainShadowSlang[];
extern const char kTerrainColorSlang[];
/// Gradients and LOD leaf ranges of heightfields written on the GPU.
extern const char kTerrainUpdateSlang[];

/// Planet modules: shared resources and cube-sphere grid vertex, and the passes.
extern const char kPlanetSlang[];
extern const char kPlanetColorSlang[];
extern const char kPlanetShadowSlang[];

}  // namespace ernie::shaders
