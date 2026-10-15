// SPDX-License-Identifier: BSD-3-Clause
/// @file embedded_shaders.h
/// @brief Ernie's Slang shaders, embedded at build time as source.
///
/// ernie/kernel/CMakeLists.txt lists the files; each is a
/// null-terminated string below and, for .slang files, a module in
/// `Modules()` named by its file stem.
#pragma once

#include <span>

namespace ernie::shaders {

/// @brief A built-in Slang module.
struct EmbeddedModule {
    /// The name other modules import it by: its file stem.
    const char* name;
    /// Its source text.
    const char* source;
};

/// @brief Every embedded .slang file as a module.
/// @return The modules, for `ShaderCompiler` to resolve imports from.
std::span<const EmbeddedModule> Modules();

/// Shared reflectance-model module (`brdf`) and the built-in models.
extern const char kBrdfSlang[];
extern const char kBrdfLambertSlang[];
extern const char kBrdfLommelSeeligerSlang[];
extern const char kBrdfLunarLambertSlang[];
extern const char kBrdfOrenNayarSlang[];
extern const char kBrdfHapkeSlang[];
extern const char kBrdfOpenPbrSlang[];
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
extern const char kFilterAgxSlang[];
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
extern const char kFilterBolometerSlang[];
extern const char kFilterThermalDisplaySlang[];
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

/// The visible depth range of a view, for sample distribution shadow
/// maps.
extern const char kDepthRangeSlang[];
extern const char kLongRangeSlang[];

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
/// Skinned meshes posed on the GPU, and subdivision surfaces refined
/// from their posed control points.
extern const char kSkinningSlang[];
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
/// Horizons of heights towards the sun, for its shadows over terrain.
extern const char kTerrainSunHorizonSlang[];
/// The cube sphere's face mapping, for kernels outside the planet
/// passes.
extern const char kCubeSphereSlang[];
/// Planet Terrains' horizons towards each distant light.
extern const char kPlanetSunHorizonSlang[];

/// Planet modules: shared resources and cube-sphere grid vertex, and
/// the passes.
extern const char kPlanetSlang[];
extern const char kPlanetColorSlang[];
extern const char kPlanetShadowSlang[];

/// The bindless heap's arrays (`BindlessHeap`), module `bindless`.
extern const char kBindlessSlang[];
/// Vector, quaternion and matrix helpers, module `math`.
extern const char kMathSlang[];

/// The G-buffer's normal and temperature encoding, module `gbuffer`.
extern const char kGBufferSlang[];

/// The volume march (`VolumePass`), and PNanoVDB's reader
/// (third_party/nanovdb) with what precedes it, which the march is
/// compiled after.
extern const char kVolumeSlang[];
extern const char kPNanoVdbPreludeSlang[];
extern const char kPNanoVdbHeader[];
/// BC7 encoding on the GPU in modes 1, 6 and 7 (`GpuBlockEncoder`).
extern const char kBc7Slang[];
/// Lasers' beams crossed by a pixel's ray, in closed form, for the
/// media that scatter them (volume.slang, shading.slang).
extern const char kLaserBeamsSlang[];

}  // namespace ernie::shaders
