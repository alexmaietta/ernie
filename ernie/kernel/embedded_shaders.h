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

/// Per-frame resources and helpers every raster pass shares.
extern const char kFrameSlang[];

/// Spectral shading, sensor filters and the present pass.
extern const char kShadingSlang[];
extern const char kSensorSlang[];
extern const char kPresentSlang[];

/// The infinite ground plane's pass.
extern const char kGroundSlang[];

/// Hierarchical depth (Hi-Z) pyramid for occlusion culling.
extern const char kHzbSlang[];
extern const char kViewsSlang[];

/// GPU instance culling for mesh draws.
extern const char kCullSlang[];

/// Mesh and PointInstancer passes.
extern const char kMeshSlang[];

/// Terrain modules: shared resources and grid vertex, the surface material, and the passes.
extern const char kTerrainSlang[];
extern const char kSurfaceSlang[];
extern const char kTerrainShadowSlang[];
extern const char kTerrainColorSlang[];

}  // namespace ernie::shaders
