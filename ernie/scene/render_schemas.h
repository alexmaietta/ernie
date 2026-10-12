// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/base_schemas.h"
#include "ernie/scene/prim_type.h"
#include "ernie/scene/sensor_filters.h"
#include "ernie/scene/tokens.h"

#include <string>
#include <vector>

/// Render prims as reflected schemas (`prim_type.h`), following
/// UsdRender: `RenderSettings` (the rasterizer's options under
/// `ernie:`) and `RenderProduct` share `RenderSettingsBase` (camera,
/// resolution and what a product may override of its settings), a
/// `RenderVar` names an output, and Ernie's `SensorFilter` is one
/// step of a product's or var's sensor chain. Each declaration's
/// fallback is the value Ernie renders an unset attribute with, the
/// default of the `FrameParams`, `ShadowSettings` or `LodSettings`
/// field it sets.
///
/// The relationships (`camera`, `products`, `ernie:lights`,
/// `orderedVars`, `ernie:filters`) are lists of targets; `camera` is
/// one target, or on a product a batch of cameras. Not declared: a
/// SensorFilter's `inputs:*`, which its filter type's model
/// (`SensorFilterRegistry`) defines.
///
/// References:
/// - Pixar, UsdRender,
///   https://openusd.org/release/api/usd_render_page_front.html
/// - NVIDIA, "RTX - Minimal Mode", Omniverse Materials and Rendering
///   documentation,
///   https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_minimal.html

// clang-format off
#define ERNIE_RENDER_SETTINGS_BASE_ATTRIBUTES(X)                                                   \
    X(std::vector<::ernie::Path>, Camera, ::ernie::tokens::kCameraRel,                             \
      std::vector<::ernie::Path>{},                                                                \
      (::ernie::AttributeMeta{.doc = "The camera rendered, or a product's batch of cameras (one "  \
                                     "view each); unset, the settings' camera, else the scene's "  \
                                     "first.",                                                     \
                              .group = "Image", .options = {::ernie::tokens::kCamera},             \
                              .plural = "cameras"}))                                               \
    X(glm::dvec2, Resolution, ::ernie::tokens::kResolution, glm::dvec2(2048.0, 1080.0),            \
      (::ernie::AttributeMeta{.doc = "Pixels, width and height; a product's, unset, its "          \
                                     "settings'.",                                                 \
                              .group = "Image"}))                                                  \
    X(std::string, Overlays, ::ernie::tokens::kOverlays, "",                                       \
      (::ernie::AttributeMeta{.doc = "\"*\" or names of overlays drawn over LdrColor; the "        \
                                     "viewport shows every overlay unless set.",                   \
                              .group = "Image"}))                                                  \
    X(std::string, MinimalMode, ::ernie::tokens::kMinimalMode, ::ernie::tokens::kOff,              \
      (::ernie::AttributeMeta{.label = "Minimal mode",                                             \
                              .doc = "Minimal shading for throughput: no colour (\"none\"), one "  \
                                     "colour or surfaces' diffuse colours, lit by the sun and "    \
                                     "ambient, with glossy highlights and emission or not, "       \
                                     "without sensor filters.",                                    \
                              .group = "Minimal",                                                  \
                              .options = {::ernie::tokens::kOff, ::ernie::tokens::kNone,           \
                                          ::ernie::tokens::kConstantDiffuse,                       \
                                          ::ernie::tokens::kTexturedDiffuse,                       \
                                          ::ernie::tokens::kDiffuseGlossyEmission}}))              \
    X(glm::dvec3, MinimalConstantColor, ::ernie::tokens::kMinimalConstantColor, glm::dvec3(1.0),   \
      (::ernie::AttributeMeta{.label = "Constant colour", .doc = "Linear RGB of constantDiffuse.", \
                              .group = "Minimal", .color = true}))

#define ERNIE_RENDER_SETTINGS_ATTRIBUTES(X)                                                        \
    X(std::vector<::ernie::Path>, Products, ::ernie::tokens::kProducts,                            \
      std::vector<::ernie::Path>{},                                                                \
      (::ernie::AttributeMeta{.doc = "The RenderProducts rendered; none, one image of the "        \
                                     "settings' own.",                                             \
                              .group = "Image", .options = {::ernie::tokens::kRenderProduct}}))    \
    X(std::vector<::ernie::Path>, Lights, ::ernie::tokens::kLightsRel,                             \
      std::vector<::ernie::Path>{},                                                                \
      (::ernie::AttributeMeta{.doc = "The lights rendered, when set (empty: none); unset, every "  \
                                     "visible light.",                                             \
                              .group = "Image"}))                                                  \
    X(double, Ambient, ::ernie::tokens::kAmbient, 0.0,                                             \
      (::ernie::AttributeMeta{.doc = "Luminance (cd/m^2) of light reaching every surface "        \
                                     "uniformly from all around, besides the lights; 0 none. "     \
                                     "A DomeLight gives the same, with a map.",                    \
                              .group = "Image", .max = 1e5}))                                      \
    X(glm::dvec3, AmbientColor, ::ernie::tokens::kAmbientColor, glm::dvec3(1.0),                   \
      (::ernie::AttributeMeta{.label = "Ambient colour",                                           \
                              .doc = "Linear RGB of the ambient light, whose luminance "           \
                                     "`ernie:ambient` sets whatever the colour's brightness.",     \
                              .group = "Image", .color = true}))                                   \
    X(glm::dvec3, ClearColor, ::ernie::tokens::kClearColor, glm::dvec3(0.0),                       \
      (::ernie::AttributeMeta{.label = "Background", .doc = "Linear display colour of LdrColor's " \
                                                            "background.",                         \
                              .group = "Image", .color = true}))                                   \
    X(std::string, Wireframe, ::ernie::tokens::kWireframe, ::ernie::tokens::kOff,                  \
      (::ernie::AttributeMeta{.doc = "Triangle edges over shaded surfaces, or on flat unlit ones; " \
                                     "a bool is \"overlay\" or \"off\". The viewport's toolbar "   \
                                     "may override it.",                                           \
                              .group = "Image",                                                    \
                              .options = {::ernie::tokens::kOff, ::ernie::tokens::kOverlay,        \
                                          ::ernie::tokens::kHiddenLine},                           \
                              .lenient = true}))                                                   \
    X(int, Frame, ::ernie::tokens::kFrame, 0,                                                      \
      (::ernie::AttributeMeta{.doc = "The frame noise derives from; counts up by itself unless "   \
                                     "set.",                                                       \
                              .group = "Image"}))                                                  \
    X(double, ExposureCompensation, ::ernie::tokens::kExposureCompensation, 0.0,                   \
      (::ernie::AttributeMeta{.label = "Exposure", .doc = "Stops on the photographic outputs, on " \
                                                          "top of the camera's exposure.",         \
                              .group = "Image", .min = -16.0, .max = 16.0}))                       \
    X(bool, Meter, ::ernie::tokens::kMeter, false,                                                 \
      (::ernie::AttributeMeta{.doc = "Measure each frame's luminance, for automatic exposure and " \
                                     "meters.",                                                    \
                              .group = "Image"}))                                                  \
    X(std::string, Culling, ::ernie::tokens::kCulling, ::ernie::tokens::kGpu,                      \
      (::ernie::AttributeMeta{.doc = "What lies outside a view is skipped: whole draws (cpu), "    \
                                     "then every instance (gpu).",                                 \
                              .group = "Culling and LOD",                                          \
                              .options = {::ernie::tokens::kOff, ::ernie::tokens::kCpu,            \
                                          ::ernie::tokens::kGpu}}))                                \
    X(std::string, Occlusion, ::ernie::tokens::kOcclusion, ::ernie::tokens::kOff,                  \
      (::ernie::AttributeMeta{.doc = "With GPU culling, also skip instances hidden behind "        \
                                     "occluders, or behind instances seen last frame (twoPass).",  \
                              .group = "Culling and LOD",                                          \
                              .options = {::ernie::tokens::kOff, ::ernie::tokens::kOccluders,      \
                                          ::ernie::tokens::kTwoPass}}))                            \
    X(double, LodMeshError, ::ernie::tokens::kLodMeshError, 1.0,                                   \
      (::ernie::AttributeMeta{.label = "Mesh LOD error",                                           \
                              .doc = "Pixels of error a coarser mesh level may add.",              \
                              .group = "Culling and LOD", .max = 16.0}))                           \
    X(bool, LodEnable, ::ernie::tokens::kLodEnable, true,                                          \
      (::ernie::AttributeMeta{.label = "Terrain LOD", .group = "Culling and LOD"}))                \
    X(double, LodDetail, ::ernie::tokens::kLodDetail, 4.0,                                         \
      (::ernie::AttributeMeta{.label = "Terrain detail",                                           \
                              .doc = "First LOD range in level-0 node widths.",                    \
                              .group = "Culling and LOD", .min = 1.0, .max = 16.0}))               \
    X(double, LodMorphStart, ::ernie::tokens::kLodMorphStart, 0.67,                                \
      (::ernie::AttributeMeta{.label = "Morph start",                                              \
                              .doc = "Fraction of each level's band before morphing starts.",      \
                              .group = "Culling and LOD", .max = 1.0}))                            \
    X(bool, LodFreeze, ::ernie::tokens::kLodFreeze, false,                                         \
      (::ernie::AttributeMeta{.label = "Freeze LOD", .group = "Culling and LOD"}))                 \
    X(bool, LodShowLevels, ::ernie::tokens::kLodShowLevels, false,                                 \
      (::ernie::AttributeMeta{.label = "Show LOD levels", .group = "Culling and LOD"}))            \
    X(std::string, ShadowFit, ::ernie::tokens::kShadowFit, ::ernie::tokens::kScene,                \
      (::ernie::AttributeMeta{.label = "Fit",                                                      \
                              .doc = "scene: the casters' bounds, an image the same every time; " \
                                     "visible: sample distribution shadow maps, from the depths a " \
                                     "recent frame saw.",                                           \
                              .group = "Shadows",                                                  \
                              .options = {::ernie::tokens::kScene, ::ernie::tokens::kVisible}}))   \
    X(int, ShadowCascades, ::ernie::tokens::kShadowCascades, 3,                                    \
      (::ernie::AttributeMeta{.label = "Cascades", .group = "Shadows", .min = 1.0, .max = 4.0}))   \
    X(int, ShadowResolution, ::ernie::tokens::kShadowResolution, 2048,                             \
      (::ernie::AttributeMeta{.label = "Resolution", .doc = "Texels along each cascade's side.",   \
                              .group = "Shadows", .min = 256.0, .max = 8192.0}))                   \
    X(double, ShadowSoftness, ::ernie::tokens::kShadowSoftness, 1.5,                               \
      (::ernie::AttributeMeta{.label = "Softness", .doc = "Filter radius in texels.",              \
                              .group = "Shadows", .max = 8.0}))                                    \
    X(double, ShadowNormalOffset, ::ernie::tokens::kShadowNormalOffset, 1.5,                       \
      (::ernie::AttributeMeta{.label = "Normal offset", .doc = "Texels along the normal.",         \
                              .group = "Shadows", .max = 8.0}))                                    \
    X(double, ShadowSplitLambda, ::ernie::tokens::kShadowSplitLambda, 0.8,                         \
      (::ernie::AttributeMeta{.label = "Split lambda",                                             \
                              .doc = "Logarithmic weight of the cascades' splits.",                \
                              .group = "Shadows", .max = 1.0}))                                    \
    X(bool, ShadowShowCascades, ::ernie::tokens::kShadowShowCascades, false,                       \
      (::ernie::AttributeMeta{.label = "Show cascades", .group = "Shadows"}))                      \
    X(std::string, ShadowWorlds, ::ernie::tokens::kShadowWorlds, ::ernie::tokens::kSeparate,       \
      (::ernie::AttributeMeta{.label = "Worlds",                                                   \
                              .doc = "Batches of views of several worlds: a map per world, one "   \
                                     "of what they share, or one of every world's casters.",       \
                              .group = "Shadows",                                                  \
                              .options = {::ernie::tokens::kSeparate, ::ernie::tokens::kShared,    \
                                          ::ernie::tokens::kMerged}}))                             \
    X(std::string, ShadowWorldFit, ::ernie::tokens::kShadowWorldFit, ::ernie::tokens::kScene,      \
      (::ernie::AttributeMeta{.label = "World fit",                                                \
                              .doc = "What each world's map covers: the scene, or the world's "    \
                                     "casters and the margin.",                                    \
                              .group = "Shadows",                                                  \
                              .options = {::ernie::tokens::kScene, ::ernie::tokens::kWorldFit}}))  \
    X(int, ShadowWorldResolution, ::ernie::tokens::kShadowWorldResolution, 0,                      \
      (::ernie::AttributeMeta{.label = "World resolution", .doc = "0 for automatic.",              \
                              .group = "Shadows", .max = 8192.0}))                                 \
    X(double, ShadowWorldMargin, ::ernie::tokens::kShadowWorldMargin, 1.0,                         \
      (::ernie::AttributeMeta{.label = "World margin", .doc = "Metres past the world's casters.",  \
                              .group = "Shadows", .max = 100.0}))                                  \
    X(double, VolumeStepsPerVoxel, ::ernie::tokens::kVolumeStepsPerVoxel, 2.0,                     \
      (::ernie::AttributeMeta{.label = "Steps per voxel",                                          \
                              .doc = "Steps through a volume per voxel of its densest field, "     \
                                     "fewer for speed.",                                           \
                              .group = "Volumes", .min = 0.1, .max = 8.0}))                        \
    X(int, VolumeMaxSteps, ::ernie::tokens::kVolumeMaxSteps, 256,                                  \
      (::ernie::AttributeMeta{.label = "Most steps", .doc = "Steps through one volume, at most.",  \
                              .group = "Volumes", .min = 1.0, .max = 4096.0}))                     \
    X(int, VolumeShadowSteps, ::ernie::tokens::kVolumeShadowSteps, 12,                             \
      (::ernie::AttributeMeta{.label = "Shadow steps",                                             \
                              .doc = "Steps towards each light through each volume on the way.",   \
                              .group = "Volumes", .min = 1.0, .max = 64.0}))                       \
    X(int, VolumeSkyLightSteps, ::ernie::tokens::kVolumeSkyLightSteps, 6,                          \
      (::ernie::AttributeMeta{.label = "Sky light steps",                                          \
                              .doc = "Steps up and down through the volumes for how much of the "  \
                                     "sky's and the ground's light they let through; 0, all.",     \
                              .group = "Volumes", .max = 64.0}))                                   \
    X(int, VolumeAirSteps, ::ernie::tokens::kVolumeAirSteps, 32,                                   \
      (::ernie::AttributeMeta{.label = "Air steps",                                                \
                              .doc = "Steps through the atmosphere in front of the volumes; 0 "    \
                                     "leaves volumes the air of what is behind them.",             \
                              .group = "Volumes", .max = 128.0}))                                  \
    X(int, VolumeMaxOctaves, ::ernie::tokens::kVolumeMaxOctaves, 8,                                \
      (::ernie::AttributeMeta{.label = "Most octaves",                                             \
                              .doc = "Orders of multiple scattering any material approximates, "  \
                                     "at most.",                                                   \
                              .group = "Volumes", .min = 1.0, .max = 8.0}))                        \
    X(int, VolumeShadowResolution, ::ernie::tokens::kVolumeShadowResolution, 256,                  \
      (::ernie::AttributeMeta{.label = "Shadow resolution",                                        \
                              .doc = "Texels along each side of the volumes' shadow map of each "  \
                                     "light shadow view, which surfaces read.",                    \
                              .group = "Volumes", .min = 16.0, .max = 2048.0}))                    \
    X(std::string, DomeLightMode, ::ernie::tokens::kDomeLightMode, "approximate",                  \
      (::ernie::AttributeMeta{.label = "Mode",                                                     \
                              .doc = "approximate: unoccluded; full: also shadowed; directions: "  \
                                     "full, diffuse from shadowed directions; limited: "           \
                                     "reflections only.",                                          \
                              .group = "Dome light",                                               \
                              .options = {"approximate", "full", "directions", "limited"}}))       \
    X(double, DomeShadowRadius, ::ernie::tokens::kDomeShadowRadius, 1.0,                           \
      (::ernie::AttributeMeta{.label = "Shadow radius",                                            \
                              .doc = "World distance within which geometry shadows the dome.",     \
                              .group = "Dome light", .max = 100.0}))                               \
    X(int, DomeDirections, ::ernie::tokens::kDomeDirections, 16,                                   \
      (::ernie::AttributeMeta{.label = "Directions",                                               \
                              .doc = "Shadowed directions lighting the dome's diffuse light.",     \
                              .group = "Dome light", .min = 1.0, .max = 1024.0}))

#define ERNIE_RENDER_PRODUCT_ATTRIBUTES(X)                                                         \
    X(std::vector<::ernie::Path>, OrderedVars, ::ernie::tokens::kOrderedVars,                      \
      std::vector<::ernie::Path>{},                                                                \
      (::ernie::AttributeMeta{.label = "Render vars",                                              \
                              .doc = "The RenderVars written; none, LdrColor alone.",              \
                              .group = "Image", .options = {::ernie::tokens::kRenderVar}}))        \
    X(std::vector<::ernie::Path>, Filters, ::ernie::tokens::kFilters,                              \
      std::vector<::ernie::Path>{},                                                                \
      (::ernie::AttributeMeta{.doc = "The SensorFilter prims of its sensor chain, in order.",      \
                              .group = "Image", .options = {::ernie::tokens::kSensorFilter}}))

#define ERNIE_RENDER_VAR_ATTRIBUTES(X)                                                             \
    X(std::string, SourceName, ::ernie::tokens::kSourceName, ::ernie::tokens::kLdrColor,           \
      (::ernie::AttributeMeta{                                                                     \
          .label = "Source",                                                                       \
          .doc = "The output: a render var name, or \"aov:<output>\" of a custom filter.",         \
          .group = "Var",                                                                          \
          .options = std::vector<std::string_view>(::ernie::kRenderVarNames.begin(),               \
                                                   ::ernie::kRenderVarNames.end())}))              \
    X(std::vector<::ernie::Path>, Filters, ::ernie::tokens::kFilters,                              \
      std::vector<::ernie::Path>{},                                                                \
      (::ernie::AttributeMeta{.doc = "Its own sensor chain's SensorFilter prims, in place of "     \
                                     "its product's, when set.",                                   \
                              .group = "Var", .options = {::ernie::tokens::kSensorFilter}}))

#define ERNIE_SENSOR_FILTER_ATTRIBUTES(X)                                                          \
    X(std::string, FilterType, ::ernie::tokens::kFilterType, "",                                   \
      (::ernie::AttributeMeta{.label = "Type",                                                     \
                              .doc = "A built-in filter (lens, qe, noise, adc, cie, exposure) or " \
                                     "a custom filter's id; its parameters are inputs:*.",         \
                              .group = "Filter",                                                   \
                              .options = {"lens", "qe", "noise", "adc", "cie", "exposure"}}))

// Every attribute of each type, its bases' first, for its accessors.
#define ERNIE_RENDER_SETTINGS_BASE_ALL(X)                                                          \
    ERNIE_IMAGEABLE_ATTRIBUTES(X) ERNIE_RENDER_SETTINGS_BASE_ATTRIBUTES(X)
#define ERNIE_RENDER_SETTINGS_ALL(X)                                                               \
    ERNIE_RENDER_SETTINGS_BASE_ALL(X) ERNIE_RENDER_SETTINGS_ATTRIBUTES(X)
#define ERNIE_RENDER_PRODUCT_ALL(X)                                                                \
    ERNIE_RENDER_SETTINGS_BASE_ALL(X) ERNIE_RENDER_PRODUCT_ATTRIBUTES(X)
#define ERNIE_RENDER_VAR_ALL(X) ERNIE_IMAGEABLE_ATTRIBUTES(X) ERNIE_RENDER_VAR_ATTRIBUTES(X)
#define ERNIE_SENSOR_FILTER_ALL(X) ERNIE_IMAGEABLE_ATTRIBUTES(X) ERNIE_SENSOR_FILTER_ATTRIBUTES(X)
// clang-format on

namespace ernie {

/// What render settings and products share (UsdRender's
/// RenderSettingsBase): the camera, the resolution, and what a
/// product sets in place of its settings' (overlays, minimal
/// shading).
class RenderSettingsBase : public Imageable {
    ERNIE_SCHEMA_BODY(RenderSettingsBase, Imageable,
                      ERNIE_RENDER_SETTINGS_BASE_ALL)
};

/// A render's settings: its camera, resolution and products, and the
/// rasterizer's options.
class RenderSettings : public RenderSettingsBase {
    ERNIE_SCHEMA_BODY(RenderSettings, RenderSettingsBase,
                      ERNIE_RENDER_SETTINGS_ALL)
    ERNIE_SCHEMA_DEFINE(RenderSettings)

    /// A view of the RenderSettings prim at `path` or, for the empty
    /// path or one naming no prim, of a prim with nothing authored,
    /// read-only, whose reads give the declared fallbacks: the
    /// settings of a render without a RenderSettings prim.
    /// @throws std::invalid_argument if `path` names a prim of
    /// another type.
    static RenderSettings OrFallbacks(const Scene& scene,
                                      const Path& path);
};

/// One image a render writes: its camera (or a batch of cameras),
/// resolution, render vars and sensor filter chain.
class RenderProduct : public RenderSettingsBase {
    ERNIE_SCHEMA_BODY(RenderProduct, RenderSettingsBase,
                      ERNIE_RENDER_PRODUCT_ALL)
    ERNIE_SCHEMA_DEFINE(RenderProduct)
};

/// An output of a render product, named by its source, optionally
/// with its own sensor chain.
class RenderVar : public Imageable {
    ERNIE_SCHEMA_BODY(RenderVar, Imageable, ERNIE_RENDER_VAR_ALL)
    ERNIE_SCHEMA_DEFINE(RenderVar)
};

/// A SensorFilter prim (Ernie), one step of a sensor filter chain: a
/// filter type and its `inputs:*` parameters. Named apart from the
/// built-in filters' enum, `SensorFilter`.
class SensorFilterPrim : public Imageable {
    ERNIE_SCHEMA_BODY(SensorFilterPrim, Imageable,
                      ERNIE_SENSOR_FILTER_ALL)
    ERNIE_SCHEMA_DEFINE(SensorFilterPrim)
};

/// This file's types' registry entries, bases first (`PrimTypes`).
std::vector<const PrimTypeDef*> RenderSchemaTypes();

}  // namespace ernie
