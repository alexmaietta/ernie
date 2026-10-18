// SPDX-License-Identifier: BSD-3-Clause
/// @file render.h
/// @brief UsdRender's settings, products, vars and passes, with Ernie's settings on every product.
///
/// References:
/// - Pixar, UsdRender, https://openusd.org/release/api/usd_render_page_front.html
#pragma once

#include "ernie/scene/view.h"

#include <string>
#include <vector>

// clang-format off
// Ernie's settings are on RenderSettingsBase, so each product may override its settings'. Tiers
// give the defaults: 0 or -1 below means the tier's.
#define ERNIE_RENDER_SETTINGS_BASE_ATTRIBUTES(X)                                                   \
    X(Path, Camera, "camera", Path(), "The camera it renders.", .group = "Render")                 \
    X(glm::dvec2, Resolution, "resolution", glm::dvec2(2048.0, 1080.0), "Pixels across and down.", \
      .group = "Render", .lower = AtLeast(1.0), .finite = true)                                    \
    X(double, PixelAspectRatio, "pixelAspectRatio", 1.0, "A pixel's width over its height.",       \
      .group = "Render", .lower = Above(0.0))                                                      \
    X(std::string, AspectRatioConformPolicy, "aspectRatioConformPolicy", "expandAperture",         \
      "How the camera's aperture meets the image's aspect.", .group = "Render",                    \
      .options = {"expandAperture", "cropAperture", "adjustApertureWidth", "adjustApertureHeight", \
                  "adjustPixelAspectRatio"})                                                       \
    X(glm::dvec4, DataWindowNdc, "dataWindowNDC", glm::dvec4(0.0, 0.0, 1.0, 1.0),                  \
      "The window of the image rendered, in normalized device coordinates.", .group = "Render")    \
    X(bool, DisableMotionBlur, "disableMotionBlur", false, "Renders the shutter's open instant.",  \
      .group = "Render")                                                                           \
    X(bool, DisableDepthOfField, "disableDepthOfField", false, "Renders as through a pinhole.",    \
      .group = "Render")                                                                           \
    X(std::string, Fidelity, "ernie:fidelity", "realtime",                                         \
      "The tier: minimal for learning at scale, realtime, or predictive for reference images.",    \
      .group = "Ernie", .options = {"minimal", "realtime", "predictive"})                          \
    X(int, Samples, "ernie:samples", 0, "Samples per pixel per frame; 0 the tier's.",              \
      .group = "Ernie", .lower = AtLeast(0.0))                                                     \
    X(int, Bounces, "ernie:bounces", -1, "Path bounces; -1 the tier's.", .group = "Ernie",         \
      .lower = AtLeast(-1.0))                                                                      \
    X(int, Accumulation, "ernie:accumulation", -1,                                                 \
      "Frames of temporal history; 0 none, -1 the tier's.", .group = "Ernie",                      \
      .lower = AtLeast(-1.0))                                                                      \
    X(std::string, Denoiser, "ernie:denoiser", "auto",                                             \
      "The denoiser; auto the tier's (none in minimal and predictive, svgf in realtime).",         \
      .group = "Ernie",                                                                            \
      .options = {"auto", "none", "svgf", "oidn", "optix", "dlss", "fidelityfx"})                  \
    X(int, Seed, "ernie:seed", 0, "The seed every random decision of its frames derives from.",    \
      .group = "Ernie")                                                                            \
    X(double, LodError, "ernie:lod:error", 1.0, "Pixels of geometric error LOD allows.",           \
      .group = "Ernie", .lower = Above(0.0))                                                       \
    X(std::string, SpectralMode, "ernie:spectral:mode", "spectral",                                \
      "spectral: bands over the range; rgb: three bands of linear sRGB, faster.",                  \
      .group = "Spectral", .options = {"spectral", "rgb"})                                         \
    X(int, SpectralBands, "ernie:spectral:bands", 16,                                              \
      "Bands sampled across the camera's sensor range.", .group = "Spectral",                      \
      .lower = AtLeast(1.0))

#define ERNIE_RENDER_SETTINGS_ATTRIBUTES(X)                                                        \
    X(std::vector<Path>, Products, "products", std::vector<Path>{}, "The products it renders.",    \
      .group = "Render")                                                                           \
    X(std::vector<std::string>, IncludedPurposes, "includedPurposes",                              \
      (std::vector<std::string>{"default", "render"}), "The purposes of the prims rendered.",      \
      .group = "Render", .dirty = Dirty::Visibility)                                               \
    X(std::vector<std::string>, MaterialBindingPurposes, "materialBindingPurposes",                \
      (std::vector<std::string>{"full", ""}), "The binding purposes tried, in order.",             \
      .group = "Render", .dirty = Dirty::Material)                                                 \
    X(std::string, RenderingColorSpace, "renderingColorSpace", "",                                 \
      "The colour space rendering happens in; unset, linear Rec.709.", .group = "Render")

#define ERNIE_RENDER_PRODUCT_ATTRIBUTES(X)                                                         \
    X(std::string, ProductType, "productType", "raster", "The kind of output.",                    \
      .group = "Product")                                                                          \
    X(std::string, ProductName, "productName", "", "The output's name, as a file path.",           \
      .group = "Product")                                                                          \
    X(std::vector<Path>, OrderedVars, "orderedVars", std::vector<Path>{},                          \
      "The vars it holds, in order.", .group = "Product")

#define ERNIE_RENDER_VAR_ATTRIBUTES(X)                                                             \
    X(std::string, DataType, "dataType", "color3f", "Each pixel's value type.", .group = "Var")    \
    X(std::string, SourceName, "sourceName", "",                                                   \
      "What it holds: an Ernie output (LdrColor, depth, normal...), a primvar or an LPE.",         \
      .group = "Var")                                                                              \
    X(std::string, SourceType, "sourceType", "raw", "How the source name is read.",                \
      .group = "Var", .options = {"raw", "primvar", "lpe", "intrinsic"})

#define ERNIE_RENDER_PASS_ATTRIBUTES(X)                                                            \
    X(std::string, PassType, "passType", "", "The kind of pass.", .group = "Pass")                 \
    X(std::vector<std::string>, Command, "command", std::vector<std::string>{},                    \
      "The command a pipeline runs it with.", .group = "Pass")                                     \
    X(std::string, FileName, "fileName", "", "The file it renders from.", .group = "Pass",        \
      .asset = true)                                                                               \
    X(std::vector<Path>, RenderSource, "renderSource", std::vector<Path>{},                        \
      "The render settings it renders.", .group = "Pass")                                          \
    X(std::vector<Path>, InputPasses, "inputPasses", std::vector<Path>{},                          \
      "Passes it depends on.", .group = "Pass")
// clang-format on

namespace ernie {

ERNIE_SCHEMA(RenderSettingsBase, SchemaView, ERNIE_RENDER_SETTINGS_BASE_ATTRIBUTES,
             "What settings and products share: the camera, image and Ernie's settings.");
ERNIE_CONCRETE_SCHEMA(RenderSettings, RenderSettingsBase, ERNIE_RENDER_SETTINGS_ATTRIBUTES,
                      "A render's settings and the products it makes.");
ERNIE_CONCRETE_SCHEMA(RenderProduct, RenderSettingsBase, ERNIE_RENDER_PRODUCT_ATTRIBUTES,
                      "One output, an image or a sensor's data, overriding its settings' where it "
                      "authors them.");
ERNIE_CONCRETE_SCHEMA(RenderVar, SchemaView, ERNIE_RENDER_VAR_ATTRIBUTES,
                      "One channel set of a product: colour, depth, normals, ids or an Ernie "
                      "output.");
ERNIE_CONCRETE_SCHEMA(RenderPass, SchemaView, ERNIE_RENDER_PASS_ATTRIBUTES,
                      "A render of a pipeline, with the prims and cameras it sees as collections.");

}  // namespace ernie
