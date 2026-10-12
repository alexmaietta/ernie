// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/base_schemas.h"
#include "ernie/scene/prim_type.h"
#include "ernie/scene/tokens.h"

#include <string>
#include <vector>

/// Materials as reflected schemas (`prim_type.h`): UsdShade's
/// Material, with Ernie's own surface inputs (a BRDF model, an
/// albedo, an albedo map and detail textures), and Shader, the nodes
/// of a UsdShade network (UsdPreviewSurface, UsdUVTexture,
/// UsdPrimvarReader_float2, UsdTransform2d) by `info:id`. Each
/// declaration's fallback is the value Ernie shades an unset input
/// with (`MaterialSettings`, `DetailSettings`, `TextureReader`, the
/// `preview_surface` model's defaults).
///
/// Not declared, and read generically: a BRDF's own parameters
/// (`inputs:<param>` per model in the `BrdfRegistry`, an open
/// namespace), connections (`<input>.connect`, through
/// `GetConnection`), and `inputs:scale`, which is UsdUVTexture's
/// 4-vector scale and UsdTransform2d's 2-vector one under one name on
/// the one Shader type: a declaration of either type would author the
/// other Shader's scale as the wrong USD type.
///
/// References:
/// - Pixar, UsdShade,
///   https://openusd.org/release/api/usd_shade_page_front.html
/// - Pixar, "UsdPreviewSurface Specification",
///   https://openusd.org/release/spec_usdpreviewsurface.html

// clang-format off
#define ERNIE_MATERIAL_ATTRIBUTES(X)                                                               \
    X(std::string, Brdf, ::ernie::tokens::kBrdf, "lambert",                                        \
      (::ernie::AttributeMeta{.label = "BRDF",                                                     \
                              .doc = "The reflectance model, a BrdfRegistry id; unset with a "     \
                                     "UsdPreviewSurface connected, \"preview_surface\".",          \
                              .group = "Surface"}))                                                \
    X(glm::dvec3, Albedo, ::ernie::tokens::kAlbedo, glm::dvec3(1.0),                               \
      (::ernie::AttributeMeta{.doc = "Linear RGB", .group = "Surface", .color = true}))            \
    X(std::vector<float>, Reflectance, ::ernie::tokens::kReflectance, std::vector<float>{},        \
      (::ernie::AttributeMeta{.doc = "(nm, value) pairs, wavelengths increasing, of a measured "  \
                                     "reflectance spectrum in [0, 1], replacing the albedo; the "  \
                                     "nearest value outside them.",                                \
                              .group = "Surface"}))                                                \
    X(std::string, ReflectanceLibrary, ::ernie::tokens::kReflectanceLibrary, "",                   \
      (::ernie::AttributeMeta{.label = "Measured reflectance",                                     \
                              .doc = "A measured spectrum Ernie embeds (ECOSTRESS), 0.35 to 2.5 "  \
                                     "um and on into the thermal infrared, replacing the albedo "  \
                                     "unless a reflectance is given.",                             \
                              .group = "Surface",                                                  \
                              .options = {"", "grass", "chaparral", "dry_grass", "loam", "sand",   \
                                          "concrete", "asphalt", "brick", "pine_wood",             \
                                          "black_paint", "olive_paint", "white_coating",           \
                                          "aluminium", "galvanized_steel", "copper", "fine_snow",  \
                                          "granular_snow", "ice", "water"}}))                      \
    X(bool, AlbedoMapEnable, ::ernie::tokens::kAlbedoMapEnable, true,                              \
      (::ernie::AttributeMeta{.label = "Use albedo map", .group = "Surface"}))                     \
    X(::ernie::Tensor, AlbedoMap, ::ernie::tokens::kAlbedoMap, nullptr,                            \
      (::ernie::AttributeMeta{.doc = "float32 (rows, cols) albedo per cell of the grid below.",    \
                              .group = "Surface"}))                                                \
    X(glm::dvec2, AlbedoMapOrigin, ::ernie::tokens::kAlbedoMapOrigin, glm::dvec2(0.0),             \
      (::ernie::AttributeMeta{.doc = "World x and y of the albedo map's north-west corner.",       \
                              .group = "Surface"}))                                                \
    X(double, AlbedoMapCellSize, ::ernie::tokens::kAlbedoMapCellSize, 1.0,                         \
      (::ernie::AttributeMeta{.doc = "World units per albedo map cell.", .group = "Surface",       \
                              .max = 1e5}))                                                        \
    X(double, AlbedoMapNoData, ::ernie::tokens::kAlbedoMapNoData, 0.0,                             \
      (::ernie::AttributeMeta{.doc = "The albedo map value marking missing cells; none unless "    \
                                     "set.",                                                       \
                              .group = "Surface"}))                                                \
    X(glm::dvec3, EmissiveColor, ::ernie::tokens::kEmissiveColor, glm::dvec3(0.0),                 \
      (::ernie::AttributeMeta{.label = "Emission",                                                 \
                              .doc = "Linear RGB radiance in cd/m^2 for white, without a "         \
                                     "UsdPreviewSurface (whose own is read then).",                \
                              .group = "Surface", .color = true}))                                 \
    X(::ernie::Tensor, DetailAlbedo, ::ernie::tokens::kDetailAlbedo, nullptr,                      \
      (::ernie::AttributeMeta{.doc = "float32 linear or uint8 sRGB (rows, cols, 1, 3 or 4) albedo " \
                                     "tiled over the surface.",                                    \
                              .group = "Detail textures"}))                                        \
    X(::ernie::Tensor, DetailNormal, ::ernie::tokens::kDetailNormal, nullptr,                      \
      (::ernie::AttributeMeta{.doc = "float32 or uint8 linear (rows, cols, 3 or 4) tangent-space " \
                                     "normal map tiled over the surface.",                         \
                              .group = "Detail textures"}))                                        \
    X(double, DetailTileSize, ::ernie::tokens::kDetailTileSize, 4.0,                               \
      (::ernie::AttributeMeta{.label = "Tile size",                                                \
                              .doc = "World units covered by one repeat of the textures.",         \
                              .group = "Detail textures", .max = 1000.0}))                         \
    X(double, DetailAlbedoStrength, ::ernie::tokens::kDetailAlbedoStrength, 1.0,                   \
      (::ernie::AttributeMeta{.label = "Albedo strength",                                          \
                              .doc = "0 ignores the detail albedo, 1 applies it fully.",           \
                              .group = "Detail textures", .max = 1.0}))                            \
    X(double, DetailNormalStrength, ::ernie::tokens::kDetailNormalStrength, 1.0,                   \
      (::ernie::AttributeMeta{.label = "Normal strength",                                          \
                              .doc = "Scale on the detail normal map's slopes.",                   \
                              .group = "Detail textures", .max = 4.0}))                            \
    X(bool, DetailDirectxNormals, ::ernie::tokens::kDetailDirectXNormals, false,                   \
      (::ernie::AttributeMeta{.label = "DirectX normals",                                          \
                              .doc = "Green points down the image (DirectX) rather than up.",      \
                              .group = "Detail textures"}))                                        \
    X(bool, DetailHexTiling, ::ernie::tokens::kDetailHexTiling, true,                              \
      (::ernie::AttributeMeta{.label = "Hex tiling",                                               \
                              .doc = "Hide repetition with hex-tiling (Mikkelsen 2022).",          \
                              .group = "Detail textures"}))                                        \
    X(double, DetailHexRotation, ::ernie::tokens::kDetailHexRotation, 1.0,                         \
      (::ernie::AttributeMeta{.label = "Hex rotation",                                             \
                              .doc = "Random tile rotation, 0 none to 1 any angle.",               \
                              .group = "Detail textures", .max = 1.0}))                            \
    X(double, DetailHexContrast, ::ernie::tokens::kDetailHexContrast, 0.6,                         \
      (::ernie::AttributeMeta{.label = "Hex contrast",                                             \
                              .doc = "Weight of sample luminance in the tile blend, 0 to 1.",      \
                              .group = "Detail textures", .max = 1.0}))                            \
    X(double, DetailHexSharpness, ::ernie::tokens::kDetailHexSharpness, 7.0,                       \
      (::ernie::AttributeMeta{.label = "Hex sharpness",                                            \
                              .doc = "Exponent on the barycentric blend weights.",                 \
                              .group = "Detail textures", .min = 1.0, .max = 32.0}))               \
    X(double, Density, ::ernie::tokens::kVolumeDensity, 1.0,                                       \
      (::ernie::AttributeMeta{.doc = "Scales the density field into the extinction coefficient, "  \
                                     "1/m.",                                                       \
                              .group = "Volume", .max = 1e6}))                                     \
    X(std::string, DensityField, ::ernie::tokens::kVolumeDensityField, "density",                  \
      (::ernie::AttributeMeta{.doc = "The Volume's field (field:<name>) of density.",              \
                              .group = "Volume"}))                                                 \
    X(glm::dvec3, VolumeAlbedo, ::ernie::tokens::kVolumeAlbedo, glm::dvec3(0.0),                   \
      (::ernie::AttributeMeta{.label = "Scattering albedo",                                        \
                              .doc = "Linear RGB share of the extinction that scatters; the rest " \
                                     "absorbs. 0, smoke that only absorbs.",                       \
                              .group = "Volume", .color = true}))                                  \
    X(std::string, PhaseFunction, ::ernie::tokens::kVolumePhaseFunction,                           \
      ::ernie::tokens::kHenyeyGreenstein,                                                          \
      (::ernie::AttributeMeta{.label = "Phase function",                                           \
                              .doc = "How what scatters spreads its light: Henyey-Greenstein by "  \
                                     "the anisotropy; Hapke's double Henyey-Greenstein by the b "  \
                                     "and c a Hapke surface reads (inputs:b, inputs:c), so dust "  \
                                     "and the regolith it rose from scatter alike; Rayleigh; or "  \
                                     "isotropic.",                                                 \
                              .group = "Volume",                                                   \
                              .options = {::ernie::tokens::kHenyeyGreenstein,                      \
                                          ::ernie::tokens::kDoubleHenyeyGreenstein,                \
                                          ::ernie::tokens::kRayleigh,                              \
                                          ::ernie::tokens::kIsotropic}}))                          \
    X(double, Anisotropy, ::ernie::tokens::kVolumeAnisotropy, 0.0,                                 \
      (::ernie::AttributeMeta{.doc = "Henyey-Greenstein g: above 0 scatters forwards, below 0 "    \
                                     "backwards.",                                                 \
                              .group = "Volume", .min = -0.99, .max = 0.99}))                      \
    X(glm::dvec3, EmissionColor, ::ernie::tokens::kVolumeEmissionColor, glm::dvec3(1.0),           \
      (::ernie::AttributeMeta{.label = "Emission colour", .doc = "Linear RGB.", .group = "Volume", \
                              .color = true}))                                                     \
    X(double, EmissionStrength, ::ernie::tokens::kVolumeEmissionStrength, 0.0,                     \
      (::ernie::AttributeMeta{.doc = "Radiance added per metre, cd/m^2 per m for a white "         \
                                     "colour, times the emission field.",                          \
                              .group = "Volume", .max = 1e9}))                                     \
    X(std::string, EmissionField, ::ernie::tokens::kVolumeEmissionField, "",                       \
      (::ernie::AttributeMeta{.doc = "The field the emission follows; unset, the density field.",  \
                              .group = "Volume"}))                                                 \
    X(double, Blackbody, ::ernie::tokens::kVolumeBlackbody, 1.0,                                   \
      (::ernie::AttributeMeta{.label = "Black-body emission",                                      \
                              .doc = "Scales the glow of what absorbs at its temperature, "        \
                                     "Planck's law times the absorption coefficient (Kirchhoff); " \
                                     "1 is physical, 0 none.",                                     \
                              .group = "Volume", .max = 1e3}))                                     \
    X(double, Temperature, ::ernie::tokens::kVolumeTemperature, 0.0,                               \
      (::ernie::AttributeMeta{.doc = "Kelvin, of the whole volume when it has no temperature "     \
                                     "field; 0, no glow.",                                         \
                              .group = "Volume", .max = 1e5}))                                     \
    X(std::string, TemperatureField, ::ernie::tokens::kVolumeTemperatureField, "temperature",      \
      (::ernie::AttributeMeta{.doc = "The Volume's field of temperature in kelvin, times the "     \
                                     "temperature scale.",                                         \
                              .group = "Volume"}))                                                 \
    X(double, TemperatureScale, ::ernie::tokens::kVolumeTemperatureScale, 1.0,                     \
      (::ernie::AttributeMeta{.doc = "Takes the temperature field's values to kelvin.",            \
                              .group = "Volume", .max = 1e5}))

#define ERNIE_SHADER_ATTRIBUTES(X)                                                                 \
    X(std::string, InfoId, ::ernie::tokens::kInfoId, "",                                           \
      (::ernie::AttributeMeta{.label = "Shader",                                                   \
                              .doc = "The shader's type; Ernie reads the four below.",             \
                              .group = "Shader",                                                   \
                              .options = {::ernie::tokens::kUsdPreviewSurface,                     \
                                          ::ernie::tokens::kUsdUVTexture,                          \
                                          ::ernie::tokens::kUsdPrimvarReaderFloat2,                \
                                          ::ernie::tokens::kUsdTransform2d}}))                     \
    X(glm::dvec3, DiffuseColor, ::ernie::tokens::kDiffuseColor, glm::dvec3(0.18),                  \
      (::ernie::AttributeMeta{.label = "Diffuse colour", .doc = "Linear RGB albedo.",              \
                              .group = "Preview surface", .color = true}))                         \
    X(double, Roughness, ::ernie::tokens::kRoughness, 0.5,                                         \
      (::ernie::AttributeMeta{.group = "Preview surface", .max = 1.0}))                            \
    X(double, Metallic, ::ernie::tokens::kMetallic, 0.0,                                           \
      (::ernie::AttributeMeta{.group = "Preview surface", .max = 1.0}))                            \
    X(double, Ior, ::ernie::tokens::kIor, 1.5,                                                     \
      (::ernie::AttributeMeta{.label = "Index of refraction", .group = "Preview surface",          \
                              .min = 1.0, .max = 3.0}))                                            \
    X(bool, UseSpecularWorkflow, ::ernie::tokens::kUseSpecularWorkflow, false,                     \
      (::ernie::AttributeMeta{.label = "Specular workflow",                                        \
                              .doc = "Specular colour in place of metallic; USD's int 0 or 1.",    \
                              .group = "Preview surface"}))                                        \
    X(glm::dvec3, SpecularColor, ::ernie::tokens::kSpecularColor, glm::dvec3(0.0),                 \
      (::ernie::AttributeMeta{.label = "Specular colour",                                          \
                              .doc = "Linear RGB reflectance at normal incidence, in the "         \
                                     "specular workflow.",                                         \
                              .group = "Preview surface", .color = true}))                         \
    X(double, Clearcoat, ::ernie::tokens::kClearcoat, 0.0,                                         \
      (::ernie::AttributeMeta{.group = "Preview surface", .max = 1.0}))                            \
    X(double, ClearcoatRoughness, ::ernie::tokens::kClearcoatRoughness, 0.01,                      \
      (::ernie::AttributeMeta{.group = "Preview surface", .max = 1.0}))                            \
    X(glm::dvec3, EmissiveColor, ::ernie::tokens::kEmissiveColor, glm::dvec3(0.0),                 \
      (::ernie::AttributeMeta{.label = "Emission",                                                 \
                              .doc = "Linear RGB radiance in cd/m^2 for white.",                   \
                              .group = "Preview surface", .color = true}))                         \
    X(double, Opacity, ::ernie::tokens::kOpacity, 1.0,                                             \
      (::ernie::AttributeMeta{.group = "Preview surface", .max = 1.0}))                            \
    X(double, OpacityThreshold, ::ernie::tokens::kOpacityThreshold, 0.0,                           \
      (::ernie::AttributeMeta{.doc = "Above 0, surfaces less opaque are cut out.",                 \
                              .group = "Preview surface", .max = 1.0}))                            \
    X(std::string, File, ::ernie::tokens::kFile, "",                                               \
      (::ernie::AttributeMeta{.doc = "Image file: PNG, JPEG or EXR.", .group = "Texture",          \
                              .asset = true}))                                                     \
    X(::ernie::Tensor, Texture, ::ernie::tokens::kShaderTexture, nullptr,                          \
      (::ernie::AttributeMeta{.doc = "float32 or uint8 (rows, cols, channels) image, rows top to " \
                                     "bottom, in place of the file.",                              \
                              .group = "Texture"}))                                                \
    X(std::string, WrapS, ::ernie::tokens::kWrapS, "useMetadata",                                  \
      (::ernie::AttributeMeta{.label = "Wrap s", .doc = "useMetadata repeats.",                    \
                              .group = "Texture",                                                  \
                              .options = {"repeat", "mirror", "clamp", "black", "useMetadata"}}))  \
    X(std::string, WrapT, ::ernie::tokens::kWrapT, "useMetadata",                                  \
      (::ernie::AttributeMeta{.label = "Wrap t", .doc = "useMetadata repeats.",                    \
                              .group = "Texture",                                                  \
                              .options = {"repeat", "mirror", "clamp", "black", "useMetadata"}}))  \
    X(std::string, SourceColorSpace, ::ernie::tokens::kSourceColorSpace, "auto",                   \
      (::ernie::AttributeMeta{.label = "Colour space",                                             \
                              .doc = "auto: sRGB for 8-bit images, raw otherwise.",                \
                              .group = "Texture", .options = {"auto", "raw", "sRGB"}}))            \
    X(glm::dvec4, Bias, ::ernie::tokens::kShaderBias, glm::dvec4(0.0),                             \
      (::ernie::AttributeMeta{.doc = "Added to the texture's linear values after its scale.",      \
                              .group = "Texture"}))                                                \
    X(double, Rotation, ::ernie::tokens::kRotation, 0.0,                                           \
      (::ernie::AttributeMeta{.doc = "Degrees, counter-clockwise, after the scale.",               \
                              .group = "Transform 2D", .min = -360.0, .max = 360.0}))              \
    X(glm::dvec2, Translation, ::ernie::tokens::kTranslation, glm::dvec2(0.0),                     \
      (::ernie::AttributeMeta{.doc = "Added after the rotation.", .group = "Transform 2D"}))       \
    X(std::string, Varname, ::ernie::tokens::kVarname, "st",                                       \
      (::ernie::AttributeMeta{.label = "Primvar",                                                  \
                              .doc = "The mesh primvar of the texture coordinates.",               \
                              .group = "Primvar reader"}))

#define ERNIE_MATERIAL_ALL(X) ERNIE_IMAGEABLE_ATTRIBUTES(X) ERNIE_MATERIAL_ATTRIBUTES(X)
#define ERNIE_SHADER_ALL(X) ERNIE_IMAGEABLE_ATTRIBUTES(X) ERNIE_SHADER_ATTRIBUTES(X)
// clang-format on

namespace ernie {

/// How a surface reflects light: Ernie's own inputs, or a
/// UsdPreviewSurface its `outputs:surface` connects to. A BRDF's
/// parameters are `inputs:<param>` of its model.
class Material : public Imageable {
    ERNIE_SCHEMA_BODY(Material, Imageable, ERNIE_MATERIAL_ALL)
    ERNIE_SCHEMA_DEFINE(Material)
};

/// A node of a material's network, of the type `info:id` names, its
/// inputs constants or connections (`<input>.connect`) to other
/// shaders' outputs.
class Shader : public Imageable {
    ERNIE_SCHEMA_BODY(Shader, Imageable, ERNIE_SHADER_ALL)
    ERNIE_SCHEMA_DEFINE(Shader)
};

/// This file's types' registry entries, bases first (`PrimTypes`).
std::vector<const PrimTypeDef*> MaterialSchemaTypes();

}  // namespace ernie
