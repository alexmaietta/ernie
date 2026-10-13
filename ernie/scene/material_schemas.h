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
    X(std::string, MaterialXFile, ::ernie::tokens::kMaterialXFile, "",                             \
      (::ernie::AttributeMeta{.label = "MaterialX file",                                           \
                              .doc = "A MaterialX document (.mtlx) whose material this is, in "    \
                                     "place of a network on outputs:mtlx:surface.",                \
                              .group = "MaterialX", .asset = true}))                               \
    X(std::string, MaterialXElement, ::ernie::tokens::kMaterialXElement, "",                       \
      (::ernie::AttributeMeta{.label = "MaterialX element",                                        \
                              .doc = "The material (or shader) of the file to render, by name "    \
                                     "path; empty, its first renderable one.",                     \
                              .group = "MaterialX"}))                                              \
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
    X(double, PolarizationRetention, ::ernie::tokens::kPolarizationRetention, 0.0,                 \
      (::ernie::AttributeMeta{.label = "Polarization kept",                                        \
                              .doc = "Share of incident polarization diffuse reflection keeps, 0 " \
                                     "to 1; dark, single-scattering surfaces keep more.",          \
                              .group = "Surface", .max = 1.0}))                                    \
    X(double, SpeckleRoughness, ::ernie::tokens::kSpeckleRoughness, 1e-5,                          \
      (::ernie::AttributeMeta{.label = "Speckle roughness",                                        \
                              .doc = "RMS surface height, m: rough (the default) shows laser "     \
                                     "speckle fully, polished (0) none.",                          \
                              .group = "Surface", .max = 1e-3}))                                   \
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
    X(double, ThermalTemperature, ::ernie::tokens::kThermalTemperature, 0.0,                       \
      (::ernie::AttributeMeta{.label = "Temperature",                                              \
                              .doc = "Kelvin; above 0 the surface glows by Planck's law times "    \
                                     "its emissivity, in every band.",                             \
                              .group = "Thermal", .max = 5000.0}))                                 \
    X(std::vector<float>, ThermalEmissivity, ::ernie::tokens::kThermalEmissivity,                  \
      std::vector<float>{},                                                                        \
      (::ernie::AttributeMeta{.label = "Emissivity",                                               \
                              .doc = "(nm, value) pairs in [0, 1], wavelengths increasing; the "   \
                                     "nearest value outside them. Unset, 1 minus the "             \
                                     "reflectance (Kirchhoff).",                                   \
                              .group = "Thermal"}))                                                \
    X(::ernie::Tensor, ThermalTemperatureTexture, ::ernie::tokens::kThermalTemperatureTexture,     \
      nullptr,                                                                                     \
      (::ernie::AttributeMeta{.label = "Temperature texture",                                      \
                              .doc = "float32 (rows, cols) or (rows, cols, 1) kelvin through "     \
                                     "mesh texture coordinates, row 0 at the top, in place of "    \
                                     "the temperature; 0 or below there for the temperature.",     \
                              .group = "Thermal"}))                                                \
    X(std::string, ThermalModel, ::ernie::tokens::kThermalModel, "",                               \
      (::ernie::AttributeMeta{.label = "Temperature model",                                        \
                              .doc = "A temperature model computing its pixels' temperature as "   \
                                     "they render: balance (zero-capacitance, spacecraft), "       \
                                     "diurnal (a latitude and local time table, the Moon's by "    \
                                     "default) or a registered one; empty for none.",              \
                              .group = "Thermal"}))                                                \
    X(std::vector<float>, ThermalModelParams, ::ernie::tokens::kThermalModelParams,                \
      std::vector<float>{},                                                                        \
      (::ernie::AttributeMeta{.label = "Model parameters",                                         \
                              .doc = "The temperature model's parameters in its order; missing "   \
                                     "ones its defaults.",                                         \
                              .group = "Thermal"}))                                                \
    X(std::vector<float>, ThermalModelData, ::ernie::tokens::kThermalModelData,                    \
      std::vector<float>{},                                                                        \
      (::ernie::AttributeMeta{.label = "Model data",                                               \
                              .doc = "Floats the temperature model reads, such as a diurnal "      \
                                     "table (DiurnalTable::ModelData).",                           \
                              .group = "Thermal"}))                                                \
    X(double, ThermalInertia, ::ernie::tokens::kThermalInertia, 0.0,                               \
      (::ernie::AttributeMeta{.label = "Thermal inertia",                                          \
                              .doc = "J m^-2 K^-1 s^-1/2, which the temperature model reads (the " \
                                     "diurnal table's inertia axis); 0 for none.",                 \
                              .group = "Thermal", .min = 0.0, .max = 10000.0}))                    \
    X(::ernie::Tensor, ThermalInertiaTexture, ::ernie::tokens::kThermalInertiaTexture, nullptr,    \
      (::ernie::AttributeMeta{.label = "Thermal inertia texture",                                  \
                              .doc = "float32 (rows, cols) or (rows, cols, 1) thermal inertia "    \
                                     "through mesh texture coordinates, row 0 at the top, in "     \
                                     "place of the thermal inertia where above 0.",                \
                              .group = "Thermal"}))                                                \
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
                                     "and the regolith it rose from scatter alike; Rayleigh; "     \
                                     "isotropic; or cloud droplets' Mie scattering by their "      \
                                     "radius and spread, with clouds' silver linings, fogbows "    \
                                     "and glories.",                                               \
                              .group = "Volume",                                                   \
                              .options = {::ernie::tokens::kHenyeyGreenstein,                      \
                                          ::ernie::tokens::kDoubleHenyeyGreenstein,                \
                                          ::ernie::tokens::kDualHenyeyGreenstein,                  \
                                          ::ernie::tokens::kRayleigh,                              \
                                          ::ernie::tokens::kIsotropic,                             \
                                          ::ernie::tokens::kCloudDroplets}}))                      \
    X(double, DropletRadius, ::ernie::tokens::kVolumeDropletRadius, 7.0,                           \
      (::ernie::AttributeMeta{.label = "Droplet radius (um)",                                      \
                              .doc = "Cloud droplets' effective radius, micrometres: about 5 to "  \
                                     "10 in stratus, 10 to 15 in cumulus; larger droplets "        \
                                     "scatter more forwards and tighten the glory.",               \
                              .group = "Volume", .min = 1.0, .max = 30.0}))                        \
    X(double, DropletSpread, ::ernie::tokens::kVolumeDropletSpread, 2.0,                           \
      (::ernie::AttributeMeta{.label = "Droplet spread",                                           \
                              .doc = "The gamma of their size distribution: larger, more alike "   \
                                     "in size, their fogbow and glory more coloured.",             \
                              .group = "Volume", .min = 0.5, .max = 20.0}))                        \
    X(double, SurfaceDensity, ::ernie::tokens::kVolumeSurfaceDensity, 0.1,                         \
      (::ernie::AttributeMeta{.label = "Surface density",                                          \
                              .doc = "Cloud droplets' surface, the density field's value where "   \
                                     "it begins, for multiple scattering by collectors: their "    \
                                     "light found where it entered there.",                        \
                              .group = "Volume", .min = 0.0}))                                     \
    X(double, Anisotropy, ::ernie::tokens::kVolumeAnisotropy, 0.0,                                 \
      (::ernie::AttributeMeta{.doc = "Henyey-Greenstein g: above 0 scatters forwards, below 0 "    \
                                     "backwards.",                                                 \
                              .group = "Volume", .min = -0.99, .max = 0.99}))                      \
    X(double, SecondAnisotropy, ::ernie::tokens::kVolumeSecondAnisotropy, -0.2,                    \
      (::ernie::AttributeMeta{.label = "Second anisotropy",                                        \
                              .doc = "The dual Henyey-Greenstein phase function's second lobe's "  \
                                     "g, backwards below 0 (clouds' 0.7 and -0.2 by halves, "      \
                                     "three-geospatial's).",                                       \
                              .group = "Volume", .min = -0.99, .max = 0.99}))                      \
    X(double, LobeMix, ::ernie::tokens::kVolumeLobeMix, 0.5,                                       \
      (::ernie::AttributeMeta{.label = "Lobe mix",                                                 \
                              .doc = "The second lobe's share of the dual Henyey-Greenstein "      \
                                     "phase function.",                                            \
                              .group = "Volume", .min = 0.0, .max = 1.0}))                         \
    X(double, Powder, ::ernie::tokens::kVolumePowder, 0.0,                                         \
      (::ernie::AttributeMeta{.label = "Powder",                                                   \
                              .doc = "The powder effect's strength: light scattered in dimmed by " \
                                     "1 - powder e^(-sigma length), darkening thin edges facing "  \
                                     "the light as light scattered many times has yet to build "   \
                                     "up there (Schneider and Vos 2015). 0, none.",                \
                              .group = "Volume", .min = 0.0, .max = 1.0}))                         \
    X(double, PowderLength, ::ernie::tokens::kVolumePowderLength, 150.0,                           \
      (::ernie::AttributeMeta{.label = "Powder length",                                            \
                              .doc = "The powder effect's length (world units) times the "         \
                                     "extinction: over how much medium it fades.",                 \
                              .group = "Volume", .min = 0.0}))                                     \
    X(int, ScatteringOctaves, ::ernie::tokens::kVolumeScatteringOctaves, 1,                        \
      (::ernie::AttributeMeta{.doc = "Orders of scattering approximated, after Wrenninge et al. "  \
                                     "(2013): 1 is single scattering; dense media such as clouds " \
                                     "want 4 to 8, which brighten their insides as light that "    \
                                     "scattered many times does.",                                 \
                              .group = "Volume", .min = 1.0, .max = 8.0}))                         \
    X(double, OctaveAttenuation, ::ernie::tokens::kVolumeOctaveAttenuation, 0.5,                   \
      (::ernie::AttributeMeta{.doc = "Each order's scattering, of the one before's (a); at most "  \
                                     "the octave extinction, to conserve energy.",                 \
                              .group = "Volume", .max = 1.0}))                                     \
    X(double, OctaveExtinction, ::ernie::tokens::kVolumeOctaveExtinction, 0.5,                     \
      (::ernie::AttributeMeta{.doc = "Each order's extinction towards the light, of the one "      \
                                     "before's (b): light that scattered many times gets "         \
                                     "deeper.",                                                    \
                              .group = "Volume", .max = 1.0}))                                     \
    X(double, OctavePhase, ::ernie::tokens::kVolumeOctavePhase, 0.5,                               \
      (::ernie::AttributeMeta{.doc = "Each order's anisotropy, of the one before's (c): light "    \
                                     "that scattered many times spreads more evenly.",             \
                              .group = "Volume", .max = 1.0}))                                     \
    X(double, SkyLight, ::ernie::tokens::kVolumeSkyLight, 1.0,                                     \
      (::ernie::AttributeMeta{.doc = "Scales the light of the sky and the ground around the "      \
                                     "medium, from the atmosphere's distant lights; 1 is "         \
                                     "physical, 0 none.",                                          \
                              .group = "Volume", .max = 10.0}))                                    \
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
                                     "field, or what the field's 0 is with one (the ambient "      \
                                     "air's); 0, no glow without a field.",                        \
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
                              .doc = "The shader's type; Ernie reads the seven below.",            \
                              .group = "Shader",                                                   \
                              .options = {::ernie::tokens::kUsdPreviewSurface,                     \
                                          ::ernie::tokens::kUsdUVTexture,                          \
                                          ::ernie::tokens::kUsdPrimvarReaderFloat2,                \
                                          ::ernie::tokens::kUsdTransform2d,                        \
                                          ::ernie::tokens::kErnieThermal,                          \
                                          ::ernie::tokens::kUsdPrimvarReaderFloat,                 \
                                          ::ernie::tokens::kUsdPrimvarReaderFloat3}}))             \
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
    X(std::string, VirtualTexture, ::ernie::tokens::kVirtualTexture, "auto",                       \
      (::ernie::AttributeMeta{.label = "Virtual",                                                  \
                              .doc = "Only the tiles in view resident, for textures past device "  \
                                     "memory or image size (DDS or KTX2 with every mip, devices "  \
                                     "with sparse residency). auto: albedo textures over 8192 "    \
                                     "texels a side.",                                             \
                              .group = "Texture", .options = {"auto", "on", "off"}}))              \
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
                              .doc = "The mesh primvar it reads: texture coordinates, or the "     \
                                     "colours a diffuse colour takes (float, float3 readers).",    \
                              .group = "Primvar reader"}))                                         \
    X(double, ThermalTemperature, ::ernie::tokens::kThermalTemperatureInput, 0.0,                  \
      (::ernie::AttributeMeta{.label = "Temperature",                                              \
                              .doc = "Kelvin, 0 for none; connectable to one channel of a "        \
                                     "UsdUVTexture, kelvin after its scale and bias.",             \
                              .group = "Thermal", .max = 5000.0}))                                 \
    X(double, ThermalEmissivity, ::ernie::tokens::kThermalEmissivityInput, -1.0,                   \
      (::ernie::AttributeMeta{.label = "Emissivity",                                               \
                              .doc = "0 to 1 at every wavelength; below 0, 1 minus the "           \
                                     "reflectance (Kirchhoff).",                                   \
                              .group = "Thermal", .min = -1.0, .max = 1.0}))                       \
    X(double, ThermalInertia, ::ernie::tokens::kThermalInertiaInput, 0.0,                          \
      (::ernie::AttributeMeta{.label = "Thermal inertia",                                          \
                              .doc = "J m^-2 K^-1 s^-1/2 for the temperature model, 0 for none; "  \
                                     "connectable to one channel of a UsdUVTexture.",              \
                              .group = "Thermal", .min = 0.0, .max = 10000.0}))                    \
    X(std::string, ThermalModel, ::ernie::tokens::kThermalModelInput, "",                          \
      (::ernie::AttributeMeta{.label = "Temperature model",                                        \
                              .doc = "As the Material's ernie:thermal:model.",                     \
                              .group = "Thermal"}))                                                \
    X(std::vector<float>, ThermalModelParams, ::ernie::tokens::kThermalModelParamsInput,           \
      std::vector<float>{},                                                                        \
      (::ernie::AttributeMeta{.label = "Model parameters", .group = "Thermal"}))                   \
    X(std::vector<float>, ThermalModelData, ::ernie::tokens::kThermalModelDataInput,               \
      std::vector<float>{},                                                                        \
      (::ernie::AttributeMeta{.label = "Model data", .group = "Thermal"}))

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
