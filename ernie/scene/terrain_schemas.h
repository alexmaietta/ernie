// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/base_schemas.h"
#include "ernie/scene/prim_type.h"
#include "ernie/scene/tokens.h"
#include "ernie/util/astronomy.h"

#include <vector>

/// Ernie's Terrain (a heightfield, or a whole planet of cube faces)
/// and StarField (stars from a catalogue at infinity), as reflected
/// schemas (`prim_type.h`). Each declaration's fallback is the value
/// Ernie renders an unset attribute with.

// clang-format off
#define ERNIE_TERRAIN_ATTRIBUTES(X)                                                                \
    X(::ernie::Tensor, Heights, ::ernie::tokens::kHeights, nullptr,                                \
      (::ernie::AttributeMeta{.doc = "float32 (rows, cols) heights in world units, the first row " \
                                     "the northern; a planet's float32 (6, n + 1, n + 1) cube "    \
                                     "faces.",                                                     \
                              .group = "Terrain", .dirty = ::ernie::Dirty::Points}))               \
    X(glm::dvec2, Origin, ::ernie::tokens::kOrigin, glm::dvec2(0.0),                               \
      (::ernie::AttributeMeta{.doc = "World x and y of the north-west corner", .group = "Terrain"})) \
    X(double, CellSize, ::ernie::tokens::kCellSize, 1.0,                                           \
      (::ernie::AttributeMeta{.doc = "World units between neighbouring heights.",                  \
                              .group = "Terrain", .max = 1e5}))                                    \
    X(double, Exaggeration, ::ernie::tokens::kExaggeration, 1.0,                                   \
      (::ernie::AttributeMeta{.doc = "Scales the heights as drawn.", .group = "Terrain",           \
                              .max = 100.0}))                                                      \
    X(std::string, ColorMode, ::ernie::tokens::kColorMode, ::ernie::tokens::kShaded,               \
      (::ernie::AttributeMeta{.label = "Colour", .doc = "shaded: the material, lit; "              \
                                     "colormap: a height ramp, unlit; shadedColormap: the ramp "   \
                                     "times the material, lit (the ramps for showing heights); "   \
                                     "hillshade: a GIS hillshade, grey and unlit.",                \
                              .group = "Terrain",                                                  \
                              .options = {::ernie::tokens::kShaded, ::ernie::tokens::kColormap,    \
                                          ::ernie::tokens::kShadedColormap,                        \
                                          ::ernie::tokens::kHillshade}}))                          \
    X(double, HillshadeAzimuth, ::ernie::tokens::kHillshadeAzimuth, 315.0,                         \
      (::ernie::AttributeMeta{.label = "Hillshade azimuth", .doc = "Degrees clockwise from "      \
                                     "north (+y; on a planet, towards +z) of the hillshade's "     \
                                     "light; the GIS default 315, from the north-west.",          \
                              .group = "Terrain", .max = 360.0}))                                  \
    X(double, HillshadeAltitude, ::ernie::tokens::kHillshadeAltitude, 45.0,                        \
      (::ernie::AttributeMeta{.label = "Hillshade altitude", .doc = "Degrees above the horizon "  \
                                     "of the hillshade's light.",                                  \
                              .group = "Terrain", .max = 90.0}))                                   \
    X(double, NoData, ::ernie::tokens::kNoData, 0.0,                                               \
      (::ernie::AttributeMeta{.doc = "Height marking missing cells; unset, every cell is a "       \
                                     "height.",                                                    \
                              .group = "Terrain"}))                                                \
    X(double, PlanetRadius, ::ernie::tokens::kPlanetRadius, 0.0,                                   \
      (::ernie::AttributeMeta{.doc = "World units. Set, the Terrain is a whole planet of this "    \
                                     "radius centred at its transform's translation (its "         \
                                     "rotation not yet applied), its heights cube faces, and the " \
                                     "origin and cell size do not apply.",                         \
                              .group = "Terrain"}))                                                \
    X(glm::dvec3, PlanetRadii, ::ernie::tokens::kPlanetRadii, glm::dvec3(0.0),                     \
      (::ernie::AttributeMeta{.label = "Radii", .doc = "World units. Set, the Terrain is a planet " \
                                     "on the ellipsoid of these semi-axes along its x, y and z "   \
                                     "(oblate as a, a, c; triaxial too), heights rising along "    \
                                     "its normals; overrides the radius.",                          \
                              .group = "Terrain"}))                                                \
    X(double, FinestCell, ::ernie::tokens::kFinestCell, 1.0,                                       \
      (::ernie::AttributeMeta{.label = "Finest cell", .doc = "World units. A planet's LOD "        \
                                     "refines below its heights' quads, bilinear between them, "   \
                                     "down to quads about this size; 0 stops at the heights'.",   \
                              .group = "Terrain", .max = 1e6}))                                    \
    X(std::string, HeightsFile, ::ernie::tokens::kHeightsFile, "",                                 \
      (::ernie::AttributeMeta{.label = "Heights file", .doc = "A planet's heights as a "          \
                                     "latitude-longitude raster (TIFF, PDS3, PNG or JPEG; the "    \
                                     "whole body from 180 W unless the file says), in place of "   \
                                     "the heights: faces of up to 1024 quads read at once, finer " \
                                     "ones paged in as views need them.",                          \
                              .group = "Terrain", .asset = true}))                                 \
    X(double, HeightsUnit, ::ernie::tokens::kHeightsUnit, 1.0,                                     \
      (::ernie::AttributeMeta{.label = "Heights unit", .doc = "World units in a unit of the "     \
                                     "heights file's values: 1000 for kilometres.",                \
                              .group = "Terrain", .max = 1e6}))                                    \
    X(double, HeightsReference, ::ernie::tokens::kHeightsReference, 0.0,                           \
      (::ernie::AttributeMeta{.label = "Heights reference", .doc = "World units subtracted from " \
                                     "the heights file's values: the radius, for files of radii.", \
                              .group = "Terrain"}))

#define ERNIE_STAR_FIELD_ATTRIBUTES(X)                                                             \
    X(double, MagnitudeLimit, ::ernie::tokens::kStarsMagnitudeLimit, 99.0,                         \
      (::ernie::AttributeMeta{.doc = "Stars fainter than this are left out", .group = "Stars",     \
                              .min = -30.0, .max = 99.0}))                                         \
    X(double, Exposure, ::ernie::tokens::kExposure, 0.0,                                           \
      (::ernie::AttributeMeta{.doc = "Stops; scales every star by 2^exposure", .group = "Stars",   \
                              .min = -16.0, .max = 16.0}))                                         \
    X(double, ZeroPoint, ::ernie::tokens::kStarsZeroPoint, ::ernie::kMagnitudeZeroLux,             \
      (::ernie::AttributeMeta{.doc = "Lux of a magnitude 0 star (Allen's Astrophysical "           \
                                     "Quantities)",                                                \
                              .group = "Stars", .max = 1.0}))                                      \
    X(::ernie::Tensor, Magnitude, ::ernie::tokens::kStarsMagnitude, nullptr,                       \
      (::ernie::AttributeMeta{.doc = "float32 (stars): apparent visual (Johnson V) magnitudes.",   \
                              .group = "Stars"}))                                                  \
    X(::ernie::Tensor, RightAscension, ::ernie::tokens::kStarsRightAscension, nullptr,             \
      (::ernie::AttributeMeta{.doc = "float32 (stars), degrees, in the catalogue's frame.",        \
                              .group = "Stars"}))                                                  \
    X(::ernie::Tensor, Declination, ::ernie::tokens::kStarsDeclination, nullptr,                   \
      (::ernie::AttributeMeta{.doc = "float32 (stars), degrees, in the catalogue's frame.",        \
                              .group = "Stars"}))                                                  \
    X(::ernie::Tensor, Direction, ::ernie::tokens::kStarsDirection, nullptr,                       \
      (::ernie::AttributeMeta{.doc = "float32 (stars, 3): unit directions to the stars in the "    \
                                     "prim's frame, in place of right ascension and declination.", \
                              .group = "Stars"}))                                                  \
    X(::ernie::Tensor, ColorIndex, ::ernie::tokens::kStarsColorIndex, nullptr,                     \
      (::ernie::AttributeMeta{.doc = "float32 (stars): B-V colour indices, giving each star's "    \
                                     "temperature.",                                               \
                              .group = "Stars"}))                                                  \
    X(::ernie::Tensor, Temperature, ::ernie::tokens::kStarsTemperature, nullptr,                   \
      (::ernie::AttributeMeta{.doc = "float32 (stars), kelvin, in place of the colour indices. "   \
                                     "Without either, stars are 5772 K, the Sun's.",               \
                              .group = "Stars"}))

#define ERNIE_ATMOSPHERE_ATTRIBUTES(X)                                                             \
    X(std::string, Preset, ::ernie::tokens::kAtmospherePreset, "earth",                            \
      (::ernie::AttributeMeta{.doc = "The planet's physical model: Earth's air, ozone and haze, "  \
                                     "or Mars's carbon dioxide and dust.",                         \
                              .group = "Atmosphere",                                               \
                              .options = {::ernie::tokens::kEarth, ::ernie::tokens::kMars}}))      \
    X(double, PlanetRadius, ::ernie::tokens::kAtmospherePlanetRadius, 0.0,                         \
      (::ernie::AttributeMeta{.doc = "Metres; 0 takes the planet Terrain's radius, or else the "   \
                                     "preset planet's, centred under the world's origin.",         \
                              .group = "Atmosphere", .max = 1e9}))                                 \
    X(double, GroundAlbedo, ::ernie::tokens::kAtmosphereGroundAlbedo, 0.3,                         \
      (::ernie::AttributeMeta{.doc = "The ground's reflectance for light scattered in the air.",   \
                              .group = "Atmosphere", .max = 1.0}))                                 \
    X(double, HazeOpticalDepth, ::ernie::tokens::kAtmosphereHazeOpticalDepth, 0.1,                 \
      (::ernie::AttributeMeta{.doc = "Earth: the haze's vertical optical depth at 550 nm.",        \
                              .group = "Atmosphere", .max = 5.0}))                                 \
    X(double, OzoneColumn, ::ernie::tokens::kAtmosphereOzoneColumn, 300.0,                         \
      (::ernie::AttributeMeta{.doc = "Earth: total column ozone, Dobson units.",                   \
                              .group = "Atmosphere", .max = 1000.0}))                              \
    X(double, SodiumColumn, ::ernie::tokens::kAtmosphereSodiumColumn, 4e13,                        \
      (::ernie::AttributeMeta{.doc = "Earth: the mesospheric sodium layer's atoms per square "     \
                                     "metre (about 2e13 to 8e13), which a laser tuned to the "     \
                                     "589 nm D2 line lights as a guide star 92 km up.",            \
                              .group = "Atmosphere", .max = 1e15}))                                \
    X(double, DustOpticalDepth, ::ernie::tokens::kAtmosphereDustOpticalDepth, 0.5,                 \
      (::ernie::AttributeMeta{.doc = "Mars: the dust's vertical optical depth at 880 nm (about "   \
                                     "0.3 to 0.9 in clear seasons, above 1 in dust storms).",      \
                              .group = "Atmosphere", .max = 10.0}))                                \
    X(double, CloudTop, ::ernie::tokens::kAtmosphereCloudTop, 0.0,                                 \
      (::ernie::AttributeMeta{.doc = "Metres: clouds block light grazing the planet below this, "  \
                                     "darkening its shadow, as in a lunar eclipse's umbra; 0 "     \
                                     "for clear air.",                                             \
                              .group = "Atmosphere", .max = 20e3}))                                \
    X(double, CloudCover, ::ernie::tokens::kAtmosphereCloudCover, 0.5,                             \
      (::ernie::AttributeMeta{.doc = "The fraction of the planet's limb the clouds cover.",        \
                              .group = "Atmosphere", .max = 1.0}))                                 \
    X(double, PolarizationDegree, ::ernie::tokens::kPolarizationDegree, 0.75,                      \
      (::ernie::AttributeMeta{.label = "Sky polarization",                                         \
                              .doc = "The sky's greatest degree of linear polarization, 90 "       \
                                     "degrees from the sun: Rayleigh scattering's 1, lowered by "  \
                                     "multiple scattering and haze (a clear sky about 0.75, a "    \
                                     "dusty one far less).",                                       \
                              .group = "Atmosphere", .max = 1.0}))                                 \
    X(std::string, Anchor, ::ernie::tokens::kAtmosphereAnchor, ::ernie::tokens::kAnchorGround,     \
      (::ernie::AttributeMeta{.doc = "ground: its origin is on the planet's surface, the centre "  \
                                     "a radius below along its -z; centre: its origin is the "     \
                                     "planet's centre. Its transform places it; anchored on "      \
                                     "the ground, a planet Terrain's air, centred on it.",         \
                              .group = "Atmosphere",                                               \
                              .options = {::ernie::tokens::kAnchorGround,                          \
                                          ::ernie::tokens::kAnchorCentre}}))

#define ERNIE_TERRAIN_ALL(X) ERNIE_GPRIM_ALL(X) ERNIE_TERRAIN_ATTRIBUTES(X)
#define ERNIE_STAR_FIELD_ALL(X) ERNIE_IMAGEABLE_ATTRIBUTES(X) ERNIE_STAR_FIELD_ATTRIBUTES(X)
#define ERNIE_ATMOSPHERE_ALL(X) ERNIE_IMAGEABLE_ATTRIBUTES(X) ERNIE_ATMOSPHERE_ATTRIBUTES(X)
// clang-format on

namespace ernie {

/// A north-up heightfield (x east, y north, z up) drawn with its
/// bound material, or, with a planet radius, a whole planet of six
/// cube faces (`CubeSphereLod`). Renders draw the first visible one.
///
/// The cells that changed with the last write of heights on a GPU,
/// `ernie:heights:dirty` (four numbers), are read generically: the
/// registry has no four-vector type.
class Terrain : public Gprim {
    ERNIE_SCHEMA_BODY(Terrain, Gprim, ERNIE_TERRAIN_ALL)
    ERNIE_SCHEMA_DEFINE(Terrain)
};

/// Stars from a catalogue as points of light at infinity, behind
/// everything else, each with a black-body spectrum. The prim's
/// transform takes the catalogue's frame (J2000 equatorial, x towards
/// right ascension 0, z the north celestial pole) into the world; a
/// star's illuminance is the zero point times 10^(-0.4 magnitude)
/// times 2^exposure. Renders draw the first visible one.
///
/// The star arrays are written as tensors; `StarsOf` also reads them
/// as float array values.
class StarField : public Imageable {
    ERNIE_SCHEMA_BODY(StarField, Imageable, ERNIE_STAR_FIELD_ALL)
    ERNIE_SCHEMA_DEFINE(StarField)
};

/// A planet's atmosphere from a physical model (`EarthAtmosphere`,
/// `MarsAtmosphere`): its gases and aerosols scatter and absorb the
/// sun's light on its way to the ground, the sun's DistantLight then
/// giving the irradiance above the atmosphere. It surrounds the
/// planet Terrain, or without one a planet of `planetRadius` centred
/// that far below the world's origin (z up), so a flat scene's ground
/// at z = 0 is its surface. Renders use the first visible one.
class Atmosphere : public Imageable {
    ERNIE_SCHEMA_BODY(Atmosphere, Imageable, ERNIE_ATMOSPHERE_ALL)
    ERNIE_SCHEMA_DEFINE(Atmosphere)
};

/// This file's types' registry entries, bases first (`PrimTypes`).
std::vector<const PrimTypeDef*> TerrainSchemaTypes();

}  // namespace ernie
