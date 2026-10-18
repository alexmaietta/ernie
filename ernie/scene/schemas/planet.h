// SPDX-License-Identifier: BSD-3-Clause
/// @file planet.h
/// @brief Ernie's planetary prims: terrains on bodies or planes, atmospheres around bodies, star
/// fields and ground planes. A body is any Sphere, an ellipsoid when scaled unevenly.
#pragma once

#include "ernie/physics/astronomy.h"
#include "ernie/scene/schemas/lux.h"

// clang-format off
#define ERNIE_TERRAIN_ATTRIBUTES(X)                                                                \
    X(Path, Body, "ernie:terrain:body", Path(),                                                    \
      "The Sphere whose surface the heights displace, along its geodetic normals; unset, a plane " \
      "in the terrain's xy plane.",                                                                \
      .group = "Terrain", .dirty = Dirty::Topology)                                                \
    X(Tensor, Heights, "ernie:terrain:heights", Tensor(),                                          \
      "float32 (rows, columns) heights in metres, the first row the northern; on a body, "         \
      "(6, n + 1, n + 1) cube faces.",                                                             \
      .group = "Terrain", .ranks = {2, 3}, .dirty = Dirty::Points)                                 \
    X(std::string, HeightsFile, "ernie:terrain:heights:file", "",                                  \
      "Heights as a raster (TIFF, PDS3, PNG or JPEG) in place of the array, paged as views need "  \
      "them.",                                                                                     \
      .group = "Terrain", .asset = true, .dirty = Dirty::Points)                                   \
    X(double, HeightsUnit, "ernie:terrain:heights:unit", 1.0,                                      \
      "Metres in a unit of the file's values: 1000 for kilometres.", .group = "Terrain",           \
      .lower = Above(0.0), .dirty = Dirty::Points)                                                 \
    X(double, HeightsReference, "ernie:terrain:heights:reference", 0.0,                            \
      "Metres subtracted from the file's values: the radius, for files of radii.",                 \
      .group = "Terrain", .finite = true, .dirty = Dirty::Points)                                  \
    X(double, NoData, "ernie:terrain:noData", 0.0,                                                 \
      "The height marking missing cells; unauthored, every cell is a height.", .group = "Terrain") \
    X(glm::dvec2, Origin, "ernie:terrain:origin", glm::dvec2(0.0),                                 \
      "A plane's north-west corner in its x and y, metres.", .group = "Terrain", .finite = true)   \
    X(double, CellSize, "ernie:terrain:cellSize", 1.0, "Metres between a plane's heights.",        \
      .group = "Terrain", .lower = Above(0.0), .finite = true)                                     \
    X(double, Exaggeration, "ernie:terrain:exaggeration", 1.0, "Scales the heights as drawn.",     \
      .group = "Terrain", .lower = AtLeast(0.0), .finite = true)                                   \
    X(double, FinestCell, "ernie:terrain:finestCell", 1.0,                                         \
      "Metres: a body's LOD refines below its heights, bilinearly, to quads this size; 0 stops at " \
      "the heights'.",                                                                             \
      .group = "Terrain", .lower = AtLeast(0.0))                                                   \
    X(std::string, ColorMode, "ernie:terrain:colorMode", "shaded",                                 \
      "shaded: its material, lit; colormap: a height ramp, unlit; shadedColormap: both; "          \
      "hillshade: a GIS hillshade, grey and unlit.",                                               \
      .group = "Display", .options = {"shaded", "colormap", "shadedColormap", "hillshade"})        \
    X(double, HillshadeAzimuth, "ernie:terrain:hillshade:azimuth", 315.0,                          \
      "Degrees clockwise from north of the hillshade's light; GIS's 315, from the north-west.",    \
      .group = "Display", .lower = AtLeast(0.0), .upper = AtMost(360.0))                           \
    X(double, HillshadeAltitude, "ernie:terrain:hillshade:altitude", 45.0,                         \
      "Degrees above the horizon of the hillshade's light.", .group = "Display",                   \
      .lower = AtLeast(0.0), .upper = AtMost(90.0))

#define ERNIE_ATMOSPHERE_COLUMN(X, Name, usdName, fallback, text)                                  \
    X(double, Name, usdName, fallback, text, .group = "Atmosphere", .lower = AtLeast(0.0),         \
      .finite = true)

#define ERNIE_ATMOSPHERE_ATTRIBUTES(X)                                                             \
    X(Path, Body, "ernie:atmosphere:body", Path(), "The Sphere it surrounds.",                     \
      .group = "Atmosphere", .dirty = Dirty::Topology)                                             \
    X(std::string, Preset, "ernie:atmosphere:preset", "earth",                                     \
      "Its physical model: Earth's air, ozone and haze, or Mars's carbon dioxide and dust.",       \
      .group = "Atmosphere", .options = {"earth", "mars"})                                         \
    ERNIE_ATMOSPHERE_COLUMN(X, Top, "ernie:atmosphere:top", 100000.0,                              \
                            "Metres above the surface it reaches; Earth's 100 km.")                \
    X(double, GroundAlbedo, "ernie:atmosphere:groundAlbedo", 0.3,                                  \
      "The ground's reflectance for light scattered in the air.", .group = "Atmosphere",           \
      .lower = AtLeast(0.0), .upper = AtMost(1.0))                                                 \
    ERNIE_ATMOSPHERE_COLUMN(X, HazeOpticalDepth, "ernie:atmosphere:hazeOpticalDepth", 0.1,         \
                            "Earth: the haze's vertical optical depth at 550 nm.")                 \
    ERNIE_ATMOSPHERE_COLUMN(X, OzoneColumn, "ernie:atmosphere:ozoneColumn", 300.0,                 \
                            "Earth: total column ozone, Dobson units.")                            \
    ERNIE_ATMOSPHERE_COLUMN(X, SodiumColumn, "ernie:atmosphere:sodiumColumn", 4e13,                \
                            "Earth: mesospheric sodium atoms per square metre, lit 92 km up by a " \
                            "589 nm laser as a guide star.")                                       \
    ERNIE_ATMOSPHERE_COLUMN(X, DustOpticalDepth, "ernie:atmosphere:dustOpticalDepth", 0.5,         \
                            "Mars: the dust's vertical optical depth at 880 nm.")                  \
    ERNIE_ATMOSPHERE_COLUMN(X, CloudTop, "ernie:atmosphere:cloudTop", 0.0,                         \
                            "Metres below which clouds block light grazing the limb, darkening "   \
                            "its shadow as in a lunar eclipse; 0 clear.")                          \
    X(double, CloudCover, "ernie:atmosphere:cloudCover", 0.5,                                      \
      "The fraction of the limb the clouds cover.", .group = "Atmosphere", .lower = AtLeast(0.0),  \
      .upper = AtMost(1.0))                                                                        \
    X(double, PolarizationDegree, "ernie:polarization:degree", 0.75,                               \
      "The sky's greatest degree of linear polarization, 90 degrees from the sun.",                \
      .label = "Sky polarization", .group = "Atmosphere", .lower = AtLeast(0.0),                   \
      .upper = AtMost(1.0))

#define ERNIE_STAR_FIELD_ATTRIBUTES(X)                                                             \
    X(double, MagnitudeLimit, "ernie:stars:magnitudeLimit", 99.0,                                  \
      "Stars fainter than this are left out.", .group = "Stars", .finite = true)                   \
    X(double, ZeroPoint, "ernie:stars:zeroPoint", kMagnitudeZeroLux,                               \
      "Lux of a magnitude 0 star (Allen's Astrophysical Quantities).", .group = "Stars",           \
      .lower = Above(0.0))                                                                         \
    X(Tensor, Magnitude, "ernie:stars:magnitude", Tensor(),                                        \
      "float32 (stars): apparent visual (Johnson V) magnitudes.", .group = "Stars")                \
    X(Tensor, RightAscension, "ernie:stars:rightAscension", Tensor(),                              \
      "float32 (stars), degrees, in the catalogue's frame.", .group = "Stars")                     \
    X(Tensor, Declination, "ernie:stars:declination", Tensor(),                                    \
      "float32 (stars), degrees, in the catalogue's frame.", .group = "Stars")                     \
    X(Tensor, Direction, "ernie:stars:direction", Tensor(),                                        \
      "float32 (stars, 3) unit directions in its own space, in place of right ascension and "      \
      "declination.",                                                                              \
      .group = "Stars", .components = 3)                                                           \
    X(Tensor, ColorIndex, "ernie:stars:colorIndex", Tensor(),                                      \
      "float32 (stars): B-V colour indices, giving each star's temperature.", .group = "Stars")    \
    X(Tensor, Temperature, "ernie:stars:temperature", Tensor(),                                    \
      "float32 (stars) kelvin, in place of the colour indices; without either, the Sun's 5772 K.", \
      .group = "Stars")

#define ERNIE_GROUND_PLANE_ATTRIBUTES(X)                                                           \
    X(bool, Checker, "ernie:checker", false, "Multiplies its material's albedo by checker squares.", \
      .group = "Ground")                                                                           \
    X(double, CheckerSize, "ernie:checker:size", 1.0, "Metres along a square's edge.",             \
      .group = "Ground", .lower = Above(0.0), .finite = true)                                      \
    X(glm::dvec3, CheckerColor0, "ernie:checker:color0", glm::dvec3(1.0), "Linear RGB.",           \
      .group = "Ground", .lower = AtLeast(0.0), .color = true)                                     \
    X(glm::dvec3, CheckerColor1, "ernie:checker:color1", glm::dvec3(0.5), "Linear RGB.",           \
      .group = "Ground", .lower = AtLeast(0.0), .color = true)
// clang-format on

namespace ernie {

ERNIE_CONCRETE_SCHEMA(Terrain, Gprim, ERNIE_TERRAIN_ATTRIBUTES,
                      "Heights displacing a body's surface, or a plane.");
ERNIE_CONCRETE_SCHEMA(Atmosphere, Xformable, ERNIE_ATMOSPHERE_ATTRIBUTES,
                      "The air around a body: its constituents, haze and clouds.");
ERNIE_CONCRETE_SCHEMA(StarField, NonboundableLightBase, ERNIE_STAR_FIELD_ATTRIBUTES,
                      "The stars, from a catalogue or arrays, at infinity: a light, unbounded.");
ERNIE_CONCRETE_SCHEMA(GroundPlane, Gprim, ERNIE_GROUND_PLANE_ATTRIBUTES,
                      "An infinite plane in its xy plane, facing +z.");

}  // namespace ernie
