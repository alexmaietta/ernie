// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/base_schemas.h"
#include "ernie/scene/prim_type.h"
#include "ernie/scene/tokens.h"
#include "ernie/util/astronomy.h"

#include <vector>

/// Ernie's Terrain (a heightfield, or a whole planet of cube faces) and StarField (stars from a
/// catalogue at infinity), as reflected schemas (`prim_type.h`). Each declaration's fallback is the
/// value Ernie renders an unset attribute with.

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
    X(double, NoData, ::ernie::tokens::kNoData, 0.0,                                               \
      (::ernie::AttributeMeta{.doc = "Height marking missing cells; unset, every cell is a "       \
                                     "height.",                                                    \
                              .group = "Terrain"}))                                                \
    X(double, PlanetRadius, ::ernie::tokens::kPlanetRadius, 0.0,                                   \
      (::ernie::AttributeMeta{.doc = "World units. Set, the Terrain is a whole planet of this "    \
                                     "radius centred at the world's origin, its heights cube "     \
                                     "faces, and the origin and cell size do not apply.",          \
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

#define ERNIE_TERRAIN_ALL(X) ERNIE_GPRIM_ALL(X) ERNIE_TERRAIN_ATTRIBUTES(X)
#define ERNIE_STAR_FIELD_ALL(X) ERNIE_IMAGEABLE_ATTRIBUTES(X) ERNIE_STAR_FIELD_ATTRIBUTES(X)
// clang-format on

namespace ernie {

/// A north-up heightfield (x east, y north, z up) drawn with its bound material, or, with a planet
/// radius, a whole planet of six cube faces (`CubeSphereLod`). Renders draw the first visible one.
///
/// The cells that changed with the last write of heights on a GPU, `ernie:heights:dirty` (four
/// numbers), are read generically: the registry has no four-vector type.
class Terrain : public Gprim {
    ERNIE_SCHEMA_BODY(Terrain, Gprim, ERNIE_TERRAIN_ALL)
    ERNIE_SCHEMA_DEFINE(Terrain)
};

/// Stars from a catalogue as points of light at infinity, behind everything else, each with a
/// black-body spectrum. The prim's transform takes the catalogue's frame (J2000 equatorial, x
/// towards right ascension 0, z the north celestial pole) into the world; a star's illuminance is
/// the zero point times 10^(-0.4 magnitude) times 2^exposure. Renders draw the first visible one.
///
/// The star arrays are written as tensors; `StarsOf` also reads them as float array values.
class StarField : public Imageable {
    ERNIE_SCHEMA_BODY(StarField, Imageable, ERNIE_STAR_FIELD_ALL)
    ERNIE_SCHEMA_DEFINE(StarField)
};

/// This file's types' registry entries, bases first (`PrimTypes`).
std::vector<const PrimTypeDef*> TerrainSchemaTypes();

}  // namespace ernie
