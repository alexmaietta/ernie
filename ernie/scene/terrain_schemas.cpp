// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/terrain_schemas.h"

namespace ernie {

ERNIE_DEFINE_SCHEMA(Terrain, tokens::kTerrain, &Gprim::Definition(),
                    false,
                    "A north-up heightfield (x east, y north, z up), "
                    "or with a planet radius a "
                    "whole planet of six cube faces.",
                    ERNIE_TERRAIN_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(
    StarField, tokens::kStarField, &Imageable::Definition(), false,
    "Stars from a catalogue as points of light at infinity, behind "
    "everything "
    "else; its transform takes the catalogue's frame into the world.",
    ERNIE_STAR_FIELD_ATTRIBUTES)

ERNIE_DEFINE_SCHEMA(
    Atmosphere, tokens::kAtmosphere, &Imageable::Definition(), false,
    "A planet's atmosphere from a physical model, around the planet "
    "Terrain or a planet under the world's origin.",
    ERNIE_ATMOSPHERE_ATTRIBUTES)

ERNIE_DEFINE_SCHEMA(
    CelestialBody, tokens::kCelestialBody, &Imageable::Definition(),
    false,
    "A sphere far beyond the scene, such as a moon, drawn in the sky "
    "and casting eclipse shadows.",
    ERNIE_CELESTIAL_BODY_ATTRIBUTES)

std::vector<const PrimTypeDef*> TerrainSchemaTypes() {
    return {&Terrain::Definition(), &StarField::Definition(),
            &Atmosphere::Definition(), &CelestialBody::Definition()};
}

}  // namespace ernie
