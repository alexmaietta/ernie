// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/util/raster_source.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace ernie {

/// How a latitude-longitude raster's samples become heights over a
/// planet (`PlanetHeightsOf`).
struct PlanetHeightsOptions {
    /// World units (metres) in a unit of the raster's physical values
    /// (sample * Scale() + Offset()): 1000 for kilometres.
    double unit = 1.0;
    /// Subtracted after `unit`: the planet's radius for rasters of
    /// radii (LOLA's PDS LDEMs), 0 for rasters of heights.
    double reference = 0.0;
    /// Where the raster lies, when not where its file says
    /// (`RasterSource::Grid`). With neither, it covers the whole body
    /// from 180 degrees west and the north pole.
    std::optional<LatLongGrid> grid;
};

/// The heights of a planet's six cube faces of `faceCells` quads
/// each, resampled from the first channel of `source`, a raster in
/// simple cylindrical (equirectangular) projection: Terrain's planet
/// heights, a (6, faceCells + 1, faceCells + 1) grid of floats face
/// by face, row by row, at `CubeSphereLod`'s vertices.
///
/// The raster is read a band of rows at a time and box-filtered down
/// to about the faces' spacing (90 / faceCells degrees), so a source
/// far finer than the faces neither aliases nor need fit in memory,
/// then sampled bilinearly between the filtered cells' centres. A
/// raster spanning 360 degrees wraps in longitude. Samples without
/// data are left out of both; a vertex with none about it, or off the
/// raster, is at height 0.
///
/// References:
/// - J. P. Snyder, "Map Projections: A Working Manual", USGS
///   Professional Paper 1395, 1987 (the equirectangular projection).
/// @throws std::invalid_argument for no face cells or a grid without
/// positive spacing; what `source.Read` throws.
std::vector<float>
PlanetHeightsOf(const RasterSource& source, std::uint32_t faceCells,
                const PlanetHeightsOptions& options = {});

}  // namespace ernie
