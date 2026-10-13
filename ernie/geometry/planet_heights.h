// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/util/raster_source.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <mutex>
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

/// A square of a planet's cube-face grid read from a latitude-
/// longitude raster (`PlanetTileOf`), its points row by row.
struct PlanetTile {
    /// (cells + 1)^2 heights: the raster's, bilinear between its
    /// samples' centres, at each point's direction. Not filtered, so
    /// a coarser grid's points have the same heights as a finer
    /// grid's there: patches of any level agree where they meet.
    std::vector<float> heights;
    /// The heights' slopes along increasing column and row at each
    /// point, per world unit across the body's surface: of the raster
    /// box-filtered to about the grid's spacing, for normals without
    /// aliasing.
    std::vector<glm::vec2> slopes;
    /// When asked for, cells^2 quads' (min, max) heights, bounding
    /// the surface within each: of the raster's samples around it.
    std::vector<glm::vec2> ranges;
};

/// The tile of `cells` quads square from grid point (`column`,
/// `row`) of face `face` of a planet's cube-face grid of `faceCells`
/// quads a face (`CubeSphereLod`), on the ellipsoid of semi-axes
/// `radii`, read from the first channel of `source` as
/// `PlanetHeightsOf` reads it: only the window of the raster under
/// the tile (whole rows across the seam or a pole), with `reading`
/// locked while it is read when given, so tiles are made on several
/// threads from one raster.
///
/// References:
/// - J. P. Snyder, "Map Projections: A Working Manual", USGS
///   Professional Paper 1395, 1987 (the equirectangular projection).
/// @throws std::invalid_argument for no cells, a tile past the face
/// or a grid without positive spacing; what `source.Read` throws.
PlanetTile PlanetTileOf(const RasterSource& source,
                        const PlanetHeightsOptions& options,
                        const glm::dvec3& radii,
                        std::uint32_t faceCells, std::uint32_t face,
                        std::uint32_t column, std::uint32_t row,
                        std::uint32_t cells, bool ranges = false,
                        std::mutex* reading = nullptr);

/// A whole planet's heights at `faceCells` quads a face, made a
/// `PlanetTileOf` tile of 256 quads at a time on every hardware
/// thread: the (6, faceCells + 1, faceCells + 1) heights and slopes
/// and the (6, faceCells, faceCells) quads' ranges, face by face.
/// @throws as `PlanetTileOf`.
PlanetTile PlanetFacesOf(const RasterSource& source,
                         const PlanetHeightsOptions& options,
                         const glm::dvec3& radii,
                         std::uint32_t faceCells);

}  // namespace ernie
