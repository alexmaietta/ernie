// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/util/tensor.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

/// Planetary digital elevation models, as the PDS archives them, and
/// the local heightfields Ernie's Terrain draws of them.
///
/// References:
/// - NASA Planetary Data System, "Planetary Data System Standards
///   Reference", version 3.8, 2009: chapter 12 (the object
///   description language of PDS3 labels).
/// - G. A. Neumann, "Lunar Reconnaissance Orbiter Lunar Orbiter Laser
///   Altimeter Reduced Data Record and Derived Products Software
///   Interface Specification", version 2.42, 2010 (the LDEM products,
///   their labels and simple cylindrical grids).
/// - J. P. Snyder, "Map Projections: A Working Manual", USGS
///   Professional Paper 1395, 1987 (the equirectangular projection).
namespace ernie {

/// A PDS3 label's keywords, by name; a keyword inside an OBJECT
/// prefixed with the object's name and a dot (`IMAGE.LINES`). Values
/// keep their text, units (`<km>`) and quotes removed.
struct Pds3Label {
    std::map<std::string, std::string, std::less<>> values;

    /// The keyword's text, or none.
    std::optional<std::string> Get(std::string_view key) const;
    /// The keyword as a number.
    /// @throws std::invalid_argument when missing or not a number.
    double Number(std::string_view key) const;
    /// The keyword's text.
    /// @throws std::invalid_argument when missing.
    std::string Text(std::string_view key) const;
};

/// Parses label text: `KEY = value` lines, `OBJECT`/`END_OBJECT`
/// nesting, comments (`/* */`) and values continued over lines inside
/// quotes or parentheses.
Pds3Label ParsePds3Label(std::string_view text);

/// A planet's radius on a simple cylindrical (equirectangular) grid,
/// as LOLA's LDEM products hold it: `heights` (metres above
/// `referenceRadius`) row by row from the northern, each row from the
/// western sample eastwards, `pixelsPerDegree` both ways. Sample s of
/// line l is centred at latitude maxLatitude - (l + 0.5) / ppd and
/// east longitude westLongitude + (s + 0.5) / ppd.
struct CylindricalDem {
    std::uint32_t lines = 0;
    std::uint32_t samples = 0;
    double pixelsPerDegree = 1.0;
    double maxLatitudeDeg = 90.0;
    double westLongitudeDeg = 0.0;
    double referenceRadius = 0.0;  ///< Metres.
    std::vector<float> heights;

    /// The height at latitude `latitudeDeg` and east longitude
    /// `longitudeDeg` (any multiple of 360 apart), bilinear between
    /// the cells' centres, clamped to the grid's edges.
    double HeightAt(double latitudeDeg, double longitudeDeg) const;
};

/// Reads the DEM a PDS3 label at `labelPath` describes: its IMAGE
/// object (16- or 32-bit LSB integers, or 32-bit LSB reals, each
/// times SCALING_FACTOR, plus the label's OFFSET where the label says
/// the heights are radii) in the file its ^IMAGE pointer names beside
/// the label, on its IMAGE_MAP_PROJECTION's simple cylindrical grid.
/// @throws std::runtime_error if a file cannot be read, or
/// std::invalid_argument for another projection, sample type or a
/// label it cannot follow.
CylindricalDem ReadCylindricalDem(const std::string& labelPath);

/// A heightfield of a DEM about one point, in a local frame there:
/// x east, y north and z up, its origin on the reference sphere at
/// the point. Each cell holds the height above that tangent plane of
/// the surface straight below or above it, so the planet's curvature
/// is in the heights (a drop of about x^2 / 2R).
struct LocalTerrain {
    Tensor heights;     ///< float32 (rows, cols), the first northern.
    glm::dvec2 origin;  ///< x and y of the north-western cell.
    double cellSize = 1.0;
    /// Body-fixed coordinates (metres, the planet's centre at the
    /// origin) to the local frame.
    glm::dmat4 bodyToLocal{1.0};
};

/// The `cells` x `cells` heightfield of `dem`, `cellSize` metres
/// apart, about latitude `latitudeDeg` and east longitude
/// `longitudeDeg` (`LocalTerrain`).
/// @throws std::invalid_argument for fewer than 2 cells or a
/// non-positive cell size.
LocalTerrain LocalTerrainOf(const CylindricalDem& dem,
                            double latitudeDeg, double longitudeDeg,
                            std::uint32_t cells, double cellSize);

/// The body-fixed unit vector of planetocentric latitude
/// `latitudeDeg` and east longitude `longitudeDeg`.
glm::dvec3 BodyDirection(double latitudeDeg, double longitudeDeg);

}  // namespace ernie
