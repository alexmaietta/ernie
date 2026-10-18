// SPDX-License-Identifier: BSD-3-Clause
/// @file planet.cpp
/// @brief The planetary prims' registry entries and extents: terrains from their heights and
/// bodies, atmospheres from their bodies and tops.
#include "ernie/scene/schemas/planet.h"

#include <algorithm>
#include <optional>
#include <vector>

namespace ernie {
namespace {

/// @return The lowest and highest of a host array's heights; none on a device or when empty.
std::optional<std::pair<double, double>> HeightRange(const Tensor& heights) {
    if (heights.Empty() || !heights.OnHost()) {
        return std::nullopt;
    }
    const std::vector<float> values = CopyFloats(heights, "heights");
    const auto [low, high] = std::ranges::minmax_element(values);
    return std::pair<double, double>(*low, *high);
}

/// A plane's footprint: x east from the origin, y south from it, a cell between heights.
Box PlaneFootprint(const Tensor& heights, glm::dvec2 origin, double cell, double low, double high) {
    const double rows = static_cast<double>(heights.Shape()[0]);
    const double columns = static_cast<double>(heights.Shape()[1]);
    return {{origin.x, origin.y - (rows - 1.0) * cell, low},
            {origin.x + (columns - 1.0) * cell, origin.y, high}};
}

void TerrainExtent(const ExtentRows& rows, RowRange range, std::span<Box> out) {
    const auto heights = rows.Column<Tensor>(Terrain::kHeights);
    const auto origin = rows.Column<glm::dvec2>(Terrain::kOrigin);
    const auto cell = rows.Column<double>(Terrain::kCellSize);
    const auto exaggeration = rows.Column<double>(Terrain::kExaggeration);
    const auto authored = rows.Column<std::vector<float>>(Boundable::kExtent);
    const auto bodies = rows.Column<Target>(Terrain::kBody);
    ComputeRows(range, out, [&](std::uint32_t row) {
        const std::optional<std::pair<double, double>> span = HeightRange(heights[row]);
        const double low = span ? span->first * exaggeration[row] : 0.0;
        const double high = span ? span->second * exaggeration[row] : 0.0;
        if (const std::optional<Box> body = rows.TargetExtent(bodies[row], row)) {
            return body->Grown(std::max(high, 0.0));
        }
        return span && heights[row].Rank() == 2
                   ? PlaneFootprint(heights[row], origin[row], cell[row], low, high)
                   : AuthoredExtent(authored[row]);
    });
}

void AtmosphereExtent(const ExtentRows& rows, RowRange range, std::span<Box> out) {
    const auto top = rows.Column<double>(Atmosphere::kTop);
    const auto bodies = rows.Column<Target>(Atmosphere::kBody);
    ComputeRows(range, out, [&](std::uint32_t row) {
        const std::optional<Box> body = rows.TargetExtent(bodies[row], row);
        return body ? body->Grown(top[row]) : Box();
    });
}

}  // namespace

ERNIE_DEFINE_SCHEMA(Terrain, Concrete, ComputedFromOtherPrims(TerrainExtent),
                    ERNIE_TERRAIN_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Atmosphere, Concrete, ComputedFromOtherPrims(AtmosphereExtent),
                    ERNIE_ATMOSPHERE_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(StarField, Concrete, kInheritExtent, ERNIE_STAR_FIELD_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(GroundPlane, Concrete, kUnbounded, ERNIE_GROUND_PLANE_ATTRIBUTES)

}  // namespace ernie
