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

/// @return The body a row's relationship names, its extent brought into the row's own space; none
/// when it names none.
std::optional<Box> BodyInOwnSpace(const ExtentRows& rows, std::string_view relationship,
                                  std::uint32_t row) {
    const Target target = rows.Column<Target>(relationship)[row];
    const std::optional<PrimId> body =
        target.path != kNoPath ? rows.store.Find(target.path) : std::nullopt;
    const std::optional<Box> extent = body ? rows.scene->Extent(*body) : std::nullopt;
    if (!extent) {
        return std::nullopt;
    }
    const PrimId own = rows.store.GetBucket(rows.bucket).prims[row];
    return extent->Transformed(glm::inverse(rows.scene->WorldTransform(own)) *
                               rows.scene->WorldTransform(*body));
}

/// @return The lowest and highest of a host array's heights; none on a device or when empty.
std::optional<std::pair<double, double>> HeightRange(const Tensor& heights) {
    if (heights.Empty() || IsGpu(heights.Location().kind)) {
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
    for (std::uint32_t row = range.begin; row < range.end; ++row) {
        const std::optional<std::pair<double, double>> span = HeightRange(heights[row]);
        const double low = span ? span->first * exaggeration[row] : 0.0;
        const double high = span ? span->second * exaggeration[row] : 0.0;
        Box& box = out[row - range.begin];
        if (const std::optional<Box> body = BodyInOwnSpace(rows, Terrain::kBody, row)) {
            box = body->Grown(std::max(high, 0.0));
        } else if (span && heights[row].Rank() == 2) {
            box = PlaneFootprint(heights[row], origin[row], cell[row], low, high);
        } else {
            box = AuthoredExtent(authored[row]);
        }
    }
}

void AtmosphereExtent(const ExtentRows& rows, RowRange range, std::span<Box> out) {
    const auto top = rows.Column<double>(Atmosphere::kTop);
    for (std::uint32_t row = range.begin; row < range.end; ++row) {
        const std::optional<Box> body = BodyInOwnSpace(rows, Atmosphere::kBody, row);
        out[row - range.begin] = body ? body->Grown(top[row]) : Box();
    }
}

}  // namespace

ERNIE_DEFINE_SCHEMA(Terrain, "Terrain", &Gprim::Definition(), Concrete,
                    ComputedFromOtherPrims(TerrainExtent),
                    "Heights displacing a body's surface, or a plane.", ERNIE_TERRAIN_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Atmosphere, "Atmosphere", &Xformable::Definition(), Concrete,
                    ComputedFromOtherPrims(AtmosphereExtent),
                    "The air around a body: its constituents, haze and clouds.",
                    ERNIE_ATMOSPHERE_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(StarField, "StarField", &NonboundableLightBase::Definition(), Concrete,
                    kInheritExtent, "The stars, from a catalogue or arrays, at infinity.",
                    ERNIE_STAR_FIELD_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(GroundPlane, "GroundPlane", &Gprim::Definition(), Concrete, kUnbounded,
                    "An infinite plane in its xy plane, facing +z.", ERNIE_GROUND_PLANE_ATTRIBUTES)

}  // namespace ernie
