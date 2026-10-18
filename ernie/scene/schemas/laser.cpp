// SPDX-License-Identifier: BSD-3-Clause
/// @file laser.cpp
/// @brief The laser's registry entry and extent: its emitter, the larger of its aperture and waist.
#include "ernie/scene/schemas/laser.h"

#include <algorithm>

namespace ernie {
namespace {

void LaserExtent(const ExtentRows& rows, RowRange range, std::span<Box> out) {
    const auto aperture = rows.Column<double>(LaserLight::kApertureRadius);
    const auto waist = rows.Column<glm::dvec2>(LaserLight::kWaistRadius);
    for (std::uint32_t row = range.begin; row < range.end; ++row) {
        const double radius = std::max({aperture[row], waist[row].x, waist[row].y});
        out[row - range.begin] = Box::Around(glm::dvec3(0.0), {radius, radius, 0.0});
    }
}

}  // namespace

ERNIE_DEFINE_SCHEMA(LaserLight, "LaserLight", &BoundableLightBase::Definition(), Concrete,
                    ComputedExtent(LaserExtent),
                    "A laser along -z: a Gaussian beam and its pattern.",
                    ERNIE_LASER_LIGHT_ATTRIBUTES,
                    .fallbackOverrides = {{"ernie:power", 0.001},
                                          {"ernie:polarization:degree", 0.98}})

}  // namespace ernie
