// SPDX-License-Identifier: BSD-3-Clause
/// @file laser.cpp
/// @brief The laser's registry entry and extent: its emitter, the larger of its aperture and waist.
#include "ernie/scene/schemas/laser.h"

#include <algorithm>

namespace ernie {
namespace {

auto EmitterHalves(const ExtentRows& rows) {
    return [aperture = rows.Column<double>(LaserLight::kApertureRadius),
            waist = rows.Column<glm::dvec2>(LaserLight::kWaistRadius)](std::uint32_t row) {
        const double radius = std::max({aperture[row], waist[row].x, waist[row].y});
        return glm::dvec3(radius, radius, 0.0);
    };
}

}  // namespace

ERNIE_DEFINE_SCHEMA(LaserLight, Concrete, ComputedExtent(CentredExtent<EmitterHalves>),
                    ERNIE_LASER_LIGHT_ATTRIBUTES,
                    .fallbackOverrides = {{"ernie:power", 0.001},
                                          {"ernie:polarization:degree", 0.98}})

}  // namespace ernie
