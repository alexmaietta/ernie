// SPDX-License-Identifier: BSD-3-Clause
/// @file lux.cpp
/// @brief UsdLux's registry entries and the shaped lights' extents, as UsdLux computes them.
#include "ernie/scene/schemas/lux.h"

namespace ernie {
namespace {

/// Each row's box from its half-size, `half(columns, row)`: shaped lights are centred.
template <auto kHalf>
void CentredExtent(const ExtentRows& rows, RowRange range, std::span<Box> out) {
    for (std::uint32_t row = range.begin; row < range.end; ++row) {
        out[row - range.begin] = Box::Around(glm::dvec3(0.0), kHalf(rows, row));
    }
}

glm::dvec3 DiskHalf(const ExtentRows& rows, std::uint32_t row) {
    const double radius = rows.Column<double>(DiskLight::kRadius)[row];
    return {radius, radius, 0.0};
}

/// Rect and portal lights alike: their width and height in the xy plane.
glm::dvec3 RectHalf(const ExtentRows& rows, std::uint32_t row) {
    return {0.5 * rows.Column<double>(RectLight::kWidth)[row],
            0.5 * rows.Column<double>(RectLight::kHeight)[row], 0.0};
}

glm::dvec3 SphereHalf(const ExtentRows& rows, std::uint32_t row) {
    return glm::dvec3(rows.Column<double>(SphereLight::kRadius)[row]);
}

glm::dvec3 CylinderHalf(const ExtentRows& rows, std::uint32_t row) {
    const double radius = rows.Column<double>(CylinderLight::kRadius)[row];
    return {0.5 * rows.Column<double>(CylinderLight::kLength)[row], radius, radius};
}

}  // namespace

// Every light has LightAPI and its light-linking collections built in: which prims it lights and
// which cast its shadows, everything unless they say otherwise.
#define ERNIE_LIGHT_BUILTINS                                                                       \
    .builtinApis = {"LightAPI", "CollectionAPI:lightLink", "CollectionAPI:shadowLink"},            \
    .fallbackOverrides = {{"collection:lightLink:includeRoot", true},                              \
                          {"collection:shadowLink:includeRoot", true}}

ERNIE_DEFINE_SCHEMA(LightAPI, "LightAPI", nullptr, SingleApply, kInheritExtent,
                    "What every light has: brightness, colour, spectrum, polarization and pulses.",
                    ERNIE_LIGHT_API_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(BoundableLightBase, "BoundableLightBase", &Boundable::Definition(), Abstract,
                    kInheritExtent, "A light with a shape, bounded by it.",
                    ERNIE_BOUNDABLE_LIGHT_BASE_ATTRIBUTES, ERNIE_LIGHT_BUILTINS)
ERNIE_DEFINE_SCHEMA(NonboundableLightBase, "NonboundableLightBase", &Xformable::Definition(),
                    Abstract, kUnbounded, "A light at infinity or around everything.",
                    ERNIE_NO_ATTRIBUTES, ERNIE_LIGHT_BUILTINS)
ERNIE_DEFINE_SCHEMA(DistantLight, "DistantLight", &NonboundableLightBase::Definition(), Concrete,
                    kInheritExtent, "A light from infinitely far along +z, such as the Sun's.",
                    ERNIE_DISTANT_LIGHT_ATTRIBUTES,
                    .fallbackOverrides = {{"inputs:intensity", 50000.0}})
ERNIE_DEFINE_SCHEMA(DiskLight, "DiskLight", &BoundableLightBase::Definition(), Concrete,
                    ComputedExtent(CentredExtent<DiskHalf>), "A disk emitting towards -z.",
                    ERNIE_DISK_LIGHT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(RectLight, "RectLight", &BoundableLightBase::Definition(), Concrete,
                    ComputedExtent(CentredExtent<RectHalf>), "A rectangle emitting towards -z.",
                    ERNIE_RECT_LIGHT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(SphereLight, "SphereLight", &BoundableLightBase::Definition(), Concrete,
                    ComputedExtent(CentredExtent<SphereHalf>), "A sphere emitting outwards.",
                    ERNIE_SPHERE_LIGHT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(CylinderLight, "CylinderLight", &BoundableLightBase::Definition(), Concrete,
                    ComputedExtent(CentredExtent<CylinderHalf>),
                    "A cylinder along x emitting outwards.", ERNIE_CYLINDER_LIGHT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(DomeLight, "DomeLight", &NonboundableLightBase::Definition(), Concrete,
                    kInheritExtent, "Light from all around, from an environment image.",
                    ERNIE_DOME_LIGHT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(PortalLight, "PortalLight", &BoundableLightBase::Definition(), Concrete,
                    ComputedExtent(CentredExtent<RectHalf>),
                    "A rectangle a dome light's light comes in through.",
                    ERNIE_PORTAL_LIGHT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(ShapingAPI, "ShapingAPI", nullptr, SingleApply, kInheritExtent,
                    "Focuses a light, limits it to a cone, or shapes it by an IES profile.",
                    ERNIE_SHAPING_API_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(ShadowAPI, "ShadowAPI", nullptr, SingleApply, kInheritExtent,
                    "Controls a light's shadows.", ERNIE_SHADOW_API_ATTRIBUTES)

}  // namespace ernie
