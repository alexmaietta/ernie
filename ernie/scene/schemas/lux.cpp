// SPDX-License-Identifier: BSD-3-Clause
/// @file lux.cpp
/// @brief UsdLux's registry entries and the shaped lights' extents, as UsdLux computes them, and
/// the laser's: its emitter.
#include "ernie/scene/schemas/lux.h"

#include <algorithm>

namespace ernie {
namespace {

/// Sphere and disk lights: their radius, in the xy plane for a disk.
template <bool kDisk> auto RadiusHalves(const ExtentRows& rows) {
    return [radius = rows.Column<double>(SphereLight::kRadius)](std::uint32_t row) {
        return glm::dvec3(radius[row], radius[row], kDisk ? 0.0 : radius[row]);
    };
}

/// Rect and portal lights alike: their width and height in the xy plane.
auto RectHalves(const ExtentRows& rows) {
    return [width = rows.Column<double>(RectLight::kWidth),
            height = rows.Column<double>(RectLight::kHeight)](std::uint32_t row) {
        return glm::dvec3(0.5 * width[row], 0.5 * height[row], 0.0);
    };
}

auto CylinderLightHalves(const ExtentRows& rows) {
    return [radius = rows.Column<double>(CylinderLight::kRadius),
            length = rows.Column<double>(CylinderLight::kLength)](std::uint32_t row) {
        return glm::dvec3(0.5 * length[row], radius[row], radius[row]);
    };
}

/// A laser's emitter: the larger of its aperture and waist.
auto EmitterHalves(const ExtentRows& rows) {
    return [aperture = rows.Column<double>(LaserLight::kApertureRadius),
            waist = rows.Column<glm::dvec2>(LaserLight::kWaistRadius)](std::uint32_t row) {
        const double radius = std::max({aperture[row], waist[row].x, waist[row].y});
        return glm::dvec3(radius, radius, 0.0);
    };
}

}  // namespace

// LightAPI carries its light-linking collections, as USD's does: which prims a light lights and
// which cast its shadows, everything unless they say otherwise. Every light has it built in.
ERNIE_DEFINE_SCHEMA(LightAPI, SingleApply, kInheritExtent, ERNIE_LIGHT_API_ATTRIBUTES,
                    .builtinApis = {"CollectionAPI:lightLink", "CollectionAPI:shadowLink"},
                    .fallbackOverrides = {{"collection:lightLink:includeRoot", true},
                                          {"collection:shadowLink:includeRoot", true}})
ERNIE_DEFINE_SCHEMA(BoundableLightBase, Abstract, kInheritExtent,
                    ERNIE_BOUNDABLE_LIGHT_BASE_ATTRIBUTES, .builtinApis = {"LightAPI"})
ERNIE_DEFINE_SCHEMA(NonboundableLightBase, Abstract, kUnbounded, ERNIE_NO_ATTRIBUTES,
                    .builtinApis = {"LightAPI"})
// Emitting geometry: its LightAPI is built in, its material's glow tinting the light.
ERNIE_DEFINE_SCHEMA(MeshLightAPI, SingleApply, kInheritExtent, ERNIE_NO_ATTRIBUTES,
                    .builtinApis = {"LightAPI"},
                    .fallbackOverrides = {{"light:shaderId", "MeshLight"},
                                          {"light:materialSyncMode", "materialGlowTintsLight"}},
                    .appliesTo = {"Mesh"})
ERNIE_DEFINE_SCHEMA(VolumeLightAPI, SingleApply, kInheritExtent, ERNIE_NO_ATTRIBUTES,
                    .builtinApis = {"LightAPI"},
                    .fallbackOverrides = {{"light:shaderId", "VolumeLight"},
                                          {"light:materialSyncMode", "materialGlowTintsLight"}},
                    .appliesTo = {"Volume"})
ERNIE_DEFINE_SCHEMA(DistantLight, Concrete, kInheritExtent, ERNIE_DISTANT_LIGHT_ATTRIBUTES,
                    .fallbackOverrides = {{"inputs:intensity", 50000.0}})
ERNIE_DEFINE_SCHEMA(DiskLight, Concrete, ComputedExtent(CentredExtent<RadiusHalves<true>>),
                    ERNIE_DISK_LIGHT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(RectLight, Concrete, ComputedExtent(CentredExtent<RectHalves>),
                    ERNIE_RECT_LIGHT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(SphereLight, Concrete, ComputedExtent(CentredExtent<RadiusHalves<false>>),
                    ERNIE_SPHERE_LIGHT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(CylinderLight, Concrete, ComputedExtent(CentredExtent<CylinderLightHalves>),
                    ERNIE_CYLINDER_LIGHT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(DomeLight, Concrete, kInheritExtent, ERNIE_DOME_LIGHT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(PortalLight, Concrete, ComputedExtent(CentredExtent<RectHalves>),
                    ERNIE_PORTAL_LIGHT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(ShapingAPI, SingleApply, kInheritExtent, ERNIE_SHAPING_API_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(ShadowAPI, SingleApply, kInheritExtent, ERNIE_SHADOW_API_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(PhotometricAreaLightAPI, SingleApply, kInheritExtent,
                    ERNIE_PHOTOMETRIC_AREA_LIGHT_API_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(PhotometricDistantLightAPI, SingleApply, kInheritExtent,
                    ERNIE_PHOTOMETRIC_DISTANT_LIGHT_API_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(PhotometricDomeLightAPI, SingleApply, kInheritExtent,
                    ERNIE_PHOTOMETRIC_DOME_LIGHT_API_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(PhysicalLightIlluminantAPI, SingleApply, kInheritExtent,
                    ERNIE_PHYSICAL_LIGHT_ILLUMINANT_API_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(LaserLight, Concrete, ComputedExtent(CentredExtent<EmitterHalves>),
                    ERNIE_LASER_LIGHT_ATTRIBUTES,
                    .fallbackOverrides = {{"ernie:power", 0.001},
                                          {"ernie:polarization:degree", 0.98}})

}  // namespace ernie
