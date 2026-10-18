// SPDX-License-Identifier: BSD-3-Clause
/// @file lux.cpp
/// @brief UsdLux's registry entries and the shaped lights' extents, as UsdLux computes them.
#include "ernie/scene/schemas/lux.h"

namespace ernie {
namespace {

auto DiskHalves(const ExtentRows& rows) {
    return [radius = rows.Column<double>(DiskLight::kRadius)](std::uint32_t row) {
        return glm::dvec3(radius[row], radius[row], 0.0);
    };
}

/// Rect and portal lights alike: their width and height in the xy plane.
auto RectHalves(const ExtentRows& rows) {
    return [width = rows.Column<double>(RectLight::kWidth),
            height = rows.Column<double>(RectLight::kHeight)](std::uint32_t row) {
        return glm::dvec3(0.5 * width[row], 0.5 * height[row], 0.0);
    };
}

auto SphereLightHalves(const ExtentRows& rows) {
    return [radius = rows.Column<double>(SphereLight::kRadius)](std::uint32_t row) {
        return glm::dvec3(radius[row]);
    };
}

auto CylinderLightHalves(const ExtentRows& rows) {
    return [radius = rows.Column<double>(CylinderLight::kRadius),
            length = rows.Column<double>(CylinderLight::kLength)](std::uint32_t row) {
        return glm::dvec3(0.5 * length[row], radius[row], radius[row]);
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
ERNIE_DEFINE_SCHEMA(DiskLight, Concrete, ComputedExtent(CentredExtent<DiskHalves>),
                    ERNIE_DISK_LIGHT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(RectLight, Concrete, ComputedExtent(CentredExtent<RectHalves>),
                    ERNIE_RECT_LIGHT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(SphereLight, Concrete, ComputedExtent(CentredExtent<SphereLightHalves>),
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

}  // namespace ernie
