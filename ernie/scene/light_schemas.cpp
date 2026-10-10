// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/light_schemas.h"

namespace ernie::schema {

ERNIE_DEFINE_SCHEMA(Light, "Light", nullptr, true,
                    "Any light: UsdLux's LightAPI inputs and the shape of its spectrum.",
                    ERNIE_LIGHT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(BoundableLight, "BoundableLightBase", &Light::Definition(), true,
                    "An area light (sphere, rect, disk), with a power.",
                    ERNIE_BOUNDABLE_LIGHT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(NonboundableLight, "NonboundableLightBase", &Light::Definition(), true,
                    "A light at infinity (distant, dome), with an illuminance.",
                    ERNIE_NONBOUNDABLE_LIGHT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(DistantLight, tokens::kDistantLight, &NonboundableLight::Definition(), false,
                    "A light at infinity shining along the -z axis of its transform.",
                    ERNIE_DISTANT_LIGHT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(DomeLight, tokens::kDomeLight, &NonboundableLight::Definition(), false,
                    "Light from every direction, optionally from a latitude-longitude map.",
                    ERNIE_DOME_LIGHT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(SphereLight, tokens::kSphereLight, &BoundableLight::Definition(), false,
                    "A sphere or point light, optionally shaped to a cone around its -z axis.",
                    ERNIE_SPHERE_LIGHT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(RectLight, tokens::kRectLight, &BoundableLight::Definition(), false,
                    "A rectangle in its xy plane emitting along -z.", ERNIE_RECT_LIGHT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(DiskLight, tokens::kDiskLight, &BoundableLight::Definition(), false,
                    "A disk in its xy plane emitting along -z.", ERNIE_DISK_LIGHT_ATTRIBUTES)

bool Light::IsPhotometric() const {
    return Has(tokens::kPhotometricIlluminance) || Has(tokens::kPhotometricIlluminanceDistance) ||
           Has(tokens::kPhotometricPower);
}

}  // namespace ernie::schema
