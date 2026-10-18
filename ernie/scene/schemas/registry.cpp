// SPDX-License-Identifier: BSD-3-Clause
/// @file registry.cpp
/// @brief The list of every registered schema.
#include "ernie/scene/schemas/camera.h"
#include "ernie/scene/schemas/collection.h"
#include "ernie/scene/schemas/geom.h"
#include "ernie/scene/schemas/instancer.h"
#include "ernie/scene/schemas/laser.h"
#include "ernie/scene/schemas/lux.h"
#include "ernie/scene/schemas/mesh.h"
#include "ernie/scene/schemas/planet.h"
#include "ernie/scene/schemas/render.h"
#include "ernie/scene/schemas/shade.h"
#include "ernie/scene/schemas/vol.h"

namespace ernie {

std::span<const SchemaDef* const> Schemas() {
    static const std::array kSchemas{
        &Imageable::Definition(),
        &Xformable::Definition(),
        &Boundable::Definition(),
        &Gprim::Definition(),
        &Xform::Definition(),
        &Scope::Definition(),
        &Cube::Definition(),
        &Sphere::Definition(),
        &Cylinder::Definition(),
        &Cone::Definition(),
        &Capsule::Definition(),
        &Plane::Definition(),
        &PointBased::Definition(),
        &Mesh::Definition(),
        &Points::Definition(),
        &PointInstancer::Definition(),
        &Volume::Definition(),
        &FieldBase::Definition(),
        &FieldAsset::Definition(),
        &OpenVDBAsset::Definition(),
        &NanoVDBField::Definition(),
        &DenseField::Definition(),
        &Camera::Definition(),
        &ErnieLensAPI::Definition(),
        &ErniePolarizerAPI::Definition(),
        &ErnieSensorAPI::Definition(),
        &LightAPI::Definition(),
        &BoundableLightBase::Definition(),
        &NonboundableLightBase::Definition(),
        &DistantLight::Definition(),
        &DiskLight::Definition(),
        &RectLight::Definition(),
        &SphereLight::Definition(),
        &CylinderLight::Definition(),
        &DomeLight::Definition(),
        &PortalLight::Definition(),
        &LaserLight::Definition(),
        &Terrain::Definition(),
        &Atmosphere::Definition(),
        &StarField::Definition(),
        &GroundPlane::Definition(),
        &ShapingAPI::Definition(),
        &ShadowAPI::Definition(),
        &PhotometricAreaLightAPI::Definition(),
        &PhotometricDistantLightAPI::Definition(),
        &PhotometricDomeLightAPI::Definition(),
        &PhysicalLightIlluminantAPI::Definition(),
        &RenderSettingsBase::Definition(),
        &RenderSettings::Definition(),
        &RenderProduct::Definition(),
        &RenderVar::Definition(),
        &RenderPass::Definition(),
        &NodeGraph::Definition(),
        &Material::Definition(),
        &Shader::Definition(),
        &CollectionAPI::Definition(),
        &MaterialBindingAPI::Definition(),
    };
    return kSchemas;
}

}  // namespace ernie
