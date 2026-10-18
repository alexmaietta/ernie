// SPDX-License-Identifier: BSD-3-Clause
/// @file registry.cpp
/// @brief Every registered schema, by family, bases before the types deriving from them.
#include "ernie/scene/schema/camera.h"
#include "ernie/scene/schema/collection.h"
#include "ernie/scene/schema/geom.h"
#include "ernie/scene/schema/instancer.h"
#include "ernie/scene/schema/lux.h"
#include "ernie/scene/schema/mesh.h"
#include "ernie/scene/schema/planet.h"
#include "ernie/scene/schema/render.h"
#include "ernie/scene/schema/shade.h"
#include "ernie/scene/schema/vol.h"

namespace ernie {

ERNIE_DEFINE_SCHEMA(CollectionAPI, MultipleApply, kInheritExtent, ERNIE_COLLECTION_API_ATTRIBUTES)

std::span<const SchemaFamily> SchemaFamilies() {
    static const std::vector<SchemaFamily> kFamilies{
        {"geometry",
         "Geometry",
         "UsdGeom",
         "https://openusd.org/release/api/usd_geom_page_front.html",
         {&Imageable::Definition(),     &Xformable::Definition(),   &Boundable::Definition(),
          &Gprim::Definition(),         &Xform::Definition(),       &Scope::Definition(),
          &GeomModelAPI::Definition(),  &GeomSubset::Definition(),  &Cube::Definition(),
          &Sphere::Definition(),        &Cylinder::Definition(),    &Cone::Definition(),
          &Capsule::Definition(),       &Plane::Definition(),       &PointBased::Definition(),
          &Mesh::Definition(),          &Points::Definition(),      &Curves::Definition(),
          &BasisCurves::Definition(),   &NurbsCurves::Definition(), &HermiteCurves::Definition(),
          &PointInstancer::Definition()},
         {}},
        {"volumes",
         "Volumes",
         "UsdVol",
         "https://openusd.org/release/api/usd_vol_page_front.html",
         {&Volume::Definition(), &FieldBase::Definition(), &FieldAsset::Definition(),
          &OpenVDBAsset::Definition()},
         {&NanoVDBField::Definition(), &DenseField::Definition()}},
        {"cameras",
         "Cameras and lenses",
         "UsdGeom",
         "https://openusd.org/release/api/class_usd_geom_camera.html",
         {&Camera::Definition()},
         {&ErnieLensAPI::Definition(), &ErniePolarizerAPI::Definition(),
          &ErnieSensorAPI::Definition()}},
        {"lights",
         "Lights",
         "UsdLux",
         "https://openusd.org/release/api/usd_lux_page_front.html",
         {&LightAPI::Definition(), &BoundableLightBase::Definition(),
          &NonboundableLightBase::Definition(), &DistantLight::Definition(),
          &DiskLight::Definition(), &RectLight::Definition(), &SphereLight::Definition(),
          &CylinderLight::Definition(), &DomeLight::Definition(), &PortalLight::Definition(),
          &MeshLightAPI::Definition(), &VolumeLightAPI::Definition(), &ShapingAPI::Definition(),
          &ShadowAPI::Definition()},
         {&PhotometricAreaLightAPI::Definition(), &PhotometricDistantLightAPI::Definition(),
          &PhotometricDomeLightAPI::Definition(), &PhysicalLightIlluminantAPI::Definition(),
          &LaserLight::Definition()}},
        {"materials",
         "Materials",
         "UsdShade",
         "https://openusd.org/release/api/usd_shade_page_front.html",
         {&MaterialBindingAPI::Definition(), &NodeGraph::Definition(), &Material::Definition(),
          &Shader::Definition()},
         {}},
        {"render",
         "Render settings",
         "UsdRender",
         "https://openusd.org/release/api/usd_render_page_front.html",
         {&RenderSettingsBase::Definition(), &RenderSettings::Definition(),
          &RenderProduct::Definition(), &RenderVar::Definition(), &RenderPass::Definition()},
         {}},
        {"planetary",
         "Planetary",
         "",
         "",
         {},
         {&Terrain::Definition(), &Atmosphere::Definition(), &StarField::Definition(),
          &GroundPlane::Definition()}},
        {"collections",
         "Collections",
         "Usd",
         "https://openusd.org/release/api/class_usd_collection_a_p_i.html",
         {&CollectionAPI::Definition()},
         {}},
    };
    return kFamilies;
}

std::span<const SchemaDef* const> Schemas() {
    static const std::vector<const SchemaDef*> kSchemas = [] {
        std::vector<const SchemaDef*> schemas;
        for (const SchemaFamily& family : SchemaFamilies()) {
            schemas.insert(schemas.end(), family.usd.begin(), family.usd.end());
            schemas.insert(schemas.end(), family.ernie.begin(), family.ernie.end());
        }
        return schemas;
    }();
    return kSchemas;
}

}  // namespace ernie
