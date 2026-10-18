// SPDX-License-Identifier: BSD-3-Clause
/// @file registry.cpp
/// @brief The list of every registered schema.
#include "ernie/scene/schemas/collection.h"
#include "ernie/scene/schemas/geom.h"
#include "ernie/scene/schemas/mesh.h"
#include "ernie/scene/schemas/shade.h"

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
        &CollectionAPI::Definition(),
        &MaterialBindingAPI::Definition(),
    };
    return kSchemas;
}

}  // namespace ernie
