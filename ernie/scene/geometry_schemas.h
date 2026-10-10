// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/base_schemas.h"
#include "ernie/scene/prim_type.h"
#include "ernie/scene/tokens.h"

#include <vector>

/// Geometry: UsdGeom's shapes (Cube, Sphere, Cylinder, Cone, Capsule, Plane), Mesh, Points, PointInstancer and Volume, and Ernie's GroundPlane, as reflected schemas (`prim_type.h`).

namespace ernie {

/// This file's types' registry entries, bases first (`PrimTypes`).
std::vector<const PrimTypeDef*> GeometrySchemaTypes();

}  // namespace ernie
