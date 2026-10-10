// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/base_schemas.h"
#include "ernie/scene/prim_type.h"
#include "ernie/scene/tokens.h"

#include <vector>

/// Cameras: UsdGeom's Camera with Ernie's lens, exposure, shutter, spectral and sensor attributes, as reflected schemas (`prim_type.h`).

namespace ernie {

/// This file's types' registry entries, bases first (`PrimTypes`).
std::vector<const PrimTypeDef*> CameraSchemaTypes();

}  // namespace ernie
