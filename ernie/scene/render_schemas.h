// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/base_schemas.h"
#include "ernie/scene/prim_type.h"
#include "ernie/scene/tokens.h"

#include <vector>

/// Render prims: UsdRender's RenderSettings, RenderProduct and RenderVar, and Ernie's SensorFilter, as reflected schemas (`prim_type.h`).

namespace ernie {

/// This file's types' registry entries, bases first (`PrimTypes`).
std::vector<const PrimTypeDef*> RenderSchemaTypes();

}  // namespace ernie
