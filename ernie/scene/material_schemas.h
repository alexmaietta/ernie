// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/base_schemas.h"
#include "ernie/scene/prim_type.h"
#include "ernie/scene/tokens.h"

#include <vector>

/// Materials: UsdShade's Material and Shader with Ernie's BRDF and detail-texture inputs, as reflected schemas (`prim_type.h`).

namespace ernie {

/// This file's types' registry entries, bases first (`PrimTypes`).
std::vector<const PrimTypeDef*> MaterialSchemaTypes();

}  // namespace ernie
