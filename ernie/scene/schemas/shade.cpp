// SPDX-License-Identifier: BSD-3-Clause
/// @file shade.cpp
/// @brief UsdShade's registry entries.
#include "ernie/scene/schemas/shade.h"

namespace ernie {

ERNIE_DEFINE_SCHEMA(MaterialBindingAPI, "MaterialBindingAPI", nullptr, SingleApply, kInheritExtent,
                    "Binds a material to a prim and its descendants.",
                    ERNIE_MATERIAL_BINDING_API_ATTRIBUTES, "material:binding:")

}  // namespace ernie
