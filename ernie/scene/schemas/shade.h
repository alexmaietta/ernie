// SPDX-License-Identifier: BSD-3-Clause
/// @file shade.h
/// @brief UsdShade's schemas.
///
/// References:
/// - Pixar, UsdShade, https://openusd.org/release/api/usd_shade_page_front.html
#pragma once

#include "ernie/scene/view.h"

// clang-format off
#define ERNIE_MATERIAL_BINDING_API_ATTRIBUTES(X)                                                   \
    X(Path, MaterialBinding, "material:binding", Path(),                                           \
      "The Material the prim and its descendants are drawn with.", .label = "Material",            \
      .group = "Material", .dirty = Dirty::Material)
// clang-format on

namespace ernie {

ERNIE_SCHEMA_ACCESSORS(MaterialBindingAPI, SchemaView, ERNIE_MATERIAL_BINDING_API_ATTRIBUTES);

/// Binds a material to a prim; `material:binding:<purpose>` binds one for a purpose.
class MaterialBindingAPI : public MaterialBindingAPIAccessors<MaterialBindingAPI, SchemaView> {
    ERNIE_API_SCHEMA_BODY(MaterialBindingAPI)
};

}  // namespace ernie
