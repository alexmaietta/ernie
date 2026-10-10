// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/material_schemas.h"

namespace ernie {

// A Material's inputs include its BRDF's parameters (`BrdfRegistry`), and a Shader's depend on its
// info:id, so both open their inputs.
ERNIE_DEFINE_OPEN_SCHEMA(Material, tokens::kMaterial, &Imageable::Definition(), false,
                         "How a surface reflects light: Ernie's own inputs, or a UsdPreviewSurface "
                         "its outputs:surface connects to.",
                         ERNIE_MATERIAL_ATTRIBUTES, "inputs", "outputs")
ERNIE_DEFINE_OPEN_SCHEMA(Shader, tokens::kShader, &Imageable::Definition(), false,
                         "A node of a material's network, of the type info:id names.",
                         ERNIE_SHADER_ATTRIBUTES, "inputs", "outputs")

std::vector<const PrimTypeDef*> MaterialSchemaTypes() {
    return {&Material::Definition(), &Shader::Definition()};
}

}  // namespace ernie
