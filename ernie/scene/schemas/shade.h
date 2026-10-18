// SPDX-License-Identifier: BSD-3-Clause
/// @file shade.h
/// @brief UsdShade's schemas.
///
/// References:
/// - Pixar, UsdShade, https://openusd.org/release/api/usd_shade_page_front.html
#pragma once

#include "ernie/scene/view.h"

#include <optional>
#include <string>
#include <string_view>

// clang-format off
#define ERNIE_MATERIAL_BINDING_API_ATTRIBUTES(X)                                                   \
    X(Path, MaterialBinding, "material:binding", Path(),                                           \
      "The Material the prim and its descendants are drawn with.", .label = "Material",            \
      .group = "Material", .dirty = Dirty::Material)

#define ERNIE_SHADER_ATTRIBUTES(X)                                                                 \
    X(std::string, Id, "info:id", "",                                                              \
      "The node it is: UsdPreviewSurface, UsdUVTexture, a primvar reader or an Ernie model.",      \
      .group = "Shader", .dirty = Dirty::Material)                                                 \
    X(std::string, ImplementationSource, "info:implementationSource", "id",                        \
      "Where its definition comes from: its id, a source asset or source code.",                   \
      .group = "Shader", .options = {"id", "sourceAsset", "sourceCode"}, .dirty = Dirty::Material) \
    X(std::string, SourceAsset, "info:sourceAsset", "", "A file defining it, as a MaterialX one.", \
      .group = "Shader", .asset = true, .dirty = Dirty::Material)                                  \
    X(std::string, SourceCode, "info:sourceCode", "", "Code defining it, as a Slang model's.",     \
      .group = "Shader", .dirty = Dirty::Material)
// clang-format on

namespace ernie {

ERNIE_SCHEMA_ACCESSORS(MaterialBindingAPI, SchemaView, ERNIE_MATERIAL_BINDING_API_ATTRIBUTES);
ERNIE_SCHEMA_ACCESSORS(NodeGraph, SchemaView, ERNIE_NO_ATTRIBUTES);
ERNIE_SCHEMA_ACCESSORS(Material, NodeGraph, ERNIE_NO_ATTRIBUTES);
ERNIE_SCHEMA_ACCESSORS(Shader, SchemaView, ERNIE_SHADER_ATTRIBUTES);

/// Binds a material to a prim; `material:binding:<purpose>` binds one for a purpose.
class MaterialBindingAPI : public MaterialBindingAPIAccessors<MaterialBindingAPI, SchemaView> {
    ERNIE_API_SCHEMA_BODY(MaterialBindingAPI)
};

/// A network of shaders with `inputs:` its users set and `outputs:` connected inside it.
class NodeGraph : public NodeGraphAccessors<NodeGraph, SchemaView> {
    ERNIE_SCHEMA_BODY(NodeGraph, SchemaView)
    ERNIE_SCHEMA_DEFINE(NodeGraph)
};

/// What prims are drawn with: its terminals `outputs:surface`, `outputs:displacement` and
/// `outputs:volume`, and Ernie's `outputs:ernie:thermal`, `outputs:ernie:lidar` and
/// `outputs:ernie:radar`, connect to the shaders that compute them.
class Material : public MaterialAccessors<Material, NodeGraph> {
    ERNIE_SCHEMA_BODY(Material, NodeGraph)
    ERNIE_SCHEMA_DEFINE(Material)
};

/// One node of a network: a model or a texture, by its `info:id`, with its `inputs:` and
/// `outputs:`.
class Shader : public ShaderAccessors<Shader, SchemaView> {
    ERNIE_SCHEMA_BODY(Shader, SchemaView)
    ERNIE_SCHEMA_DEFINE(Shader)
};

/// @brief The source of a connection: a shader or graph and one of its outputs.
struct Connection {
    Path prim;
    std::string output;
    bool operator==(const Connection&) const = default;
};

/// @brief Connects an input or output of a Material, NodeGraph or Shader to another's output, as
/// USD's `<input>.connect` holds it: `/Looks/Red/Surface.outputs:surface`.
void Connect(Scene& scene, const Path& path, std::string_view input, const Connection& source);
/// @return The input's connection, or none.
/// @throws std::invalid_argument if its text is not a prim path, a dot and a property name.
std::optional<Connection> FindConnection(const Scene& scene, const Path& path,
                                         std::string_view input);

}  // namespace ernie
