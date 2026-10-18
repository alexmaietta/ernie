// SPDX-License-Identifier: BSD-3-Clause
/// @file shade.cpp
/// @brief UsdShade's registry entries and connections.
#include "ernie/scene/schema/shade.h"

#include <stdexcept>

namespace ernie {
namespace {

constexpr std::string_view kConnect = ".connect";

/// Inputs and outputs are open on every connectable prim, as UsdShadeConnectableAPI makes them.
#define ERNIE_CONNECTABLE .openNamespaces = {"inputs:", "outputs:"}

std::string ConnectionAttribute(std::string_view input) {
    return std::string(input) + std::string(kConnect);
}

}  // namespace

ERNIE_DEFINE_SCHEMA(MaterialBindingAPI, SingleApply, kInheritExtent,
                    ERNIE_MATERIAL_BINDING_API_ATTRIBUTES, .openNamespaces = {"material:binding:"})
ERNIE_DEFINE_SCHEMA(NodeGraph, Concrete, kNoExtent, ERNIE_NO_ATTRIBUTES, ERNIE_CONNECTABLE)
ERNIE_DEFINE_SCHEMA(Material, Concrete, kInheritExtent, ERNIE_NO_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Shader, Concrete, kNoExtent, ERNIE_SHADER_ATTRIBUTES, ERNIE_CONNECTABLE)

void Connect(Scene& scene, const Path& path, std::string_view input, const Connection& source) {
    scene.Set(path, ConnectionAttribute(input), source.prim.String() + "." + source.output);
}

std::optional<Connection> FindConnection(const Scene& scene, const Path& path,
                                         std::string_view input) {
    const std::optional<Value> value = scene.Get(path, ConnectionAttribute(input));
    if (!value) {
        return std::nullopt;
    }
    const std::string& text = *value->Get<std::string>();
    const std::size_t dot =
        text.find('.', text.rfind('/') == std::string::npos ? 0 : text.rfind('/'));
    if (dot == std::string::npos || dot + 1 == text.size()) {
        throw std::invalid_argument("'" + path.String() + "': " + ConnectionAttribute(input) +
                                    " must be a prim path, a dot and a property, not '" + text +
                                    "'");
    }
    return Connection{Path(text.substr(0, dot)), text.substr(dot + 1)};
}

}  // namespace ernie
