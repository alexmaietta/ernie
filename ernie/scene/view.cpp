// SPDX-License-Identifier: BSD-3-Clause
/// @file view.cpp
/// @brief Schema views' checks.
#include "ernie/scene/view.h"

#include <stdexcept>
#include <string>

namespace ernie {
namespace {

PrimId Check(const Scene& scene, const Path& path, const SchemaDef& schema) {
    const PrimId prim = scene.Require(path);
    const SchemaDef* type = scene.Schema(prim);
    if (type == nullptr || !type->IsA(schema)) {
        const std::string_view name = type != nullptr ? type->name : "a prim without a type";
        throw std::invalid_argument("'" + path.String() + "' is " + std::string(name) + ", not a " +
                                    std::string(schema.name));
    }
    return prim;
}

}  // namespace

SchemaView::SchemaView(const Scene& scene, const Path& path, const SchemaDef& schema)
    : scene_(&scene), prim_(Check(scene, path, schema)), schema_(&schema) {}

SchemaView::SchemaView(Scene& scene, const Path& path, const SchemaDef& schema)
    : scene_(&scene), mutable_(&scene), prim_(Check(scene, path, schema)), schema_(&schema) {}

Scene& SchemaView::Edit() const {
    if (mutable_ == nullptr) {
        throw std::logic_error("'" + GetPath().String() + "' was viewed in a const scene");
    }
    return *mutable_;
}

}  // namespace ernie
