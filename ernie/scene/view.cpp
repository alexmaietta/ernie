// SPDX-License-Identifier: BSD-3-Clause
/// @file view.cpp
/// @brief Schema views' checks.
#include "ernie/scene/view.h"

#include <stdexcept>
#include <string>

namespace ernie {
namespace {

bool IsApi(const SchemaDef& schema) {
    return schema.kind == SchemaKind::SingleApply || schema.kind == SchemaKind::MultipleApply;
}

PrimId Check(const Scene& scene, const Path& path, const SchemaDef& schema,
             std::string_view instance) {
    const PrimId prim = scene.Require(path);
    if (IsApi(schema)) {
        const std::string api = ApiName{&schema, instance}.ToString();
        if (!scene.HasApi(prim, api)) {
            throw std::invalid_argument("'" + path.String() + "' has no " + api + " applied");
        }
        return prim;
    }
    const SchemaDef* type = scene.Schema(prim);
    if (type == nullptr || !type->IsA(schema)) {
        const std::string_view name = type != nullptr ? type->name : "a prim without a type";
        throw std::invalid_argument("'" + path.String() + "' is " + std::string(name) + ", not a " +
                                    std::string(schema.name));
    }
    return prim;
}

}  // namespace

SchemaView::SchemaView(const Scene& scene, const Path& path, const SchemaDef& schema,
                       std::string_view instance)
    : scene_(&scene), prim_(Check(scene, path, schema, instance)), schema_(&schema),
      instance_(instance) {}

SchemaView::SchemaView(Scene& scene, const Path& path, const SchemaDef& schema,
                       std::string_view instance)
    : scene_(&scene), mutable_(&scene), prim_(Check(scene, path, schema, instance)),
      schema_(&schema), instance_(instance) {}

Scene& SchemaView::Edit() const {
    if (mutable_ == nullptr) {
        throw std::logic_error("'" + GetPath().String() + "' was viewed in a const scene");
    }
    return *mutable_;
}

}  // namespace ernie
