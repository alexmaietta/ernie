// SPDX-License-Identifier: BSD-3-Clause
/// @file reference.h
/// @brief Every view of the schema registry, generated from its declarations: the JSON Schema of
/// scene files, the reference pages, Ernie's codeless USD schema plugin and scene validation.
#pragma once

#include "ernie/scene/json.h"
#include "ernie/scene/schema.h"

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// @return The family of a schema, or null for an unlisted one.
const SchemaFamily* FindFamily(std::string_view schema);
/// @return Whether a schema is Ernie's own rather than one mirroring USD.
bool IsErnieSchema(std::string_view schema);

/// @return An attribute's USD value type, as `double3`, `token` or `float3[]`.
std::string UsdTypeName(const AttributeDef& def);

/// @return The JSON Schema (draft 2020-12) of scene files: each prim type and API schema with its
/// attributes' types, fallbacks, ranges, allowed tokens and descriptions.
std::string SceneJsonSchema();
/// @return A family's reference page in Markdown: each schema's description, base and attribute
/// table, with an example scene.
std::string ReferencePage(const SchemaFamily& family);
/// @return A minimal scene file using a prim type or API schema, which loads without problems.
std::string ExampleScene(const SchemaDef& schema);

/// @brief A codeless USD schema plugin: its two files.
struct UsdSchemaPlugin {
    std::string generatedSchema;
    std::string plugInfo;
};
/// @return The plugin declaring Ernie's own schemas to USD, as `usdGenSchema` lays one out.
UsdSchemaPlugin ErnieUsdSchemaPlugin();

/// @brief Checks a scene file against the registry as loading does, and past what JSON Schema
/// can: unknown attributes (with the nearest declared names), wrong types, out-of-range values,
/// misapplied API schemas and relationships to prims the file doesn't have.
/// @return Every problem; empty for a valid file.
std::vector<SceneProblem> ValidateScene(std::string_view json);

}  // namespace ernie
