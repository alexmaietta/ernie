// SPDX-License-Identifier: BSD-3-Clause
/// @file usd_plugin.cpp
/// @brief Ernie's own schemas as a codeless USD schema plugin: `generatedSchema.usda`, its
/// classes flattened as `usdGenSchema` writes them, and `plugInfo.json`.
///
/// References:
/// - Pixar, "Generating New Schema Classes" (codeless schemas),
///   https://openusd.org/release/tut_generating_new_schema.html
#include "ernie/scene/json_value.h"
#include "ernie/scene/reference.h"

#include <glm/gtc/matrix_access.hpp>

#include <string>

namespace ernie {
namespace {

std::string Number(double number) {
    return Json(number).dump();
}

template <typename Vector> std::string Tuple(const Vector& vector) {
    std::string out = "(";
    for (glm::length_t i = 0; i < vector.length(); ++i) {
        out += (i == 0 ? "" : ", ") + Number(static_cast<double>(vector[i]));
    }
    return out + ")";
}

/// @return A fallback as USD's text format writes it, or empty when it has none there.
std::string UsdValue(const Value& value, bool asset) {
    return std::visit(
        [&](const auto& v) -> std::string {
            using V = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<V, bool>) {
                return v ? "1" : "0";
            } else if constexpr (std::is_same_v<V, int>) {
                return std::to_string(v);
            } else if constexpr (std::is_same_v<V, float> || std::is_same_v<V, double>) {
                return Number(v);
            } else if constexpr (std::is_same_v<V, glm::dquat>) {
                return Tuple(glm::dvec4(v.w, v.x, v.y, v.z));
            } else if constexpr (std::is_same_v<V, glm::dmat4>) {
                // USD's rows are its row vectors: the columns of Ernie's matrices.
                std::string out = "(";
                for (int column = 0; column < 4; ++column) {
                    out += (column == 0 ? "" : ", ") + Tuple(v[column]);
                }
                return out + ")";
            } else if constexpr (std::is_same_v<V, std::string>) {
                return asset ? (v.empty() ? "" : "@" + v + "@") : "\"" + v + "\"";
            } else if constexpr (std::is_same_v<V, std::vector<float>>) {
                std::string out = "[";
                for (std::size_t i = 0; i < v.size(); ++i) {
                    out += (i == 0 ? "" : ", ") + Number(v[i]);
                }
                return out + "]";
            } else if constexpr (std::is_same_v<V, std::vector<std::string>>) {
                std::string out = "[";
                for (std::size_t i = 0; i < v.size(); ++i) {
                    out += (i == 0 ? "\"" : ", \"") + v[i] + "\"";
                }
                return out + "]";
            } else if constexpr (requires { V::length(); }) {
                return Tuple(v);
            } else {
                return "";
            }
        },
        value.Raw());
}

std::string UsdType(const AttributeDef& def) {
    const AttributeMeta& meta = def.meta;
    switch (def.type) {
    case ColumnType::Bool:
        return "bool";
    case ColumnType::Int:
        return "int";
    case ColumnType::Double:
        return "double";
    case ColumnType::Vec2:
        return "double2";
    case ColumnType::Vec3:
        return meta.color ? "color3d" : "double3";
    case ColumnType::Vec4:
        return "double4";
    case ColumnType::Quat:
        return "quatd";
    case ColumnType::Matrix:
        return "matrix4d";
    case ColumnType::TokenValue:
        return meta.asset ? "asset" : "token";
    case ColumnType::Relationship:
    case ColumnType::Relationships:
        return "rel";
    case ColumnType::FloatArray:
        return "float[]";
    case ColumnType::TokenArray:
        return "token[]";
    case ColumnType::Tensor:
        break;
    }
    const std::string element = meta.bytes ? "uchar" : meta.integer ? "int" : "float";
    return element + (meta.components > 1 ? std::to_string(meta.components) : "") + "[]";
}

std::string Quoted(std::string_view text) {
    return "\"\"\"" + std::string(text) + "\"\"\"";
}

std::string Property(const SchemaDef& schema, const AttributeDef& def) {
    const std::string type = UsdType(def);
    std::string out = "    " + type + " " + std::string(def.name);
    const Value* override = schema.FallbackOverride(def.name);
    if (const std::string fallback =
            type == "rel" ? "" : UsdValue(override ? *override : def.fallback, def.meta.asset);
        !fallback.empty()) {
        out += " = " + fallback;
    }
    out += " (\n";
    if (!def.meta.options.empty()) {
        std::string tokens;
        for (const std::string_view option : def.meta.options) {
            tokens += (tokens.empty() ? "\"" : ", \"") + std::string(option) + "\"";
        }
        out += "        allowedTokens = [" + tokens + "]\n";
    }
    return out + "        doc = " + Quoted(def.meta.doc) + "\n    )\n";
}

/// @return The plugin type name of a schema, as `UsdLuxBoundableLightBase` or `ErnieLaserLight`.
std::string TypeName(const SchemaDef& schema) {
    const std::string name(schema.name);
    if (IsErnieSchema(name)) {
        return name.starts_with("Ernie") ? name : "Ernie" + name;
    }
    const SchemaFamily* family = FindFamily(name);
    return std::string(family != nullptr ? family->usdModule : "Usd") + name;
}

std::string_view Kind(SchemaKind kind) {
    switch (kind) {
    case SchemaKind::Abstract:
        return "abstractTyped";
    case SchemaKind::Concrete:
        return "concreteTyped";
    case SchemaKind::SingleApply:
        return "singleApplyAPI";
    case SchemaKind::MultipleApply:
        break;
    }
    return "multipleApplyAPI";
}

std::string Class(const SchemaDef& schema) {
    const bool typed = schema.kind == SchemaKind::Abstract || schema.kind == SchemaKind::Concrete;
    const std::string name(schema.name);
    std::string out = typed && schema.kind == SchemaKind::Concrete
                          ? "class " + name + " \"" + name + "\" (\n"
                          : "class \"" + name + "\" (\n";
    if (const std::vector<std::string_view> builtins = schema.AllBuiltinApis(); !builtins.empty()) {
        std::string apis;
        for (const std::string_view api : builtins) {
            apis += (apis.empty() ? "\"" : ", \"") + std::string(api) + "\"";
        }
        out += "    prepend apiSchemas = [" + apis + "]\n";
    }
    out += "    doc = " + Quoted(schema.doc) + "\n)\n{\n";
    for (const AttributeDef* def : schema.AllAttributes()) {
        out += Property(schema, *def);
    }
    return out + "}\n\n";
}

Json PlugInfoType(const SchemaDef& schema) {
    const bool typed = schema.kind == SchemaKind::Abstract || schema.kind == SchemaKind::Concrete;
    Json type{{"alias", {{"UsdSchemaBase", schema.name}}},
              {"autoGenerated", true},
              {"bases", Json::array({typed ? (schema.base != nullptr ? TypeName(*schema.base)
                                                                     : std::string("UsdTyped"))
                                           : std::string("UsdAPISchemaBase")})},
              {"schemaIdentifier", schema.name},
              {"schemaKind", Kind(schema.kind)}};
    if (!schema.appliesTo.empty()) {
        Json types = Json::array();
        for (const std::string_view to : schema.appliesTo) {
            types.push_back(to);
        }
        type["apiSchemaCanOnlyApplyTo"] = std::move(types);
    }
    return type;
}

}  // namespace

UsdSchemaPlugin ErnieUsdSchemaPlugin() {
    UsdSchemaPlugin plugin;
    plugin.generatedSchema = "#usda 1.0\n(\n    \"WARNING: THIS FILE IS GENERATED BY ernie schema. "
                             "DO NOT EDIT.\"\n)\n\n";
    Json types = Json::object();
    for (const SchemaDef* schema : Schemas()) {
        if (IsErnieSchema(schema->name)) {
            plugin.generatedSchema += Class(*schema);
            types[TypeName(*schema)] = PlugInfoType(*schema);
        }
    }
    const Json info{{"Plugins", Json::array({{{"Info", {{"Types", std::move(types)}}},
                                              {"LibraryPath", ""},
                                              {"Name", "ernie"},
                                              {"ResourcePath", "."},
                                              {"Root", "."},
                                              {"Type", "resource"}}})}};
    plugin.plugInfo = PrintJson(info);
    return plugin;
}

}  // namespace ernie
