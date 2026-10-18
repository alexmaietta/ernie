// SPDX-License-Identifier: BSD-3-Clause
/// @file reference.cpp
/// @brief The reference families, the JSON Schema of scene files and the reference pages.
#include "ernie/scene/reference.h"

#include "ernie/scene/json_value.h"
#include "ernie/util/text.h"

#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>

namespace ernie {
namespace {

constexpr std::string_view kInstancePattern = "[A-Za-z_][A-Za-z0-9_]*";
constexpr std::string_view kExampleInstance = "example";

/// @return Text as a regular expression matching it literally.
std::string Literal(std::string_view text) {
    std::string out;
    for (const char c : text) {
        if (std::string_view("\\^$.|?*+()[]{}").find(c) != std::string_view::npos) {
            out += '\\';
        }
        out += c;
    }
    return out;
}

std::optional<Json> JsonValue(const Value& value) {
    if (value.Empty()) {
        return std::nullopt;
    }
    try {
        return ToJson(value);
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    }
}

/// @return A type's fallback for an attribute as scene files write it; none if it has none.
std::optional<Json> JsonFallback(const SchemaDef& schema, const AttributeDef& def) {
    return JsonValue(schema.Fallback(def));
}

bool IsInstanced(std::string_view name) {
    return name.find(kInstanceName) != std::string_view::npos;
}

std::vector<std::string_view> OpenNamespaces(const SchemaDef& schema) {
    std::vector<std::string_view> namespaces;
    for (const SchemaDef* type = &schema; type != nullptr; type = type->base) {
        namespaces.insert(namespaces.end(), type->openNamespaces.begin(),
                          type->openNamespaces.end());
    }
    return namespaces;
}

std::vector<std::string> Names(SchemaKind kind) {
    std::vector<std::string> names;
    for (const SchemaDef* schema : Schemas()) {
        if (schema->kind == kind) {
            names.emplace_back(schema->name);
        }
    }
    return names;
}

// The JSON Schema.

Json Ref(std::string_view name) {
    return {{"$ref", "#/$defs/" + std::string(name)}};
}

Json Number(const AttributeMeta& meta, bool integer) {
    Json number{{"type", integer ? "integer" : "number"}};
    if (meta.lower) {
        number[meta.lower->exclusive ? "exclusiveMinimum" : "minimum"] = meta.lower->value;
    }
    if (meta.upper) {
        number[meta.upper->exclusive ? "exclusiveMaximum" : "maximum"] = meta.upper->value;
    }
    return number;
}

Json ArrayOf(Json items, std::optional<std::size_t> count = std::nullopt) {
    Json array{{"type", "array"}, {"items", std::move(items)}};
    if (count) {
        array["minItems"] = *count;
        array["maxItems"] = *count;
    }
    return array;
}

Json String() {
    return {{"type", "string"}};
}

Json ValueSchema(const AttributeDef& def) {
    const AttributeMeta& meta = def.meta;
    switch (def.type) {
    case ColumnType::Bool:
        return {{"type", "boolean"}};
    case ColumnType::Int:
        return Number(meta, true);
    case ColumnType::Double:
        return Number(meta, false);
    case ColumnType::Vec2:
        return ArrayOf(Number(meta, false), 2);
    case ColumnType::Vec3:
        return ArrayOf(Number(meta, false), 3);
    case ColumnType::Vec4:
        return ArrayOf(Number(meta, false), 4);
    case ColumnType::Quat:
        return ArrayOf(Number({}, false), 4);
    case ColumnType::Matrix:
        return ArrayOf(ArrayOf(Number({}, false), 4), 4);
    case ColumnType::TokenValue: {
        Json token = String();
        if (!meta.options.empty()) {
            token["enum"] = meta.options;
        }
        return token;
    }
    case ColumnType::Relationship:
        return String();
    case ColumnType::Relationships:
        return {{"anyOf", Json::array({String(), ArrayOf(String())})}};
    case ColumnType::FloatArray:
        return ArrayOf(Number(meta, false));
    case ColumnType::TokenArray:
        return ArrayOf(String());
    case ColumnType::Tensor:
        break;
    }
    const Json shared{{"type", "object"},
                      {"required", Json::array({"$array"})},
                      {"properties", {{"$array", {{"type", "integer"}, {"minimum", 0}}}}}};
    return {{"anyOf", Json::array({{{"type", "array"}}, shared})}};
}

Json SchemaDefinition(const SchemaDef& schema) {
    Json properties = Json::object();
    Json patterns = Json::object();
    for (const AttributeDef* def : schema.AllAttributes()) {
        Json attribute = ValueSchema(*def);
        attribute["description"] = def->meta.doc;
        if (const std::optional<Json> fallback = JsonFallback(schema, *def)) {
            attribute["default"] = *fallback;
        }
        if (IsInstanced(def->name)) {
            patterns["^" + Replace(Literal(def->name), kInstanceName, kInstancePattern) + "$"] =
                std::move(attribute);
        } else {
            properties[std::string(def->name)] = std::move(attribute);
        }
    }
    for (const std::string_view space : OpenNamespaces(schema)) {
        patterns[space == "*" ? std::string(".*") : "^" + Literal(space)] = Json::object();
    }
    Json definition{{"description", schema.doc}};
    if (!properties.empty()) {
        definition["properties"] = std::move(properties);
    }
    if (!patterns.empty()) {
        definition["patternProperties"] = std::move(patterns);
    }
    Json builtins = Json::array();
    for (const std::string_view api : schema.AllBuiltinApis()) {
        builtins.push_back(Ref(ApiName::Parse(api).schema->name));
    }
    if (!builtins.empty()) {
        definition["allOf"] = std::move(builtins);
    }
    return definition;
}

Json XformOpsSchema() {
    const Json vector = ArrayOf(Number({}, false), 3);
    return {{"type", "object"},
            {"description", "The prim's transform ops, as SetXform authors them."},
            {"properties",
             {{"translate", vector},
              {"rotate", vector},
              {"rotationOrder", {{"enum", {"XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"}}}},
              {"orient", ArrayOf(Number({}, false), 4)},
              {"scale", {{"anyOf", Json::array({Number({}, false), vector})}}}}},
            {"additionalProperties", false}};
}

/// @return When the prim's type is `name`, or it applies the API schema `name`, check it so.
Json Condition(const SchemaDef& schema) {
    Json test;
    if (schema.kind == SchemaKind::Concrete) {
        test = {{"required", Json::array({kTypeName})},
                {"properties", {{kTypeName, {{"const", schema.name}}}}}};
    } else {
        const Json item = schema.kind == SchemaKind::SingleApply
                              ? Json{{"const", schema.name}}
                              : Json{{"pattern", "^" + Literal(schema.name) + ":"}};
        test = {{"required", Json::array({kApis})},
                {"properties", {{kApis, {{"contains", item}}}}}};
    }
    return {{"if", test}, {"then", Ref(schema.name)}};
}

Json PrimSchema() {
    const std::string instanced = Join(Names(SchemaKind::MultipleApply), "|", Literal);
    const Json api{
        {"anyOf", Json::array({{{"enum", Names(SchemaKind::SingleApply)}},
                               {{"pattern", "^(" + instanced +
                                                "):" + std::string(kInstancePattern) + "$"}}})}};
    Json conditions = Json::array();
    // A prim without a type takes attributes of any name.
    conditions.push_back({{"if", {{"not", {{"required", Json::array({kTypeName})}}}}},
                          {"then", {{"patternProperties", {{".*", Json::object()}}}}}});
    for (const SchemaDef* schema : Schemas()) {
        if (schema->kind != SchemaKind::Abstract) {
            conditions.push_back(Condition(*schema));
        }
    }
    return {{"type", "object"},
            {"description", "A prim: its type, applied API schemas, transform and attributes."},
            {"properties",
             {{kTypeName, {{"enum", Names(SchemaKind::Concrete)}}},
              {kApis, ArrayOf(api)},
              {kXform, Ref("XformOps")}}},
            {"patternProperties",
             {{Literal(kTimeSamples) + "$",
               {{"type", "object"},
                {"description", "The attribute's values by time code."},
                {"propertyNames", {{"pattern", "^-?[0-9]+(\\.[0-9]+)?([eE][-+]?[0-9]+)?$"}}}}}}},
            {"allOf", std::move(conditions)},
            {"unevaluatedProperties", false}};
}

// The reference pages.

std::string Anchor(std::string_view schema) {
    return ToLower(schema);
}

std::string Link(std::string_view schema, const SchemaFamily& from) {
    const SchemaFamily* family = FindFamily(schema);
    const std::string page =
        family == nullptr || family == &from ? std::string() : std::string(family->name) + ".md";
    return "[" + std::string(schema) + "](" + page + "#" + Anchor(schema) + ")";
}

/// @return The type a reference page shows: USD's, with relationship lists and array shapes
/// spelt out.
std::string TypeText(const AttributeDef& def) {
    const AttributeMeta& meta = def.meta;
    if (def.type == ColumnType::Relationships) {
        return "rel[]";
    }
    if (def.type != ColumnType::Tensor || (meta.components == 0 && meta.ranks.empty())) {
        return UsdTypeName(def);
    }
    const std::string element = meta.bytes ? "uchar" : meta.integer ? "int" : "float";
    if (meta.components != 0) {
        return element + "[n, " + std::to_string(meta.components) + "]";
    }
    return element + " tensor, rank " +
           Join(meta.ranks, " or ", [](std::int64_t rank) { return std::to_string(rank); });
}

std::string ValuesText(const AttributeMeta& meta) {
    std::vector<std::string> values;
    for (const std::string_view option : meta.options) {
        values.push_back("`" + std::string(option) + "`");
    }
    if (meta.lower) {
        values.push_back((meta.lower->exclusive ? "> " : "≥ ") + ToText(meta.lower->value));
    }
    if (meta.upper) {
        values.push_back((meta.upper->exclusive ? "< " : "≤ ") + ToText(meta.upper->value));
    }
    return Join(values, ", ");
}

std::string Cell(std::string_view text) {
    return Replace(Replace(text, "|", "\\|"), "\n", " ");
}

std::string Code(const std::optional<Json>& json) {
    return json ? "`" + Cell(json->dump()) + "`" : "";
}

std::string KindText(const SchemaDef& schema) {
    switch (schema.kind) {
    case SchemaKind::Abstract:
        return "Abstract base";
    case SchemaKind::Concrete:
        return "Prim type";
    case SchemaKind::SingleApply:
        return "API schema";
    case SchemaKind::MultipleApply:
        break;
    }
    return "Multiple-apply API schema, applied as `" + std::string(schema.name) + ":<name>`";
}

std::string SchemaSection(const SchemaDef& schema, const SchemaFamily& family) {
    std::string out = "## " + std::string(schema.name) + "\n\n" + std::string(schema.doc) + "\n\n";
    std::string facts = KindText(schema);
    if (schema.base != nullptr) {
        facts += " · Inherits " + Link(schema.base->name, family);
    }
    for (const std::string_view api : schema.builtinApis) {
        facts += " · Built in: " + Link(ApiName::Parse(api).schema->name, family);
    }
    for (const std::string_view type : schema.appliesTo) {
        facts += " · Applies to " + Link(type, family);
    }
    out += facts + "\n\n";
    if (!schema.attributes.empty()) {
        out += "| Attribute | Type | Fallback | Values | Source | Description |\n"
               "|---|---|---|---|---|---|\n";
        for (const AttributeDef& def : schema.attributes) {
            const bool ernie = IsErnieSchema(schema.name) || def.name.starts_with("ernie:");
            out += "| `" + std::string(def.name) + "` | " + TypeText(def) + " | " +
                   Code(JsonFallback(schema, def)) + " | " + ValuesText(def.meta) + " | " +
                   std::string(ernie ? "Ernie" : family.usdModule) + " | " + Cell(def.meta.doc) +
                   " |\n";
        }
        out += "\n";
    }
    if (!schema.fallbackOverrides.empty()) {
        out +=
            "Fallbacks it changes: " +
            Join(schema.fallbackOverrides, "; ",
                 [](const auto& entry) {
                     return "`" + std::string(entry.first) + "` " + Code(JsonValue(entry.second));
                 }) +
            ".\n\n";
    }
    if (const std::string example = ExampleScene(schema); !example.empty()) {
        out += "```json\n" + example + "```\n\n";
    }
    return out;
}

}  // namespace

std::span<const SchemaFamily> SchemaFamilies() {
    static const std::vector<SchemaFamily> kFamilies{
        {"geometry",
         "Geometry",
         "UsdGeom",
         "https://openusd.org/release/api/usd_geom_page_front.html",
         {"Imageable",    "Xformable",   "Boundable",     "Gprim",         "Xform",    "Scope",
          "GeomModelAPI", "GeomSubset",  "Cube",          "Sphere",        "Cylinder", "Cone",
          "Capsule",      "Plane",       "PointBased",    "Mesh",          "Points",   "Curves",
          "BasisCurves",  "NurbsCurves", "HermiteCurves", "PointInstancer"},
         {}},
        {"volumes",
         "Volumes",
         "UsdVol",
         "https://openusd.org/release/api/usd_vol_page_front.html",
         {"Volume", "FieldBase", "FieldAsset", "OpenVDBAsset"},
         {"NanoVDBField", "DenseField"}},
        {"cameras",
         "Cameras and lenses",
         "UsdGeom",
         "https://openusd.org/release/api/class_usd_geom_camera.html",
         {"Camera"},
         {"ErnieLensAPI", "ErniePolarizerAPI", "ErnieSensorAPI"}},
        {"lights",
         "Lights",
         "UsdLux",
         "https://openusd.org/release/api/usd_lux_page_front.html",
         {"LightAPI", "BoundableLightBase", "NonboundableLightBase", "DistantLight", "DiskLight",
          "RectLight", "SphereLight", "CylinderLight", "DomeLight", "PortalLight", "MeshLightAPI",
          "VolumeLightAPI", "ShapingAPI", "ShadowAPI"},
         {"PhotometricAreaLightAPI", "PhotometricDistantLightAPI", "PhotometricDomeLightAPI",
          "PhysicalLightIlluminantAPI", "LaserLight"}},
        {"materials",
         "Materials",
         "UsdShade",
         "https://openusd.org/release/api/usd_shade_page_front.html",
         {"MaterialBindingAPI", "NodeGraph", "Material", "Shader"},
         {}},
        {"render",
         "Render settings",
         "UsdRender",
         "https://openusd.org/release/api/usd_render_page_front.html",
         {"RenderSettingsBase", "RenderSettings", "RenderProduct", "RenderVar", "RenderPass"},
         {}},
        {"planetary",
         "Planetary",
         "",
         "",
         {},
         {"Terrain", "Atmosphere", "StarField", "GroundPlane"}},
        {"collections",
         "Collections",
         "Usd",
         "https://openusd.org/release/api/class_usd_collection_a_p_i.html",
         {"CollectionAPI"},
         {}},
    };
    return kFamilies;
}

const SchemaFamily* FindFamily(std::string_view schema) {
    for (const SchemaFamily& family : SchemaFamilies()) {
        if (std::ranges::find(family.usd, schema) != family.usd.end() ||
            std::ranges::find(family.ernie, schema) != family.ernie.end()) {
            return &family;
        }
    }
    return nullptr;
}

bool IsErnieSchema(std::string_view schema) {
    const SchemaFamily* family = FindFamily(schema);
    return family != nullptr && std::ranges::find(family->ernie, schema) != family->ernie.end();
}

std::string SceneJsonSchema() {
    Json defs = Json::object();
    defs["Prim"] = PrimSchema();
    defs["XformOps"] = XformOpsSchema();
    for (const SchemaDef* schema : Schemas()) {
        defs[std::string(schema->name)] = SchemaDefinition(*schema);
    }
    Json root{{"$schema", "https://json-schema.org/draft/2020-12/schema"},
              {"$id", kSceneSchemaUrl},
              {"title", "Ernie scene"},
              {"description", "A scene file: prims by path, each with its type, API schemas and "
                              "attributes by USD name."},
              {"type", "object"},
              {"required", Json::array({"prims"})},
              {"properties",
               {{"$schema", String()},
                {"prims",
                 {{"type", "object"},
                  {"propertyNames", {{"pattern", "^/"}}},
                  {"additionalProperties", Ref("Prim")}}}}},
              {"additionalProperties", false},
              {"$defs", std::move(defs)}};
    return PrintJson(root);
}

std::string ReferencePage(const SchemaFamily& family) {
    std::string out = "# " + std::string(family.title) + "\n\n";
    if (!family.usd.empty()) {
        out += "These mirror [" + std::string(family.usdModule) + "](" +
               std::string(family.usdUrl) +
               ") one to one: the same names, attributes, types, fallbacks and allowed tokens.";
        out += family.ernie.empty() ? "\n\n" : " Ernie's own schemas follow them.\n\n";
    } else {
        out += "Ernie's own schemas.\n\n";
    }
    out += "Generated from the schema registry by `ernie schema --markdown`.\n\n";
    for (const auto* names : {&family.usd, &family.ernie}) {
        for (const std::string_view name : *names) {
            out += SchemaSection(*FindSchema(name), family);
        }
    }
    return out;
}

std::string ExampleScene(const SchemaDef& schema) {
    if (schema.kind == SchemaKind::Abstract) {
        return {};
    }
    const bool api = schema.kind != SchemaKind::Concrete;
    const ApiName name{&schema, schema.kind == SchemaKind::MultipleApply ? kExampleInstance : ""};
    Json prim = Json::object();
    prim[kTypeName] =
        api ? (schema.appliesTo.empty() ? "Xform" : schema.appliesTo.front()) : schema.name;
    if (api) {
        prim[kApis] = Json::array({name.ToString()});
    }
    for (const AttributeDef& def : schema.attributes) {
        const std::optional<Json> fallback = JsonFallback(schema, def);
        // Arrays, targets and empty tokens have nothing to show.
        const bool shown = fallback && def.type != ColumnType::Tensor &&
                           def.type != ColumnType::Relationship &&
                           def.type != ColumnType::Relationships && *fallback != Json("");
        if (shown) {
            prim[name.Attribute(def.name)] = *fallback;
        }
    }
    Json prims = Json::object();
    prims["/World/" + std::string(schema.name)] = std::move(prim);
    return PrintJson(SceneFile(std::move(prims)));
}

std::string UsdTypeName(const AttributeDef& def) {
    static constexpr std::array<std::string_view, 14> kTypes{
        "bool",     "int",   "double", "double2", "double3", "double4", "quatd",
        "matrix4d", "token", "rel",    "rel",     "float[]", "token[]", ""};
    const AttributeMeta& meta = def.meta;
    switch (def.type) {
    case ColumnType::Vec3:
        return meta.color ? "color3d" : "double3";
    case ColumnType::TokenValue:
        return meta.asset ? "asset" : "token";
    case ColumnType::Tensor: {
        const std::string element = meta.bytes ? "uchar" : meta.integer ? "int" : "float";
        return element + (meta.components > 1 ? std::to_string(meta.components) : "") + "[]";
    }
    default:
        return std::string(kTypes[static_cast<std::size_t>(def.type)]);
    }
}

}  // namespace ernie
