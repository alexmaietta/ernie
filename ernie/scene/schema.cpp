// SPDX-License-Identifier: BSD-3-Clause
/// @file schema.cpp
/// @brief Schema lookups and value validation.
#include "ernie/scene/schema.h"

#include "ernie/scene/convert.h"
#include "ernie/util/text.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ernie {
namespace {

/// What the bounds take: "above 0 and below 170", "from 0 to 1", "finite and 1 or more".
std::string RangeText(const AttributeMeta& meta) {
    const std::optional<Bound>& lower = meta.lower;
    const std::optional<Bound>& upper = meta.upper;
    std::string text;
    if (lower && upper && !lower->exclusive && !upper->exclusive) {
        text = "from " + ToText(lower->value) + " to " + ToText(upper->value);
    } else {
        if (lower) {
            text = lower->exclusive ? "above " + ToText(lower->value)
                                    : ToText(lower->value) + " or more";
        }
        if (upper) {
            text += (text.empty() ? "" : " and ") + (upper->exclusive
                                                         ? "below " + ToText(upper->value)
                                                         : ToText(upper->value) + " or less");
        }
    }
    if (meta.finite) {
        text = text.empty() ? "finite" : "finite and " + text;
    }
    return text;
}

bool Fits(const AttributeMeta& meta, double number) {
    // Written to fail for NaN.
    return (!meta.finite || std::isfinite(number)) &&
           (!meta.lower ||
            (meta.lower->exclusive ? number > meta.lower->value : number >= meta.lower->value)) &&
           (!meta.upper ||
            (meta.upper->exclusive ? number < meta.upper->value : number <= meta.upper->value));
}

/// @return Why `value` doesn't fit the attribute, or empty when it does.
std::string Problem(const AttributeDef& def, const Value& value) {
    const std::vector<double> numbers = ToNumbers(value);
    for (const double number : numbers) {
        if (!Fits(def.meta, number)) {
            return "must be " + RangeText(def.meta) +
                   (numbers.size() > 1 ? " in each component" : "") + ", not " + ToText(number);
        }
    }
    const std::vector<std::string_view>& options = def.meta.options;
    if (const auto* token = value.Get<std::string>();
        token && !options.empty() && std::ranges::find(options, *token) == options.end()) {
        std::string list;
        for (const std::string_view option : options) {
            list += (list.empty() ? "" : ", ") + std::string(option);
        }
        return "must be one of " + list + ", not " + *token;
    }
    return {};
}

/// @return Why `array` doesn't fit the tensor attribute, or empty when it does.
std::string ArrayProblem(const AttributeDef& def, const Tensor& array) {
    if (def.type != ColumnType::Tensor) {
        return "takes one value, not an array; set it";
    }
    if (def.meta.bytes) {
        return {};
    }
    const DType type = array.Type();
    const bool integer = type == DType::Int32 || type == DType::Int64 || type == DType::UInt32;
    const bool floating = type == DType::Float32 || type == DType::Float64;
    if (def.meta.integer ? !integer : !floating) {
        return def.meta.integer ? "must hold integers" : "must hold float32 or float64 numbers";
    }
    if (def.meta.rank != 0) {
        return array.Rank() == def.meta.rank
                   ? std::string()
                   : "must have " + std::to_string(def.meta.rank) + " dimensions";
    }
    const std::int64_t components = def.meta.components;
    const bool rows =
        array.Rank() == 2 && array.Shape()[1] == std::max<std::int64_t>(components, 1);
    const bool flat =
        array.Rank() == 1 && array.Elements() % std::max<std::int64_t>(components, 1) == 0;
    if (!rows && !flat) {
        return components > 1 ? "must be (n, " + std::to_string(components) + ") or flat"
                              : "must be flat";
    }
    return {};
}

void ThrowIf(const std::string& problem, std::string_view attribute, const Path& path) {
    if (!problem.empty()) {
        throw std::invalid_argument("'" + path.String() + "': '" + std::string(attribute) + "' " +
                                    problem);
    }
}

}  // namespace

bool SchemaDef::IsA(const SchemaDef& other) const {
    return FirstInLineage([&](const SchemaDef& type) { return &type == &other; }) != nullptr;
}

bool SchemaDef::IsOpen(std::string_view attribute) const {
    return FirstInLineage([&](const SchemaDef& type) {
               return std::ranges::any_of(type.openNamespaces, [&](std::string_view space) {
                   return space == "*" || attribute.starts_with(space);
               });
           }) != nullptr;
}

const AttributeDef* SchemaDef::Find(std::string_view attribute) const {
    const AttributeDef* found = nullptr;
    FirstInLineage([&](const SchemaDef& type) {
        const auto def = std::ranges::find(type.attributes, attribute, &AttributeDef::name);
        found = def != type.attributes.end() ? &*def : nullptr;
        return found != nullptr;
    });
    return found;
}

std::vector<const AttributeDef*> SchemaDef::AllAttributes() const {
    std::vector<const AttributeDef*> all =
        base != nullptr ? base->AllAttributes() : std::vector<const AttributeDef*>{};
    for (const AttributeDef& def : attributes) {
        all.push_back(&def);
    }
    return all;
}

std::vector<std::string_view> SchemaDef::AllBuiltinApis() const {
    std::vector<std::string_view> all =
        base != nullptr ? base->AllBuiltinApis() : std::vector<std::string_view>{};
    all.insert(all.end(), builtinApis.begin(), builtinApis.end());
    return all;
}

const Value* SchemaDef::FallbackOverride(std::string_view attribute) const {
    const Value* found = nullptr;
    FirstInLineage([&](const SchemaDef& type) {
        const auto override = std::ranges::find(type.fallbackOverrides, attribute,
                                                &std::pair<std::string_view, Value>::first);
        found = override != type.fallbackOverrides.end() ? &override->second : nullptr;
        return found != nullptr;
    });
    return found;
}

ExtentRule SchemaDef::Extent() const {
    const SchemaDef* ruled = FirstInLineage(
        [](const SchemaDef& type) { return type.extent.kind != ExtentRule::Kind::Inherit; });
    return ruled != nullptr ? ruled->extent : kInheritExtent;
}

const SchemaDef* FindSchema(std::string_view name) {
    const std::span<const SchemaDef* const> schemas = Schemas();
    const auto found = std::ranges::find_if(
        schemas, [&](const SchemaDef* schema) { return schema->name == name; });
    return found != schemas.end() ? *found : nullptr;
}

ApiName ApiName::Parse(std::string_view api) {
    const std::size_t colon = api.find(':');
    const std::string_view name = api.substr(0, colon);
    const ApiName parsed{FindSchema(name), colon == std::string_view::npos ? std::string_view()
                                                                           : api.substr(colon + 1)};
    const bool single = parsed.schema != nullptr && parsed.schema->kind == SchemaKind::SingleApply;
    const bool multiple =
        parsed.schema != nullptr && parsed.schema->kind == SchemaKind::MultipleApply;
    if (!(single && parsed.instance.empty()) && !(multiple && IsIdentifier(parsed.instance))) {
        throw std::invalid_argument(
            "'" + std::string(api) + "' is no API schema" +
            (multiple ? ": a multiple-apply one takes an instance name, as CollectionAPI:lights"
                      : ""));
    }
    return parsed;
}

std::string ApiName::Attribute(std::string_view declared) const {
    std::string name(declared);
    if (const std::size_t at = name.find(kInstanceName); at != std::string::npos) {
        name.replace(at, kInstanceName.size(), instance);
    }
    return name;
}

std::string ApiName::ToString() const {
    return std::string(schema->name) + (instance.empty() ? "" : ":" + std::string(instance));
}

void Validate(const AttributeDef& def, std::string_view attribute, const Value& value,
              const Path& path) {
    ThrowIf(Problem(def, value), attribute, path);
}

void ValidateArray(const AttributeDef& def, std::string_view attribute, const Tensor& array,
                   const Path& path) {
    ThrowIf(ArrayProblem(def, array), attribute, path);
}

}  // namespace ernie
