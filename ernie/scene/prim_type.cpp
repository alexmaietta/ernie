// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/prim_type.h"

#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/terrain_schemas.h"

#include <array>
#include <cctype>
#include <stdexcept>
#include <unordered_map>

namespace ernie {

namespace {

std::optional<double> NumberOf(const Value& value) {
    if (const auto* d = value.Get<double>()) {
        return *d;
    }
    if (const auto* f = value.Get<float>()) {
        return static_cast<double>(*f);
    }
    if (const auto* i = value.Get<int>()) {
        return static_cast<double>(*i);
    }
    return std::nullopt;
}

/// "ColorTemperature" as "Color temperature".
std::string Words(std::string_view member) {
    std::string words;
    for (std::size_t i = 0; i < member.size(); ++i) {
        const char c = member[i];
        if (i > 0 && std::isupper(static_cast<unsigned char>(c)) != 0) {
            words += ' ';
            words += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        } else {
            words += c;
        }
    }
    return words;
}

}  // namespace

std::string SnakeCase(std::string_view member) {
    std::string snake;
    for (std::size_t i = 0; i < member.size(); ++i) {
        const char c = member[i];
        if (std::isupper(static_cast<unsigned char>(c)) != 0) {
            if (i > 0) {
                snake += '_';
            }
            snake += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        } else {
            snake += c;
        }
    }
    return snake;
}

std::optional<bool> AttributeTraits<bool>::From(const Value& value) {
    if (const auto* b = value.Get<bool>()) {
        return *b;
    }
    if (const auto* i = value.Get<int>(); i != nullptr && (*i == 0 || *i == 1)) {
        return *i != 0;
    }
    return std::nullopt;
}

std::optional<int> AttributeTraits<int>::From(const Value& value) {
    if (const auto* i = value.Get<int>()) {
        return *i;
    }
    return std::nullopt;
}

std::optional<double> AttributeTraits<double>::From(const Value& value) {
    return NumberOf(value);
}

std::optional<glm::dvec2> AttributeTraits<glm::dvec2>::From(const Value& value) {
    if (const auto* d = value.Get<glm::dvec2>()) {
        return *d;
    }
    if (const auto* f = value.Get<glm::vec2>()) {
        return glm::dvec2(*f);
    }
    if (const auto* i = value.Get<glm::ivec2>()) {
        return glm::dvec2(*i);
    }
    return std::nullopt;
}

std::optional<glm::dvec3> AttributeTraits<glm::dvec3>::From(const Value& value) {
    if (const auto* d = value.Get<glm::dvec3>()) {
        return *d;
    }
    if (const auto* f = value.Get<glm::vec3>()) {
        return glm::dvec3(*f);
    }
    return std::nullopt;
}

std::optional<std::string> AttributeTraits<std::string>::From(const Value& value) {
    if (const auto* s = value.Get<std::string>()) {
        return *s;
    }
    return std::nullopt;
}

std::optional<Path> AttributeTraits<Path>::From(const Value& value) {
    if (const auto* p = value.Get<Path>()) {
        return *p;
    }
    if (const auto* s = value.Get<std::string>()) {
        try {
            return Path(*s);
        } catch (const std::invalid_argument&) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

std::vector<float> AttributeTraits<std::vector<float>>::Read(const Scene& scene, const Path& path,
                                                             std::string_view name,
                                                             const std::vector<float>& fallback) {
    return GetFloats(scene, path, name).value_or(fallback);
}

std::optional<std::vector<float>> AttributeTraits<std::vector<float>>::From(const Value& value) {
    if (const auto* v = value.Get<std::vector<float>>()) {
        return *v;
    }
    // Two to four numbers arrive from Python as a vector.
    if (const auto* v = value.Get<glm::dvec2>()) {
        return std::vector<float>{static_cast<float>(v->x), static_cast<float>(v->y)};
    }
    if (const auto* v = value.Get<glm::ivec2>()) {
        return std::vector<float>{static_cast<float>(v->x), static_cast<float>(v->y)};
    }
    if (const auto* v = value.Get<glm::dvec3>()) {
        return std::vector<float>{static_cast<float>(v->x), static_cast<float>(v->y),
                                  static_cast<float>(v->z)};
    }
    if (const auto* v = value.Get<glm::vec4>()) {
        return std::vector<float>{v->x, v->y, v->z, v->w};
    }
    return std::nullopt;
}

AttributeDef AttributeDef::Describe(std::string_view name, std::string_view member,
                                    AttributeKind kind, AttributeMeta meta) {
    AttributeDef def;
    def.name = name;
    def.member = std::string(member);
    def.pyName = SnakeCase(member);
    def.label = meta.label.empty() ? Words(member) : std::string(meta.label);
    def.kind = kind;
    def.meta = std::move(meta);
    return def;
}

AttributeDef AttributeDef::MakeTensor(std::string_view name, std::string_view member,
                                      AttributeMeta meta) {
    AttributeDef def = Describe(name, member, AttributeKind::Tensor, std::move(meta));
    def.read_ = [](const AttributeDef&, const Scene&, const Path&) { return Value(); };
    return def;
}

Value AttributeDef::Coerce(const Value& value) const {
    if (from_ != nullptr) {
        if (std::optional<Value> converted = from_(value)) {
            return std::move(*converted);
        }
    }
    static constexpr std::array<std::string_view, 9> kKinds{
        "a bool",  "an int",      "a number",      "a 2-vector",         "a 3-vector",
        "a token", "a prim path", "a float array", "an array (write it)"};
    throw std::invalid_argument("'" + std::string(name) + "' must be " +
                                std::string(kKinds[static_cast<std::size_t>(kind)]));
}

bool PrimTypeDef::IsA(const PrimTypeDef& other) const {
    for (const PrimTypeDef* type = this; type != nullptr; type = type->base) {
        if (type == &other) {
            return true;
        }
    }
    return false;
}

const AttributeDef* PrimTypeDef::Find(std::string_view attribute) const {
    for (const PrimTypeDef* type = this; type != nullptr; type = type->base) {
        for (const AttributeDef& def : type->attributes) {
            if (def.name == attribute) {
                return &def;
            }
        }
    }
    return nullptr;
}

const AttributeDef* PrimTypeDef::FindByPyName(std::string_view attribute) const {
    for (const PrimTypeDef* type = this; type != nullptr; type = type->base) {
        for (const AttributeDef& def : type->attributes) {
            if (def.pyName == attribute) {
                return &def;
            }
        }
    }
    return nullptr;
}

std::vector<const AttributeDef*> PrimTypeDef::AllAttributes() const {
    std::vector<const PrimTypeDef*> chain;
    for (const PrimTypeDef* type = this; type != nullptr; type = type->base) {
        chain.push_back(type);
    }
    std::vector<const AttributeDef*> all;
    for (auto type = chain.rbegin(); type != chain.rend(); ++type) {
        for (const AttributeDef& def : (*type)->attributes) {
            all.push_back(&def);
        }
    }
    return all;
}

std::span<const PrimTypeDef* const> PrimTypes() {
    // Each schema file's types, bases first, after the bases' file.
    static const std::vector<const PrimTypeDef*> types = [] {
        std::vector<const PrimTypeDef*> all;
        for (std::vector<const PrimTypeDef*> family :
             {BaseSchemaTypes(), LightSchemaTypes(), GeometrySchemaTypes(), CameraSchemaTypes(),
              MaterialSchemaTypes(), RenderSchemaTypes(), TerrainSchemaTypes()}) {
            all.insert(all.end(), family.begin(), family.end());
        }
        return all;
    }();
    return types;
}

const PrimTypeDef* FindPrimType(std::string_view name) {
    static const std::unordered_map<std::string_view, const PrimTypeDef*> byName = [] {
        std::unordered_map<std::string_view, const PrimTypeDef*> map;
        for (const PrimTypeDef* type : PrimTypes()) {
            map.emplace(type->name, type);
        }
        return map;
    }();
    const auto found = byName.find(name);
    return found != byName.end() ? found->second : nullptr;
}

const PrimTypeDef& GetPrimType(std::string_view name) {
    if (const PrimTypeDef* type = FindPrimType(name)) {
        return *type;
    }
    throw std::invalid_argument("No prim type '" + std::string(name) + "' is registered");
}

const PrimTypeDef* PrimTypeOf(const Scene& scene, const Path& path) {
    return FindPrimType(scene.TypeOf(path));
}

namespace {

const PrimTypeDef& Checked(const Scene& scene, const Path& path, const PrimTypeDef& type) {
    const std::string& name = scene.TypeOf(path);
    const PrimTypeDef* found = FindPrimType(name);
    if (found == nullptr || !found->IsA(type)) {
        throw std::invalid_argument("'" + path.String() + "' is " +
                                    (name.empty() ? std::string("untyped") : "a " + name) +
                                    ", not a " + std::string(type.name));
    }
    return *found;
}

}  // namespace

SchemaView::SchemaView(const Scene& scene, Path path, const PrimTypeDef& type)
    : scene_(&scene), path_(std::move(path)), type_(&Checked(scene, path_, type)) {}

SchemaView::SchemaView(Scene& scene, Path path, const PrimTypeDef& type)
    : scene_(&scene), mutable_(&scene), path_(std::move(path)),
      type_(&Checked(scene, path_, type)) {}

Scene& SchemaView::Edit() const {
    if (mutable_ == nullptr) {
        throw std::logic_error("'" + path_.String() + "' was viewed through a const scene");
    }
    return *mutable_;
}

}  // namespace ernie
