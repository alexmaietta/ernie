// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/prim_type.h"

#include "ernie/scene/base_schemas.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/light_schemas.h"
#include "ernie/scene/material_schemas.h"
#include "ernie/scene/render_schemas.h"
#include "ernie/scene/terrain_schemas.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

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

glm::dvec4 AttributeTraits<glm::dvec4>::Read(const Scene& scene, const Path& path,
                                             std::string_view name, glm::dvec4 fallback) {
    const Value* value = scene.Get(path, name);
    return value != nullptr ? From(*value).value_or(fallback) : fallback;
}

std::optional<glm::dvec4> AttributeTraits<glm::dvec4>::From(const Value& value) {
    if (const auto* d = value.Get<glm::dvec4>()) {
        return *d;
    }
    if (const auto* f = value.Get<glm::vec4>()) {
        return glm::dvec4(*f);
    }
    return std::nullopt;
}

glm::dquat AttributeTraits<glm::dquat>::Read(const Scene& scene, const Path& path,
                                             std::string_view name, glm::dquat fallback) {
    const Value* value = scene.Get(path, name);
    return value != nullptr ? From(*value).value_or(fallback) : fallback;
}

std::optional<glm::dquat> AttributeTraits<glm::dquat>::From(const Value& value) {
    if (const auto* q = value.Get<glm::dquat>()) {
        return *q;
    }
    return std::nullopt;
}

glm::dmat4 AttributeTraits<glm::dmat4>::Read(const Scene& scene, const Path& path,
                                             std::string_view name, const glm::dmat4& fallback) {
    const Value* value = scene.Get(path, name);
    return value != nullptr ? From(*value).value_or(fallback) : fallback;
}

std::optional<glm::dmat4> AttributeTraits<glm::dmat4>::From(const Value& value) {
    if (const auto* m = value.Get<glm::dmat4>()) {
        return *m;
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

std::vector<Path> AttributeTraits<std::vector<Path>>::Read(const Scene& scene, const Path& path,
                                                           std::string_view name,
                                                           const std::vector<Path>& fallback) {
    const Value* value = scene.Get(path, name);
    if (value == nullptr) {
        return fallback;
    }
    if (const auto* list = value->Get<std::vector<Path>>()) {
        return *list;
    }
    if (const auto* single = value->Get<Path>()) {
        return {*single};
    }
    return fallback;
}

std::optional<std::vector<Path>> AttributeTraits<std::vector<Path>>::From(const Value& value) {
    if (const auto* list = value.Get<std::vector<Path>>()) {
        return *list;
    }
    if (const auto* texts = value.Get<std::vector<std::string>>()) {
        std::vector<Path> paths;
        try {
            for (const std::string& text : *texts) {
                paths.emplace_back(text);
            }
        } catch (const std::invalid_argument&) {
            return std::nullopt;
        }
        return paths;
    }
    if (const auto* floats = value.Get<std::vector<float>>();
        floats != nullptr && floats->empty()) {
        return std::vector<Path>{};
    }
    if (std::optional<Path> single = AttributeTraits<Path>::From(value)) {
        return std::vector<Path>{std::move(*single)};
    }
    return std::nullopt;
}

std::optional<Value> AttributeTraits<std::vector<Path>>::Store(const Value& value) {
    if (std::optional<Path> single = AttributeTraits<Path>::From(value)) {
        return Value(std::move(*single));
    }
    if (std::optional<std::vector<Path>> list = From(value)) {
        return Value(std::move(*list));
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
    if (const auto* v = value.Get<glm::dvec4>()) {
        return std::vector<float>{static_cast<float>(v->x), static_cast<float>(v->y),
                                  static_cast<float>(v->z), static_cast<float>(v->w)};
    }
    // One number is an array of one, as `GetFloats` reads it.
    if (const std::optional<double> number = NumberOf(value)) {
        return std::vector<float>{static_cast<float>(*number)};
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
    static constexpr std::array<std::string_view, 13> kKinds{"a bool",
                                                             "an int",
                                                             "a number",
                                                             "a 2-vector",
                                                             "a 3-vector",
                                                             "a 4-vector",
                                                             "a token",
                                                             "a prim path",
                                                             "a prim path or a list of them",
                                                             "a float array",
                                                             "an array (write it)",
                                                             "a quaternion",
                                                             "a 4 x 4 matrix"};
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

bool PrimTypeDef::IsOpen(std::string_view space) const {
    for (const PrimTypeDef* type = this; type != nullptr; type = type->base) {
        for (const std::string_view open : type->openNamespaces) {
            if (open == "*" || space == open ||
                (space.size() > open.size() && space.starts_with(open) &&
                 space[open.size()] == ':')) {
                return true;
            }
        }
    }
    return false;
}

namespace {

/// Edit distance between `a` and `b`, for suggesting the declared name an author meant.
std::size_t Distance(std::string_view a, std::string_view b) {
    std::vector<std::size_t> row(b.size() + 1);
    for (std::size_t j = 0; j <= b.size(); ++j) {
        row[j] = j;
    }
    for (std::size_t i = 1; i <= a.size(); ++i) {
        std::size_t diagonal = row[0];
        row[0] = i;
        for (std::size_t j = 1; j <= b.size(); ++j) {
            const std::size_t above = row[j];
            row[j] =
                std::min({row[j] + 1, row[j - 1] + 1, diagonal + (a[i - 1] == b[j - 1] ? 0u : 1u)});
            diagonal = above;
        }
    }
    return row[b.size()];
}

/// "; did you mean 'radius'?" for the declared names closest to `name`, or nothing.
std::string Suggestion(const PrimTypeDef& type, std::string_view name) {
    // Names compare whole and by their last component, so "radus" finds "inputs:radius".
    const auto last = [](std::string_view text) { return text.substr(text.rfind(':') + 1); };
    std::vector<std::pair<std::size_t, std::string_view>> close;
    for (const AttributeDef* def : type.AllAttributes()) {
        const std::size_t distance =
            std::min(Distance(name, def->name), Distance(last(name), last(def->name)));
        if (distance <= std::max<std::size_t>(2, name.size() / 4)) {
            close.emplace_back(distance, def->name);
        }
    }
    if (close.empty()) {
        return {};
    }
    std::sort(close.begin(), close.end());
    std::string text = "; did you mean ";
    for (std::size_t i = 0; i < std::min<std::size_t>(close.size(), 3); ++i) {
        text += (i > 0 ? ", '" : "'") + std::string(close[i].second) + "'";
    }
    return text + "?";
}

/// Attributes any prim may carry, whatever its type: undeclared primvars, transform ops Ernie does
/// not know and their order, and UsdShade connections.
bool AlwaysAllowed(std::string_view attribute) {
    return attribute.starts_with("xformOp") || attribute.starts_with("primvars:") ||
           attribute.ends_with(".connect");
}

/// The namespace of `attribute`: all but its last component ("inputs:shaping:cone" of
/// "inputs:shaping:cone:angle"), or empty.
std::string_view NamespaceOf(std::string_view attribute) {
    const std::size_t colon = attribute.rfind(':');
    return colon == std::string_view::npos ? std::string_view() : attribute.substr(0, colon);
}

/// A registered type's attributes by name, for `Scene::Set`'s check on every edit: the type's
/// own declarations before its bases' (as `Find` walks them), the namespaces they are in, and
/// whether the type takes any name.
struct TypeIndex {
    std::unordered_map<std::string_view, const AttributeDef*> byName;
    std::unordered_map<std::string_view, const AttributeDef*> byPyName;
    std::unordered_set<std::string_view> spaces;
    bool openAll = false;
};

/// The index of `type` when it is registered (`PrimTypes`), built for every registered type the
/// first time any is asked for; null for others.
const TypeIndex* IndexOf(const PrimTypeDef& type) {
    static const std::unordered_map<const PrimTypeDef*, TypeIndex> indices = [] {
        std::unordered_map<const PrimTypeDef*, TypeIndex> built;
        for (const PrimTypeDef* registered : PrimTypes()) {
            TypeIndex& index = built[registered];
            for (const PrimTypeDef* level = registered; level != nullptr; level = level->base) {
                for (const AttributeDef& def : level->attributes) {
                    index.byName.emplace(def.name, &def);
                    index.byPyName.emplace(def.pyName, &def);
                    if (const std::string_view space = NamespaceOf(def.name); !space.empty()) {
                        index.spaces.insert(space);
                    }
                }
            }
            index.openAll = registered->IsOpen("*");
        }
        return built;
    }();
    const auto it = indices.find(&type);
    return it != indices.end() ? &it->second : nullptr;
}

/// Checks `value`, or an array written in its place when null, against the declared `def` of a
/// prim of `type` (null: untyped).
void CheckValue(const AttributeDef& def, const PrimTypeDef* type, const Path& path,
                const Value* value) {
    if (def.meta.lenient) {
        return;
    }
    const auto fail = [&](std::string_view message) {
        throw std::invalid_argument(
            "'" + path.String() + "'" +
            (type != nullptr ? " (" + std::string(type->name) + ")" : std::string()) + ": " +
            std::string(message));
    };
    if (value == nullptr) {
        if (def.kind != AttributeKind::Tensor && !def.meta.array) {
            fail("'" + std::string(def.name) + "' takes one value, not an array; set it");
        }
        return;
    }
    if (def.kind == AttributeKind::Tensor) {
        return;
    }
    // Numbers of any type, as `GetNumber` reads them (an int attribute's reader rounds, or
    // checks, them itself), and ints for bools, as `GetBool` reads them.
    const bool number = value->Is<double>() || value->Is<float>() || value->Is<int>();
    if (((def.kind == AttributeKind::Int || def.kind == AttributeKind::Double) && number) ||
        (def.kind == AttributeKind::Bool && (value->Is<bool>() || value->Is<int>()))) {
        return;
    }
    try {
        def.Coerce(*value);
    } catch (const std::invalid_argument& error) {
        fail(error.what());
    }
}

}  // namespace

const AttributeDef* FindXformOp(std::string_view attribute) {
    constexpr std::string_view kPrefix = "xformOp:";
    if (!attribute.starts_with(kPrefix)) {
        return nullptr;
    }
    std::string_view op = attribute.substr(kPrefix.size());
    op = op.substr(0, op.find(':'));
    static const std::vector<AttributeDef> kOps = [] {
        const auto make = [](std::string_view name, std::string_view member, const auto& fallback,
                             std::string_view doc) {
            return AttributeDef::Make(
                name, member, fallback,
                AttributeMeta{.doc = doc, .group = "Transform", .array = true});
        };
        std::vector<AttributeDef> ops;
        ops.push_back(make(tokens::kTranslate, "Translate", glm::dvec3(0.0), "Metres."));
        AttributeDef scale =
            make(tokens::kScale, "Scale", glm::dvec3(1.0), "Factors along x, y and z, or one.");
        scale.from_ = [](const Value& value) -> std::optional<Value> {
            if (const std::optional<glm::dvec3> factors =
                    AttributeTraits<glm::dvec3>::From(value)) {
                return Value(*factors);
            }
            if (const std::optional<double> factor = NumberOf(value)) {
                return Value(glm::dvec3(*factor));
            }
            return std::nullopt;
        };
        ops.push_back(std::move(scale));
        for (const auto& [name, member] :
             {std::pair{tokens::kRotateX, "RotateX"}, std::pair{tokens::kRotateY, "RotateY"},
              std::pair{tokens::kRotateZ, "RotateZ"}}) {
            ops.push_back(make(name, member, 0.0, "Degrees."));
        }
        for (const auto& [name, member] : {std::pair{tokens::kRotateXYZ, "RotateXYZ"},
                                           std::pair{tokens::kRotateXZY, "RotateXZY"},
                                           std::pair{tokens::kRotateYXZ, "RotateYXZ"},
                                           std::pair{tokens::kRotateYZX, "RotateYZX"},
                                           std::pair{tokens::kRotateZXY, "RotateZXY"},
                                           std::pair{tokens::kRotateZYX, "RotateZYX"}}) {
            ops.push_back(make(name, member, glm::dvec3(0.0), "Euler angles in degrees."));
        }
        ops.push_back(make(tokens::kOrient, "Orient", glm::dquat(1.0, 0.0, 0.0, 0.0),
                           "A rotation as a quaternion."));
        ops.push_back(make(tokens::kTransform, "Transform", glm::dmat4(1.0),
                           "A matrix, column vectors with the translation in the last column."));
        return ops;
    }();
    for (const AttributeDef& def : kOps) {
        if (def.name.substr(kPrefix.size()) == op) {
            return &def;
        }
    }
    return nullptr;
}

void Validate(const PrimTypeDef* type, const Path& path, std::string_view attribute,
              const Value* value) {
    if (type == nullptr) {
        if (const AttributeDef* op = FindXformOp(attribute)) {
            CheckValue(*op, nullptr, path, value);
        }
        return;
    }
    if (const AttributeDef* def = type->Find(attribute)) {
        CheckValue(*def, type, path, value);
        return;
    }
    if (const AttributeDef* op = FindXformOp(attribute)) {
        CheckValue(*op, type, path, value);
        return;
    }
    if (AlwaysAllowed(attribute)) {
        return;
    }
    const TypeIndex* index = IndexOf(*type);
    if (index != nullptr ? index->openAll : type->IsOpen("*")) {
        return;
    }
    const std::string_view space = NamespaceOf(attribute);
    if (!space.empty()) {
        if (type->IsOpen(space)) {
            return;
        }
        // Custom data in a namespace of its own is the author's; a typo in one of the type's
        // own namespaces is not.
        bool declared = false;
        if (index != nullptr) {
            declared = index->spaces.contains(space);
        } else {
            for (const AttributeDef* def : type->AllAttributes()) {
                declared = declared || NamespaceOf(def->name) == space;
            }
        }
        if (!declared) {
            return;
        }
    }
    throw std::invalid_argument("'" + path.String() + "' is a " + std::string(type->name) +
                                ", which has no attribute '" + std::string(attribute) + "'" +
                                Suggestion(*type, attribute));
}

const AttributeDef* PrimTypeDef::Find(std::string_view attribute) const {
    if (const TypeIndex* index = IndexOf(*this)) {
        const auto it = index->byName.find(attribute);
        return it != index->byName.end() ? it->second : nullptr;
    }
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
    if (const TypeIndex* index = IndexOf(*this)) {
        const auto it = index->byPyName.find(attribute);
        return it != index->byPyName.end() ? it->second : nullptr;
    }
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
