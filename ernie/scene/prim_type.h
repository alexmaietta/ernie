// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/scene.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/value.h"
#include "ernie/util/tensor.h"

#include <glm/glm.hpp>

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

/// Prim types as reflected schemas: each type declares its attributes once, as a list of
/// (C++ type, name, USD token, fallback, metadata) entries, and that list gives both a typed view
/// class over `Scene` (`GetRadius`, `HasRadius`, `SetRadius`) and the type's entry in a runtime
/// registry (`PrimTypeDef`), which generic code walks: Python's typed prim classes, the viewer's
/// property panels, loaders. The view reads and writes the scene's attributes by their USD names,
/// so it changes nothing about storage or change tracking; unset attributes read as the declared
/// fallback, the one place Ernie keeps it.
///
/// A schema's header declares its attributes with an X-macro list and its class with
/// `ERNIE_SCHEMA_BODY`; its source defines the registry entry with `ERNIE_DEFINE_SCHEMA`, and
/// `prim_type.cpp` lists it in the registry. See `light_schemas.h`.
///
/// References:
/// - Blender, Cycles' node types and sockets (`NODE_DEFINE`, `SOCKET_*`, `NODE_SOCKET_API`),
///   https://projects.blender.org/blender/cycles/src/branch/main/src/graph/node_type.h
/// - Pixar, UsdLux and generated schema classes,
///   https://openusd.org/release/api/usd_lux_page_front.html
namespace ernie {

/// What an attribute holds, from its declared C++ type.
enum class AttributeKind {
    Bool,
    Int,
    Double,
    Vec2,
    Vec3,
    Token,         ///< A string: a token, or an asset path (`AttributeMeta::asset`).
    Relationship,  ///< A target `Path`.
    FloatArray,    ///< A small `std::vector<float>`, such as (nm, value) pairs.
    Tensor,        ///< Bulk data, written and read as a `Tensor`.
};

/// How an attribute is described and edited. Declared with designated initializers in this order.
struct AttributeMeta {
    std::string_view label;  ///< For panels; derived from the name ("Color temperature") if empty.
    std::string_view doc;
    std::string_view group;  ///< The panel section, such as "Light" or "Shape".
    double min = 0.0;        ///< Edit limits; none when equal.
    double max = 0.0;
    std::vector<std::string_view> options;  ///< A token's choices, or a relationship's target type.
    bool color = false;                     ///< A `Vec3` that is a linear RGB colour.
    bool asset = false;                     ///< A `Token` that is a file path.
};

class AttributeDef;

/// How the declared C++ type `T` reads, writes and converts: `Read` with the fallback (accepting
/// any numeric or vector type an author used, as `GetNumber` does), `Write`, and `From`, a value
/// as `T` or nullopt when it cannot be one.
template <typename T> struct AttributeTraits;

template <> struct AttributeTraits<bool> {
    static constexpr AttributeKind kKind = AttributeKind::Bool;
    using Result = bool;
    using Arg = bool;
    static bool Read(const Scene& scene, const Path& path, std::string_view name, bool fallback) {
        return GetBool(scene, path, name, fallback);
    }
    static std::optional<bool> From(const Value& value);
};

template <> struct AttributeTraits<int> {
    static constexpr AttributeKind kKind = AttributeKind::Int;
    using Result = int;
    using Arg = int;
    static int Read(const Scene& scene, const Path& path, std::string_view name, int fallback) {
        return static_cast<int>(GetNumber(scene, path, name, fallback));
    }
    static std::optional<int> From(const Value& value);
};

template <> struct AttributeTraits<double> {
    static constexpr AttributeKind kKind = AttributeKind::Double;
    using Result = double;
    using Arg = double;
    static double Read(const Scene& scene, const Path& path, std::string_view name,
                       double fallback) {
        return GetNumber(scene, path, name, fallback);
    }
    static std::optional<double> From(const Value& value);
};

template <> struct AttributeTraits<glm::dvec2> {
    static constexpr AttributeKind kKind = AttributeKind::Vec2;
    using Result = glm::dvec2;
    using Arg = glm::dvec2;
    static glm::dvec2 Read(const Scene& scene, const Path& path, std::string_view name,
                           glm::dvec2 fallback) {
        return GetVec2(scene, path, name, fallback);
    }
    static std::optional<glm::dvec2> From(const Value& value);
};

template <> struct AttributeTraits<glm::dvec3> {
    static constexpr AttributeKind kKind = AttributeKind::Vec3;
    using Result = glm::dvec3;
    using Arg = glm::dvec3;
    static glm::dvec3 Read(const Scene& scene, const Path& path, std::string_view name,
                           glm::dvec3 fallback) {
        return GetVec3(scene, path, name, fallback);
    }
    static std::optional<glm::dvec3> From(const Value& value);
};

template <> struct AttributeTraits<std::string> {
    static constexpr AttributeKind kKind = AttributeKind::Token;
    using Result = std::string;
    using Arg = std::string_view;
    static std::string Read(const Scene& scene, const Path& path, std::string_view name,
                            std::string_view fallback) {
        return GetToken(scene, path, name, fallback);
    }
    static std::optional<std::string> From(const Value& value);
};

template <> struct AttributeTraits<Path> {
    static constexpr AttributeKind kKind = AttributeKind::Relationship;
    using Result = Path;
    using Arg = const Path&;
    static Path Read(const Scene& scene, const Path& path, std::string_view name, const Path&) {
        return GetTarget(scene, path, name);
    }
    static std::optional<Path> From(const Value& value);
};

template <> struct AttributeTraits<std::vector<float>> {
    static constexpr AttributeKind kKind = AttributeKind::FloatArray;
    /// The authored array or the fallback; empty when neither has values.
    using Result = std::vector<float>;
    using Arg = std::vector<float>;
    static std::vector<float> Read(const Scene& scene, const Path& path, std::string_view name,
                                   const std::vector<float>& fallback);
    static std::optional<std::vector<float>> From(const Value& value);
};

/// Bulk data: reads the written tensor or null (there is no fallback), writes with `Scene::Write`.
template <> struct AttributeTraits<Tensor> {
    static constexpr AttributeKind kKind = AttributeKind::Tensor;
    using Result = const Tensor*;
    using Arg = Tensor;
    static const Tensor* Read(const Scene& scene, const Path& path, std::string_view name,
                              std::nullptr_t) {
        return scene.Read(path, name);
    }
};

/// One declared attribute of a prim type.
class AttributeDef {
  public:
    std::string_view name;  ///< Its USD name, such as "inputs:radius".
    std::string member;     ///< Its C++ accessor stem, such as "Radius".
    std::string pyName;     ///< Its snake_case name, such as "radius".
    std::string label;      ///< `meta.label`, or the member name in words ("Color temperature").
    AttributeKind kind = AttributeKind::Double;
    Value fallback;  ///< Empty for relationships and tensors.
    AttributeMeta meta;

    /// The attribute of the prim at `path` as its declared type, or the fallback when unset;
    /// empty for tensors (read them with `Scene::Read`).
    Value Read(const Scene& scene, const Path& path) const {
        return read_(*this, scene, path);
    }

    /// `value` as the declared type, converting as `Read` would (an int to a double, a `vec3` to
    /// a `dvec3`, a string to a `Path`).
    /// @throws std::invalid_argument if it cannot be one, and for tensors.
    Value Coerce(const Value& value) const;

    template <typename T>
    static AttributeDef Make(std::string_view name, std::string_view member, const T& fallback,
                             AttributeMeta meta);
    static AttributeDef MakeTensor(std::string_view name, std::string_view member,
                                   AttributeMeta meta);

  private:
    /// An attribute without its fallback or functions: names, label and metadata.
    static AttributeDef Describe(std::string_view name, std::string_view member, AttributeKind kind,
                                 AttributeMeta meta);

    Value (*read_)(const AttributeDef&, const Scene&, const Path&) = nullptr;
    std::optional<Value> (*from_)(const Value&) = nullptr;
};

/// A registered prim type: its name (`Scene::Define`'s type), its base, and the attributes it
/// declares itself; abstract ones, such as "Light", are bases only.
struct PrimTypeDef {
    std::string_view name;
    const PrimTypeDef* base = nullptr;
    bool abstract = false;
    std::string_view doc;
    std::vector<AttributeDef> attributes;

    /// Whether this is `other` or derives from it.
    bool IsA(const PrimTypeDef& other) const;

    /// The attribute named `name` (its USD name), here or on a base; null when it has none.
    const AttributeDef* Find(std::string_view name) const;

    /// The attribute whose `pyName` is `name`, here or on a base; null when it has none.
    const AttributeDef* FindByPyName(std::string_view name) const;

    /// Every attribute, the root base's first.
    std::vector<const AttributeDef*> AllAttributes() const;
};

/// Every registered prim type, bases before the types derived from them.
std::span<const PrimTypeDef* const> PrimTypes();

/// The registered prim type named `name`, or null.
const PrimTypeDef* FindPrimType(std::string_view name);

/// The registered prim type named `name`.
/// @throws std::invalid_argument if none is.
const PrimTypeDef& GetPrimType(std::string_view name);

/// The registered type of the prim at `path`, or null if its type is not registered.
/// @throws std::invalid_argument if the prim does not exist.
const PrimTypeDef* PrimTypeOf(const Scene& scene, const Path& path);

/// "ColorTemperature" as "color_temperature".
std::string SnakeCase(std::string_view member);

namespace schema {

/// Base of the typed views: a handle to one prim of a registered type, read-only when made from a
/// const scene. Copies are cheap and name the same prim.
class SchemaView {
  public:
    const Path& GetPath() const {
        return path_;
    }

    const Scene& GetScene() const {
        return *scene_;
    }

    /// The prim's registered type: this view's type or one derived from it.
    const PrimTypeDef& PrimType() const {
        return *type_;
    }

  protected:
    /// @throws std::invalid_argument if the prim does not exist or is not a `type`.
    SchemaView(const Scene& scene, Path path, const PrimTypeDef& type);
    SchemaView(Scene& scene, Path path, const PrimTypeDef& type);

    bool Has(std::string_view name) const {
        return scene_->Get(path_, name) != nullptr || scene_->Read(path_, name) != nullptr;
    }

    /// The scene to edit.
    /// @throws std::logic_error if the view was made from a const scene.
    Scene& Edit() const;

  private:
    const Scene* scene_ = nullptr;
    Scene* mutable_ = nullptr;
    Path path_;
    const PrimTypeDef* type_ = nullptr;
};

}  // namespace schema

}  // namespace ernie

/// One attribute's accessors in a view class: `Get<Name>` (the fallback when unset), `Has<Name>`
/// (whether it is authored) and `Set<Name>` (returning the view, for chaining).
#define ERNIE_ATTRIBUTE_ACCESSORS(Type, Name, token, fallback, meta)                               \
    ::ernie::AttributeTraits<Type>::Result Get##Name() const {                                     \
        return ::ernie::AttributeTraits<Type>::Read(GetScene(), GetPath(), token, fallback);       \
    }                                                                                              \
    bool Has##Name() const {                                                                       \
        return Has(token);                                                                         \
    }                                                                                              \
    Self& Set##Name(::ernie::AttributeTraits<Type>::Arg value) {                                   \
        ::ernie::schema::WriteAttribute<Type>(Edit(), GetPath(), token, value);                    \
        return static_cast<Self&>(*this);                                                          \
    }

/// The body of a view class `Class` deriving from `Base` (`SchemaView` for a root), with the
/// accessors of `ATTRIBUTES`, every attribute of the type and its bases.
#define ERNIE_SCHEMA_BODY(Class, Base, ATTRIBUTES)                                                 \
  public:                                                                                          \
    using Self = Class;                                                                            \
    /** @throws std::invalid_argument if the prim does not exist or is not a `Class`. */           \
    Class(const ::ernie::Scene& scene, ::ernie::Path path)                                         \
        : Base(scene, std::move(path), Definition()) {}                                            \
    Class(::ernie::Scene& scene, ::ernie::Path path)                                               \
        : Base(scene, std::move(path), Definition()) {}                                            \
    /** The type's registry entry. */                                                              \
    static const ::ernie::PrimTypeDef& Definition();                                               \
    ATTRIBUTES(ERNIE_ATTRIBUTE_ACCESSORS)                                                          \
                                                                                                   \
  protected:                                                                                       \
    Class(const ::ernie::Scene& scene, ::ernie::Path path, const ::ernie::PrimTypeDef& type)       \
        : Base(scene, std::move(path), type) {}                                                    \
    Class(::ernie::Scene& scene, ::ernie::Path path, const ::ernie::PrimTypeDef& type)             \
        : Base(scene, std::move(path), type) {}                                                    \
                                                                                                   \
  public:

/// `Define` for a concrete view class: defines the prim (`Scene::Define`) and returns its view.
#define ERNIE_SCHEMA_DEFINE(Class)                                                                 \
    static Class Define(::ernie::Scene& scene, const ::ernie::Path& path) {                        \
        scene.Define(path, Definition().name);                                                     \
        return Class(scene, path);                                                                 \
    }

/// One `AttributeDef` of an attribute list entry, for `ERNIE_DEFINE_SCHEMA`.
#define ERNIE_ATTRIBUTE_DEF(Type, Name, token, fallback, meta)                                     \
    ::ernie::schema::MakeAttribute<Type>(token, #Name, fallback, meta),

/// Defines `Class::Definition()`: the type `typeName`, deriving from `base` (a `PrimTypeDef*`, or
/// nullptr), declaring the attributes of `OWN_ATTRIBUTES`.
#define ERNIE_DEFINE_SCHEMA(Class, typeName, base, isAbstract, typeDoc, OWN_ATTRIBUTES)            \
    const ::ernie::PrimTypeDef& Class::Definition() {                                              \
        static const ::ernie::PrimTypeDef definition{                                              \
            typeName, base, isAbstract, typeDoc, {OWN_ATTRIBUTES(ERNIE_ATTRIBUTE_DEF)}};           \
        return definition;                                                                         \
    }

namespace ernie::schema {

template <typename T>
void WriteAttribute(Scene& scene, const Path& path, std::string_view name,
                    typename AttributeTraits<T>::Arg value) {
    if constexpr (std::is_same_v<T, Tensor>) {
        scene.Write(path, name, std::move(value));
    } else if constexpr (std::is_same_v<T, std::string>) {
        scene.Set(path, name, Value(std::string(value)));
    } else {
        scene.Set(path, name, Value(value));
    }
}

/// The `AttributeDef` of an attribute declared as `T` with `fallback` (anything `T` is made from;
/// nullptr for a tensor).
template <typename T, typename Fallback>
AttributeDef MakeAttribute(std::string_view name, std::string_view member, const Fallback& fallback,
                           AttributeMeta meta) {
    if constexpr (std::is_same_v<T, Tensor>) {
        return AttributeDef::MakeTensor(name, member, std::move(meta));
    } else {
        return AttributeDef::Make<T>(name, member, T(fallback), std::move(meta));
    }
}

}  // namespace ernie::schema

namespace ernie {

template <typename T>
AttributeDef AttributeDef::Make(std::string_view name, std::string_view member, const T& fallback,
                                AttributeMeta meta) {
    AttributeDef def = Describe(name, member, AttributeTraits<T>::kKind, std::move(meta));
    if constexpr (!std::is_same_v<T, Path>) {
        def.fallback = Value(fallback);
    }
    def.read_ = [](const AttributeDef& self, const Scene& scene, const Path& path) -> Value {
        if constexpr (std::is_same_v<T, Path>) {
            const Path target = AttributeTraits<Path>::Read(scene, path, self.name, Path());
            return target.IsEmpty() ? Value() : Value(target);
        } else {
            return Value(AttributeTraits<T>::Read(scene, path, self.name, *self.fallback.Get<T>()));
        }
    };
    def.from_ = [](const Value& value) -> std::optional<Value> {
        if (auto converted = AttributeTraits<T>::From(value)) {
            return Value(std::move(*converted));
        }
        return std::nullopt;
    };
    return def;
}

}  // namespace ernie
