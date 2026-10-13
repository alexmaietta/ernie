// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/scene.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/value.h"
#include "ernie/scene/xform.h"
#include "ernie/util/numbers.h"
#include "ernie/util/tensor.h"

#include <glm/glm.hpp>

#include <cstddef>
#include <initializer_list>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

/// Prim types as reflected schemas: each type declares its attributes
/// once, as a list of (C++ type, name, USD token, fallback, metadata)
/// entries, and that list gives both a typed view class over `Scene`
/// (`GetRadius`, `HasRadius`, `SetRadius`) and the type's entry in a
/// runtime registry (`PrimTypeDef`), which generic code walks:
/// Python's typed prim classes, the viewer's property panels,
/// loaders. The view reads and writes the scene's attributes by their
/// USD names, so it changes nothing about storage or change tracking;
/// unset attributes read as the declared fallback, the one place
/// Ernie keeps it.
///
/// A schema's header declares its attributes with an X-macro list and
/// its class with `ERNIE_SCHEMA_BODY`; its source defines the
/// registry entry with `ERNIE_DEFINE_SCHEMA`, and `prim_type.cpp`
/// lists it in the registry. See `light_schemas.h`.
///
/// References:
/// - Blender, Cycles' node types and sockets (`NODE_DEFINE`,
///   `SOCKET_*`, `NODE_SOCKET_API`),
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
    Vec4,
    Token,          ///< A string: a token, or an asset path
                    ///< (`AttributeMeta::asset`).
    Relationship,   ///< A target `Path`.
    Relationships,  ///< A list of target `Path`s; one `Path` authored
                    ///< alone is a list of one.
    FloatArray,     ///< A small `std::vector<float>`, such as (nm,
                    ///< value) pairs.
    Tensor,         ///< Bulk data, written and read as a `Tensor`.
    Quat,           ///< A `glm::dquat`, written [w, x, y, z] in scene
                    ///< descriptions.
    Matrix,  ///< A `glm::dmat4` (math convention), written as 4 rows
             ///< of 4.
    TokenArray,  ///< A `std::vector<std::string>`, such as joint
                 ///< names.
};

/// How an attribute is described and edited. Declared with designated
/// initializers in this order; every member has a default
/// initializer, so those naming only some draw no
/// -Wmissing-field-initializers.
struct AttributeMeta {
    std::string_view label{};  ///< For panels; derived from the name
                               ///< ("Color temperature") if empty.
    std::string_view doc{};
    std::string_view
        group{};  ///< The panel section, such as "Light" or "Shape".
    double min = 0.0;  ///< Edit limits; none when equal.
    double max = 0.0;
    std::vector<std::string_view>
        options{};  ///< A token's choices, or a relationship's target
                    ///< type.
    bool color =
        false;  ///< A `Vec3` (`Vec4`) that is linear RGB (RGBA).
    bool asset = false;  ///< A `Token` that is a file path.
    /// A `Tensor` of whole numbers, such as indices: scene loaders
    /// make int32 arrays of it, and float32 arrays of other tensors,
    /// whatever the numbers look like.
    bool integer = false;
    /// Its readers also take values of other types (a legacy form),
    /// so `Scene::Set` checks only its name, not its value's type.
    bool lenient = false;
    /// What changing it stamps (`DirtyOf`): `Dirty::Points` for bulk
    /// positions, say. Unset, any other edit (`Dirty::Params`).
    std::optional<Dirty> dirty{};
    /// For a list of targets usually holding one (a product's camera,
    /// or a batch's cameras): the plural name Python reads the list
    /// by, its own name then reading the first target alone.
    std::string_view plural{};
    /// It also takes an array written in place of its value
    /// (`Scene::Write`): colours per vertex for one `displayColor`, a
    /// width per point, a pose per prim for a transform op. Scene
    /// loaders make float32 arrays of nested arrays for it; others
    /// take no array. Declare such an attribute as `OrArray<T>`,
    /// which sets it and gives its view `Read<Name>`.
    bool array = false;
};

class AttributeDef;

/// How the declared C++ type `T` reads, writes and converts: `Read`
/// with the fallback (accepting any numeric or vector type an author
/// used, as `GetNumber` does), `Write`, and `From`, a value as `T` or
/// nullopt when it cannot be one.
template <typename T> struct AttributeTraits;

template <> struct AttributeTraits<bool> {
    static constexpr AttributeKind kKind = AttributeKind::Bool;
    using Result = bool;
    using Arg = bool;
    static bool Read(const Scene& scene, const Path& path,
                     std::string_view name, bool fallback) {
        return GetBool(scene, path, name, fallback);
    }
    static std::optional<bool> From(const Value& value);
};

template <> struct AttributeTraits<int> {
    static constexpr AttributeKind kKind = AttributeKind::Int;
    using Result = int;
    using Arg = int;
    static int Read(const Scene& scene, const Path& path,
                    std::string_view name, int fallback) {
        return static_cast<int>(
            GetNumber(scene, path, name, fallback));
    }
    static std::optional<int> From(const Value& value);
};

template <> struct AttributeTraits<double> {
    static constexpr AttributeKind kKind = AttributeKind::Double;
    using Result = double;
    using Arg = double;
    static double Read(const Scene& scene, const Path& path,
                       std::string_view name, double fallback) {
        return GetNumber(scene, path, name, fallback);
    }
    static std::optional<double> From(const Value& value);
};

template <> struct AttributeTraits<glm::dvec2> {
    static constexpr AttributeKind kKind = AttributeKind::Vec2;
    using Result = glm::dvec2;
    using Arg = glm::dvec2;
    static glm::dvec2 Read(const Scene& scene, const Path& path,
                           std::string_view name,
                           glm::dvec2 fallback) {
        return GetVec2(scene, path, name, fallback);
    }
    static std::optional<glm::dvec2> From(const Value& value);
};

template <> struct AttributeTraits<glm::dvec3> {
    static constexpr AttributeKind kKind = AttributeKind::Vec3;
    using Result = glm::dvec3;
    using Arg = glm::dvec3;
    static glm::dvec3 Read(const Scene& scene, const Path& path,
                           std::string_view name,
                           glm::dvec3 fallback) {
        return GetVec3(scene, path, name, fallback);
    }
    static std::optional<glm::dvec3> From(const Value& value);
};

template <> struct AttributeTraits<glm::dvec4> {
    static constexpr AttributeKind kKind = AttributeKind::Vec4;
    using Result = glm::dvec4;
    using Arg = glm::dvec4;
    /// From a `vec4` or `dvec4`; `fallback` when unset or of another
    /// type.
    static glm::dvec4 Read(const Scene& scene, const Path& path,
                           std::string_view name,
                           glm::dvec4 fallback);
    static std::optional<glm::dvec4> From(const Value& value);
};

template <> struct AttributeTraits<glm::dquat> {
    static constexpr AttributeKind kKind = AttributeKind::Quat;
    using Result = glm::dquat;
    using Arg = glm::dquat;
    /// A `dquat`; `fallback` when unset or of another type (four
    /// numbers are not one: w may be first or last).
    static glm::dquat Read(const Scene& scene, const Path& path,
                           std::string_view name,
                           glm::dquat fallback);
    static std::optional<glm::dquat> From(const Value& value);
};

template <> struct AttributeTraits<glm::dmat4> {
    static constexpr AttributeKind kKind = AttributeKind::Matrix;
    using Result = glm::dmat4;
    using Arg = const glm::dmat4&;
    /// A `dmat4`; `fallback` when unset or of another type.
    static glm::dmat4 Read(const Scene& scene, const Path& path,
                           std::string_view name,
                           const glm::dmat4& fallback);
    static std::optional<glm::dmat4> From(const Value& value);
};

template <> struct AttributeTraits<std::string> {
    static constexpr AttributeKind kKind = AttributeKind::Token;
    using Result = std::string;
    using Arg = std::string_view;
    static std::string Read(const Scene& scene, const Path& path,
                            std::string_view name,
                            std::string_view fallback) {
        return GetToken(scene, path, name, fallback);
    }
    static std::optional<std::string> From(const Value& value);
};

template <> struct AttributeTraits<Path> {
    static constexpr AttributeKind kKind =
        AttributeKind::Relationship;
    using Result = Path;
    using Arg = const Path&;
    static Path Read(const Scene& scene, const Path& path,
                     std::string_view name, const Path&) {
        return GetTarget(scene, path, name);
    }
    static std::optional<Path> From(const Value& value);
};

/// Relationship targets given to a setter: one target (a `Path` or
/// its text), stored as a single `Path`, or a list of them, stored as
/// a list; reads take either.
class Targets {
  public:
    Targets(Path target) : paths_{std::move(target)}, single_(true) {}
    Targets(const char* target) : Targets(Path(target)) {}
    Targets(std::string_view target) : Targets(Path(target)) {}
    Targets(const std::string& target) : Targets(Path(target)) {}
    Targets(std::vector<Path> targets) : paths_(std::move(targets)) {}
    Targets(std::initializer_list<Path> targets) : paths_(targets) {}

    const std::vector<Path>& Paths() const {
        return paths_;
    }

    /// As the scene stores it: a `Path` for one target given alone,
    /// else the list.
    Value ToValue() const {
        return single_ ? Value(paths_.front()) : Value(paths_);
    }

  private:
    std::vector<Path> paths_;
    bool single_ = false;
};

template <> struct AttributeTraits<std::vector<Path>> {
    static constexpr AttributeKind kKind =
        AttributeKind::Relationships;
    using Result = std::vector<Path>;
    using Arg = const Targets&;
    /// The targets of one `Path` or a list; `fallback` when unset or
    /// of another type.
    static std::vector<Path> Read(const Scene& scene,
                                  const Path& path,
                                  std::string_view name,
                                  const std::vector<Path>& fallback);
    /// A `Path`, a string, or a list of either (an empty list arrives
    /// from Python as an empty float array) as the list of targets.
    static std::optional<std::vector<Path>> From(const Value& value);
    /// `value` as the scene stores it (`AttributeDef::Coerce`): a
    /// single target (a `Path` or a string) as a `Path`, a list as a
    /// list of `Path`s.
    static std::optional<Value> Store(const Value& value);
};

template <> struct AttributeTraits<std::vector<float>> {
    static constexpr AttributeKind kKind = AttributeKind::FloatArray;
    /// The authored array or the fallback; empty when neither has
    /// values.
    using Result = std::vector<float>;
    using Arg = std::vector<float>;
    static std::vector<float>
    Read(const Scene& scene, const Path& path, std::string_view name,
         const std::vector<float>& fallback);
    /// A float array, a 2- to 4-vector or a single number (an array
    /// of one) as an array.
    static std::optional<std::vector<float>> From(const Value& value);
};

template <> struct AttributeTraits<std::vector<std::string>> {
    static constexpr AttributeKind kKind = AttributeKind::TokenArray;
    using Result = std::vector<std::string>;
    using Arg = std::vector<std::string>;
    /// The authored tokens, or `fallback` when unset or of another
    /// type.
    static std::vector<std::string>
    Read(const Scene& scene, const Path& path, std::string_view name,
         const std::vector<std::string>& fallback);
    /// A list of strings, or one string as a list of one (an empty
    /// list arrives from Python as an empty float array).
    static std::optional<std::vector<std::string>>
    From(const Value& value);
};

/// Bulk data: reads the written tensor or null (there is no
/// fallback), writes with `Scene::Write`.
template <> struct AttributeTraits<Tensor> {
    static constexpr AttributeKind kKind = AttributeKind::Tensor;
    using Result = const Tensor*;
    using Arg = Tensor;
    static const Tensor* Read(const Scene& scene, const Path& path,
                              std::string_view name, std::nullptr_t) {
        return scene.Read(path, name);
    }
};

/// An attribute declared as `T` that also takes an array written in
/// place of its value: colours per vertex for one `displayColor`, a
/// width per point. Declaring one so sets `AttributeMeta::array`; its
/// view reads the value with `Get<Name>` as for `T`, and the array
/// with `Read<Name>` (null when none is written) and `Write<Name>`.
template <typename T> struct OrArray {
    using Value = T;
};

template <typename T> inline constexpr bool kIsOrArray = false;
template <typename T>
inline constexpr bool kIsOrArray<OrArray<T>> = true;

template <typename T>
struct AttributeTraits<OrArray<T>> : AttributeTraits<T> {};

/// One declared attribute of a prim type.
class AttributeDef {
  public:
    std::string_view
        name;            ///< Its USD name, such as "inputs:radius".
    std::string member;  ///< Its C++ accessor stem, such as "Radius".
    std::string pyName;  ///< Its snake_case name, such as "radius".
    std::string label;  ///< `meta.label`, or the member name in words
                        ///< ("Color temperature").
    AttributeKind kind = AttributeKind::Double;
    Value fallback;  ///< Empty for single relationships and tensors.
    AttributeMeta meta;

    /// The attribute of the prim at `path` as its declared type, or
    /// the fallback when unset; empty for tensors (read them with
    /// `Scene::Read`).
    Value Read(const Scene& scene, const Path& path) const {
        return read_(*this, scene, path);
    }

    /// `value` as the declared type, converting as `Read` would (an
    /// int to a double, a `vec3` to a `dvec3`, a string to a `Path`);
    /// for a list of relationship targets, as the scene stores it:
    /// one target given alone a `Path`, a list a list of `Path`s.
    /// @throws std::invalid_argument if it cannot be one, and for
    /// tensors.
    Value Coerce(const Value& value) const;

    template <typename T>
    static AttributeDef Make(std::string_view name,
                             std::string_view member,
                             const T& fallback, AttributeMeta meta);
    static AttributeDef MakeTensor(std::string_view name,
                                   std::string_view member,
                                   AttributeMeta meta);

  private:
    /// An attribute without its fallback or functions: names, label
    /// and metadata.
    static AttributeDef Describe(std::string_view name,
                                 std::string_view member,
                                 AttributeKind kind,
                                 AttributeMeta meta);

    friend const AttributeDef*
    FindXformOp(std::string_view attribute);

    Value (*read_)(const AttributeDef&, const Scene&,
                   const Path&) = nullptr;
    std::optional<Value> (*from_)(const Value&) = nullptr;
};

/// The transform op `attribute` names ("xformOp:orient",
/// "xformOp:translate:pivot") as a declared attribute, any prim's:
/// `xformOp:translate` and the Euler rotations a `Vec3`,
/// `xformOp:scale` a `Vec3` or one number, `rotateX` (Y, Z) a
/// `Double` in degrees, `xformOp:orient` a `Quat` and
/// `xformOp:transform` a `Matrix`, each also taking an array of them,
/// a pose per prim (`AttributeMeta::array`); null for other names and
/// ops Ernie does not know.
const AttributeDef* FindXformOp(std::string_view attribute);

/// A registered prim type: its name (`Scene::Define`'s type), its
/// base, and the attributes it declares itself; abstract ones, such
/// as "Light", are bases only.
struct PrimTypeDef {
    std::string_view name;
    const PrimTypeDef* base = nullptr;
    bool abstract = false;
    std::string_view doc;
    std::vector<AttributeDef> attributes;
    /// Namespaces whose attributes are data-driven, so any name in
    /// them is allowed (`Validate`): "inputs" on a Shader, whose
    /// inputs depend on its `info:id`, say.
    std::vector<std::string_view> openNamespaces;

    /// Whether this is `other` or derives from it.
    bool IsA(const PrimTypeDef& other) const;

    /// Whether attributes in `space` (such as "inputs" or
    /// "ernie:sensor") may have any name here or on a base.
    bool IsOpen(std::string_view space) const;

    /// The attribute named `name` (its USD name), here or on a base;
    /// null when it has none.
    const AttributeDef* Find(std::string_view name) const;

    /// The attribute whose `pyName` is `name`, here or on a base;
    /// null when it has none.
    const AttributeDef* FindByPyName(std::string_view name) const;

    /// Every attribute, the root base's first.
    std::vector<const AttributeDef*> AllAttributes() const;
};

/// Checks that `attribute` may be set on a prim of `type` (at `path`,
/// for messages), and that `value` is of its declared type, or, when
/// null (`Scene::Write`), that it takes an array, as `Scene::Set` and
/// `Scene::Write` do:
/// - a declared attribute, or a transform op (`FindXformOp`), takes
///   what its readers take (`AttributeDef::Coerce`; anything for
///   `AttributeMeta::lenient` ones), and an array only when it is a
///   tensor or `AttributeMeta::array` (a tensor also takes small
///   arrays as values);
/// - an undeclared one is an error when its name has no namespace
///   (`radus`), or is in a namespace the type declares attributes in
///   and does not open (`inputs:radus` on a light,
///   `ernie:sensor:noize` on a camera), with the closest declared
///   names suggested;
/// - everything else is allowed: other namespaces (custom data such
///   as `user:speed`), the type's open namespaces
///   (`PrimTypeDef::openNamespaces`), undeclared `primvars:*`,
///   `xformOpOrder` and ops Ernie does not know, and UsdShade
///   connections (`<input>.connect`). A null `type` (an untyped prim,
///   or one of a type not registered) checks only what Ernie reads on
///   every prim: transform ops and Imageable's attributes
///   (`visibility`, `ernie:world`, `ernie:quaternionOrder`), any
///   other name being allowed.
/// @throws std::invalid_argument naming the prim and the attribute.
void Validate(const PrimTypeDef* type, const Path& path,
              std::string_view attribute, const Value* value);

/// Every registered prim type, bases before the types derived from
/// them.
std::span<const PrimTypeDef* const> PrimTypes();

/// The registered prim type named `name`, or null.
const PrimTypeDef* FindPrimType(std::string_view name);

/// The registered prim type named `name`.
/// @throws std::invalid_argument if none is.
const PrimTypeDef& GetPrimType(std::string_view name);

/// The registered type of the prim at `path`, or null if its type is
/// not registered.
/// @throws std::invalid_argument if the prim does not exist.
const PrimTypeDef* PrimTypeOf(const Scene& scene, const Path& path);

/// "ColorTemperature" as "color_temperature".
std::string SnakeCase(std::string_view member);

/// Base of the typed views: a handle to one prim of a registered
/// type, read-only when made from a const scene. Copies are cheap and
/// name the same prim.
class SchemaView {
  public:
    const Path& GetPath() const {
        return path_;
    }

    const Scene& GetScene() const {
        return *scene_;
    }

    /// A handle to the prim, for what the view does not cover
    /// (`Prim::Get`, `Prim::Define`).
    /// @throws std::logic_error if the view was made from a const
    /// scene.
    Prim GetPrim() const {
        return Prim(&Edit(), path_);
    }

    /// The prim's registered type: this view's type or one derived
    /// from it.
    const PrimTypeDef& PrimType() const {
        return *type_;
    }

  protected:
    /// @throws std::invalid_argument if the prim does not exist or is
    /// not a `type`.
    SchemaView(const Scene& scene, Path path,
               const PrimTypeDef& type);
    SchemaView(Scene& scene, Path path, const PrimTypeDef& type);

    bool Has(std::string_view name) const {
        return scene_->Get(path_, name) != nullptr ||
               scene_->Read(path_, name) != nullptr;
    }

    /// The scene to edit.
    /// @throws std::logic_error if the view was made from a const
    /// scene.
    Scene& Edit() const;

  private:
    const Scene* scene_ = nullptr;
    Scene* mutable_ = nullptr;
    Path path_;
    const PrimTypeDef* type_ = nullptr;
};

}  // namespace ernie

/// One attribute's accessors in a view class: `Get<Name>` (the
/// fallback when unset), `Has<Name>` (whether it is authored), and
/// `Set<Name>` and `Clear<Name>` (removing it, `Scene::Clear`), which
/// return the view for chaining; for an `OrArray` attribute,
/// `Read<Name>` and `Write<Name>` too, its array.
#define ERNIE_ATTRIBUTE_ACCESSORS(Type, Name, token, fallback, meta) \
    ::ernie::AttributeTraits<Type>::Result Get##Name() const {       \
        return ::ernie::AttributeTraits<Type>::Read(                 \
            GetScene(), GetPath(), token, fallback);                 \
    }                                                                \
    bool Has##Name() const {                                         \
        return Has(token);                                           \
    }                                                                \
    Self& Set##Name(::ernie::AttributeTraits<Type>::Arg value) {     \
        ::ernie::detail::WriteAttribute<Type>(Edit(), GetPath(),     \
                                              token, value);         \
        return static_cast<Self&>(*this);                            \
    }                                                                \
    /** For a vector or float array attribute: its numbers from any  \
     * contiguous range (`std::array`, `std::vector`, `std::span`),  \
     * copied. @throws std::invalid_argument if a vector's numbers   \
     * are not as many as it has. */                                 \
    template <typename R>                                            \
        requires ::ernie::detail::SetsFromNumbers<Type, R>           \
    Self& Set##Name(const R& values) {                               \
        return Set##Name(                                            \
            ::ernie::detail::FromNumbers<Type>(values));             \
    }                                                                \
    /** For a tensor attribute: a copy of the numbers of any         \
     * contiguous range, of `shape` (one dimension when empty), on   \
     * the host; GPU arrays are written as tensors (DLPack). */      \
    template <typename R, typename T = Type>                         \
        requires(std::is_same_v<T, ::ernie::Tensor> &&               \
                 ::ernie::NumberRange<R>)                            \
    Self& Set##Name(const R& values,                                 \
                    std::vector<std::int64_t> shape = {}) {          \
        return Set##Name(::ernie::detail::TensorFromNumbers(         \
            values, std::move(shape)));                              \
    }                                                                \
    Self& Clear##Name() {                                            \
        Edit().Clear(GetPath(), token);                              \
        return static_cast<Self&>(*this);                            \
    }                                                                \
    /** For an `OrArray` attribute: the array written in place of    \
     * its value, or null. */                                        \
    template <typename T = Type>                                     \
        requires ::ernie::kIsOrArray<T>                              \
    const ::ernie::Tensor* Read##Name() const {                      \
        return GetScene().Read(GetPath(), token);                    \
    }                                                                \
    /** For an `OrArray` attribute: writes an array in place of its  \
     * value (`Scene::Write`). */                                    \
    template <typename T = Type>                                     \
        requires ::ernie::kIsOrArray<T>                              \
    Self& Write##Name(::ernie::Tensor array) {                       \
        Edit().Write(GetPath(), token, std::move(array));            \
        return static_cast<Self&>(*this);                            \
    }                                                                \
    /** For an `OrArray` attribute: writes a float32 copy of the     \
     * numbers of any contiguous range as its array, rows of a       \
     * vector's numbers. @throws std::invalid_argument if they do    \
     * not fill whole rows. */                                       \
    template <typename R, typename T = Type>                         \
        requires(::ernie::kIsOrArray<T> &&                           \
                 ::ernie::NumberRange<R>)                            \
    Self& Write##Name(const R& values) {                             \
        return Write##Name(                                          \
            ::ernie::detail::ArrayFromNumbers<T>(values));           \
    }

/// The body of a view class `Class` deriving from `Base`
/// (`SchemaView` for a root), with the accessors of `ATTRIBUTES`,
/// every attribute of the type and its bases.
#define ERNIE_SCHEMA_BODY(Class, Base, ATTRIBUTES)                   \
  public:                                                            \
    using Self = Class;                                              \
    /** @throws std::invalid_argument if the prim does not exist or  \
     * is not a `Class`. */                                          \
    Class(const ::ernie::Scene& scene, ::ernie::Path path)           \
        : Base(scene, std::move(path), Definition()) {}              \
    Class(::ernie::Scene& scene, ::ernie::Path path)                 \
        : Base(scene, std::move(path), Definition()) {}              \
    /** The type's registry entry. */                                \
    static const ::ernie::PrimTypeDef& Definition();                 \
    ATTRIBUTES(ERNIE_ATTRIBUTE_ACCESSORS)                            \
    /** Sets the prim's transform as one matrix op                   \
     * (`tokens::kTransform`, column vectors). */                    \
    Self& SetTransform(const glm::dmat4& matrix) {                   \
        Edit().Set(GetPath(), ::ernie::tokens::kTransform,           \
                   ::ernie::Value(matrix));                          \
        return static_cast<Self&>(*this);                            \
    }                                                                \
    /** Sets the prim's transform as translate, rotate or orient,    \
     * and scale ops (`SetXform`). */                                \
    Self& SetXform(const ::ernie::XformOps& ops) {                   \
        ::ernie::SetXform(Edit(), GetPath(), ops);                   \
        return static_cast<Self&>(*this);                            \
    }                                                                \
    /** The prim's local transform, from its ops                     \
     * (`Scene::LocalTransform`). */                                 \
    glm::dmat4 GetLocalTransform() const {                           \
        return GetScene().LocalTransform(GetPath());                 \
    }                                                                \
    /** Sets any attribute by USD name, as `Scene::Set`: what the    \
     * declarations do not cover. */                                 \
    Self& Set(std::string_view attribute, ::ernie::Value value) {    \
        Edit().Set(GetPath(), attribute, std::move(value));          \
        return static_cast<Self&>(*this);                            \
    }                                                                \
    /** Writes any array attribute by USD name, as `Scene::Write`.   \
     */                                                              \
    Self& Write(std::string_view attribute,                          \
                ::ernie::Tensor tensor) {                            \
        Edit().Write(GetPath(), attribute, std::move(tensor));       \
        return static_cast<Self&>(*this);                            \
    }                                                                \
                                                                     \
  protected:                                                         \
    Class(const ::ernie::Scene& scene, ::ernie::Path path,           \
          const ::ernie::PrimTypeDef& type)                          \
        : Base(scene, std::move(path), type) {}                      \
    Class(::ernie::Scene& scene, ::ernie::Path path,                 \
          const ::ernie::PrimTypeDef& type)                          \
        : Base(scene, std::move(path), type) {}                      \
                                                                     \
  public:

/// `Define` for a concrete view class: defines the prim
/// (`Scene::Define`) and returns its view.
#define ERNIE_SCHEMA_DEFINE(Class)                                   \
    static Class Define(::ernie::Scene& scene,                       \
                        const ::ernie::Path& path) {                 \
        scene.Define(path, Definition().name);                       \
        return Class(scene, path);                                   \
    }

/// One `AttributeDef` of an attribute list entry, for
/// `ERNIE_DEFINE_SCHEMA`.
#define ERNIE_ATTRIBUTE_DEF(Type, Name, token, fallback, meta)       \
    ::ernie::detail::MakeAttribute<Type>(token, #Name, fallback,     \
                                         meta),

/// Defines `Class::Definition()`: the type `typeName`, deriving from
/// `base` (a `PrimTypeDef*`, or nullptr), declaring the attributes of
/// `OWN_ATTRIBUTES`.
#define ERNIE_DEFINE_SCHEMA(Class, typeName, base, isAbstract,       \
                            typeDoc, OWN_ATTRIBUTES)                 \
    ERNIE_DEFINE_OPEN_SCHEMA(Class, typeName, base, isAbstract,      \
                             typeDoc, OWN_ATTRIBUTES)

/// `ERNIE_DEFINE_SCHEMA` for a type whose attributes in the
/// namespaces listed after `OWN_ATTRIBUTES` (such as "inputs") are
/// data-driven (`PrimTypeDef::openNamespaces`).
#define ERNIE_DEFINE_OPEN_SCHEMA(Class, typeName, base, isAbstract,  \
                                 typeDoc, OWN_ATTRIBUTES, ...)       \
    const ::ernie::PrimTypeDef& Class::Definition() {                \
        static const ::ernie::PrimTypeDef definition{                \
            typeName,                                                \
            base,                                                    \
            isAbstract,                                              \
            typeDoc,                                                 \
            {OWN_ATTRIBUTES(ERNIE_ATTRIBUTE_DEF)},                   \
            {__VA_ARGS__}};                                          \
        return definition;                                           \
    }

/// What the view macros expand to.
namespace ernie::detail {

template <typename T>
void WriteAttribute(Scene& scene, const Path& path,
                    std::string_view name,
                    typename AttributeTraits<T>::Arg value) {
    if constexpr (kIsOrArray<T>) {
        WriteAttribute<typename T::Value>(scene, path, name, value);
    } else if constexpr (std::is_same_v<T, Tensor>) {
        scene.Write(path, name, std::move(value));
    } else if constexpr (std::is_same_v<T, std::string>) {
        scene.Set(path, name, Value(std::string(value)));
    } else if constexpr (std::is_same_v<T, std::vector<Path>>) {
        scene.Set(path, name, value.ToValue());
    } else {
        scene.Set(path, name, Value(value));
    }
}

/// The numbers in a vector attribute of type `T`; 0 for other types.
/// Quaternions and matrices take none: four numbers may put w first
/// or last, sixteen may be rows or columns.
template <typename T> inline constexpr std::size_t kVectorSize = 0;
template <> inline constexpr std::size_t kVectorSize<glm::dvec2> = 2;
template <> inline constexpr std::size_t kVectorSize<glm::dvec3> = 3;
template <> inline constexpr std::size_t kVectorSize<glm::dvec4> = 4;

/// The type an attribute declared as `T` holds as a value.
template <typename T> struct ValueTypeOf {
    using Type = T;
};
template <typename T> struct ValueTypeOf<OrArray<T>> {
    using Type = T;
};
template <typename T>
using ValueType = typename ValueTypeOf<T>::Type;

/// Whether the setter of an attribute declared as `T` takes `R` as
/// numbers: a vector's, or a float array's, where `R` is not already
/// its own type.
template <typename T, typename R>
concept SetsFromNumbers =
    NumberRange<R> &&
    (kVectorSize<ValueType<T>> > 0 ||
     std::is_same_v<ValueType<T>, std::vector<float>>) &&
    !std::is_convertible_v<const R&, typename AttributeTraits<T>::Arg>;

/// `values` as the value of an attribute declared as `T`.
/// @throws std::invalid_argument if a vector's numbers are not as
/// many as it has.
template <typename T, NumberRange R>
ValueType<T> FromNumbers(const R& values) {
    using V = ValueType<T>;
    if constexpr (kVectorSize<V> > 0) {
        return VectorOf<static_cast<glm::length_t>(kVectorSize<V>)>(
            values);
    } else {
        const std::span numbers(values);
        return V(numbers.begin(), numbers.end());
    }
}

/// `values` as the float32 array of an `OrArray<T>` attribute: rows
/// of a vector's numbers, else one dimension.
/// @throws std::invalid_argument if the numbers do not fill whole
/// rows.
template <typename T, NumberRange R> Tensor ArrayFromNumbers(const R& values) {
    const std::span numbers(values);
    const std::vector<float> floats(numbers.begin(), numbers.end());
    const auto count = static_cast<std::int64_t>(floats.size());
    constexpr auto n =
        static_cast<std::int64_t>(kVectorSize<ValueType<T>>);
    if constexpr (n > 0) {
        if (count % n != 0) {
            throw std::invalid_argument(
                "Expected rows of " + std::to_string(n) +
                " numbers, not " + std::to_string(count) + " numbers");
        }
        return Tensor::Copy<float>(floats, {count / n, n});
    } else {
        return Tensor::Copy<float>(floats, {count});
    }
}

/// `values` as a tensor of `shape`, one dimension when empty, of the
/// range's own element type.
/// @throws std::invalid_argument if the shape's elements are not as
/// many as the values.
template <NumberRange R>
Tensor TensorFromNumbers(const R& values,
                         std::vector<std::int64_t> shape) {
    const std::span numbers(values);
    using E = std::remove_cv_t<typename decltype(numbers)::element_type>;
    if (shape.empty()) {
        shape = {static_cast<std::int64_t>(numbers.size())};
    }
    return Tensor::Copy<E>(std::span<const E>(numbers), std::move(shape));
}

/// The `AttributeDef` of an attribute declared as `T` with `fallback`
/// (anything `T` is made from; nullptr for a tensor).
template <typename T, typename Fallback>
AttributeDef
MakeAttribute(std::string_view name, std::string_view member,
              const Fallback& fallback, AttributeMeta meta) {
    if constexpr (kIsOrArray<T>) {
        meta.array = true;
        return MakeAttribute<typename T::Value>(
            name, member, fallback, std::move(meta));
    } else if constexpr (std::is_same_v<T, Tensor>) {
        return AttributeDef::MakeTensor(name, member,
                                        std::move(meta));
    } else {
        return AttributeDef::Make<T>(name, member, T(fallback),
                                     std::move(meta));
    }
}

}  // namespace ernie::detail

namespace ernie {

template <typename T>
AttributeDef
AttributeDef::Make(std::string_view name, std::string_view member,
                   const T& fallback, AttributeMeta meta) {
    AttributeDef def = Describe(
        name, member, AttributeTraits<T>::kKind, std::move(meta));
    if constexpr (!std::is_same_v<T, Path>) {
        def.fallback = Value(fallback);
    }
    def.read_ = [](const AttributeDef& self, const Scene& scene,
                   const Path& path) -> Value {
        if constexpr (std::is_same_v<T, Path>) {
            const Path target = AttributeTraits<Path>::Read(
                scene, path, self.name, Path());
            return target.IsEmpty() ? Value() : Value(target);
        } else {
            return Value(AttributeTraits<T>::Read(
                scene, path, self.name, *self.fallback.Get<T>()));
        }
    };
    def.from_ = [](const Value& value) -> std::optional<Value> {
        if constexpr (std::is_same_v<T, std::vector<Path>>) {
            return AttributeTraits<T>::Store(value);
        } else if (auto converted = AttributeTraits<T>::From(value)) {
            return Value(std::move(*converted));
        }
        return std::nullopt;
    };
    return def;
}

}  // namespace ernie
