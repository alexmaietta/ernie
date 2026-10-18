// SPDX-License-Identifier: BSD-3-Clause
/// @file view.h
/// @brief Typed views of prims, with accessors generated from their schemas' declarations.
#pragma once

#include "ernie/scene/convert.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/schema.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace ernie {

/// The column element a declared type is stored as: itself, unless listed below.
template <typename T> struct StoredAs {
    using type = T;
};
template <> struct StoredAs<bool> {
    using type = std::uint8_t;
};
template <> struct StoredAs<int> {
    using type = std::int32_t;
};
template <> struct StoredAs<std::string> {
    using type = Token;
};
template <> struct StoredAs<Path> {
    using type = Target;
};
template <> struct StoredAs<std::vector<Path>> {
    using type = std::vector<Target>;
};
template <> struct StoredAs<std::vector<std::string>> {
    using type = std::vector<Token>;
};

/// @brief How a declared type is stored in a column and read back.
template <typename T> struct AttributeTraits {
    using Stored = typename StoredAs<T>::type;
    using Result = T;

    static T Load(const PathTable& paths, const Stored& stored) {
        return *ToValue(paths, stored).template Get<T>();
    }
    static Value Save(const T& value) {
        return Value(value);
    }
    static std::vector<std::string_view> Options() {
        return {};
    }
};

template <typename E> struct AttributeTraits<Choice<E>> {
    using Stored = Token;
    using Result = E;

    static E Load(const PathTable& paths, Token token) {
        const auto& tokens = ChoiceTokens<E>::kTokens;
        const auto found = std::ranges::find(tokens, paths.Names().Text(token));
        return static_cast<E>(found != tokens.end() ? found - tokens.begin() : 0);
    }
    static Value Save(E value) {
        return std::string(ChoiceTokens<E>::kTokens[static_cast<std::size_t>(value)]);
    }
    static std::vector<std::string_view> Options() {
        return {ChoiceTokens<E>::kTokens.begin(), ChoiceTokens<E>::kTokens.end()};
    }
};

/// @brief An attribute declaration from a schema row: its column type, fallback and metadata.
template <typename T, typename Fallback>
AttributeDef DeclareAttribute(std::string_view name, const Fallback& fallback, AttributeMeta meta) {
    using Traits = AttributeTraits<T>;
    if (meta.options.empty()) {
        meta.options = Traits::Options();
    }
    return {name, kColumnType<typename Traits::Stored>, Traits::Save(fallback), std::move(meta)};
}

/// @brief A prim seen through its schema: what generated accessors read and write.
class SchemaView {
  public:
    PrimId Id() const {
        return prim_;
    }
    Path GetPath() const {
        return scene_->PrimPath(prim_);
    }
    const Scene& GetScene() const {
        return *scene_;
    }
    const SchemaDef& Schema() const {
        return *schema_;
    }

    /// The instance name of a multiple-apply API schema's view; empty otherwise.
    std::string_view Instance() const {
        return instance_;
    }

  protected:
    /// @throws std::invalid_argument if the prim is missing, is not of `schema`'s type or lacks
    /// the applied API schema.
    SchemaView(const Scene& scene, const Path& path, const SchemaDef& schema,
               std::string_view instance = {});
    SchemaView(Scene& scene, const Path& path, const SchemaDef& schema,
               std::string_view instance = {});

    /// Calls `use` with a declared attribute's name, its instance's for a multiple-apply schema.
    template <typename Use> decltype(auto) Named(std::string_view declared, const Use& use) const {
        if (instance_.empty()) {
            return use(declared);
        }
        const std::string name = ApiName{schema_, instance_}.Attribute(declared);
        return use(std::string_view(name));
    }

    template <typename T>
    typename AttributeTraits<T>::Result Read(std::string_view declared) const {
        return Named(declared, [&](std::string_view name) {
            using Traits = AttributeTraits<T>;
            const PrimStore& store = scene_->Store();
            const Token token = *store.Tokens().Find(name);
            const auto column = store.Read<typename Traits::Stored>(store.PrimBucket(prim_), token);
            return Traits::Load(store.Paths(), column[store.PrimRow(prim_)]);
        });
    }
    template <typename T>
    void Write(std::string_view declared, const typename AttributeTraits<T>::Result& value) {
        Named(declared, [&](std::string_view name) {
            Edit().Set(prim_, name, AttributeTraits<T>::Save(value));
        });
    }
    bool Has(std::string_view declared) const {
        return Named(declared,
                     [&](std::string_view name) { return scene_->IsAuthored(prim_, name); });
    }
    void Unset(std::string_view declared) {
        Named(declared, [&](std::string_view name) { Edit().Clear(prim_, name); });
    }
    Scene& Edit() const;

  private:
    const Scene* scene_ = nullptr;
    Scene* mutable_ = nullptr;
    PrimId prim_;
    const SchemaDef* schema_ = nullptr;
    std::string instance_;
};

/// The root of every generated accessor chain.
template <typename Self, typename Base> class SchemaViewAccessors : public Base {
  public:
    using Base::Base;
};

}  // namespace ernie

/// For schemas that declare no attributes of their own.
#define ERNIE_NO_ATTRIBUTES(X)

#define ERNIE_ATTRIBUTE_ACCESSORS(Type, Name, usdName, fallback, ...)                              \
    static constexpr std::string_view k##Name = usdName;                                           \
    typename ::ernie::AttributeTraits<Type>::Result Get##Name() const {                            \
        return this->template Read<Type>(k##Name);                                                 \
    }                                                                                              \
    bool Has##Name() const {                                                                       \
        return this->Has(k##Name);                                                                 \
    }                                                                                              \
    Self& Set##Name(const typename ::ernie::AttributeTraits<Type>::Result& value) {                \
        this->template Write<Type>(k##Name, value);                                                \
        return static_cast<Self&>(*this);                                                          \
    }                                                                                              \
    Self& Clear##Name() {                                                                          \
        this->Unset(k##Name);                                                                      \
        return static_cast<Self&>(*this);                                                          \
    }

#define ERNIE_ATTRIBUTE_DECLARE(Type, Name, usdName, fallback, text, ...)                          \
    ::ernie::DeclareAttribute<Type>(usdName, fallback,                                             \
                                    ::ernie::AttributeMeta{.doc = text, __VA_ARGS__}),

/// Declares `Class##Accessors`, the getters and setters of a schema's own attributes over its
/// base's, each returning the most derived view.
#define ERNIE_SCHEMA_ACCESSORS(Class, BaseClass, OWN_ATTRIBUTES)                                   \
    template <typename Self, typename Base>                                                        \
    class Class##Accessors : public BaseClass##Accessors<Self, Base> {                             \
      public:                                                                                      \
        using BaseClass##Accessors<Self, Base>::BaseClass##Accessors;                              \
        OWN_ATTRIBUTES(ERNIE_ATTRIBUTE_ACCESSORS)                                                  \
    }

/// The constructors and registry entry of a schema view class.
#define ERNIE_SCHEMA_BODY(Class, Base)                                                             \
  public:                                                                                          \
    using Self = Class;                                                                            \
    /** @throws std::invalid_argument if the prim does not exist or is not a `Class`. */           \
    Class(const ::ernie::Scene& scene, const ::ernie::Path& path)                                  \
        : Class##Accessors<Class, Base>(scene, path, Definition()) {}                              \
    Class(::ernie::Scene& scene, const ::ernie::Path& path)                                        \
        : Class##Accessors<Class, Base>(scene, path, Definition()) {}                              \
    static const ::ernie::SchemaDef& Definition();                                                 \
                                                                                                   \
  protected:                                                                                       \
    Class(const ::ernie::Scene& scene, const ::ernie::Path& path,                                  \
          const ::ernie::SchemaDef& schema)                                                        \
        : Class##Accessors<Class, Base>(scene, path, schema) {}                                    \
    Class(::ernie::Scene& scene, const ::ernie::Path& path, const ::ernie::SchemaDef& schema)      \
        : Class##Accessors<Class, Base>(scene, path, schema) {}                                    \
                                                                                                   \
  public:

/// `Class::Define(scene, path)` for a concrete schema.
#define ERNIE_SCHEMA_DEFINE(Class)                                                                 \
    static Class Define(::ernie::Scene& scene, const ::ernie::Path& path) {                        \
        scene.Define(path, Definition().name);                                                     \
        return Class(scene, path);                                                                 \
    }

/// The constructors, `Apply` and registry entry of an API schema's view; a multiple-apply
/// schema's take its instance name.
#define ERNIE_API_SCHEMA_BODY(Class)                                                               \
  public:                                                                                          \
    using Self = Class;                                                                            \
    /** @throws std::invalid_argument if the prim is missing or lacks the API schema. */           \
    Class(const ::ernie::Scene& scene, const ::ernie::Path& path, std::string_view instance = {})  \
        : Class##Accessors<Class, ::ernie::SchemaView>(scene, path, Definition(), instance) {}     \
    Class(::ernie::Scene& scene, const ::ernie::Path& path, std::string_view instance = {})        \
        : Class##Accessors<Class, ::ernie::SchemaView>(scene, path, Definition(), instance) {}     \
    static const ::ernie::SchemaDef& Definition();                                                 \
    static Class Apply(::ernie::Scene& scene, const ::ernie::Path& path,                           \
                       std::string_view instance = {}) {                                           \
        scene.ApplyApi(scene.Require(path), ::ernie::ApiName{&Definition(), instance}.ToString()); \
        return Class(scene, path, instance);                                                       \
    }

/// Defines `Class::Definition()`, the schema's registry entry.
#define ERNIE_DEFINE_SCHEMA(Class, name, base, kind, extentRule, text, OWN_ATTRIBUTES, ...)        \
    const ::ernie::SchemaDef& Class::Definition() {                                                \
        static const ::ernie::SchemaDef definition{name,                                           \
                                                   base,                                           \
                                                   ::ernie::SchemaKind::kind,                      \
                                                   text,                                           \
                                                   {OWN_ATTRIBUTES(ERNIE_ATTRIBUTE_DECLARE)},      \
                                                   {__VA_ARGS__},                                  \
                                                   extentRule};                                    \
        return definition;                                                                         \
    }
