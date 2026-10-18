// SPDX-License-Identifier: BSD-3-Clause
/// @file scene.cpp
/// @brief Prims, attributes, transforms and bounds over the columnar store.
#include "ernie/scene/scene.h"

#include "ernie/scene/convert.h"
#include "ernie/scene/schemas/geom.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "ernie/util/algorithm.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ernie {
namespace {

/// @return A column layout for the attribute `name` holding `fallback` in unauthored rows.
ColumnLayout MakeLayout(PathTable& paths, const AttributeDef& def, std::string_view name,
                        const Value& fallback) {
    const AttributeId attribute = paths.Names().Intern(name);
    return WithColumnType(def.type, [&]<typename T>(std::type_identity<T>) {
        if constexpr (std::is_same_v<T, Tensor>) {
            return ColumnLayout::Of<T>(attribute);
        } else {
            return ColumnLayout::Of<T>(attribute, ToElement<T>(paths, fallback).value_or(T{}));
        }
    });
}

std::string Describe(const Scene& scene, PrimId prim) {
    const std::string_view type = scene.TypeName(prim);
    return "'" + scene.PrimPath(prim).String() + "'" +
           (type.empty() ? std::string() : " (" + std::string(type) + ")");
}

}  // namespace

Scene::Scene()
    : inherited_(store_.Tokens().Intern(tokens::kInherited)),
      invisible_(store_.Tokens().Intern(tokens::kInvisible)),
      defaultPurpose_(store_.Tokens().Intern(tokens::kDefault)) {
    FindOrAddBucket(nullptr, {});
}

PrimId Scene::Define(const Path& path, std::string_view type) {
    if (path.IsEmpty() || path.IsRoot()) {
        throw std::invalid_argument("Cannot define the prim at '" + path.String() + "'");
    }
    const SchemaDef* schema = type.empty() ? nullptr : FindSchema(type);
    if (!type.empty() && (schema == nullptr || schema->kind != SchemaKind::Concrete)) {
        throw std::invalid_argument("Cannot define '" + path.String() + "' as '" +
                                    std::string(type) + "', which is no prim type");
    }
    PathTable& paths = store_.Paths();
    const PathId id = paths.Intern(path);
    for (PathId above = paths.Parent(id); above != kRootPath; above = paths.Parent(above)) {
        if (!store_.Find(above)) {
            store_.Define(above, 0);
        }
    }
    if (const std::optional<PrimId> existing = store_.Find(id)) {
        MoveTo(*existing, schema, Apis(*existing));
        return *existing;
    }
    return store_.Define(id, FindOrAddBucket(schema, {}));
}

bool Scene::Remove(const Path& path) {
    if (path.IsRoot()) {
        throw std::invalid_argument("Cannot remove the root prim");
    }
    const std::optional<PathId> id = store_.Paths().Find(path);
    if (!id || !store_.Find(*id)) {
        return false;
    }
    std::vector<PathId> pending{*id};
    while (!pending.empty()) {
        const PathId next = pending.back();
        pending.pop_back();
        if (const std::optional<PrimId> prim = store_.Find(next)) {
            animated_.erase(prim->slot);
            store_.Remove(*prim);
        }
        store_.Paths().ForEachChild(next, [&](PathId child) { pending.push_back(child); });
    }
    return true;
}

std::optional<PrimId> Scene::Find(const Path& path) const {
    const std::optional<PathId> id = store_.Paths().Find(path);
    return id ? store_.Find(*id) : std::nullopt;
}

Path Scene::PrimPath(PrimId prim) const {
    return store_.Paths().ToPath(store_.PrimPath(prim));
}

std::string_view Scene::TypeName(PrimId prim) const {
    const SchemaDef* schema = Schema(prim);
    return schema != nullptr ? schema->name : std::string_view();
}

const SchemaDef* Scene::Schema(PrimId prim) const {
    return buckets_[store_.PrimBucket(prim)].schema;
}

std::vector<Path> Scene::Children(const Path& path) const {
    std::vector<Path> children;
    const PathTable& paths = store_.Paths();
    if (const std::optional<PathId> id = path.IsRoot() ? kRootPath : paths.Find(path)) {
        paths.ForEachChild(*id, [&](PathId child) {
            if (store_.Find(child)) {
                children.push_back(paths.ToPath(child));
            }
        });
    }
    return children;
}

std::vector<Path> Scene::Prims(std::string_view type) const {
    const SchemaDef* base = type.empty() ? nullptr : FindSchema(type);
    if (!type.empty() && base == nullptr) {
        throw std::invalid_argument("'" + std::string(type) + "' is no prim type");
    }
    std::vector<Path> prims;
    for (BucketId bucket = 0; bucket < buckets_.size(); ++bucket) {
        const SchemaDef* schema = buckets_[bucket].schema;
        if (base == nullptr || (schema != nullptr && schema->IsA(*base))) {
            for (const PrimId prim : store_.GetBucket(bucket).prims) {
                prims.push_back(PrimPath(prim));
            }
        }
    }
    std::ranges::sort(prims);
    return prims;
}

void Scene::Set(PrimId prim, std::string_view attribute, const Value& value) {
    ClearSamples(prim, attribute);
    Author(prim, attribute, value);
}

void Scene::Set(const Path& path, std::string_view attribute, const Value& value) {
    Set(Require(path), attribute, value);
}

void Scene::Author(PrimId prim, std::string_view attribute, const Value& value, Value* checked) {
    const AttributeId token = store_.Tokens().Intern(attribute);
    const AttributeDef* def = Declared(store_.PrimBucket(prim), token);
    if (def != nullptr) {
        Validate(*def, attribute, value, PrimPath(prim));
    }
    const ColumnType type = def != nullptr ? def->type : TypeForUndeclared(prim, attribute, value);
    WithColumnType(type, [&]<typename T>(std::type_identity<T>) {
        std::optional<T> element = ToElement<T>(store_.Paths(), value);
        if (!element) {
            throw std::invalid_argument(Describe(*this, prim) + ": '" + std::string(attribute) +
                                        "' takes another type of value");
        }
        if (checked != nullptr) {
            *checked = ToValue(store_.Paths(), *element);
        } else {
            store_.Set(prim, token, std::move(*element));
        }
    });
}

void Scene::Write(PrimId prim, std::string_view attribute, Tensor array) {
    ClearSamples(prim, attribute);
    CheckArray(prim, attribute, array);
    store_.Set(prim, store_.Tokens().Intern(attribute), std::move(array));
}

void Scene::CheckArray(PrimId prim, std::string_view attribute, const Tensor& array) const {
    const std::optional<Token> token = store_.Tokens().Find(attribute);
    const AttributeDef* def = token ? Declared(store_.PrimBucket(prim), *token) : nullptr;
    if (def == nullptr && !TakesUndeclared(prim, attribute)) {
        throw std::invalid_argument(Describe(*this, prim) + " has no attribute '" +
                                    std::string(attribute) + "'");
    }
    if (def != nullptr) {
        ValidateArray(*def, attribute, array, PrimPath(prim));
    }
}

const Tensor* Scene::Read(PrimId prim, std::string_view attribute) const {
    return Authored<Tensor>(prim, attribute);
}

std::optional<Value> Scene::Get(PrimId prim, std::string_view attribute) const {
    return CellValue(prim, AuthoredColumn(prim, attribute));
}

std::optional<Value> Scene::GetResolved(PrimId prim, std::string_view attribute) const {
    const std::optional<Token> token = store_.Tokens().Find(ResolvedName(attribute));
    return CellValue(prim,
                     token ? store_.GetBucket(store_.PrimBucket(prim)).Find(*token) : nullptr);
}

std::optional<Value> Scene::CellValue(PrimId prim, const Column* column) const {
    if (column == nullptr) {
        return std::nullopt;
    }
    return WithColumnType(column->values->Type(), [&]<typename T>(std::type_identity<T>) {
        const auto values = store_.Read<T>(store_.PrimBucket(prim), column->attribute);
        return ToValue(store_.Paths(), values[store_.PrimRow(prim)]);
    });
}

std::optional<Value> Scene::Get(const Path& path, std::string_view attribute) const {
    return Get(Require(path), attribute);
}

bool Scene::Clear(PrimId prim, std::string_view attribute) {
    ClearSamples(prim, attribute);
    const Column* column = AuthoredColumn(prim, attribute);
    if (column != nullptr) {
        store_.Clear(prim, column->attribute);
    }
    return column != nullptr;
}

bool Scene::IsAuthored(PrimId prim, std::string_view attribute) const {
    return AuthoredColumn(prim, attribute) != nullptr;
}

std::vector<std::string> Scene::Attributes(const Path& path) const {
    const PrimId prim = Require(path);
    std::vector<std::string> names;
    for (const Column& column : store_.GetBucket(store_.PrimBucket(prim)).columns) {
        if (column.authored.Test(store_.PrimRow(prim))) {
            names.emplace_back(store_.Tokens().Text(column.attribute));
        }
    }
    return names;
}

glm::dmat4 Scene::LocalTransform(PrimId prim) const {
    const SchemaDef* schema = Schema(prim);
    const bool xformable = schema != nullptr && schema->IsA(Xformable::Definition());
    return xformable ? ComposeXformOps(*this, prim) : glm::dmat4(1.0);
}

glm::dmat4 Scene::WorldTransform(PrimId prim) const {
    glm::dmat4 world = LocalTransform(prim);
    const PathTable& paths = store_.Paths();
    for (PathId above = paths.Parent(store_.PrimPath(prim));
         above != kRootPath && !ResetsXformStack(*this, prim); above = paths.Parent(above)) {
        prim = *store_.Find(above);
        world = LocalTransform(prim) * world;
    }
    return world;
}

std::optional<Box> Scene::Extent(PrimId prim) const {
    const SchemaDef* schema = Schema(prim);
    const ExtentRule rule = schema != nullptr ? schema->Extent() : kNoExtent;
    if (rule.kind != ExtentRule::Kind::Computed) {
        return std::nullopt;
    }
    const BucketId bucket = store_.PrimBucket(prim);
    const std::uint32_t row = store_.PrimRow(prim);
    if (const std::optional<Box> cached = extents_.Find(store_, bucket, row)) {
        return cached;
    }
    Box extent;
    rule.compute({store_, bucket, this}, {row, row + 1}, std::span(&extent, 1));
    return extent;
}

std::vector<PrimId> Scene::Holders(const Path& target) const {
    const std::optional<PathId> id = store_.Paths().Find(target);
    return id ? holders_.Find(store_, *id) : std::vector<PrimId>{};
}

Box Scene::WorldBounds(const Path& path) const {
    const PrimId prim = Require(path);
    const PathId id = store_.PrimPath(prim);
    const PathId parent = store_.Paths().Parent(id);
    const glm::dmat4 above =
        parent == kRootPath ? glm::dmat4(1.0) : WorldTransform(*store_.Find(parent));
    Box bounds;
    AddTo(bounds, id, above, true);
    return bounds;
}

Box Scene::BoundsInParent(PrimId prim) const {
    Box bounds;
    AddTo(bounds, store_.PrimPath(prim), glm::dmat4(1.0), false);
    return bounds;
}

void Scene::AddTo(Box& bounds, PathId path, const glm::dmat4& parent, bool visibleOnly) const {
    const PrimId prim = *store_.Find(path);
    if (visibleOnly && LeavesOutOfBounds(prim)) {
        return;
    }
    const glm::dmat4 local = LocalTransform(prim);
    const glm::dmat4 world = ResetsXformStack(*this, prim) ? local : parent * local;
    // A prim with an extent bounds what it is made of, so its children aren't added again.
    if (const std::optional<Box> extent = Extent(prim)) {
        bounds.Extend(extent->Transformed(world));
        return;
    }
    bool loaded = false;
    store_.Paths().ForEachChild(path, [&](PathId child) {
        if (store_.Find(child)) {
            loaded = true;
            AddTo(bounds, child, world, visibleOnly);
        }
    });
    // A model whose subtree isn't loaded is bounded by its hint.
    if (!loaded) {
        bounds.Extend(ExtentsHint(prim, !visibleOnly).Transformed(world));
    }
}

PrimId Scene::Require(const Path& path) const {
    const std::optional<PrimId> prim = Find(path);
    if (!prim) {
        throw std::invalid_argument("No prim at '" + path.String() + "'");
    }
    return *prim;
}

BucketId Scene::FindOrAddBucket(const SchemaDef* schema, std::vector<Token> apis) {
    PathTable& paths = store_.Paths();
    const BucketKey key{paths.Names().Intern(schema != nullptr ? schema->name : ""),
                        std::move(apis)};
    if (const std::optional<BucketId> found = store_.FindBucket(key)) {
        return *found;
    }
    BucketSchema entry{schema, {}};
    const std::vector<ApiName> allApis = AllApis(schema, key.apiSchemas);
    // The type's fallback overrides win over its API schemas', as USD's do.
    const auto fallback = [&](const AttributeDef& def, const std::string& name) -> const Value& {
        const Value* override = schema != nullptr ? schema->FallbackOverride(name) : nullptr;
        for (auto api = allApis.begin(); override == nullptr && api != allApis.end(); ++api) {
            override = api->schema->FallbackOverride(name);
        }
        return override != nullptr ? *override : def.fallback;
    };
    std::vector<ColumnLayout> layout;
    const auto add = [&](const SchemaDef& declaring, const ApiName& api) {
        for (const AttributeDef* def : declaring.AllAttributes()) {
            const std::string name = api.Attribute(def->name);
            layout.push_back(MakeLayout(paths, *def, name, fallback(*def, name)));
            entry.declared.emplace(layout.back().attribute, def);
        }
    };
    if (schema != nullptr) {
        add(*schema, {});
    }
    for (const ApiName& api : allApis) {
        add(*api.schema, api);
    }
    if (schema != nullptr && schema->IsA(Imageable::Definition())) {
        for (const AttributeDef* def : InheritedAttributes()) {
            layout.push_back(MakeLayout(paths, *def, ResolvedName(def->name), def->fallback));
        }
    }
    const BucketId bucket = store_.AddBucket(key, layout);
    buckets_.resize(std::max<std::size_t>(buckets_.size(), bucket + 1));
    buckets_[bucket] = std::move(entry);
    const ExtentRule rule = schema != nullptr ? schema->Extent() : kNoExtent;
    const bool cached = rule.kind == ExtentRule::Kind::Computed && !rule.readsOtherPrims;
    extents_.Track(bucket, cached ? rule.compute : nullptr);
    return bucket;
}

void Scene::MoveTo(PrimId prim, const SchemaDef* schema, std::vector<Token> apis) {
    store_.Move(prim, FindOrAddBucket(schema, std::move(apis)));
}

const std::vector<Token>& Scene::Apis(PrimId prim) const {
    return store_.GetBucket(store_.PrimBucket(prim)).key.apiSchemas;
}

void Scene::ApplyApi(PrimId prim, std::string_view api) {
    const ApiName parsed = ApiName::Parse(api);
    if (HasApi(prim, api)) {
        return;
    }
    const std::vector<std::string_view>& allowed = parsed.schema->appliesTo;
    const SchemaDef* type = Schema(prim);
    if (!allowed.empty() && std::ranges::none_of(allowed, [&](std::string_view name) {
            return type != nullptr && type->IsA(*FindSchema(name));
        })) {
        throw std::invalid_argument(std::string(api) + " applies only to " +
                                    std::string(allowed.front()) + " prims, not " +
                                    Describe(*this, prim));
    }
    std::vector<Token> apis = Apis(prim);
    apis.push_back(store_.Tokens().Intern(api));
    MoveTo(prim, Schema(prim), std::move(apis));
}

bool Scene::RemoveApi(PrimId prim, std::string_view api) {
    if (!HasApi(prim, api)) {
        return false;
    }
    std::vector<Token> apis = Apis(prim);
    std::erase(apis, *store_.Tokens().Find(api));
    MoveTo(prim, Schema(prim), std::move(apis));
    return true;
}

bool Scene::HasApi(PrimId prim, std::string_view api) const {
    return std::ranges::any_of(AllApis(Schema(prim), Apis(prim)),
                               [&](const ApiName& name) { return name.ToString() == api; });
}

std::vector<ApiName> Scene::AllApis(const SchemaDef* schema, std::span<const Token> applied) const {
    std::vector<ApiName> apis;
    // Each API schema once, followed by those built into it (MeshLightAPI's LightAPI, LightAPI's
    // light-link collections).
    const auto add = [&](const auto& self, std::string_view name) -> void {
        const ApiName api = ApiName::Parse(name);
        if (std::ranges::find(apis, api) != apis.end()) {
            return;
        }
        apis.push_back(api);
        for (const std::string_view builtin : api.schema->AllBuiltinApis()) {
            self(self, builtin);
        }
    };
    for (const std::string_view builtin :
         schema != nullptr ? schema->AllBuiltinApis() : std::vector<std::string_view>{}) {
        add(add, builtin);
    }
    for (const Token token : applied) {
        add(add, store_.Tokens().Text(token));
    }
    return apis;
}

std::vector<std::string> Scene::AppliedApis(PrimId prim) const {
    return Map(Apis(prim), [&](Token token) { return std::string(store_.Tokens().Text(token)); });
}

bool Scene::LeavesOutOfBounds(PrimId prim) const {
    const Token* visibility = Authored<Token>(prim, Imageable::kVisibility);
    // Purpose is inherited, so a subtree under another purpose is left out whole.
    const Token* purpose = Authored<Token>(prim, Imageable::kPurpose);
    return (visibility && *visibility == invisible_) || (purpose && *purpose != defaultPurpose_);
}

Box Scene::ExtentsHint(PrimId prim, bool allPurposes) const {
    const auto* hint = Authored<std::vector<float>>(prim, GeomModelAPI::kExtentsHint);
    if (hint == nullptr) {
        return {};
    }
    // Six floats per purpose, the default purpose first.
    const std::size_t floats = allPurposes ? hint->size() : std::min<std::size_t>(6, hint->size());
    Box box;
    for (std::size_t i = 0; i + 6 <= floats; i += 6) {
        box.Extend(AuthoredExtent(std::span(*hint).subspan(i, 6)));
    }
    return box;
}

const Column* Scene::AuthoredColumn(PrimId prim, std::string_view attribute) const {
    const std::optional<Token> token = store_.Tokens().Find(attribute);
    const Column* column = token ? store_.GetBucket(store_.PrimBucket(prim)).Find(*token) : nullptr;
    return column != nullptr && column->authored.Test(store_.PrimRow(prim)) ? column : nullptr;
}

const AttributeDef* Scene::Declaration(PrimId prim, std::string_view attribute) const {
    const std::optional<Token> token = store_.Tokens().Find(attribute);
    return token ? Declared(store_.PrimBucket(prim), *token) : nullptr;
}

const AttributeDef* Scene::Declared(BucketId bucket, AttributeId attribute) const {
    const auto& declared = buckets_[bucket].declared;
    const auto found = declared.find(attribute);
    return found != declared.end() ? found->second : nullptr;
}

Dirty Scene::Classify(const ChangeSet::Values& change) const {
    if (const AttributeDef* def = Declared(change.bucket, change.attribute)) {
        return def->meta.dirty;
    }
    const std::string_view name = store_.Tokens().Text(change.attribute);
    if (name.starts_with(tokens::kResolvedPrefix)) {
        return FindInherited(name.substr(tokens::kResolvedPrefix.size()))->meta.dirty;
    }
    return name.starts_with(tokens::kXformOpPrefix)    ? Dirty::Transform
           : name.starts_with(tokens::kPrimvarsPrefix) ? Dirty::Primvars
                                                       : Dirty::Params;
}

bool Scene::TakesUndeclared(PrimId prim, std::string_view attribute) const {
    if (attribute.starts_with(tokens::kResolvedPrefix)) {
        return false;
    }
    const SchemaDef* schema = Schema(prim);
    return schema == nullptr || schema->IsOpen(attribute) ||
           std::ranges::any_of(AllApis(schema, Apis(prim)),
                               [&](const ApiName& api) { return api.schema->IsOpen(attribute); });
}

ColumnType Scene::TypeForUndeclared(PrimId prim, std::string_view attribute,
                                    const Value& value) const {
    if (!TakesUndeclared(prim, attribute)) {
        throw std::invalid_argument(Describe(*this, prim) + " has no attribute '" +
                                    std::string(attribute) + "'");
    }
    const Token token = *store_.Tokens().Find(attribute);
    if (const Column* column = store_.GetBucket(store_.PrimBucket(prim)).Find(token)) {
        return column->values->Type();
    }
    const std::optional<ColumnType> type = InferColumnType(value);
    if (!type) {
        throw std::invalid_argument(Describe(*this, prim) + ": '" + std::string(attribute) +
                                    "' takes no empty value");
    }
    return *type;
}

std::optional<PrimId> ExtentRows::Resolve(const Target& target) const {
    return target.path != kNoPath ? store.Find(target.path) : std::nullopt;
}

std::optional<Box> ExtentRows::ExtentInRowSpace(PrimId other, std::uint32_t row) const {
    const std::optional<Box> extent = scene->Extent(other);
    if (!extent) {
        return std::nullopt;
    }
    const PrimId own = store.GetBucket(bucket).prims[row];
    return extent->Transformed(glm::inverse(scene->WorldTransform(own)) *
                               scene->WorldTransform(other));
}

}  // namespace ernie
