// SPDX-License-Identifier: BSD-3-Clause
/// @file scene.h
/// @brief The scene: typed prims at paths, stored in columns by type.
#pragma once

#include "ernie/geometry/bounds.h"
#include "ernie/scene/extent_cache.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/storage.h"
#include "ernie/scene/value.h"

#include <glm/glm.hpp>

#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ernie {

/// @brief Prims named by paths, each of a registered type, its attributes validated against the
/// type's schema and stored in the columns of the type's bucket.
class Scene {
  public:
    Scene();

    /// @brief Defines a prim of a concrete type, or `""` for none; missing ancestors are defined
    /// without a type. Redefining with another type keeps the authored values both types declare.
    /// @throws std::invalid_argument for the root, an empty path or an unknown or abstract type.
    PrimId Define(const Path& path, std::string_view type);
    /// @brief Removes a prim and its descendants.
    /// @return Whether there was a prim.
    bool Remove(const Path& path);

    std::optional<PrimId> Find(const Path& path) const;
    /// @throws std::invalid_argument if there is no prim at `path`.
    PrimId Require(const Path& path) const;
    bool Has(const Path& path) const {
        return Find(path).has_value();
    }
    Path PrimPath(PrimId prim) const;
    /// @return The prim's type name; empty for none.
    std::string_view TypeName(PrimId prim) const;
    /// @return The prim's schema, or null for a prim without a type.
    const SchemaDef* Schema(PrimId prim) const;
    std::vector<Path> Children(const Path& path) const;
    /// @return Every prim whose type is `type` or derives from it; every prim for `""`.
    std::vector<Path> Prims(std::string_view type = {}) const;

    /// @brief Applies an API schema, `"MaterialBindingAPI"`, or an instance of a multiple-apply
    /// one, `"CollectionAPI:lights"`, adding its attributes at their fallbacks.
    /// @throws std::invalid_argument for an unknown API schema or a missing or extra instance.
    void ApplyApi(PrimId prim, std::string_view api);
    /// @brief Removes an applied API schema with the values of its attributes.
    /// @return Whether it was applied.
    bool RemoveApi(PrimId prim, std::string_view api);
    bool HasApi(PrimId prim, std::string_view api) const;
    /// @return The applied API schemas, as `ApplyApi` names them.
    std::vector<std::string> AppliedApis(PrimId prim) const;

    /// @brief Authors an attribute: a declared one converted to its type and checked, or an
    /// undeclared one, typed by its first value, in a namespace the type leaves open or on a prim
    /// without a type.
    /// @throws std::invalid_argument for a missing prim, an undeclared attribute or a bad value.
    void Set(PrimId prim, std::string_view attribute, const Value& value);
    void Set(const Path& path, std::string_view attribute, const Value& value);
    /// @return The authored value, or none.
    std::optional<Value> Get(PrimId prim, std::string_view attribute) const;
    std::optional<Value> Get(const Path& path, std::string_view attribute) const;
    /// @brief Authors an array attribute with a tensor on the host or a device, shared, not copied.
    /// @throws std::invalid_argument for a non-array attribute, or a tensor of the wrong type or
    /// shape.
    void Write(PrimId prim, std::string_view attribute, Tensor array);
    /// @return The authored array, or null.
    const Tensor* Read(PrimId prim, std::string_view attribute) const;
    /// @return Whether the attribute was authored.
    bool Clear(PrimId prim, std::string_view attribute);
    bool IsAuthored(PrimId prim, std::string_view attribute) const;
    /// @return The authored attributes' names, in column order.
    std::vector<std::string> Attributes(const Path& path) const;

    /// @return The prim's own transform from its ops; identity for prims that aren't Xformable.
    glm::dmat4 LocalTransform(PrimId prim) const;
    glm::dmat4 WorldTransform(PrimId prim) const;
    /// @return The prim's extent in its own space; none when it isn't computed from attributes.
    /// Cached since `UpdateExtents` when its bucket hasn't changed, computed otherwise.
    std::optional<Box> Extent(PrimId prim) const;
    /// @brief Recomputes the cached extents of the rows that changed since the last update.
    void UpdateExtents() {
        extents_.Update(store_);
    }
    /// @return The world box of the prim and its visible descendants of the default purpose.
    /// Prims with extents bound their own subtrees, as UsdGeomBBoxCache prunes at Boundables.
    Box WorldBounds(const Path& path) const;
    /// @return The box of a prim's subtree in its parent's space, whatever its visibility: what a
    /// point instancer places at each instance.
    Box BoundsInParent(PrimId prim) const;

    PrimStore& Store() {
        return store_;
    }
    const PrimStore& Store() const {
        return store_;
    }
    std::uint64_t Version() const {
        return store_.Version();
    }
    /// @return What a column change means to renderers: the attribute's declared category, or for
    /// undeclared ones Transform for xformOps, Primvars for primvars and Params for the rest.
    Dirty Classify(const ChangeSet::Values& change) const;

  private:
    struct BucketSchema {
        const SchemaDef* schema = nullptr;
        std::unordered_map<AttributeId, const AttributeDef*> declared;
    };

    /// @return The bucket of a type and applied API schemas, made if new.
    BucketId FindOrAddBucket(const SchemaDef* schema, std::vector<Token> apis);
    /// Moves a prim to the bucket of its type and `apis`.
    void MoveTo(PrimId prim, const SchemaDef* schema, std::vector<Token> apis);
    const std::vector<Token>& Apis(PrimId prim) const;
    /// @return A type's built-in API schemas, then the applied ones.
    std::vector<ApiName> AllApis(const SchemaDef* schema, std::span<const Token> applied) const;
    /// @return The attribute's column when the prim authors it, or null.
    const Column* AuthoredColumn(PrimId prim, std::string_view attribute) const;
    const AttributeDef* Declared(BucketId bucket, AttributeId attribute) const;
    /// @return Whether the prim's type or an applied API schema leaves the attribute's namespace
    /// open; any attribute for a prim without a type.
    bool TakesUndeclared(PrimId prim, std::string_view attribute) const;
    ColumnType TypeForUndeclared(PrimId prim, std::string_view attribute, const Value& value) const;
    /// @return Whether the prim and its subtree are invisible or of a purpose bounds leave out.
    bool LeavesOutOfBounds(PrimId prim) const;
    /// Adds a subtree's extents under `parent`'s transform; with `visibleOnly`, only its visible
    /// default-purpose prims'.
    void AddTo(Box& bounds, PathId path, const glm::dmat4& parent, bool visibleOnly) const;

    PrimStore store_;
    Token invisible_ = 0;
    Token defaultPurpose_ = 0;
    std::vector<BucketSchema> buckets_;
    ExtentCache extents_;
};

}  // namespace ernie
