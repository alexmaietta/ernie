// SPDX-License-Identifier: BSD-3-Clause
/// @file scene.h
/// @brief The scene: typed prims at paths, stored in columns by type.
#pragma once

#include "ernie/geometry/bounds.h"
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

    /// @brief Authors an attribute: a declared one converted to its type and checked, or an
    /// undeclared one, typed by its first value, in a namespace the type leaves open or on a prim
    /// without a type.
    /// @throws std::invalid_argument for a missing prim, an undeclared attribute or a bad value.
    void Set(PrimId prim, std::string_view attribute, const Value& value);
    void Set(const Path& path, std::string_view attribute, const Value& value);
    /// @return The authored value, or none.
    std::optional<Value> Get(PrimId prim, std::string_view attribute) const;
    std::optional<Value> Get(const Path& path, std::string_view attribute) const;
    /// @return Whether the attribute was authored.
    bool Clear(PrimId prim, std::string_view attribute);
    bool IsAuthored(PrimId prim, std::string_view attribute) const;
    /// @return The authored attributes' names, in column order.
    std::vector<std::string> Attributes(const Path& path) const;

    /// @return The prim's own transform from its ops; identity for prims that aren't Xformable.
    glm::dmat4 LocalTransform(PrimId prim) const;
    glm::dmat4 WorldTransform(PrimId prim) const;
    /// @return The prim's extent in its own space; none when it isn't computed from attributes.
    std::optional<Box> Extent(PrimId prim) const;
    /// @return The world box of the prim and its visible descendants of the default purpose.
    Box WorldBounds(const Path& path) const;

    PrimStore& Store() {
        return store_;
    }
    const PrimStore& Store() const {
        return store_;
    }
    std::uint64_t Version() const {
        return store_.Version();
    }

  private:
    struct BucketSchema {
        const SchemaDef* schema = nullptr;
        std::unordered_map<AttributeId, const AttributeDef*> declared;
    };

    BucketId BucketFor(const SchemaDef* schema);
    /// @return The attribute's column when the prim authors it, or null.
    const Column* AuthoredColumn(PrimId prim, std::string_view attribute) const;
    const AttributeDef* Declared(PrimId prim, AttributeId attribute) const;
    ColumnType TypeForUndeclared(PrimId prim, std::string_view attribute, const Value& value) const;
    /// @return Whether the prim and its subtree are invisible or of a purpose bounds leave out.
    bool LeavesOutOfBounds(PrimId prim) const;
    /// Adds the visible default-purpose extents of a subtree under `parent`'s world transform.
    void AddTo(Box& bounds, PathId path, const glm::dmat4& parent) const;

    PrimStore store_;
    Token invisible_ = 0;
    Token defaultPurpose_ = 0;
    std::vector<BucketSchema> buckets_;
    std::unordered_map<const SchemaDef*, BucketId> bucketBySchema_;
};

}  // namespace ernie
