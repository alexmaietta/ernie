// SPDX-License-Identifier: BSD-3-Clause
/// @file scene.h
/// @brief The scene: typed prims at paths, stored in columns by type.
#pragma once

#include "ernie/geometry/bounds.h"
#include "ernie/scene/extent_cache.h"
#include "ernie/scene/holder_index.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/storage.h"
#include "ernie/scene/time_samples.h"
#include "ernie/scene/value.h"

#include <glm/glm.hpp>

#include <map>
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
    /// one, `"CollectionAPI:lights"`, adding its attributes at their fallbacks; one the prim has
    /// already, built in or applied, is left as it is.
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
    /// @brief `Get` by the attribute's token, for readers that look it up once.
    std::optional<Value> Get(PrimId prim, AttributeId attribute) const;
    /// @return The authored value, else the fallback its schema declares; none when neither.
    std::optional<Value> GetOrFallback(PrimId prim, std::string_view attribute) const;
    std::optional<Value> Get(const Path& path, std::string_view attribute) const;
    /// @return An inherited attribute's value as of the last `Update`: the prim's own or its
    /// nearest authoring ancestor's (where "inherited" isn't authoring), else the fallback; none
    /// for a prim that isn't Imageable.
    std::optional<Value> GetResolved(PrimId prim, std::string_view attribute) const;
    /// @brief Authors an array attribute with a tensor on the host or a device, shared, not copied.
    /// @throws std::invalid_argument for a non-array attribute, or a tensor of the wrong type or
    /// shape.
    void Write(PrimId prim, std::string_view attribute, Tensor array);
    /// @return The authored array, or null.
    const Tensor* Read(PrimId prim, std::string_view attribute) const;

    /// @brief Sets the attribute's value at a time code, a time sample, as `UsdAttribute::Set`
    /// with a time does. A sampled attribute holds its samples' value at `Time`, interpolated
    /// between the samples either side and held beyond the ends; `Set` or `Write` without a time
    /// drops its samples. An attribute's samples are all values or all arrays.
    /// @throws std::invalid_argument as `Set` does.
    void SetAt(PrimId prim, std::string_view attribute, double time, const Value& value);
    /// @brief `SetAt` for an array, kept and shared as `Write` keeps one. Device arrays between
    /// samples hold the lower one, for their consumers to mix on the device (`Samples`).
    void WriteAt(PrimId prim, std::string_view attribute, double time, Tensor array);
    /// @return The attribute's samples in time order; empty when it has none.
    std::span<const TimeSample> Samples(PrimId prim, std::string_view attribute) const;
    /// @return The attribute's value at `time` without moving the clock: its samples', else its
    /// authored value; none when unset or an array.
    std::optional<Value> GetAt(PrimId prim, std::string_view attribute, double time) const;
    /// @brief Drops the attribute's samples, keeping the value it holds now.
    /// @return Whether it had any.
    bool ClearSamples(PrimId prim, std::string_view attribute);
    /// @brief Moves the clock, evaluating every sampled attribute and stamping those whose value
    /// changes.
    void SetTime(double timeCode);
    double Time() const {
        return time_;
    }
    /// @brief Unauthors an attribute and drops its samples.
    /// @return Whether the attribute was authored.
    bool Clear(PrimId prim, std::string_view attribute);
    bool IsAuthored(PrimId prim, std::string_view attribute) const;
    /// @return The declaration of the prim's attribute by its type or API schemas, or null.
    const AttributeDef* Declaration(PrimId prim, std::string_view attribute) const;
    /// @return The authored attributes' names, in column order.
    std::vector<std::string> Attributes(const Path& path) const;

    /// @return The prim's own transform from its ops; identity for prims that aren't Xformable.
    glm::dmat4 LocalTransform(PrimId prim) const;
    glm::dmat4 WorldTransform(PrimId prim) const;
    /// @return The prim's extent in its own space; none when it isn't computed from attributes.
    /// Cached since `Update` when its bucket hasn't changed, computed otherwise.
    std::optional<Box> Extent(PrimId prim) const;
    /// @brief Brings derived data up to date with the edits since the last update, once per step
    /// before renderers pull changes: resolves inherited attributes for the subtrees whose
    /// authored values or prims changed, re-indexes changed relationships, then recomputes the
    /// changed rows' cached extents.
    void Update();
    /// @return The prims whose relationships target `target` as of the last `Update`, resolved
    /// material bindings included: what an edit to `target` reaches.
    std::vector<PrimId> Holders(const Path& target) const;
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
    /// @return A type's built-in API schemas, then the applied ones, each with those built into it,
    /// each once.
    std::vector<ApiName> AllApis(const SchemaDef* schema, std::span<const Token> applied) const;
    /// @return The attribute's column when the prim authors it, or null.
    const Column* AuthoredColumn(PrimId prim, std::string_view attribute) const;
    const Column* AuthoredColumn(PrimId prim, AttributeId attribute) const;
    /// @return The attribute's value when the prim authors it as a `T`, or null.
    template <typename T> const T* Authored(PrimId prim, std::string_view attribute) const {
        const std::optional<Token> token = store_.Tokens().Find(attribute);
        return token ? store_.Get<T>(prim, *token) : nullptr;
    }
    std::optional<Value> CellValue(PrimId prim, const Column* column) const;
    /// Checks a value against the attribute and authors it; with `checked`, converts it there
    /// instead, as `Get` would return it.
    void Author(PrimId prim, std::string_view attribute, const Value& value,
                Value* checked = nullptr);
    /// @throws std::invalid_argument if the attribute takes no such array.
    void CheckArray(PrimId prim, std::string_view attribute, const Tensor& array) const;
    /// Adds a checked sample, then authors the attribute's value at the clock.
    void AddSample(PrimId prim, std::string_view attribute, TimeSample sample);
    /// Authors a sampled attribute's value at the clock, if it changed.
    void Evaluate(PrimId prim, AttributeId attribute, std::span<const TimeSample> samples);

    /// Resolves every inherited attribute for the subtrees changed since the last resolve.
    void Resolve();
    const AttributeDef* Declared(BucketId bucket, AttributeId attribute) const;
    /// @return Whether the prim's type or an applied API schema leaves the attribute's namespace
    /// open; any attribute for a prim without a type.
    bool TakesUndeclared(PrimId prim, std::string_view attribute) const;
    /// @throws std::invalid_argument unless the prim takes the undeclared attribute.
    void RequireUndeclared(PrimId prim, std::string_view attribute) const;
    ColumnType TypeForUndeclared(PrimId prim, std::string_view attribute, const Value& value) const;
    /// @return Whether the prim and its subtree are invisible or of a purpose bounds leave out.
    bool LeavesOutOfBounds(PrimId prim) const;
    /// @return GeomModelAPI's extentsHint: the default purpose's box, or with `allPurposes`, all.
    Box ExtentsHint(PrimId prim, bool allPurposes) const;
    /// Adds a subtree's extents under `parent`'s transform; with `visibleOnly`, only its visible
    /// default-purpose prims'.
    void AddTo(Box& bounds, PathId path, const glm::dmat4& parent, bool visibleOnly) const;

    PrimStore store_;
    /// The store version inherited attributes were last resolved at.
    std::uint64_t resolved_ = 0;
    /// The sampled attributes of each prim, by its slot; sparse, as few attributes are animated.
    struct Animated {
        PrimId prim;
        std::map<AttributeId, std::vector<TimeSample>> samples;
    };
    std::unordered_map<std::uint32_t, Animated> animated_;
    double time_ = 0.0;
    Token inherited_ = 0;
    Token invisible_ = 0;
    Token defaultPurpose_ = 0;
    std::vector<BucketSchema> buckets_;
    ExtentCache extents_;
    HolderIndex holders_;
};

}  // namespace ernie
