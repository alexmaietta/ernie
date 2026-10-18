// SPDX-License-Identifier: BSD-3-Clause
/// @file inherit.cpp
/// @brief Resolving inherited attributes from the nearest authoring ancestor into their resolved
/// columns, for only the subtrees whose authored values or prims changed.
#include "ernie/scene/convert.h"
#include "ernie/scene/scene.h"

#include <algorithm>
#include <concepts>

namespace ernie {
namespace {

/// @brief One inherited attribute resolving over the store.
template <typename T> struct Resolver {
    PrimStore& store;
    AttributeId attribute;
    AttributeId resolved;
    T fallback;
    /// The "inherited" token, which authors nothing.
    Token inherited;

    /// @return What the prim's own value resolves to: null when it doesn't author one.
    const T* Own(PrimId prim) const {
        const T* own = store.Get<T>(prim, attribute);
        if constexpr (std::same_as<T, Token>) {
            return own != nullptr && *own == inherited ? nullptr : own;
        }
        return own;
    }

    /// @return The value `path` inherits from its ancestors.
    T Above(PathId path) const {
        const PathTable& paths = store.Paths();
        for (PathId above = paths.Parent(path); above != kRootPath; above = paths.Parent(above)) {
            const std::optional<PrimId> prim = store.Find(above);
            if (const T* own = prim ? Own(*prim) : nullptr) {
                return *own;
            }
        }
        return fallback;
    }

    /// Resolves the subtree at `root`, descending past a prim only while its value changes or it
    /// is one of `roots`, which it clears as it reaches them.
    void Subtree(PathId root, BitSet& roots) {
        std::vector<std::pair<PathId, T>> pending{{root, Above(root)}};
        while (!pending.empty()) {
            const auto [path, above] = std::move(pending.back());
            pending.pop_back();
            const std::optional<PrimId> prim = store.Find(path);
            if (!prim) {
                continue;
            }
            const bool isRoot = roots.Test(path);
            roots.Set(path, false);
            const T* own = Own(*prim);
            const T value = own != nullptr ? *own : above;
            // Prims without the column (typeless ones, materials) pass the value through.
            const bool stored = store.GetBucket(store.PrimBucket(*prim)).Find(resolved) != nullptr;
            // An unchanged prim's descendants are unchanged too, unless they are roots themselves.
            if (store.Derive(*prim, resolved, value) || isRoot || !stored) {
                store.Paths().ForEachChild(
                    path, [&](PathId child) { pending.emplace_back(child, value); });
            }
        }
    }
};

/// @return The paths of the prims whose subtrees resolve `attribute` again: those added or with
/// it changed.
BitSet Roots(const PrimStore& store, const ChangeSet& changes, AttributeId attribute) {
    BitSet roots;
    roots.Resize(store.Paths().Size());
    const auto add = [&](BucketId bucket, std::uint32_t begin, std::uint32_t end) {
        const Bucket& rows = store.GetBucket(bucket);
        for (std::uint32_t row = begin; row < std::min(end, rows.Rows()); ++row) {
            roots.Set(store.PrimPath(rows.prims[row]));
        }
    };
    for (const ChangeSet::Rows& rows : changes.rows) {
        if (rows.change.kind == StructuralChange::Kind::Added ||
            rows.change.kind == StructuralChange::Kind::Reset) {
            add(rows.bucket, rows.change.row, rows.change.row + rows.change.count);
        }
    }
    for (const ChangeSet::Values& values : changes.values) {
        if (values.attribute == attribute) {
            for (const RowRange range : values.rows) {
                add(values.bucket, range.begin, range.end);
            }
        }
    }
    return roots;
}

}  // namespace

void Scene::Resolve() {
    const ChangeSet changes = store_.ChangesSince(resolved_);
    for (const AttributeDef* def : InheritedAttributes()) {
        const AttributeId attribute = store_.Tokens().Intern(def->name);
        WithColumnType(def->type, [&]<typename T>(std::type_identity<T>) {
            if constexpr (std::equality_comparable<T> && !std::same_as<T, Tensor>) {
                Resolver<T> resolver{
                    store_, attribute, store_.Tokens().Intern(ResolvedName(def->name)),
                    ToElement<T>(store_.Paths(), def->fallback).value_or(T{}), inherited_};
                BitSet roots = Roots(store_, changes, attribute);
                // Ancestors intern first, so most roots are reached from one above them.
                const BitSet start = roots;
                start.ForEachRun([&](std::size_t begin, std::size_t end) {
                    for (std::size_t path = begin; path < end; ++path) {
                        if (roots.Test(path)) {
                            resolver.Subtree(static_cast<PathId>(path), roots);
                        }
                    }
                });
            }
        });
    }
    // Its own writes are to resolved columns, which no resolve reads.
    resolved_ = store_.Version();
}

}  // namespace ernie
