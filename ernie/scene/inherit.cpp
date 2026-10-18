// SPDX-License-Identifier: BSD-3-Clause
/// @file inherit.cpp
/// @brief Resolving inherited attributes from the nearest authoring ancestor into their resolved
/// columns, for only the subtrees whose authored values or prims changed.
#include "ernie/scene/convert.h"
#include "ernie/scene/scene.h"

#include <algorithm>
#include <concepts>
#include <limits>

namespace ernie {

void Scene::Resolve() {
    const ChangeSet changes = store_.ChangesSince(resolved_);
    for (const AttributeDef* def : InheritedAttributes()) {
        const AttributeId attribute = store_.Tokens().Intern(def->name);
        const AttributeId resolved = store_.Tokens().Intern(ResolvedName(def->name));
        WithColumnType(def->type, [&]<typename T>(std::type_identity<T>) {
            if constexpr (std::equality_comparable<T> && !std::same_as<T, Tensor>) {
                const T fallback = ToElement<T>(store_.Paths(), def->fallback).value_or(T{});
                BitSet roots = ResolveRoots(changes, attribute);
                std::vector<PathId> order;
                roots.ForEachRun([&](std::size_t begin, std::size_t end) {
                    for (std::size_t path = begin; path < end; ++path) {
                        order.push_back(static_cast<PathId>(path));
                    }
                });
                // Ancestors intern first, so most roots are reached from one above them.
                for (const PathId root : order) {
                    if (roots.Test(root)) {
                        ResolveSubtree<T>(root, attribute, resolved, fallback, roots);
                    }
                }
            }
        });
    }
    // Its own writes are to resolved columns, which no resolve reads.
    resolved_ = store_.Version();
}

BitSet Scene::ResolveRoots(const ChangeSet& changes, AttributeId attribute) const {
    BitSet roots;
    roots.Resize(store_.Paths().Size());
    const auto add = [&](BucketId bucket, std::uint32_t begin, std::uint32_t end) {
        const Bucket& rows = store_.GetBucket(bucket);
        for (std::uint32_t row = begin; row < std::min(end, rows.Rows()); ++row) {
            roots.Set(store_.PrimPath(rows.prims[row]));
        }
    };
    for (const BucketId bucket : changes.resets) {
        add(bucket, 0, std::numeric_limits<std::uint32_t>::max());
    }
    for (const ChangeSet::Rows& rows : changes.rows) {
        if (rows.change.kind == StructuralChange::Kind::Added) {
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

template <typename T> const T* Scene::Inheritable(PrimId prim, AttributeId attribute) const {
    const BucketId bucket = store_.PrimBucket(prim);
    const Column* column = store_.GetBucket(bucket).Find(attribute);
    const std::uint32_t row = store_.PrimRow(prim);
    if (column == nullptr || !column->authored.Test(row) ||
        column->values->Type() != kColumnType<T>) {
        return nullptr;
    }
    const T* own = &store_.Read<T>(bucket, attribute)[row];
    if constexpr (std::same_as<T, Token>) {
        return *own == inherited_ ? nullptr : own;
    }
    return own;
}

template <typename T> T Scene::Above(PathId path, AttributeId attribute, const T& fallback) const {
    const PathTable& paths = store_.Paths();
    for (PathId above = paths.Parent(path); above != kRootPath; above = paths.Parent(above)) {
        const std::optional<PrimId> prim = store_.Find(above);
        if (const T* own = prim ? Inheritable<T>(*prim, attribute) : nullptr) {
            return *own;
        }
    }
    return fallback;
}

template <typename T>
void Scene::ResolveSubtree(PathId root, AttributeId attribute, AttributeId resolved,
                           const T& fallback, BitSet& roots) {
    std::vector<std::pair<PathId, T>> pending{{root, Above<T>(root, attribute, fallback)}};
    while (!pending.empty()) {
        const auto [path, above] = std::move(pending.back());
        pending.pop_back();
        const std::optional<PrimId> prim = store_.Find(path);
        if (!prim) {
            continue;
        }
        const bool isRoot = roots.Test(path);
        roots.Set(path, false);
        const T* own = Inheritable<T>(*prim, attribute);
        const T value = own != nullptr ? *own : above;
        // Prims without the column (typeless ones, materials) pass the value through.
        const bool stored = store_.GetBucket(store_.PrimBucket(*prim)).Find(resolved) != nullptr;
        // An unchanged prim's descendants are unchanged too, unless they are roots themselves.
        if (store_.Derive(*prim, resolved, value) || isRoot || !stored) {
            store_.Paths().ForEachChild(path,
                                        [&](PathId child) { pending.emplace_back(child, value); });
        }
    }
}

}  // namespace ernie
