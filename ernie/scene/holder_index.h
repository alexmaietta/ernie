// SPDX-License-Identifier: BSD-3-Clause
/// @file holder_index.h
/// @brief The reverse index of relationships: which prims target each path.
#pragma once

#include "ernie/scene/storage.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace ernie {

/// @brief Which prims' relationships target each path, kept up to date from the store's changes,
/// so an edit to a target reaches its holders: a prototype its instancers, a material the prims
/// bound to it (through their resolved bindings), a body its terrains.
class HolderIndex {
  public:
    /// Re-indexes the relationship cells that changed since the last update.
    void Update(const PrimStore& store);
    /// @return The live prims with a relationship targeting `target`, each once.
    std::vector<PrimId> Find(const PrimStore& store, PathId target) const;

  private:
    /// A relationship cell: the prim's slot and the attribute, packed.
    using Cell = std::uint64_t;
    static Cell Key(PrimId prim, AttributeId attribute) {
        return (std::uint64_t{prim.slot} << 32) | attribute;
    }

    void Index(PrimId prim, AttributeId attribute, std::vector<PathId> targets);

    /// Each target's holding cells, with the generation of the prim that wrote them.
    std::unordered_map<PathId, std::unordered_map<Cell, std::uint32_t>> holders_;
    /// Each cell's targets, to unindex them when it changes.
    std::unordered_map<Cell, std::vector<PathId>> targets_;
    std::uint64_t cursor_ = 0;
};

}  // namespace ernie
