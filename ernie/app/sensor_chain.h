// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/path.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/sensor_filters.h"

#include <cstddef>
#include <string>
#include <string_view>

/// Editing a camera's sensor chain (the Camera panel's Filter Chain):
/// a RenderProduct's `ernie:filters` and its SensorFilter prims, as
/// ordinary scene edits.
namespace ernie::app {

/// Gives `camera` a render product of its own, `/Render/<camera>
/// Sensor` (numbered when taken), holding LdrColor, with a copy of
/// the default chain as SensorFilter prims under its path plus
/// "Chain"; the viewport then runs it (`ViewChainProduct`). Returns
/// the product.
Path CustomizeChain(Scene& scene, const Path& camera);

/// Adds a SensorFilter of `type` to `product`'s chain, beside its
/// other filters' prims (or under the product's path plus "Chain"):
/// right after the last filter leaving what it reads (`bayer` after
/// `noise`, `whiteBalance` after `cie`) when the chain stays valid
/// there (`CheckFilterChain` with `registry`), else at the latest
/// place it does, else at its end. Returns the new prim.
Path AddChainFilter(Scene& scene, const Path& product,
                    const std::string& type,
                    const SensorFilterRegistry& registry);

/// Moves `product`'s filter `index` up (`offset` -1) or down (+1),
/// as far as the chain's ends allow.
void MoveChainFilter(Scene& scene, const Path& product,
                     std::size_t index, int offset);

/// Removes `product`'s filter `index` from its chain, and its prim
/// when no product or var's chain names it any more.
void RemoveChainFilter(Scene& scene, const Path& product,
                       std::size_t index);

/// What filter `type` works on, and what it makes: "spectrum >
/// electrons" for `qe`, "electrons" for `noise`; "unknown" for a type
/// neither built in nor in `registry`.
std::string ChainFlowOf(std::string_view type,
                        const SensorFilterRegistry& registry);

}  // namespace ernie::app
