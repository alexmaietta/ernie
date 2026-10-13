// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/path.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/sensor_filters.h"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

/// Editing a camera's sensor chain (the Camera panel's Filter Chain):
/// a RenderProduct's `ernie:filters` and its SensorFilter prims, as
/// ordinary scene edits.
namespace ernie::app {

/// Gives `camera` a render product of its own, `/Render/<camera>
/// Sensor` (numbered when taken; under the session root for a camera
/// there, the viewer's own), holding LdrColor, with a copy of
/// the default chain as SensorFilter prims under its path plus
/// "Chain"; the viewport then runs it (`ViewChainProduct`). Returns
/// the product.
Path CustomizeChain(Scene& scene, const Path& camera);

/// The RenderProducts looking through `camera` (their first camera),
/// in scene order; those under the session root only for a camera
/// there.
std::vector<Path> ProductsOf(const Scene& scene, const Path& camera);

/// Whether `product` holds a render var of source name `name`.
bool HoldsVar(const Scene& scene, const Path& product,
              std::string_view name);

/// Adds a RenderVar of source name `name` to `product`'s ordered
/// vars (a child of the product named as it, numbered when taken),
/// or with `held` false removes those of that name and their prims.
void SetProductVar(Scene& scene, const Path& product,
                   std::string_view name, bool held);

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

/// The render var showing the image right after filter `type`, the
/// domain it leaves: SpectralRadiance, Electrons, RawDN, HdrColor or
/// LdrColor; LdrColor for a type neither built in nor in `registry`.
std::string_view StageVarOf(std::string_view type,
                            const SensorFilterRegistry& registry);

/// What filter `type` does, in a sentence or two: a registered
/// filter's description, or the built-in's; empty for a type neither
/// built in nor in `registry`.
std::string ChainDescriptionOf(std::string_view type,
                               const SensorFilterRegistry& registry);

/// What filter `type` works on, and what it makes: "spectrum >
/// electrons" for `qe`, "electrons" for `noise`; "unknown" for a type
/// neither built in nor in `registry`.
std::string ChainFlowOf(std::string_view type,
                        const SensorFilterRegistry& registry);

}  // namespace ernie::app
