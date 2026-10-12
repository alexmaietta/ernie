// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/path.h"

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace ernie {

class Scene;

/// One of a light's collections (UsdCollectionAPI with its prims
/// expanded): the whole scene when `includeRoot`, else nothing, then
/// the prims listed and their descendants in or out, the nearest
/// listed ancestor of a prim deciding.
struct LinkCollection {
    bool includeRoot = true;
    std::vector<Path> includes;
    std::vector<Path> excludes;

    /// Whether it holds every prim: the default, the whole scene with
    /// nothing excluded.
    bool Everything() const;

    /// Whether it holds the prim at `path`.
    bool Contains(const Path& path) const;
};

/// The light at `path`'s light link (`tokens::kLightLinkIncludeRoot`
/// and its relationships) and shadow link.
LinkCollection LightLinkOf(const Scene& scene, const Path& path);
LinkCollection ShadowLinkOf(const Scene& scene, const Path& path);

/// Geometry grouped by the lights that light it and whose shadows it
/// casts, as the rasterizer applies links: each surface carries its
/// group (0, lit and shadowed by every light, for prims no link
/// singles out), and each light a mask of the groups it lights and of
/// those that cast its shadows.
///
/// Reference: Pixar, UsdLux light linking (UsdLuxLightAPI's
/// collection:lightLink and collection:shadowLink).
/// https://openusd.org/release/api/class_usd_lux_light_a_p_i.html
struct LinkGroups {
    static constexpr std::uint32_t kMaxGroups = 32;

    /// Of the lights whose links single prims out: the groups they
    /// light and the groups whose geometry casts their shadows, a bit
    /// each.
    std::unordered_map<Path, std::uint32_t> lightMasks;
    std::unordered_map<Path, std::uint32_t> shadowMasks;
    /// Groups of the prims not in group 0.
    std::unordered_map<Path, std::uint32_t> groups;

    /// The group of the drawn prim at `path`.
    std::uint32_t GroupOf(const Path& path) const;

    /// The groups the light at `path` lights, and those that cast its
    /// shadows: every group for a light without links.
    std::uint32_t LightMask(const Path& light) const;
    std::uint32_t ShadowMask(const Path& light) const;
};

/// The link groups of `geometry` under `lights`.
/// @throws std::invalid_argument when the links make more than
/// `LinkGroups::kMaxGroups` groups.
LinkGroups LinkGroupsOf(const Scene& scene,
                        std::span<const Path> lights,
                        std::span<const Path> geometry);

}  // namespace ernie
