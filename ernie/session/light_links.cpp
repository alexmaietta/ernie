// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/session/light_links.h"

#include "ernie/scene/light_schemas.h"
#include "ernie/scene/scene.h"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>

namespace ernie {

bool LinkCollection::Everything() const {
    return includeRoot && excludes.empty();
}

bool LinkCollection::Contains(const Path& path) const {
    // The nearest of the prim and its ancestors that is listed
    // decides (an exclude winning a tie), else the root.
    for (Path at = path; !at.IsRoot(); at = at.Parent()) {
        if (std::find(excludes.begin(), excludes.end(), at) !=
            excludes.end()) {
            return false;
        }
        if (std::find(includes.begin(), includes.end(), at) !=
            includes.end()) {
            return true;
        }
    }
    return includeRoot;
}

namespace {

/// A light's link collections through its view: a UsdLux light's
/// (`Light`) or a LaserLight's, both `ERNIE_LIGHT_LINK_ATTRIBUTES`.
template <typename View>
LinkCollection LightLinkOfView(const View& light) {
    return {light.GetLightLinkIncludeRoot(),
            light.GetLightLinkIncludes(),
            light.GetLightLinkExcludes()};
}

template <typename View>
LinkCollection ShadowLinkOfView(const View& light) {
    return {light.GetShadowLinkIncludeRoot(),
            light.GetShadowLinkIncludes(),
            light.GetShadowLinkExcludes()};
}

}  // namespace

LinkCollection LightLinkOf(const Scene& scene, const Path& path) {
    return scene.TypeOf(path) == tokens::kLaserLight
               ? LightLinkOfView(LaserLight(scene, path))
               : LightLinkOfView(Light(scene, path));
}

LinkCollection ShadowLinkOf(const Scene& scene, const Path& path) {
    return scene.TypeOf(path) == tokens::kLaserLight
               ? ShadowLinkOfView(LaserLight(scene, path))
               : ShadowLinkOfView(Light(scene, path));
}

std::uint32_t LinkGroups::GroupOf(const Path& path) const {
    const auto found = groups.find(path);
    return found == groups.end() ? 0u : found->second;
}

std::uint32_t LinkGroups::LightMask(const Path& light) const {
    const auto found = lightMasks.find(light);
    return found == lightMasks.end() ? ~0u : found->second;
}

std::uint32_t LinkGroups::ShadowMask(const Path& light) const {
    const auto found = shadowMasks.find(light);
    return found == shadowMasks.end() ? ~0u : found->second;
}

LinkGroups LinkGroupsOf(const Scene& scene,
                        std::span<const Path> lights,
                        std::span<const Path> geometry) {
    // The collections that single any prims out.
    struct Link {
        std::size_t light;
        bool shadow;
        LinkCollection collection;
    };
    std::vector<Link> links;
    for (std::size_t i = 0; i < lights.size(); ++i) {
        LinkCollection light = LightLinkOf(scene, lights[i]);
        if (!light.Everything()) {
            links.push_back({i, false, std::move(light)});
        }
        LinkCollection shadow = ShadowLinkOf(scene, lights[i]);
        if (!shadow.Everything()) {
            links.push_back({i, true, std::move(shadow)});
        }
    }
    LinkGroups result;
    if (links.empty()) {
        return result;
    }
    // A prim's group by which of the links hold it: group 0 those all
    // hold, then each other combination in the order met.
    const std::vector<bool> all(links.size(), true);
    std::map<std::vector<bool>, std::uint32_t> groupOf{{all, 0u}};
    std::vector<std::vector<bool>> held{all};
    for (const Path& prim : geometry) {
        std::vector<bool> in(links.size());
        for (std::size_t k = 0; k < links.size(); ++k) {
            in[k] = links[k].collection.Contains(prim);
        }
        auto [at, added] = groupOf.emplace(
            in, static_cast<std::uint32_t>(held.size()));
        if (added) {
            if (held.size() == LinkGroups::kMaxGroups) {
                throw std::invalid_argument(
                    "Light links make more than " +
                    std::to_string(LinkGroups::kMaxGroups) +
                    " groups of geometry lit and shadowed alike");
            }
            held.push_back(in);
        }
        if (at->second != 0) {
            result.groups.emplace(prim, at->second);
        }
    }
    for (std::size_t k = 0; k < links.size(); ++k) {
        std::uint32_t mask = 0;
        for (std::size_t g = 0; g < held.size(); ++g) {
            mask |= held[g][k] ? 1u << g : 0u;
        }
        auto& masks =
            links[k].shadow ? result.shadowMasks : result.lightMasks;
        masks[lights[links[k].light]] = mask;
    }
    return result;
}

}  // namespace ernie
