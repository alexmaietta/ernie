// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/stage/schema/types.h"

#ifndef __SLANG__
#include <optional>
#include <string_view>
#endif

namespace ernie {

ERNIE_ENUM(RenderVarSource) {
    LdrColor,
    HdrColor,
    DistanceToCamera,
    DistanceToImagePlane,
    NonStableInstanceSegmentation,
    Normal,
    DiffuseAlbedo,
    SemanticSegmentation,
    SemanticIdMap,
    SemanticInstanceSegmentation,
    StableIdSegmentation,
    StableIdMap,
    Intensity,
    PointCloud,
    RadialVelocity,
    InstanceId,
    InstancerStableId,
    ElectronRate,
};

static const uint kRenderVarSourceCount = 18;

#ifndef __SLANG__

inline constexpr std::string_view kRenderVarSourceTokens[kRenderVarSourceCount] = {
    "LdrColor",
    "HdrColor",
    "DistanceToCamera",
    "DistanceToImagePlane",
    "NonStableInstanceSegmentation",
    "Normal",
    "DiffuseAlbedo",
    "SemanticSegmentation",
    "SemanticIdMap",
    "SemanticInstanceSegmentation",
    "StableIdSegmentation",
    "StableIdMap",
    "Intensity",
    "PointCloud",
    "RadialVelocity",
    "InstanceId",
    "InstancerStableId",
    "ElectronRate",
};

inline std::optional<RenderVarSource> FindRenderVarSource(std::string_view name) {
    for (uint32_t source = 0; source < kRenderVarSourceCount; ++source) {
        if (kRenderVarSourceTokens[source] == name) {
            return RenderVarSource(source);
        }
    }
    return std::nullopt;
}

#endif

}
