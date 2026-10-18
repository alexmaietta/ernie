// SPDX-License-Identifier: BSD-3-Clause
/// @file render.cpp
/// @brief UsdRender's registry entries.
#include "ernie/scene/schemas/render.h"

namespace ernie {

ERNIE_DEFINE_SCHEMA(RenderSettingsBase, "RenderSettingsBase", nullptr, Abstract, kNoExtent,
                    "What settings and products share: the camera, image and Ernie's settings.",
                    ERNIE_RENDER_SETTINGS_BASE_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(RenderSettings, "RenderSettings", &RenderSettingsBase::Definition(), Concrete,
                    kInheritExtent, "A render's settings and the products it makes.",
                    ERNIE_RENDER_SETTINGS_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(RenderProduct, "RenderProduct", &RenderSettingsBase::Definition(), Concrete,
                    kInheritExtent, "One output, overriding its settings' where it authors them.",
                    ERNIE_RENDER_PRODUCT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(RenderVar, "RenderVar", nullptr, Concrete, kNoExtent,
                    "One channel set of a product: colour, depth, normals, ids or an Ernie output.",
                    ERNIE_RENDER_VAR_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(RenderPass, "RenderPass", nullptr, Concrete, kNoExtent,
                    "A render of a pipeline, with the prims and cameras it sees as collections.",
                    ERNIE_RENDER_PASS_ATTRIBUTES,
                    .builtinApis = {"CollectionAPI:renderVisibility",
                                    "CollectionAPI:cameraVisibility"},
                    .fallbackOverrides = {{"collection:renderVisibility:includeRoot", true},
                                          {"collection:cameraVisibility:includeRoot", true}})

}  // namespace ernie
