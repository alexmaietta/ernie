// SPDX-License-Identifier: BSD-3-Clause
/// @file render.cpp
/// @brief UsdRender's registry entries.
#include "ernie/scene/schemas/render.h"

namespace ernie {

ERNIE_DEFINE_SCHEMA(RenderSettingsBase, Abstract, kNoExtent, ERNIE_RENDER_SETTINGS_BASE_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(RenderSettings, Concrete, kInheritExtent, ERNIE_RENDER_SETTINGS_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(RenderProduct, Concrete, kInheritExtent, ERNIE_RENDER_PRODUCT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(RenderVar, Concrete, kNoExtent, ERNIE_RENDER_VAR_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(RenderPass, Concrete, kNoExtent, ERNIE_RENDER_PASS_ATTRIBUTES,
                    .builtinApis = {"CollectionAPI:renderVisibility",
                                    "CollectionAPI:cameraVisibility"},
                    .fallbackOverrides = {{"collection:renderVisibility:includeRoot", true},
                                          {"collection:cameraVisibility:includeRoot", true}})

}  // namespace ernie
