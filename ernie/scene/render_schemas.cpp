// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/render_schemas.h"

namespace ernie {

ERNIE_DEFINE_SCHEMA(RenderSettingsBase, "RenderSettingsBase", &Imageable::Definition(), true,
                    "What render settings and products share: the camera, the resolution, and "
                    "what a product sets in place of its settings'.",
                    ERNIE_RENDER_SETTINGS_BASE_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(RenderSettings, tokens::kRenderSettings, &RenderSettingsBase::Definition(),
                    false,
                    "A render's settings: its camera, resolution and products, and the "
                    "rasterizer's options.",
                    ERNIE_RENDER_SETTINGS_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(RenderProduct, tokens::kRenderProduct, &RenderSettingsBase::Definition(), false,
                    "One image a render writes: its camera, resolution, render vars and sensor "
                    "filters.",
                    ERNIE_RENDER_PRODUCT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(RenderVar, tokens::kRenderVar, &Imageable::Definition(), false,
                    "An output of a render product, named by its source.",
                    ERNIE_RENDER_VAR_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(SensorFilterPrim, tokens::kSensorFilter, &Imageable::Definition(), false,
                    "One step of a sensor filter chain: a filter type and its inputs.",
                    ERNIE_SENSOR_FILTER_ATTRIBUTES)

RenderSettings RenderSettings::OrFallbacks(const Scene& scene, const Path& path) {
    if (!path.IsEmpty() && scene.Has(path)) {
        return RenderSettings(scene, path);
    }
    // Never destroyed, so views of it stay valid during static destruction.
    static const Scene& unset = *[] {
        auto* defaults = new Scene();
        defaults->Define(Path("/RenderSettings"), tokens::kRenderSettings);
        return defaults;
    }();
    return RenderSettings(unset, Path("/RenderSettings"));
}

std::vector<const PrimTypeDef*> RenderSchemaTypes() {
    return {&RenderSettingsBase::Definition(), &RenderSettings::Definition(),
            &RenderProduct::Definition(), &RenderVar::Definition(),
            &SensorFilterPrim::Definition()};
}

}  // namespace ernie
