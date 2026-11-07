// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.query.hit;

#include "ernie/render/kernels/render/render.h"

using namespace ernie;

struct SensorContext {
    RenderParams params;
    Capture* capture;
    HitQuery query;

    __init(RenderParams params, Capture* capture) {
        SceneData* scene = params.scene;
        this.params = params;
        this.capture = capture;
        query = HitQuery(scene, params.layout, scene->prims.world[capture->sensor]);
    }
}
