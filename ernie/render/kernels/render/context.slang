// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.query.hit;

#include "ernie/render/kernels/render/render.h"

using namespace ernie;

struct SensorContext {
    RenderParams params;
    Capture* capture;
    HitQuery query;

    __init(RenderParams params, Capture* capture) {
        this.params = params;
        this.capture = capture;
        query = HitQuery(params.tables, params.tables->prims.world[capture->plan.sensor]);
    }
}
