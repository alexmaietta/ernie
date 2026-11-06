// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.render.irender;
__exported import ernie.render.kernels.sensors.frame;

using namespace ernie;

struct SensorRay {
    Ray ray;
    float weight;
    float spread;
}
