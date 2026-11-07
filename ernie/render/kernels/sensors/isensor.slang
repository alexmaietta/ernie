// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.render.context;
__exported import ernie.render.kernels.sensors.frame;
__exported import ernie.render.kernels.trace.ray;

using namespace ernie;

struct SensorRay {
    Ray ray;
    float weight;
    float spread;
}
