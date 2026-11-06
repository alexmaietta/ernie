// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.isensor;
import ernie.render.kernels.random;

using namespace ernie;

interface IResponse {
    static This Load(RenderParams params, Capture* capture, HitQuery query);

    [mutating]
    void Add(SensorRay ray, inout Rng rng);

    void Write(RenderParams params, CapturePixel target);
}
