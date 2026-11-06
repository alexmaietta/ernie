// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.isensor;

using namespace ernie;

interface IPattern {
    static This Load(RenderParams params, Capture* capture);

    uint SampleCount();

    SensorRay Generate(CapturePixel target, uint sample);
}
