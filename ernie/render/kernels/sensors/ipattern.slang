// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.isensor;

using namespace ernie;

interface IPattern {
    static This Load(SensorContext context);

    uint SampleCount();

    SensorRay Generate(CapturePixel pixel, uint sample);
}
