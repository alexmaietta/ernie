// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.ipattern;
import ernie.render.kernels.random;

using namespace ernie;

interface IResponse<P : IPattern> {
    static This Load(SensorContext context, P pattern);

    [mutating]
    void Add(SensorRay ray, inout Rng rng);

    void Write(SensorContext context, P pattern, CapturePixel pixel);
}
