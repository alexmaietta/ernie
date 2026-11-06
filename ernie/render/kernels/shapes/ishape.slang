// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.trace.ray;
import ernie.device.kernels.device;

struct ShapeHit : IDifferentiable {
    float t;
    float3 normal;
    bool inside;
}

struct ShapeSt {
    float2 st;
    float3 dpds;
    float3 dpdt;
}

interface IShape : IDifferentiable {
    float3 Extents();

    [Differentiable]
    ShapeHit Intersect(Ray ray);

    float3 Normal(float3 position);

    ShapeSt St(float3 position, float3 normal);
}

float Around(float3 offset) {
    const float turn = atan2(offset.y, offset.x) / (2 * kPi);
    return turn < 0 ? turn + 1 : turn;
}
