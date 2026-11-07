// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.scene.kernels.scene;

using namespace ernie;

struct SensorFrame {
    float3 position;
    float3 origin;
    float3 x;
    float3 y;
    float3 z;

    [Differentiable]
    [NoDiffThis]
    float3 ToWorld(float3 v) {
        return x * v.x + y * v.y + z * v.z;
    }

    [Differentiable]
    [NoDiffThis]
    float3 ToLocal(float3 world) {
        const float3 v = world - position;
        return float3(dot(v, cross(y, z)), dot(v, cross(z, x)), dot(v, cross(x, y))) /
               dot(x, cross(y, z));
    }
}

SensorFrame LoadSensorFrame(SceneData* scene, RenderTables* tables, uint prim) {
    const WorldTransform transform = PrimWorld(scene, prim);
    SensorFrame frame;
    const uint world = scene->prims.world[prim];
    const RenderOrigin origin = TlasOrigin(tables, world);
    frame.position = RenderTranslation(tables, transform, world);
    frame.origin = origin.high.Vector() + origin.low.Vector();
    frame.x = transform.x.Vector();
    frame.y = transform.y.Vector();
    frame.z = transform.z.Vector();
    return frame;
}

float3 SphericalDirection(float azimuth, float elevation) {
    return float3(-sin(azimuth) * cos(elevation), sin(elevation), -cos(azimuth) * cos(elevation));
}

float3 PointVelocity(SceneData* scene, RenderTables* tables, uint prim, float3 position) {
    for (uint at = prim; at != kNoPrim; at = scene->prims.parent[at]) {
        if (HasApi(scene, at, kApiMotionAPI)) {
            const uint row = ApiRow(scene, at, kApiMotionAPI);
            const float3 linear = scene->schemas.motionAPI.motionVelocity[row].Vector();
            const float3 angular = radians(scene->schemas.motionAPI.motionAngularVelocity[row].Vector());
            return linear + cross(angular, position - LoadSensorFrame(scene, tables, at).position);
        }
    }
    return float3(0);
}
