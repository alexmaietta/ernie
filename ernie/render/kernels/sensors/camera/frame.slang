// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.camera.icamera;
__exported import ernie.render.kernels.sensors.frame;
import ernie.device.kernels.device;

using namespace ernie;

static const float kMillimeters = 0.001f;
static const uint kMostClippingPlanes = 6;

struct CameraClipping {
    float near;
    float far;
    float4 planes[kMostClippingPlanes];
    uint planeCount;

    bool Contains(float3 local, float distance) {
        if (distance < near || distance > far) {
            return false;
        }
        for (uint i = 0; i < planeCount; ++i) {
            const float4 plane = planes[i];
            if (dot(plane.xyz, local) + plane.w < 0) {
                return false;
            }
        }
        return true;
    }

    float2 Clip(float3 origin, float3 direction) {
        float2 interval = float2(near, far) / length(direction);
        for (uint i = 0; i < planeCount; ++i) {
            const float4 plane = planes[i];
            Narrow(interval, plane.xyz, plane.w, origin, direction);
        }
        return interval;
    }

    void Narrow(inout float2 interval, float3 normal, float offset, float3 origin,
                float3 direction) {
        const float start = dot(normal, origin) + offset;
        const float rate = dot(normal, direction);
        if (rate > 0) {
            interval.x = max(interval.x, -start / rate);
        } else if (rate < 0) {
            interval.y = min(interval.y, -start / rate);
        } else if (start < 0) {
            interval = float2(1, 0);
        }
    }
}

struct CameraShutter {
    float open;
    float close;

    float Time(float u) {
        return lerp(open, close, u);
    }
}

float2 SampleUniformDisk(float2 u) {
    const float2 offset = 2 * u - 1;
    if (offset.x == 0 && offset.y == 0) {
        return float2(0);
    }
    const bool wide = abs(offset.x) > abs(offset.y);
    const float radius = wide ? offset.x : offset.y;
    const float angle =
        wide ? kPi / 4 * (offset.y / offset.x) : kPi / 2 - kPi / 4 * (offset.x / offset.y);
    return radius * float2(cos(angle), sin(angle));
}

[Differentiable]
CameraRay FrameRay(no_diff SensorFrame frame, no_diff CameraClipping clipping, float3 origin,
                   float3 direction, no_diff float time) {
    const float2 interval = no_diff clipping.Clip(detach(origin), detach(direction));
    const float3 world = frame.ToWorld(direction);
    const float scale = length(detach(world));
    CameraRay result;
    result.ray = Ray(frame.position + frame.ToWorld(origin), world / length(world),
                     interval.x * scale, interval.y * scale, time);
    result.weight = interval.x < interval.y ? 1.0f : 0.0f;
    return result;
}

CameraClipping LoadCameraClipping(SceneData* scene, uint row) {
    CameraData* data = &scene->schemas.camera;
    const ArrayRef planes = data.clippingPlanes[row];
    CameraClipping clipping;
    clipping.near = data.clippingRange[row].x;
    clipping.far = data.clippingRange[row].y;
    clipping.planeCount = min(planes.count, kMostClippingPlanes);
    for (uint i = 0; i < kMostClippingPlanes; ++i) {
        clipping.planes[i] = i < clipping.planeCount ? Ptr<Float4>(planes.address)[i].Vector()
                                                     : float4(0);
    }
    return clipping;
}

CameraShutter LoadCameraShutter(SceneData* scene, uint row) {
    CameraShutter shutter;
    shutter.open = scene->schemas.camera.shutterOpen[row];
    shutter.close = scene->schemas.camera.shutterClose[row];
    return shutter;
}

float LoadCameraExposure(SceneData* scene, uint row) {
    CameraData* data = &scene->schemas.camera;
    const float fStop = data.exposureFStop[row];
    return data.exposureResponsivity[row] * data.exposureTime[row] *
           (data.exposureIso[row] / 100) / (fStop * fStop) * exp2(data.exposure[row]);
}

float2 LoadCameraAperture(SceneData* scene, uint row) {
    return float2(scene->schemas.camera.horizontalAperture[row],
                  scene->schemas.camera.verticalAperture[row]);
}

float2 LoadCameraApertureOffset(SceneData* scene, uint row) {
    return float2(scene->schemas.camera.horizontalApertureOffset[row],
                  scene->schemas.camera.verticalApertureOffset[row]);
}
