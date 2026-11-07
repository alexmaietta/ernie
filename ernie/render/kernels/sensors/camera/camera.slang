// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.camera.idistortion;
__exported import ernie.render.kernels.sensors.camera.orthographic;
__exported import ernie.render.kernels.sensors.camera.perspective;
__exported import ernie.render.kernels.trace.ray;
import ernie.device.kernels.device;

using namespace ernie;

static const float kMillimeters = 0.001f;

struct CameraSample : IDifferentiable {
    float2 film;
    no_diff float2 lens;
    no_diff float time;

    [Differentiable]
    __init(float2 film, no_diff float2 lens, no_diff float time) {
        this.film = film;
        this.lens = lens;
        this.time = time;
    }
}

struct CameraRay : IDifferentiable {
    Ray ray;
    no_diff float weight;
}

struct FilmPoint : IDifferentiable {
    float2 film;
    no_diff bool visible;
}

struct CameraSensor : IDifferentiable {
    float2 aperture;
    no_diff float2 offset;

    [Differentiable]
    float2 Film(float2 ndc) {
        return ndc * 0.5f * aperture + offset;
    }

    [Differentiable]
    float2 Ndc(float2 film) {
        return (film - offset) / (0.5f * aperture);
    }
}

struct CameraLens : IDifferentiable {
    SceneData* scene;
    no_diff uint row;
    no_diff Projection projection;

    [Differentiable]
    LensRay Trace(float2 film) {
        const float2 ideal = NoDistortion.Load(scene, row).Undistort(film);
        switch (projection) {
        case Projection::Orthographic:
            return OrthographicProjection.Load(scene, row).Trace(ideal);
        default:
            return PerspectiveProjection.Load(scene, row).Trace(ideal);
        }
    }

    [Differentiable]
    float2 Image(float3 local) {
        float2 ideal;
        switch (projection) {
        case Projection::Orthographic:
            ideal = OrthographicProjection.Load(scene, row).Image(local);
            break;
        default:
            ideal = PerspectiveProjection.Load(scene, row).Image(local);
            break;
        }
        return NoDistortion.Load(scene, row).Distort(ideal);
    }

    float Distance(float3 local) {
        switch (projection) {
        case Projection::Orthographic:
            return OrthographicProjection.Load(scene, row).Distance(local);
        default:
            return PerspectiveProjection.Load(scene, row).Distance(local);
        }
    }

    float Spread(float2 filmStep) {
        switch (projection) {
        case Projection::Orthographic:
            return OrthographicProjection.Load(scene, row).Spread(filmStep);
        default:
            return PerspectiveProjection.Load(scene, row).Spread(filmStep);
        }
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

struct CameraAperture {
    float radius;
    float focusDistance;

    [Differentiable]
    [NoDiffThis]
    LensRay Focus(LensRay chief, no_diff float2 u) {
        if (radius <= 0 || focusDistance <= 0) {
            return chief;
        }
        const float3 focus = chief.origin + chief.direction * focusDistance;
        LensRay ray;
        ray.origin = chief.origin + float3(radius * no_diff SampleUniformDisk(u), 0);
        ray.direction = focus - ray.origin;
        return ray;
    }
}

struct CameraClipping {
    float near;
    float far;
    ArrayRef planes;

    float4 Plane(uint i) {
        return Ptr<Float4>(planes.address)[i].Vector();
    }

    bool Contains(float3 local, float distance) {
        if (distance < near || distance > far) {
            return false;
        }
        for (uint i = 0; i < planes.count; ++i) {
            const float4 plane = Plane(i);
            if (dot(plane.xyz, local) + plane.w < 0) {
                return false;
            }
        }
        return true;
    }

    float2 Clip(float3 origin, float3 direction) {
        float2 interval = float2(near, far) / length(direction);
        for (uint i = 0; i < planes.count; ++i) {
            const float4 plane = Plane(i);
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

struct CameraModel : IDifferentiable {
    CameraSensor sensor;
    CameraLens lens;
    no_diff CameraAperture aperture;
    no_diff CameraClipping clipping;
    no_diff CameraShutter shutter;
    no_diff SensorFrame pose;
    no_diff float exposure;

    static CameraModel Load(SceneData* scene, InstanceLayout* layout, uint prim) {
        CameraData* data = &scene->schemas.camera;
        const uint row = PrimRow(scene, prim, kSlotCamera);
        CameraModel camera;

        camera.sensor.aperture = float2(data.horizontalAperture[row], data.verticalAperture[row]);
        camera.sensor.offset =
            float2(data.horizontalApertureOffset[row], data.verticalApertureOffset[row]);

        camera.lens.scene = scene;
        camera.lens.row = row;
        camera.lens.projection = Projection(data.projection[row]);

        const float fStop = data.fStop[row];
        if (camera.lens.projection == Projection::Orthographic || fStop <= 0) {
            camera.aperture.radius = 0;
        } else {
            camera.aperture.radius = data.focalLength[row] * kMillimeters / (2 * fStop);
        }
        camera.aperture.focusDistance = data.focusDistance[row];

        camera.clipping.near = data.clippingRange[row].x;
        camera.clipping.far = data.clippingRange[row].y;
        camera.clipping.planes = data.clippingPlanes[row];

        camera.shutter.open = data.shutterOpen[row];
        camera.shutter.close = data.shutterClose[row];

        camera.pose = LoadSensorFrame(scene, layout, prim);

        const float exposureFStop = data.exposureFStop[row];
        const float sensitivity = data.exposureResponsivity[row] * (data.exposureIso[row] / 100);
        const float light = data.exposureTime[row] / (exposureFStop * exposureFStop);
        camera.exposure = sensitivity * light * exp2(data.exposure[row]);

        return camera;
    }

    [Differentiable]
    CameraRay GenerateRay(CameraSample sample) {
        const LensRay ray = aperture.Focus(lens.Trace(sensor.Film(sample.film)), sample.lens);
        const float2 interval = no_diff clipping.Clip(detach(ray.origin), detach(ray.direction));
        const float3 world = pose.ToWorld(ray.direction);
        const float scale = length(detach(world));
        CameraRay result;
        result.ray = Ray(pose.position + pose.ToWorld(ray.origin), world / length(world),
                         interval.x * scale, interval.y * scale, shutter.Time(sample.time));
        result.weight = interval.x < interval.y ? 1.0f : 0.0f;
        return result;
    }

    [Differentiable]
    FilmPoint Project(float3 position) {
        const float3 local = pose.ToLocal(position);
        FilmPoint point;
        point.film = sensor.Ndc(lens.Image(local));
        point.visible = no_diff clipping.Contains(detach(local), lens.Distance(detach(local)));
        return point;
    }

    float ImagePlaneDistance(float3 position) {
        return -pose.ToLocal(position).z;
    }

    float PixelSpread(float2 filmStep) {
        return lens.Spread(filmStep * 0.5f * sensor.aperture);
    }
}
