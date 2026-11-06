// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.camera.frame;

using namespace ernie;

struct Perspective : ICamera {
    float focalLength;
    float2 aperture;
    no_diff float2 offset;
    no_diff float lensRadius;
    no_diff float focusDistance;
    no_diff SensorFrame frame;
    no_diff CameraClipping clipping;
    no_diff CameraShutter shutter;
    no_diff float exposure;

    static Perspective Load(SceneData* scene, InstanceLayout* layout, uint prim) {
        const uint row = PrimRow(scene, prim, kSlotCamera);
        const float fStop = scene->schemas.camera.fStop[row];
        Perspective camera;
        camera.focalLength = scene->schemas.camera.focalLength[row];
        camera.aperture = LoadCameraAperture(scene, row);
        camera.offset = LoadCameraApertureOffset(scene, row);
        camera.lensRadius = fStop > 0 ? camera.focalLength * kMillimeters / (2 * fStop) : 0;
        camera.focusDistance = scene->schemas.camera.focusDistance[row];
        camera.frame = LoadSensorFrame(scene, layout, prim);
        camera.clipping = LoadCameraClipping(scene, row);
        camera.shutter = LoadCameraShutter(scene, row);
        camera.exposure = LoadCameraExposure(scene, row);
        return camera;
    }

    [Differentiable]
    CameraRay GenerateRay(CameraSample sample) {
        const float3 film = float3(sample.film * 0.5f * aperture + offset, -focalLength);
        const float3 pinhole = film / focalLength;
        const float time = shutter.Time(sample.time);
        if (lensRadius <= 0 || focusDistance <= 0) {
            return FrameRay(frame, clipping, float3(0), pinhole, time);
        }
        const float3 lens = float3(lensRadius * no_diff SampleUniformDisk(sample.lens), 0);
        const float3 focus = pinhole * focusDistance;
        return FrameRay(frame, clipping, lens, focus - lens, time);
    }

    [Differentiable]
    FilmPoint Project(float3 position) {
        const float3 local = frame.ToLocal(position);
        FilmPoint point;
        point.film = (local.xy * (focalLength / -local.z) - offset) / (0.5f * aperture);
        point.visible = no_diff clipping.Contains(detach(local));
        return point;
    }

    float ImagePlaneDistance(float3 position) {
        return -frame.ToLocal(position).z;
    }

    float PixelSpread(float2 filmStep) {
        return filmStep.y * 0.5f * aperture.y / focalLength;
    }

    float Exposure() {
        return exposure;
    }
}
