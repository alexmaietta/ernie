// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.camera.frame;

using namespace ernie;

static const float kOrthographicApertureUnits = 0.1f;

struct Orthographic : ICamera {
    float2 size;
    no_diff float2 offset;
    no_diff SensorFrame frame;
    no_diff CameraClipping clipping;
    no_diff CameraShutter shutter;
    no_diff float exposure;

    static Orthographic Load(SceneData* scene, InstanceLayout* layout, uint prim) {
        const uint row = PrimRow(scene, prim, kSlotCamera);
        Orthographic camera;
        camera.size = LoadCameraAperture(scene, row) * kOrthographicApertureUnits;
        camera.offset = LoadCameraApertureOffset(scene, row) * kOrthographicApertureUnits;
        camera.frame = LoadSensorFrame(scene, layout, prim);
        camera.clipping = LoadCameraClipping(scene, row);
        camera.shutter = LoadCameraShutter(scene, row);
        camera.exposure = LoadCameraExposure(scene, row);
        return camera;
    }

    [Differentiable]
    CameraRay GenerateRay(CameraSample sample) {
        const float3 origin = float3(sample.film * 0.5f * size + offset, 0);
        return FrameRay(frame, clipping, origin, float3(0, 0, -1), shutter.Time(sample.time));
    }

    [Differentiable]
    FilmPoint Project(float3 position) {
        const float3 local = frame.ToLocal(position);
        FilmPoint point;
        point.film = (local.xy - offset) / (0.5f * size);
        point.visible = no_diff clipping.Contains(detach(local));
        return point;
    }

    float ImagePlaneDistance(float3 position) {
        return -frame.ToLocal(position).z;
    }

    float PixelSpread(float2 filmStep) {
        return 0;
    }

    float Exposure() {
        return exposure;
    }
}
