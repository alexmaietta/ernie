// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.sensors.camera.orthographic;
__exported import ernie.render.kernels.sensors.camera.perspective;

using namespace ernie;

ICamera LoadCamera(SceneData* scene, InstanceLayout* layout, uint prim) {
    switch (Projection(scene->schemas.camera.projection[PrimRow(scene, prim, kSlotCamera)])) {
    case Projection::Orthographic:
        return Orthographic.Load(scene, layout, prim);
    default:
        return Perspective.Load(scene, layout, prim);
    }
}
