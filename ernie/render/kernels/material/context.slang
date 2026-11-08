// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.render.kernels.query.material;
__exported import ernie.render.kernels.surface.point;

struct ShadeContext {
    MaterialQuery materials;
    uint material;
    SurfacePoint point;
    float3 outgoing;

    __init(MaterialQuery materials, uint material, SurfacePoint point, float3 outgoing) {
        this.materials = materials;
        this.material = material;
        this.point = point;
        this.outgoing = outgoing;
    }
}
