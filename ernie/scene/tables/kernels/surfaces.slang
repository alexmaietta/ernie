// SPDX-License-Identifier: BSD-3-Clause

import ernie.device.kernels.device;

#include "ernie/scene/tables/kernels/surfaces.h"

using namespace ernie;

SurfaceRecord DescribeSurface(SurfaceParams p, SlotHit hit) {
    SurfaceRecord surface = {};
    AuthoredData* scene = p.scene;
    const uint prim = hit.prim;
    if (prim == kNoPrim || prim >= scene->primCount) {
        return surface;
    }
    const SchemaId schema = SchemaId(scene->prims.schema[prim]);
    if (SchemaIsA(uint(schema), SchemaId::Gprim)) {
        const uint gprimRow = PrimRow(scene, prim, kSlotGprim);
        surface.primvars = scene->schemas.gprim.primvars[gprimRow];
        surface.displayColor = scene->schemas.gprim.displayColor[gprimRow];
    }
    if (schema == SchemaId::Mesh) {
        const uint meshRow = PrimRow(scene, prim, kSlotMesh);
        const uint pointsRow = PrimRow(scene, prim, kSlotPointBased);
        const MeshRecord mesh = p.meshes[meshRow];
        const ArrayRef authored = scene->schemas.pointBased.normals[pointsRow];
        surface.points = scene->schemas.pointBased.points[pointsRow].address;
        surface.indices = scene->schemas.mesh.faceVertexIndices[meshRow].address;
        surface.corners = mesh.corners.address;
        surface.faces = mesh.faces.address;
        surface.smooth = mesh.normals.address;
        surface.meshRow = meshRow;
        const bool subdivided =
            scene->schemas.mesh.subdivisionScheme[meshRow] != uint(SubdivisionScheme::None);
        if (!subdivided && authored.count > 0) {
            surface.normals = authored.address;
            surface.normalsInterpolation = scene->schemas.pointBased.normalsInterpolation[pointsRow];
            surface.normalsCount = authored.count;
        }
    }
    if (hit.instancer != kNoPrim &&
        SchemaId(scene->prims.schema[hit.instancer]) == SchemaId::PointInstancer) {
        const uint row = PrimRow(scene, hit.instancer, kSlotPointInstancer);
        surface.instancerPrimvars = scene->schemas.pointInstancer.primvars[row];
        const ArrayRef ids = scene->schemas.pointInstancer.ids[row];
        surface.instanceId = hit.instance < ids.count ? Ptr<int64_t>(ids.address)[hit.instance]
                                                      : int64_t(hit.instance);
    }
    return surface;
}

[shader("compute")]
[numthreads(kSurfaceGroup, 1, 1)]
void WriteSurfaces(uniform SurfaceParams p) {
    const uint3 thread = ThreadIndex();
    InstanceLayout* layout = p.layout;
    const uint tlas = p.tlases[thread.y];
    const uint slot = layout->firstInstances[tlas] + thread.x;
    if (slot >= layout->firstInstances[tlas + 1]) {
        return;
    }
    p.surfaces[slot] = DescribeSurface(p, layout->hits[slot]);
}
