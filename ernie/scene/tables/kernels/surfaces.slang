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
        surface.displayOpacity = scene->schemas.gprim.displayOpacity[gprimRow];
        surface.orientation = scene->schemas.gprim.orientation[gprimRow];
        PrimvarEntry opacity;
        if (FindPrimvarEntry(surface.primvars.address, surface.primvars.count,
                             p.displayOpacityName, opacity)) {
            const PrimvarType type = PrimvarType(opacity.type);
            surface.opacities = surface.primvars.address + opacity.values * sizeof(uint);
            surface.opacityCount = opacity.valueCount;
            surface.opacityStride = PrimvarComponents(type) * opacity.elementSize;
            surface.opacityFlags =
                (Interpolation(opacity.interpolation) != Interpolation::Constant ? kOpacityPerPoint
                                                                                 : 0) |
                (PrimvarIsInt(type) ? kOpacityInteger : 0);
            if (opacity.indexCount > 0) {
                surface.opacityIndices = surface.primvars.address + opacity.indices * sizeof(uint);
                surface.opacityIndexCount = opacity.indexCount;
            }
        }
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
        surface.meshRow = meshRow;
        const SubdivisionScheme scheme = SubdivisionScheme(scene->schemas.mesh.subdivisionScheme[meshRow]);
        if (scheme == SubdivisionScheme::CatmullClark || scheme == SubdivisionScheme::Loop) {
            surface.smooth = mesh.normals.address;
        }
        const bool polygonal = scheme == SubdivisionScheme::None;
        PrimvarEntry normals;
        if (polygonal && FindPrimvarEntry(surface.primvars.address, surface.primvars.count, p.normalsName,
                                      normals) &&
            PrimvarType(normals.type) == PrimvarType::Float3 && normals.elementSize == 1) {
            surface.normals = surface.primvars.address + normals.values * sizeof(uint);
            surface.normalsInterpolation = normals.interpolation;
            surface.normalsCount = normals.valueCount;
            if (normals.indexCount > 0) {
                surface.normalIndices = surface.primvars.address + normals.indices * sizeof(uint);
                surface.normalIndexCount = normals.indexCount;
            }
        } else if (polygonal && authored.count > 0) {
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
