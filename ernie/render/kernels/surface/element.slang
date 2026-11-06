// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.scene.kernels.scene;
__exported import ernie.render.kernels.trace.ray;

using namespace ernie;

struct SurfaceElement {
    bool mesh;
    uint triangle;
    uint face;
    uint3 corners;
    uint3 points;
    float3 weights;

    static SurfaceElement Shape(uint primitive) {
        SurfaceElement element;
        element.mesh = false;
        element.triangle = primitive;
        element.face = 0;
        element.corners = uint3(0);
        element.points = uint3(0);
        element.weights = float3(1, 0, 0);
        return element;
    }
}

SurfaceElement FindSurfaceElement(SceneData* scene, uint prim, Hit hit) {
    SurfaceElement element = SurfaceElement.Shape(hit.primitive);
    if (SchemaId(scene->prims.schema[prim]) != SchemaId::Mesh) {
        return element;
    }
    element.mesh = true;
    const uint meshRow = PrimRow(scene, prim, kSlotMesh);
    const Ptr<uint> corners = Ptr<uint>(scene->schemas.mesh.triangleCorners[meshRow].address);
    const Ptr<int> indices = Ptr<int>(scene->schemas.mesh.faceVertexIndices[meshRow].address);
    element.corners = uint3(corners[hit.primitive * 3 + 0], corners[hit.primitive * 3 + 1],
                            corners[hit.primitive * 3 + 2]);
    element.points = uint3(indices[element.corners.x], indices[element.corners.y],
                           indices[element.corners.z]);
    element.face = Ptr<uint>(scene->schemas.mesh.triangleFaces[meshRow].address)[hit.primitive];
    element.weights = float3(1 - hit.barycentrics.x - hit.barycentrics.y, hit.barycentrics.x,
                             hit.barycentrics.y);
    return element;
}

uint3 InterpolationIndices(SurfaceElement element, Interpolation interpolation) {
    switch (interpolation) {
    case Interpolation::Constant:
        return uint3(0);
    case Interpolation::Uniform:
        return uint3(element.face);
    case Interpolation::FaceVarying:
        return element.corners;
    default:
        return element.points;
    }
}
