// SPDX-License-Identifier: BSD-3-Clause

__exported import ernie.scene.records.render_tables;
__exported import ernie.render.kernels.trace.ray;

using namespace ernie;

struct SurfaceElement {
    bool mesh;
    bool indexed;
    uint triangle;
    uint face;
    uint3 corners;
    uint3 points;
    float3 weights;

    static SurfaceElement Shape(uint primitive) {
        SurfaceElement element;
        element.mesh = false;
        element.indexed = false;
        element.triangle = primitive;
        element.face = 0;
        element.corners = uint3(0);
        element.points = uint3(0);
        element.weights = float3(1, 0, 0);
        return element;
    }

    static SurfaceElement Point(uint index) {
        SurfaceElement element = Shape(index);
        element.indexed = true;
        element.face = index;
        element.corners = uint3(index);
        element.points = uint3(index);
        return element;
    }
}

SurfaceElement FindSurfaceElement(SurfaceRecord surface, Hit hit) {
    SurfaceElement element = SurfaceElement.Shape(hit.primitive);
    if (surface.corners == 0) {
        return element;
    }
    element.mesh = true;
    element.indexed = true;
    const Ptr<uint> corners = Ptr<uint>(surface.corners);
    const Ptr<int> indices = Ptr<int>(surface.indices);
    element.corners = uint3(corners[hit.primitive * 3 + 0], corners[hit.primitive * 3 + 1],
                            corners[hit.primitive * 3 + 2]);
    element.points = uint3(indices[element.corners.x], indices[element.corners.y],
                           indices[element.corners.z]);
    element.face = Ptr<uint>(surface.faces)[hit.primitive];
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
