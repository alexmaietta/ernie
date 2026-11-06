// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/kernels/scene.h"

namespace ernie {

static const float kInfinitePlane = 1e6f;

ERNIE_ENUM(ShapeKind){
    Mesh,
    Sphere,
    Box,
    Plane,
    Cylinder,
    Cone,
    Capsule,
    Disk,
    Tube,
};

static const uint kGeometryMask = 1u;
static const uint kCameraEmitterMask = 2u;
static const uint kEmitterMask = 4u;

struct ShapeRecord {
    uint kind;
    float param;
};

ERNIE_CHECK_SIZE(ShapeRecord, 8);

ERNIE_SHARED bool IsEmitter(uint schema) {
    switch (SchemaId(schema)) {
    case SchemaId::SphereLight:
    case SchemaId::RectLight:
    case SchemaId::DiskLight:
    case SchemaId::CylinderLight:
        return true;
    default:
        return false;
    }
}

#ifdef __SLANG__

uint OrderedFloat(float value) {
    const uint bits = asuint(value);
    return (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;
}

float FloatFromOrdered(uint bits) {
    return asfloat((bits & 0x80000000u) != 0 ? bits & 0x7FFFFFFFu : ~bits);
}

struct UnitShape {
    float3x3 axes;
    ShapeRecord record;
};

UnitShape PlaceAlong(WorldTransform w, ShapeKind kind, uint axis, float3 scale, float param) {
    const float3 axes[3] = {w.x.Vector(), w.y.Vector(), w.z.Vector()};
    UnitShape shape;
    shape.axes = float3x3(axes[(axis + 1) % 3] * scale.x, axes[(axis + 2) % 3] * scale.y,
                          axes[axis] * scale.z);
    shape.record.kind = uint(kind);
    shape.record.param = param;
    return shape;
}

UnitShape PlaceShape(SceneData* scene, uint prim, WorldTransform w) {
    switch (SchemaId(scene->prims.schema[prim])) {
    case SchemaId::Sphere: {
        const float radius = scene->schemas.sphere.radius[PrimRow(scene, prim, kSlotSphere)];
        return PlaceAlong(w, ShapeKind::Sphere, uint(Axis::Z), float3(radius), 0);
    }
    case SchemaId::Cube: {
        const float half = 0.5f * scene->schemas.cube.size[PrimRow(scene, prim, kSlotCube)];
        return PlaceAlong(w, ShapeKind::Box, uint(Axis::Z), float3(half), 0);
    }
    case SchemaId::Plane: {
        const uint row = PrimRow(scene, prim, kSlotPlane);
        const float width = scene->schemas.plane.width[row];
        const float length = scene->schemas.plane.length[row];
        const float3 scale = float3(width > 0 ? width / 2 : kInfinitePlane,
                                    length > 0 ? length / 2 : kInfinitePlane, 1);
        return PlaceAlong(w, ShapeKind::Plane, scene->schemas.plane.axis[row], scale, 0);
    }
    case SchemaId::Cylinder: {
        const uint row = PrimRow(scene, prim, kSlotCylinder);
        const float radius = scene->schemas.cylinder.radius[row];
        const float3 scale = float3(radius, radius, scene->schemas.cylinder.height[row] / 2);
        return PlaceAlong(w, ShapeKind::Cylinder, scene->schemas.cylinder.axis[row], scale, 0);
    }
    case SchemaId::Cone: {
        const uint row = PrimRow(scene, prim, kSlotCone);
        const float radius = scene->schemas.cone.radius[row];
        const float3 scale = float3(radius, radius, scene->schemas.cone.height[row] / 2);
        return PlaceAlong(w, ShapeKind::Cone, scene->schemas.cone.axis[row], scale, 0);
    }
    case SchemaId::Capsule: {
        const uint row = PrimRow(scene, prim, kSlotCapsule);
        const float radius = scene->schemas.capsule.radius[row];
        const float reach = scene->schemas.capsule.height[row] / 2 + radius;
        const float3 scale = float3(radius, radius, reach);
        return PlaceAlong(w, ShapeKind::Capsule, scene->schemas.capsule.axis[row], scale,
                          reach / radius);
    }
    case SchemaId::SphereLight: {
        const float radius =
            scene->schemas.sphereLight.radius[PrimRow(scene, prim, kSlotSphereLight)];
        return PlaceAlong(w, ShapeKind::Sphere, uint(Axis::Z), float3(radius), 0);
    }
    case SchemaId::RectLight: {
        const uint row = PrimRow(scene, prim, kSlotRectLight);
        const float3 scale = float3(scene->schemas.rectLight.width[row] / 2,
                                    scene->schemas.rectLight.height[row] / 2, 1);
        return PlaceAlong(w, ShapeKind::Plane, uint(Axis::Z), scale, 0);
    }
    case SchemaId::DiskLight: {
        const float radius = scene->schemas.diskLight.radius[PrimRow(scene, prim, kSlotDiskLight)];
        return PlaceAlong(w, ShapeKind::Disk, uint(Axis::Z), float3(radius, radius, 1), 0);
    }
    case SchemaId::CylinderLight: {
        const uint row = PrimRow(scene, prim, kSlotCylinderLight);
        const float radius = scene->schemas.cylinderLight.radius[row];
        const float3 scale = float3(radius, radius, scene->schemas.cylinderLight.length[row] / 2);
        return PlaceAlong(w, ShapeKind::Tube, uint(Axis::X), scale, 0);
    }
    default:
        return PlaceAlong(w, ShapeKind::Mesh, uint(Axis::Z), float3(1), 0);
    }
}

#endif

}
