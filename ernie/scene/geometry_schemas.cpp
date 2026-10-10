// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/geometry_schemas.h"

namespace ernie {

// For types that declare no attributes of their own.
#define ERNIE_NO_GEOMETRY_ATTRIBUTES(X)

ERNIE_DEFINE_SCHEMA(Cube, tokens::kCube, &Gprim::Definition(), false,
                    "A box centred on its origin, size along each axis.", ERNIE_CUBE_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Sphere, tokens::kSphere, &Gprim::Definition(), false,
                    "A sphere centred on its origin.", ERNIE_SPHERE_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(AxisShape, "AxisShape", &Gprim::Definition(), true,
                    "An implicit shape along one of its axes: a cylinder, cone, capsule or plane.",
                    ERNIE_AXIS_SHAPE_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Cylinder, tokens::kCylinder, &AxisShape::Definition(), false,
                    "A capped cylinder along its axis.", ERNIE_CYLINDER_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Cone, tokens::kCone, &AxisShape::Definition(), false,
                    "A capped cone along its axis, its apex at +height / 2.",
                    ERNIE_CYLINDER_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Capsule, tokens::kCapsule, &AxisShape::Definition(), false,
                    "A cylinder with hemispherical caps along its axis.", ERNIE_CAPSULE_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Plane, tokens::kPlane, &AxisShape::Definition(), false,
                    "A one-sided rectangle facing along its axis.", ERNIE_PLANE_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(PointBased, "PointBased", &Gprim::Definition(), true,
                    "Geometry from an array of points: a mesh or a point cloud.",
                    ERNIE_POINT_BASED_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Mesh, tokens::kMesh, &PointBased::Definition(), false,
                    "A polygon mesh: points and the faces over them.", ERNIE_MESH_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Points, tokens::kGeomPoints, &PointBased::Definition(), false,
                    "Points drawn as spheres widths across, coloured by their display colour.",
                    ERNIE_POINTS_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(PointInstancer, tokens::kPointInstancer, &Boundable::Definition(), false,
                    "Instances of its prototypes (the prototypes relationship's targets) at rows "
                    "of positions, orientations and scales.",
                    ERNIE_POINT_INSTANCER_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Volume, tokens::kVolume, &Gprim::Definition(), false,
                    "A volume (UsdVol); not drawn yet.", ERNIE_NO_GEOMETRY_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(GroundPlane, tokens::kGroundPlane, &Imageable::Definition(), false,
                    "Ernie: an infinite horizontal floor facing +z at a world height, drawn with "
                    "its material, optionally as a checkerboard.",
                    ERNIE_GROUND_PLANE_ATTRIBUTES)

std::vector<const PrimTypeDef*> GeometrySchemaTypes() {
    return {&Cube::Definition(),       &Sphere::Definition(),         &AxisShape::Definition(),
            &Cylinder::Definition(),   &Cone::Definition(),           &Capsule::Definition(),
            &Plane::Definition(),      &PointBased::Definition(),     &Mesh::Definition(),
            &Points::Definition(),     &PointInstancer::Definition(), &Volume::Definition(),
            &GroundPlane::Definition()};
}

}  // namespace ernie
