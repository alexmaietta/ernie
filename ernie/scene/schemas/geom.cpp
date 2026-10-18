// SPDX-License-Identifier: BSD-3-Clause
/// @file geom.cpp
/// @brief UsdGeom's registry entries and the implicit shapes' extents, as UsdGeom computes them.
#include "ernie/scene/schemas/geom.h"

namespace ernie {
namespace {

/// @return The axis index (0, 1, 2) of each row's `axis` token.
int AxisIndex(const ExtentRows& rows, std::span<const Token> axes, std::uint32_t row) {
    return rows.Text(axes[row]).front() - 'X';
}

/// @return A box of `radial` about the axis and `half` along it.
Box AlongAxis(int axis, double radial, double half) {
    glm::dvec3 size(radial);
    size[axis] = half;
    return Box::Around(glm::dvec3(0.0), size);
}

void CubeExtent(const ExtentRows& rows, RowRange range, std::span<Box> out) {
    const auto size = rows.Column<double>(Cube::kSize);
    for (std::uint32_t row = range.begin; row < range.end; ++row) {
        out[row - range.begin] = Box::Around(glm::dvec3(0.0), glm::dvec3(0.5 * size[row]));
    }
}

void SphereExtent(const ExtentRows& rows, RowRange range, std::span<Box> out) {
    const auto radius = rows.Column<double>(Sphere::kRadius);
    for (std::uint32_t row = range.begin; row < range.end; ++row) {
        out[row - range.begin] = Box::Around(glm::dvec3(0.0), glm::dvec3(radius[row]));
    }
}

/// Cylinders, cones and capsules: a radius about the axis; capsules' caps add it along the axis.
template <bool kCapped>
void RoundExtent(const ExtentRows& rows, RowRange range, std::span<Box> out) {
    const auto radius = rows.Column<double>(Cylinder::kRadius);
    const auto height = rows.Column<double>(Cylinder::kHeight);
    const auto axis = rows.Column<Token>(Cylinder::kAxis);
    for (std::uint32_t row = range.begin; row < range.end; ++row) {
        const double half = 0.5 * height[row] + (kCapped ? radius[row] : 0.0);
        out[row - range.begin] = AlongAxis(AxisIndex(rows, axis, row), radius[row], half);
    }
}

void PlaneExtent(const ExtentRows& rows, RowRange range, std::span<Box> out) {
    const auto width = rows.Column<double>(Plane::kWidth);
    const auto length = rows.Column<double>(Plane::kLength);
    const auto axis = rows.Column<Token>(Plane::kAxis);
    for (std::uint32_t row = range.begin; row < range.end; ++row) {
        const int normal = AxisIndex(rows, axis, row);
        // Width lies along x, or z for an X normal; length along y, or z for a Y normal.
        glm::dvec3 half(0.0);
        half[normal == 0 ? 2 : 0] = 0.5 * width[row];
        half[normal == 1 ? 2 : 1] = 0.5 * length[row];
        out[row - range.begin] = Box::Around(glm::dvec3(0.0), half);
    }
}

}  // namespace

ERNIE_DEFINE_SCHEMA(Imageable, "Imageable", nullptr, Abstract, kInheritExtent,
                    "What every prim has: visibility, purpose and the world it is in.",
                    ERNIE_IMAGEABLE_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Xformable, "Xformable", &Imageable::Definition(), Abstract, kInheritExtent,
                    "What has a transform, composed from its xformOps.", ERNIE_XFORMABLE_ATTRIBUTES,
                    tokens::kXformOpPrefix)
ERNIE_DEFINE_SCHEMA(Boundable, "Boundable", &Xformable::Definition(), Abstract, kInheritExtent,
                    "What has an extent in its own space, computed from its attributes.",
                    ERNIE_BOUNDABLE_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Gprim, "Gprim", &Boundable::Definition(), Abstract, kInheritExtent,
                    "Geometry, drawn in a display colour or with a bound material.",
                    ERNIE_GPRIM_ATTRIBUTES, "primvars:")
// Grouping prims are where custom data lives, so they take attributes of any name.
ERNIE_DEFINE_SCHEMA(Xform, tokens::kXform, &Xformable::Definition(), Concrete, kDescendantsExtent,
                    "A transform grouping its children.", ERNIE_NO_ATTRIBUTES, "*")
ERNIE_DEFINE_SCHEMA(Scope, tokens::kScope, &Imageable::Definition(), Concrete, kDescendantsExtent,
                    "A grouping without a transform of its own.", ERNIE_NO_ATTRIBUTES, "*")
ERNIE_DEFINE_SCHEMA(Cube, tokens::kCube, &Gprim::Definition(), Concrete, ComputedExtent(CubeExtent),
                    "A box centred on its origin, size along each axis.", ERNIE_CUBE_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Sphere, tokens::kSphere, &Gprim::Definition(), Concrete,
                    ComputedExtent(SphereExtent), "A sphere centred on its origin.",
                    ERNIE_SPHERE_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Cylinder, tokens::kCylinder, &Gprim::Definition(), Concrete,
                    ComputedExtent(RoundExtent<false>), "A capped cylinder along its axis.",
                    ERNIE_CYLINDER_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Cone, tokens::kCone, &Gprim::Definition(), Concrete,
                    ComputedExtent(RoundExtent<false>),
                    "A capped cone along its axis, its apex at +height / 2.",
                    ERNIE_CYLINDER_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Capsule, tokens::kCapsule, &Gprim::Definition(), Concrete,
                    ComputedExtent(RoundExtent<true>),
                    "A cylinder with hemispherical caps along its axis.", ERNIE_CAPSULE_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Plane, tokens::kPlane, &Gprim::Definition(), Concrete,
                    ComputedExtent(PlaneExtent), "A one-sided rectangle facing along its axis.",
                    ERNIE_PLANE_ATTRIBUTES)

}  // namespace ernie
