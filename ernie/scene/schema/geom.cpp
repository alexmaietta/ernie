// SPDX-License-Identifier: BSD-3-Clause
/// @file geom.cpp
/// @brief UsdGeom's registry entries and the implicit shapes' extents, as UsdGeom computes them.
#include "ernie/scene/schema/geom.h"

#include "ernie/scene/convert.h"

namespace ernie {

std::int32_t ReadWorld(const Scene& scene, PrimId prim) {
    const std::optional<Value> value = scene.GetResolved(prim, Imageable::kWorld);
    const std::optional<double> number = value ? ToNumber(*value) : std::nullopt;
    return number ? static_cast<std::int32_t>(*number) : -1;
}

namespace {

/// @return The axis index (0, 1, 2) of an `axis` token.
int AxisIndex(const ExtentRows& rows, Token axis) {
    return rows.Text(axis).front() - 'X';
}

auto CubeHalves(const ExtentRows& rows) {
    return [size = rows.Column<double>(Cube::kSize)](std::uint32_t row) {
        return glm::dvec3(0.5 * size[row]);
    };
}

auto SphereHalves(const ExtentRows& rows) {
    return [radius = rows.Column<double>(Sphere::kRadius)](std::uint32_t row) {
        return glm::dvec3(radius[row]);
    };
}

/// Cylinders, cones and capsules: a radius about the axis; capsules' caps add it along the axis.
template <bool kCapped> auto RoundHalves(const ExtentRows& rows) {
    return [&rows, radius = rows.Column<double>(Cylinder::kRadius),
            height = rows.Column<double>(Cylinder::kHeight),
            axis = rows.Column<Token>(Cylinder::kAxis)](std::uint32_t row) {
        glm::dvec3 half(radius[row]);
        half[AxisIndex(rows, axis[row])] = 0.5 * height[row] + (kCapped ? radius[row] : 0.0);
        return half;
    };
}

auto PlaneHalves(const ExtentRows& rows) {
    return [&rows, width = rows.Column<double>(Plane::kWidth),
            length = rows.Column<double>(Plane::kLength),
            axis = rows.Column<Token>(Plane::kAxis)](std::uint32_t row) {
        const int normal = AxisIndex(rows, axis[row]);
        // Width lies along x, or z for an X normal; length along y, or z for a Y normal.
        glm::dvec3 half(0.0);
        half[normal == 0 ? 2 : 0] = 0.5 * width[row];
        half[normal == 1 ? 2 : 1] = 0.5 * length[row];
        return half;
    };
}

}  // namespace

ERNIE_DEFINE_SCHEMA(Imageable, Abstract, kInheritExtent, ERNIE_IMAGEABLE_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Xformable, Abstract, kInheritExtent, ERNIE_XFORMABLE_ATTRIBUTES,
                    .openNamespaces = {tokens::kXformOpPrefix})
ERNIE_DEFINE_SCHEMA(Boundable, Abstract, kInheritExtent, ERNIE_BOUNDABLE_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Gprim, Abstract, kInheritExtent, ERNIE_GPRIM_ATTRIBUTES,
                    .openNamespaces = {tokens::kPrimvarsPrefix})
// Grouping prims are where custom data lives, so they take attributes of any name.
ERNIE_DEFINE_SCHEMA(Xform, Concrete, kDescendantsExtent, ERNIE_NO_ATTRIBUTES,
                    .openNamespaces = {"*"})
ERNIE_DEFINE_SCHEMA(Scope, Concrete, kDescendantsExtent, ERNIE_NO_ATTRIBUTES,
                    .openNamespaces = {"*"})
ERNIE_DEFINE_SCHEMA(GeomModelAPI, SingleApply, kInheritExtent, ERNIE_MODEL_API_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(GeomSubset, Concrete, kNoExtent, ERNIE_SUBSET_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Cube, Concrete, ComputedExtent(CentredExtent<CubeHalves>),
                    ERNIE_CUBE_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Sphere, Concrete, ComputedExtent(CentredExtent<SphereHalves>),
                    ERNIE_SPHERE_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Cylinder, Concrete, ComputedExtent(CentredExtent<RoundHalves<false>>),
                    ERNIE_CYLINDER_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Cone, Concrete, ComputedExtent(CentredExtent<RoundHalves<false>>),
                    ERNIE_CYLINDER_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Capsule, Concrete, ComputedExtent(CentredExtent<RoundHalves<true>>),
                    ERNIE_CAPSULE_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Plane, Concrete, ComputedExtent(CentredExtent<PlaneHalves>),
                    ERNIE_PLANE_ATTRIBUTES)

}  // namespace ernie
