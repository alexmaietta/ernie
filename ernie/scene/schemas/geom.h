// SPDX-License-Identifier: BSD-3-Clause
/// @file geom.h
/// @brief UsdGeom's base schemas, groupings and implicit shapes.
///
/// References:
/// - Pixar, UsdGeom, https://openusd.org/release/api/usd_geom_page_front.html
#pragma once

#include "ernie/scene/tokens.h"
#include "ernie/scene/view.h"

#include <array>
#include <string>
#include <string_view>
#include <vector>

// clang-format off
#define ERNIE_IMAGEABLE_ATTRIBUTES(X)                                                              \
    X(std::string, Visibility, "visibility", "inherited",                                          \
      "\"invisible\" hides the prim and its descendants.",                                         \
      .group = "Prim", .options = {tokens::kInherited, tokens::kInvisible},                        \
      .dirty = Dirty::Visibility)                                                                  \
    X(std::string, Purpose, "purpose", "default",                                                  \
      "Which renders include it: default, render, proxy or guide; inherited when default.",        \
      .group = "Prim",                                                                             \
      .options = {tokens::kDefault, tokens::kRender, tokens::kProxy, tokens::kGuide},              \
      .dirty = Dirty::Visibility)                                                                  \
    X(int, World, "ernie:world", -1,                                                               \
      "The batch world the prim and its descendants are in; -1 for every world.",                  \
      .group = "Prim", .lower = AtLeast(-1.0), .inherited = true)

#define ERNIE_XFORMABLE_ATTRIBUTES(X)                                                              \
    X(std::vector<std::string>, XformOpOrder, "xformOpOrder", std::vector<std::string>{},         \
      "The xformOp attributes composing the local transform, outermost first.",                    \
      .group = "Transform", .dirty = Dirty::Transform)

#define ERNIE_BOUNDABLE_ATTRIBUTES(X)                                                              \
    X(std::vector<float>, Extent, "extent", std::vector<float>{},                                  \
      "Authored local bounds (min x, y, z, max x, y, z), read until the geometry is available.",   \
      .group = "Geometry")

#define ERNIE_GPRIM_ATTRIBUTES(X)                                                                  \
    X(glm::dvec3, DisplayColor, "primvars:displayColor", glm::dvec3(0.5),                          \
      "Linear RGB of the whole prim.", .label = "Display colour", .group = "Geometry",              \
      .lower = AtLeast(0.0), .finite = true, .color = true, .dirty = Dirty::Primvars)              \
    X(double, DisplayOpacity, "primvars:displayOpacity", 1.0, "Opacity of the whole prim.",        \
      .group = "Geometry", .lower = AtLeast(0.0), .upper = AtMost(1.0), .dirty = Dirty::Primvars)  \
    X(bool, DoubleSided, "doubleSided", false, "Whether its back faces are seen too.",             \
      .group = "Geometry")                                                                         \
    X(std::string, Orientation, "orientation", "rightHanded",                                      \
      "The winding of its front faces.", .group = "Geometry",                                      \
      .options = {"rightHanded", "leftHanded"}, .dirty = Dirty::Topology)

#define ERNIE_SHAPE_LENGTH .group = "Shape", .lower = AtLeast(0.0), .finite = true

#define ERNIE_AXIS_ATTRIBUTE(X)                                                                    \
    X(Choice<Axis>, Axis, "axis", Axis::Z, "The axis of its height, or a plane's normal.",         \
      .group = "Shape")

#define ERNIE_CUBE_ATTRIBUTES(X)                                                                   \
    X(double, Size, "size", 2.0, "Edge length, metres.", ERNIE_SHAPE_LENGTH)

#define ERNIE_SPHERE_ATTRIBUTES(X)                                                                 \
    X(double, Radius, "radius", 1.0, "Metres.", ERNIE_SHAPE_LENGTH)

#define ERNIE_CYLINDER_ATTRIBUTES(X)                                                               \
    X(double, Radius, "radius", 1.0, "Metres.", ERNIE_SHAPE_LENGTH)                                \
    X(double, Height, "height", 2.0, "Metres along the axis.", ERNIE_SHAPE_LENGTH)                 \
    ERNIE_AXIS_ATTRIBUTE(X)

#define ERNIE_CAPSULE_ATTRIBUTES(X)                                                                \
    X(double, Radius, "radius", 0.5, "Metres, of the cylinder and its end caps.",                  \
      ERNIE_SHAPE_LENGTH)                                                                          \
    X(double, Height, "height", 1.0, "Metres along the axis between the caps.",                    \
      ERNIE_SHAPE_LENGTH)                                                                          \
    ERNIE_AXIS_ATTRIBUTE(X)

#define ERNIE_PLANE_ATTRIBUTES(X)                                                                  \
    X(double, Width, "width", 2.0, "Metres along x, or z when the axis is X.", ERNIE_SHAPE_LENGTH) \
    X(double, Length, "length", 2.0, "Metres along y, or z when the axis is Y.",                   \
      ERNIE_SHAPE_LENGTH)                                                                          \
    ERNIE_AXIS_ATTRIBUTE(X)
// clang-format on

namespace ernie {

/// @return A Boundable's authored `extent` (min x, y, z, max x, y, z), which extent functions
/// read while the data they bound isn't on the host; empty when unset.
inline Box AuthoredExtent(const std::vector<float>& extent) {
    if (extent.size() != 6) {
        return {};
    }
    return {{extent[0], extent[1], extent[2]}, {extent[3], extent[4], extent[5]}};
}

enum class Axis { X, Y, Z };

template <> struct ChoiceTokens<Axis> {
    static constexpr std::array<std::string_view, 3> kTokens{"X", "Y", "Z"};
};

ERNIE_SCHEMA(Imageable, SchemaView, ERNIE_IMAGEABLE_ATTRIBUTES,
             "What every prim has: visibility, purpose and the world it is in.");
ERNIE_SCHEMA(Xformable, Imageable, ERNIE_XFORMABLE_ATTRIBUTES,
             "What has a transform, composed from its xformOps.");
ERNIE_SCHEMA(Boundable, Xformable, ERNIE_BOUNDABLE_ATTRIBUTES,
             "What has an extent in its own space, computed from its attributes.");
ERNIE_SCHEMA(Gprim, Boundable, ERNIE_GPRIM_ATTRIBUTES,
             "Geometry, drawn in a display colour or with a bound material.");
ERNIE_CONCRETE_SCHEMA(Xform, Xformable, ERNIE_NO_ATTRIBUTES, "A transform grouping its children.");
ERNIE_CONCRETE_SCHEMA(Scope, Imageable, ERNIE_NO_ATTRIBUTES,
                      "A grouping without a transform of its own.");
ERNIE_CONCRETE_SCHEMA(Cube, Gprim, ERNIE_CUBE_ATTRIBUTES,
                      "A box centred on its origin, size along each axis.");
ERNIE_CONCRETE_SCHEMA(Sphere, Gprim, ERNIE_SPHERE_ATTRIBUTES, "A sphere centred on its origin.");
ERNIE_CONCRETE_SCHEMA(Cylinder, Gprim, ERNIE_CYLINDER_ATTRIBUTES,
                      "A capped cylinder along its axis.");
ERNIE_CONCRETE_SCHEMA(Cone, Gprim, ERNIE_CYLINDER_ATTRIBUTES,
                      "A capped cone along its axis, its apex at +height / 2.");
ERNIE_CONCRETE_SCHEMA(Capsule, Gprim, ERNIE_CAPSULE_ATTRIBUTES,
                      "A cylinder with hemispherical caps along its axis.");
ERNIE_CONCRETE_SCHEMA(Plane, Gprim, ERNIE_PLANE_ATTRIBUTES,
                      "A one-sided rectangle facing along its axis.");

}  // namespace ernie
