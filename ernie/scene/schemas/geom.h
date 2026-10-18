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
#include <span>
#include <string>
#include <string_view>
#include <vector>

// clang-format off
#define ERNIE_IMAGEABLE_ATTRIBUTES(X)                                                              \
    X(std::string, Visibility, "visibility", "inherited",                                          \
      "\"invisible\" hides the prim and its descendants.",                                         \
      .group = "Prim", .options = {tokens::kInherited, tokens::kInvisible},                        \
      .dirty = Dirty::Visibility, .inherited = true)                                               \
    X(std::string, Purpose, "purpose", "default",                                                  \
      "Which renders include it and its descendants: default, render, proxy or guide.",            \
      .group = "Prim",                                                                             \
      .options = {tokens::kDefault, tokens::kRender, tokens::kProxy, tokens::kGuide},              \
      .dirty = Dirty::Visibility, .inherited = true)                                               \
    X(int, World, "ernie:world", -1,                                                               \
      "The batch world the prim and its descendants are in; -1 for every world.",                  \
      .group = "Prim", .lower = AtLeast(-1.0), .dirty = Dirty::Visibility, .inherited = true)

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

#define ERNIE_CARD_TEXTURE(X, Name, usdName)                                                       \
    X(std::string, Name, usdName, "", "The image on the card facing this way.", .group = "Model",  \
      .asset = true, .dirty = Dirty::Params)

#define ERNIE_MODEL_API_ATTRIBUTES(X)                                                              \
    X(std::string, DrawMode, "model:drawMode", "inherited",                                        \
      "Draws the subtree as itself, or as a stand-in: its origin axes, its bounds or cards.",      \
      .group = "Model", .options = {"origin", "bounds", "cards", "default", "inherited"},          \
      .dirty = Dirty::Visibility)                                                                  \
    X(bool, ApplyDrawMode, "model:applyDrawMode", false,                                           \
      "Whether this prim's draw mode replaces its subtree.", .group = "Model",                     \
      .dirty = Dirty::Visibility)                                                                  \
    X(glm::dvec3, DrawModeColor, "model:drawModeColor", glm::dvec3(0.18),                          \
      "Linear RGB of the stand-in.", .group = "Model", .lower = AtLeast(0.0), .color = true,       \
      .dirty = Dirty::Params)                                                                      \
    X(std::string, CardGeometry, "model:cardGeometry", "cross",                                    \
      "Cards through the centre, on the bounds' faces, or as the textures place them.",            \
      .group = "Model", .options = {"cross", "box", "fromTexture"}, .dirty = Dirty::Params)        \
    X(std::string, CardVisibility, "model:cardVisibility", "inherited",                            \
      "Every textured card, or only one per axis.", .group = "Model",                              \
      .options = {"full", "simple", "inherited"}, .dirty = Dirty::Params)                          \
    ERNIE_CARD_TEXTURE(X, CardTextureXPos, "model:cardTextureXPos")                                \
    ERNIE_CARD_TEXTURE(X, CardTextureYPos, "model:cardTextureYPos")                                \
    ERNIE_CARD_TEXTURE(X, CardTextureZPos, "model:cardTextureZPos")                                \
    ERNIE_CARD_TEXTURE(X, CardTextureXNeg, "model:cardTextureXNeg")                                \
    ERNIE_CARD_TEXTURE(X, CardTextureYNeg, "model:cardTextureYNeg")                                \
    ERNIE_CARD_TEXTURE(X, CardTextureZNeg, "model:cardTextureZNeg")                                \
    X(std::vector<float>, ExtentsHint, "extentsHint", std::vector<float>{},                        \
      "The subtree's local bounds per purpose (default, render, proxy, guide), six floats each, "  \
      "read while it isn't loaded.", .group = "Model")

#define ERNIE_SUBSET_ATTRIBUTES(X)                                                                 \
    X(std::string, ElementType, "elementType", "face", "What its indices count.",                  \
      .group = "Subset", .options = {"face", "point", "edge", "segment", "tetrahedron"},           \
      .dirty = Dirty::Topology)                                                                    \
    X(Tensor, Indices, "indices", Tensor(), "The elements in the subset.", .group = "Subset",      \
      .integer = true, .dirty = Dirty::Topology)                                                   \
    X(std::string, FamilyName, "familyName", "",                                                   \
      "The family of subsets it is one of, as materialBind for per-face materials.",               \
      .group = "Subset", .dirty = Dirty::Topology)

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
inline Box AuthoredExtent(std::span<const float> extent) {
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
ERNIE_API_SCHEMA(GeomModelAPI, ERNIE_MODEL_API_ATTRIBUTES,
                 "A model's stand-ins and the bounds hint used while its subtree isn't loaded.");
ERNIE_CONCRETE_SCHEMA(GeomSubset, SchemaView, ERNIE_SUBSET_ATTRIBUTES,
                      "Some faces, points, edges, segments or tetrahedra of its parent, to bind a "
                      "material to or select.");
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
