// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/prim_type.h"
#include "ernie/scene/tokens.h"

/// The bases every prim type derives from, as reflected schemas (`prim_type.h`): `Imageable`
/// (what every prim has: visibility and its batch world), `Boundable` (what has bounds: an
/// extent), `Gprim` (geometry: a display colour and a material), and the plain grouping prims
/// `Xform` and `Scope`. Transforms are xformOps, set and read through `xform.h`.
///
/// References:
/// - Pixar, UsdGeom (Imageable, Boundable, Xform, Scope, Gprim),
///   https://openusd.org/release/api/usd_geom_page_front.html

// clang-format off
#define ERNIE_IMAGEABLE_ATTRIBUTES(X)                                                              \
    X(std::string, Visibility, ::ernie::tokens::kVisibility, "inherited",                          \
      (::ernie::AttributeMeta{.doc = "\"invisible\" hides the prim and its descendants.",          \
                              .group = "Prim",                                                     \
                              .options = {::ernie::tokens::kInherited,                             \
                                          ::ernie::tokens::kInvisible}}))                          \
    X(int, World, ::ernie::tokens::kWorld, -1,                                                     \
      (::ernie::AttributeMeta{.doc = "The world the prim and its descendants are in, of a "        \
                                     "batch's environments; -1 shared by every world.",            \
                              .group = "Prim", .min = -1.0, .max = 1048576.0}))

#define ERNIE_BOUNDABLE_ATTRIBUTES(X)                                                              \
    X(std::vector<float>, Extent, ::ernie::tokens::kExtent, std::vector<float>{},                  \
      (::ernie::AttributeMeta{.doc = "The prim's local bounds as six numbers, (min x, y, z, max "  \
                                     "x, y, z); unset, computed from its geometry. A (2, 3) "      \
                                     "array written in its place is read too.",                    \
                              .group = "Geometry"}))

#define ERNIE_GPRIM_ATTRIBUTES(X)                                                                  \
    X(glm::dvec3, DisplayColor, ::ernie::tokens::kDisplayColor, glm::dvec3(0.5),                   \
      (::ernie::AttributeMeta{.label = "Display colour",                                           \
                              .doc = "Linear RGB of the whole prim; colours per vertex are "       \
                                     "written as an array instead.",                               \
                              .group = "Geometry", .color = true}))                                \
    X(::ernie::Path, MaterialBinding, ::ernie::tokens::kMaterialBinding, ::ernie::Path(),          \
      (::ernie::AttributeMeta{.label = "Material", .doc = "The Material prim it is drawn with.",   \
                              .group = "Geometry", .options = {::ernie::tokens::kMaterial}}))

#define ERNIE_BOUNDABLE_ALL(X) ERNIE_IMAGEABLE_ATTRIBUTES(X) ERNIE_BOUNDABLE_ATTRIBUTES(X)
#define ERNIE_GPRIM_ALL(X) ERNIE_BOUNDABLE_ALL(X) ERNIE_GPRIM_ATTRIBUTES(X)
// clang-format on

namespace ernie {

/// What every prim has: visibility and the world it is in.
class Imageable : public SchemaView {
    ERNIE_SCHEMA_BODY(Imageable, SchemaView, ERNIE_IMAGEABLE_ATTRIBUTES)
};

/// A transform grouping its children.
class Xform : public Imageable {
    ERNIE_SCHEMA_BODY(Xform, Imageable, ERNIE_IMAGEABLE_ATTRIBUTES)
    ERNIE_SCHEMA_DEFINE(Xform)
};

/// A grouping without a transform of its own.
class Scope : public Imageable {
    ERNIE_SCHEMA_BODY(Scope, Imageable, ERNIE_IMAGEABLE_ATTRIBUTES)
    ERNIE_SCHEMA_DEFINE(Scope)
};

/// What has local bounds: geometry and instancers, with an optional authored extent.
class Boundable : public Imageable {
    ERNIE_SCHEMA_BODY(Boundable, Imageable, ERNIE_BOUNDABLE_ALL)
};

/// Geometry: drawn in a display colour, or with a bound material.
class Gprim : public Boundable {
    ERNIE_SCHEMA_BODY(Gprim, Boundable, ERNIE_GPRIM_ALL)
};

/// This file's types' registry entries, bases first (`PrimTypes`).
std::vector<const PrimTypeDef*> BaseSchemaTypes();

}  // namespace ernie
