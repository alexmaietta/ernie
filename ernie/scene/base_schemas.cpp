// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/base_schemas.h"

namespace ernie {

// For types that declare no attributes of their own.
#define ERNIE_NO_ATTRIBUTES(X)

ERNIE_DEFINE_SCHEMA(Imageable, "Imageable", nullptr, true,
                    "What every prim has: visibility and the world it is in.",
                    ERNIE_IMAGEABLE_ATTRIBUTES)
// Grouping prims are where custom data lives, so they take attributes of any name.
ERNIE_DEFINE_OPEN_SCHEMA(Xform, tokens::kXform, &Imageable::Definition(), false,
                         "A transform grouping its children; takes custom attributes of any name.",
                         ERNIE_NO_ATTRIBUTES, "*")
ERNIE_DEFINE_OPEN_SCHEMA(Scope, tokens::kScope, &Imageable::Definition(), false,
                         "A grouping without a transform of its own; takes custom attributes of "
                         "any name.",
                         ERNIE_NO_ATTRIBUTES, "*")
ERNIE_DEFINE_SCHEMA(Boundable, "Boundable", &Imageable::Definition(), true,
                    "What has local bounds: geometry and instancers, with an optional authored "
                    "extent.",
                    ERNIE_BOUNDABLE_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Gprim, "Gprim", &Boundable::Definition(), true,
                    "Geometry: drawn in a display colour, or with a bound material.",
                    ERNIE_GPRIM_ATTRIBUTES)

std::vector<const PrimTypeDef*> BaseSchemaTypes() {
    return {&Imageable::Definition(), &Xform::Definition(), &Scope::Definition(),
            &Boundable::Definition(), &Gprim::Definition()};
}

}  // namespace ernie
