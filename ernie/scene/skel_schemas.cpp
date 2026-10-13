// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/skel_schemas.h"

#include "ernie/scene/schema.h"

namespace ernie {

ERNIE_DEFINE_SCHEMA(SkelRoot, tokens::kSkelRoot,
                    &Boundable::Definition(), false,
                    "The root of a skinned character (UsdSkel).",
                    ERNIE_SKEL_ROOT_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(Skeleton, tokens::kSkeleton,
                    &Boundable::Definition(), false,
                    "Joints in a hierarchy with bind and rest "
                    "transforms, posed by an animation (UsdSkel).",
                    ERNIE_SKELETON_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(
    SkelAnimation, tokens::kSkelAnimation, &Imageable::Definition(),
    false,
    "Joint transforms and blend shape weights sampled "
    "over time (UsdSkel).",
    ERNIE_SKEL_ANIMATION_ATTRIBUTES)
ERNIE_DEFINE_SCHEMA(BlendShape, tokens::kBlendShape,
                    &Imageable::Definition(), false,
                    "Offsets of a mesh's points weighted by an "
                    "animation, with inbetween shapes (UsdSkel).",
                    ERNIE_BLEND_SHAPE_ATTRIBUTES)

std::vector<const PrimTypeDef*> SkelSchemaTypes() {
    return {&SkelRoot::Definition(), &Skeleton::Definition(),
            &SkelAnimation::Definition(), &BlendShape::Definition()};
}

}  // namespace ernie
