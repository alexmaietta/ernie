// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/base_schemas.h"
#include "ernie/scene/prim_type.h"
#include "ernie/scene/tokens.h"

#include <string>
#include <vector>

/// Skeletal animation as reflected schemas (`prim_type.h`), after
/// UsdSkel: a Skeleton's joints, a SkelAnimation sampling them over
/// time, and BlendShapes. Meshes bind to a Skeleton with their
/// `skel:*` attributes (`geometry_schemas.h`). Matrices in tensors
/// are in Ernie's convention (column vectors, translation in the last
/// column), each stored row by row.
///
/// Reference: Pixar, UsdSkel,
/// https://openusd.org/release/api/usd_skel_page_front.html

// clang-format off
#define ERNIE_SKEL_ROOT_ATTRIBUTES(X)

#define ERNIE_SKELETON_ATTRIBUTES(X)                                                               \
    X(std::vector<std::string>, Joints, ::ernie::tokens::kJoints, std::vector<std::string>{},      \
      (::ernie::AttributeMeta{.doc = "The joints, each a path (\"Hips/Spine\") whose parent is "   \
                                     "the path without its last element; parents come first.",     \
                              .group = "Skeleton", .dirty = ::ernie::Dirty::Topology}))            \
    X(::ernie::Tensor, BindTransforms, ::ernie::tokens::kBindTransforms, nullptr,                  \
      (::ernie::AttributeMeta{.doc = "(joints, 4, 4) each joint's skeleton-space transform when "  \
                                     "the meshes were bound.",                                     \
                              .group = "Skeleton", .dirty = ::ernie::Dirty::Topology}))            \
    X(::ernie::Tensor, RestTransforms, ::ernie::tokens::kRestTransforms, nullptr,                  \
      (::ernie::AttributeMeta{.doc = "(joints, 4, 4) each joint's transform relative to its "      \
                                     "parent with no animation.",                                  \
                              .group = "Skeleton", .dirty = ::ernie::Dirty::Points}))              \
    X(::ernie::Path, AnimationSource, ::ernie::tokens::kSkelAnimationSource, ::ernie::Path(),      \
      (::ernie::AttributeMeta{.label = "Animation",                                                \
                              .doc = "The SkelAnimation posing its joints; unset, the rest pose.", \
                              .group = "Skeleton", .options = {::ernie::tokens::kSkelAnimation},   \
                              .dirty = ::ernie::Dirty::Points,                                     \
                              .targetDirty = ::ernie::Dirty::Points}))

#define ERNIE_SKEL_ANIMATION_ATTRIBUTES(X)                                                         \
    X(std::vector<std::string>, Joints, ::ernie::tokens::kJoints, std::vector<std::string>{},      \
      (::ernie::AttributeMeta{.doc = "The joints it animates, by the Skeleton's names, in the "    \
                                     "order of its arrays.",                                       \
                              .group = "Animation", .dirty = ::ernie::Dirty::Topology}))           \
    X(::ernie::Tensor, Times, ::ernie::tokens::kAnimationTimes, nullptr,                           \
      (::ernie::AttributeMeta{.doc = "(samples,) increasing time codes of the samples; unset, "    \
                                     "one sample for all time.",                                   \
                              .group = "Animation", .dirty = ::ernie::Dirty::Points}))             \
    X(::ernie::Tensor, Translations, ::ernie::tokens::kTranslations, nullptr,                      \
      (::ernie::AttributeMeta{.doc = "(samples, joints, 3) each joint's translation from its "     \
                                     "parent.",                                                    \
                              .group = "Animation", .dirty = ::ernie::Dirty::Points}))             \
    X(::ernie::Tensor, Rotations, ::ernie::tokens::kRotations, nullptr,                            \
      (::ernie::AttributeMeta{.doc = "(samples, joints, 4) each joint's rotation from its "        \
                                     "parent, quaternions x, y, z, w.",                            \
                              .group = "Animation", .dirty = ::ernie::Dirty::Points}))             \
    X(::ernie::Tensor, Scales, ::ernie::tokens::kScales, nullptr,                                  \
      (::ernie::AttributeMeta{.doc = "(samples, joints, 3) each joint's scale.",                   \
                              .group = "Animation", .dirty = ::ernie::Dirty::Points}))             \
    X(std::vector<std::string>, BlendShapes, ::ernie::tokens::kBlendShapes,                        \
      std::vector<std::string>{},                                                                  \
      (::ernie::AttributeMeta{.doc = "The blend shapes it weights, by the meshes' names for "      \
                                     "them.",                                                      \
                              .group = "Animation", .dirty = ::ernie::Dirty::Topology}))           \
    X(::ernie::Tensor, BlendShapeWeights, ::ernie::tokens::kBlendShapeWeights, nullptr,            \
      (::ernie::AttributeMeta{.doc = "(samples, blend shapes) weights.", .group = "Animation",     \
                              .dirty = ::ernie::Dirty::Points}))

#define ERNIE_BLEND_SHAPE_ATTRIBUTES(X)                                                            \
    X(::ernie::Tensor, Offsets, ::ernie::tokens::kOffsets, nullptr,                                \
      (::ernie::AttributeMeta{.doc = "(n, 3) offsets of the mesh's points at weight 1.",           \
                              .group = "Blend shape", .dirty = ::ernie::Dirty::Points}))           \
    X(::ernie::Tensor, NormalOffsets, ::ernie::tokens::kNormalOffsets, nullptr,                    \
      (::ernie::AttributeMeta{.doc = "(n, 3) offsets of the mesh's normals at weight 1.",          \
                              .group = "Blend shape", .dirty = ::ernie::Dirty::Points}))           \
    X(::ernie::Tensor, PointIndices, ::ernie::tokens::kPointIndices, nullptr,                      \
      (::ernie::AttributeMeta{.doc = "(n,) the points the offsets move; unset, every point in "    \
                                     "order.",                                                     \
                              .group = "Blend shape", .integer = true,                             \
                              .dirty = ::ernie::Dirty::Points}))                                   \
    X(::ernie::Tensor, InbetweenWeights, ::ernie::tokens::kInbetweenWeights, nullptr,              \
      (::ernie::AttributeMeta{.doc = "(k,) the weights of its inbetween shapes.",                  \
                              .group = "Blend shape", .dirty = ::ernie::Dirty::Points}))           \
    X(::ernie::Tensor, InbetweenOffsets, ::ernie::tokens::kInbetweenOffsets, nullptr,              \
      (::ernie::AttributeMeta{.doc = "(k, n, 3) the inbetween shapes' offsets, at its points.",    \
                              .group = "Blend shape", .dirty = ::ernie::Dirty::Points}))           \
    X(::ernie::Tensor, InbetweenNormalOffsets, ::ernie::tokens::kInbetweenNormalOffsets, nullptr,  \
      (::ernie::AttributeMeta{.doc = "(k, n, 3) the inbetween shapes' normal offsets.",            \
                              .group = "Blend shape", .dirty = ::ernie::Dirty::Points}))

#define ERNIE_SKEL_ROOT_ALL(X) ERNIE_BOUNDABLE_ALL(X) ERNIE_SKEL_ROOT_ATTRIBUTES(X)
#define ERNIE_SKELETON_ALL(X) ERNIE_BOUNDABLE_ALL(X) ERNIE_SKELETON_ATTRIBUTES(X)
#define ERNIE_SKEL_ANIMATION_ALL(X) ERNIE_IMAGEABLE_ATTRIBUTES(X) ERNIE_SKEL_ANIMATION_ATTRIBUTES(X)
#define ERNIE_BLEND_SHAPE_ALL(X) ERNIE_IMAGEABLE_ATTRIBUTES(X) ERNIE_BLEND_SHAPE_ATTRIBUTES(X)
// clang-format on

namespace ernie {

/// The root of a skinned character: the meshes and skeletons under it
/// go together.
class SkelRoot : public Boundable {
    ERNIE_SCHEMA_BODY(SkelRoot, Boundable, ERNIE_SKEL_ROOT_ALL)
    ERNIE_SCHEMA_DEFINE(SkelRoot)
};

/// Joints in a hierarchy, with the transforms their meshes were bound
/// at and their rest pose, posed by an animation source.
class Skeleton : public Boundable {
    ERNIE_SCHEMA_BODY(Skeleton, Boundable, ERNIE_SKELETON_ALL)
    ERNIE_SCHEMA_DEFINE(Skeleton)
};

/// Joint transforms and blend shape weights sampled over time.
class SkelAnimation : public Imageable {
    ERNIE_SCHEMA_BODY(SkelAnimation, Imageable,
                      ERNIE_SKEL_ANIMATION_ALL)
    ERNIE_SCHEMA_DEFINE(SkelAnimation)
};

/// Offsets of a mesh's points (and normals), weighted by an
/// animation, with inbetween shapes at weights between 0 and 1.
class BlendShape : public Imageable {
    ERNIE_SCHEMA_BODY(BlendShape, Imageable, ERNIE_BLEND_SHAPE_ALL)
    ERNIE_SCHEMA_DEFINE(BlendShape)
};

/// This file's types' registry entries, bases first (`PrimTypes`).
std::vector<const PrimTypeDef*> SkelSchemaTypes();

}  // namespace ernie
