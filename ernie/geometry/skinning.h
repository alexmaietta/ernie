// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

/// Skeletal skinning and blend shapes on the CPU, as UsdSkel defines
/// them: the reference the GPU's skinning is checked against.
/// Matrices are column-vector (glm) ones.
///
/// References:
///   Pixar, UsdSkel,
///   https://openusd.org/release/api/_usd_skel__intro.html and
///   pxr/usd/usdSkel/utils.cpp (UsdSkelSkinPointsLBS, ...DQS). N.
///   Magnenat-Thalmann, R. Laperrière, D. Thalmann, "Joint-dependent
///   local deformations for hand animation and object grasping",
///   Graphics Interface '88, 26-33, 1988.
///   https://doi.org/10.20380/GI1988.04
///   L. Kavan, S. Collins, J. Žára, C. O'Sullivan, "Geometric
///   Skinning with Approximate Dual Quaternion Blending", ACM TOG
///   27(4), 105, 2008. https://doi.org/10.1145/1409625.1409627
namespace ernie {

/// Each joint's parent among `joints` (UsdSkel joint paths such as
/// "Hips/Spine"), -1 for a root: the joint at the path without its
/// last element.
/// @throws std::invalid_argument if a parent is missing or comes
/// after its child, or a joint is named twice.
std::vector<int> JointParents(std::span<const std::string> joints);

/// translate * rotate * scale: a joint's transform from its parent.
glm::mat4 JointTransform(const glm::vec3& translation,
                         const glm::quat& rotation,
                         const glm::vec3& scale);

/// Each joint's transform in its skeleton's space: its local
/// transform after its parent's.
std::vector<glm::mat4>
SkelSpaceTransforms(std::span<const int> parents,
                    std::span<const glm::mat4> locals);

/// The transforms skinning applies: each joint's skeleton-space
/// transform after the inverse of its bind transform.
/// @throws std::invalid_argument for spans of different sizes.
std::vector<glm::mat4>
SkinningTransforms(std::span<const glm::mat4> skelSpace,
                   std::span<const glm::mat4> bindTransforms);

/// A SkelAnimation's samples interpolated at a time: empty arrays for
/// those it lacks.
struct AnimationPose {
    std::vector<glm::vec3> translations;
    std::vector<glm::quat> rotations;
    std::vector<glm::vec3> scales;
    std::vector<float> blendShapeWeights;
};

/// The pose at `time` of samples at `times` (increasing; empty for a
/// single sample): linear in translations, scales and weights,
/// spherical in rotations, held before the first sample and after the
/// last. Arrays are (samples, joints, 3 or 4) and (samples, shapes);
/// rotations are quaternions x, y, z, w; any may be empty.
/// @throws std::invalid_argument for arrays of other sizes.
AnimationPose SampleAnimation(
    std::span<const double> times, std::size_t joints,
    std::size_t shapes, std::span<const float> translations,
    std::span<const float> rotations, std::span<const float> scales,
    std::span<const float> blendShapeWeights, double time);

/// For each of `from`'s names, its index in `to`, or -1.
std::vector<int> NameMapping(std::span<const std::string> from,
                             std::span<const std::string> to);

/// A blend shape's contributions at `weight`: (sub-shape, weight)
/// pairs, sub-shape 0 its primary shape (weight 1) and 1 + k its
/// inbetween k (weight `inbetweenWeights[k]`). The weight falls
/// between the two nearest sub-shapes, the null shape at 0 among
/// them, and beyond them extrapolates from the outermost two.
std::vector<std::pair<int, float>>
SubShapeWeights(float weight,
                std::span<const float> inbetweenWeights);

/// Adds `weight` times `offsets` (x, y, z each) to the points at
/// `pointIndices`, or to every point in order without them.
/// @throws std::invalid_argument for an index out of range or offsets
/// of the wrong size.
void ApplyBlendShape(std::span<float> points,
                     std::span<const float> offsets,
                     std::span<const int> pointIndices, float weight);

/// Scales each point's `perPoint` weights to sum to 1, or zeroes them
/// all when they sum to nearly 0.
void NormalizeWeights(std::span<float> weights, std::size_t perPoint);

enum class SkinningMethod { Linear, DualQuaternion };

/// A joint's transform as a unit dual quaternion (rotation, then
/// translation) after a 3 by 3 of its scale and shear.
struct JointDualQuaternion {
    glm::quat real{1.0f, 0.0f, 0.0f, 0.0f};
    glm::quat dual{0.0f, 0.0f, 0.0f, 0.0f};
    glm::mat3 scale{1.0f};
};

/// `transform` factored for dual quaternion skinning: its rotation
/// the orthonormal polar factor of its 3 by 3, the rest its scale.
///
/// Reference: N. J. Higham, "Computing the Polar Decomposition—with
/// Applications", SIAM J. Sci. Stat. Comput. 7(4), 1160-1174, 1986.
/// https://doi.org/10.1137/0907079
JointDualQuaternion FactorJoint(const glm::mat4& transform);

/// Skins `points` (x, y, z each, in the mesh's space) into the
/// skeleton's space: each through `geomBindTransform`, then blended
/// by its joints' `skinning` transforms. `jointIndices` and
/// `jointWeights` hold `perPoint` per point, or `perPoint` for all
/// points (`constant`). Dual quaternions blend scale linearly first,
/// then the rigid part signed against the point's heaviest joint.
/// @throws std::invalid_argument for a joint index out of range or
/// influences of the wrong size.
void SkinPoints(SkinningMethod method,
                const glm::mat4& geomBindTransform,
                std::span<const glm::mat4> skinning,
                std::span<const int> jointIndices,
                std::span<const float> jointWeights,
                std::size_t perPoint, bool constant,
                std::span<float> points);

}  // namespace ernie
