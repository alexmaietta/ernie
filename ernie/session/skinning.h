// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/geometry/skinning.h"
#include "ernie/scene/path.h"
#include "ernie/scene/scene.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

/// Skinned meshes resolved from the scene as UsdSkel defines them, in
/// two parts: a mesh's binding to its skeleton, which changes rarely
/// (`SkinBindingOf`), and its pose at a time (`SkinPoseOf`), from its
/// skeleton's (`SkeletonPoseOf`, shared by the skeleton's meshes).
/// Skinning on the CPU (`SkinnedPoints`) and on the GPU both take
/// them.
namespace ernie {

/// What binds a Mesh to its Skeleton.
struct SkinBinding {
    Path skeleton;
    SkinningMethod method = SkinningMethod::Linear;
    glm::mat4 geomBindTransform{1.0f};
    /// For each joint the influences index (the mesh's `skel:joints`
    /// order, else the skeleton's), its skeleton joint, or -1.
    std::vector<int> joints;
    std::vector<std::int32_t> jointIndices;
    std::vector<float> jointWeights;
    std::size_t perPoint = 0;
    bool constant = false;  ///< One set of influences for all points.

    /// A blend shape's primary shape or one of its inbetweens.
    struct SubShape {
        std::vector<float> offsets;
        std::vector<std::int32_t>
            pointIndices;  ///< Empty: every point.
    };
    std::vector<SubShape> subShapes;
    /// Each blend shape by the name animations weight it: its first
    /// sub-shape (its primary; its inbetweens follow) and the
    /// inbetweens' weights.
    struct Shape {
        std::string name;
        std::size_t first = 0;
        std::vector<float> inbetweenWeights;
    };
    std::vector<Shape> shapes;
};

/// The binding of the Mesh `mesh`, or none when it names no Skeleton
/// (`skel:skeleton`).
/// @throws std::invalid_argument for a skeleton, binding or blend
/// shape whose arrays disagree.
std::optional<SkinBinding> SkinBindingOf(const Scene& scene,
                                         const Path& mesh);

/// A Skeleton posed at a time.
struct SkeletonPose {
    std::vector<std::string> joints;
    /// Each joint's skinning transform (skeleton space).
    std::vector<glm::mat4> skinning;
    glm::dmat4 world{1.0};
    std::vector<std::string> blendShapes;  ///< The animation's names.
    std::vector<float> blendShapeWeights;
    /// Each of `blendShapes` by name, for the meshes' lookups.
    std::unordered_map<std::string, std::size_t> blendShapeIndex;
};

/// The Skeleton `skeleton` at `timeCode`: its animation's joints over
/// its rest pose (all of them without one).
/// @throws std::invalid_argument for arrays that disagree.
SkeletonPose SkeletonPoseOf(const Scene& scene, const Path& skeleton,
                            double timeCode);

/// A skinned mesh's pose at a time.
struct SkinPose {
    /// The skinning transform of each joint the influences index.
    std::vector<glm::mat4> skinning;
    /// From the skeleton's space to the mesh's: inverse(mesh world) *
    /// skeleton world.
    glm::dmat4 skelToMesh{1.0};
    /// Each sub-shape's weight (`SkinBinding::subShapes`).
    std::vector<float> subShapeWeights;
};

SkinPose SkinPoseOf(const Scene& scene, const Path& mesh,
                    const SkinBinding& binding,
                    const SkeletonPose& skeleton);

/// `SkinPoseOf` for a mesh whose world transform is `meshWorld`, as a
/// caller posing many meshes under one parent finds it once.
SkinPose SkinPoseOf(const glm::dmat4& meshWorld,
                    const SkinBinding& binding,
                    const SkeletonPose& skeleton);

/// `restPoints` (x, y, z each) moved by the weighted sub-shapes,
/// skinned, and brought back into the mesh's space.
std::vector<float> SkinnedPoints(const SkinBinding& binding,
                                 const SkinPose& pose,
                                 std::span<const float> restPoints);

}  // namespace ernie
