// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/session/skinning.h"

#include "ernie/scene/geometry_schemas.h"
#include "ernie/scene/skel_schemas.h"
#include "ernie/scene/tokens.h"
#include "ernie/util/tensor.h"

#include <stdexcept>
#include <string>

namespace ernie {
namespace {

/// The attribute's array as floats, of any shape; empty when unset.
std::vector<float> FloatsOf(const Scene& scene, const Path& path,
                            std::string_view attribute) {
    const Tensor* tensor = scene.Read(path, attribute);
    if (tensor == nullptr) {
        return {};
    }
    if (tensor->Location().kind != DeviceKind::Cpu) {
        throw std::invalid_argument(path.String() + " " +
                                    std::string(attribute) +
                                    " must be a CPU array");
    }
    const Tensor packed = tensor->Contiguous();
    const auto count = static_cast<std::size_t>(packed.Elements());
    if (packed.Type() == DType::Float32) {
        return {packed.As<float>(), packed.As<float>() + count};
    }
    if (packed.Type() == DType::Float64) {
        return {packed.As<double>(), packed.As<double>() + count};
    }
    throw std::invalid_argument(path.String() + " " +
                                std::string(attribute) +
                                " must be float32 or float64");
}

/// The attribute's array as ints; empty when unset.
std::vector<std::int32_t> IntsOf(const Scene& scene, const Path& path,
                                 std::string_view attribute) {
    const Tensor* tensor = scene.Read(path, attribute);
    if (tensor == nullptr) {
        return {};
    }
    if (tensor->Location().kind != DeviceKind::Cpu) {
        throw std::invalid_argument(path.String() + " " +
                                    std::string(attribute) +
                                    " must be a CPU array");
    }
    const Tensor packed = tensor->Contiguous();
    const auto count = static_cast<std::size_t>(packed.Elements());
    switch (packed.Type()) {
    case DType::Int32:
        return {packed.As<std::int32_t>(),
                packed.As<std::int32_t>() + count};
    case DType::Int64:
        return {packed.As<std::int64_t>(),
                packed.As<std::int64_t>() + count};
    default:
        throw std::invalid_argument(path.String() + " " +
                                    std::string(attribute) +
                                    " must be int32 or int64");
    }
}

/// (n, 4, 4) column-vector matrices stored row by row.
std::vector<glm::mat4> MatricesOf(const std::vector<float>& values) {
    std::vector<glm::mat4> matrices(values.size() / 16);
    for (std::size_t j = 0; j < matrices.size(); ++j) {
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                matrices[j][c][r] =
                    values[16 * j +
                           static_cast<std::size_t>(4 * r + c)];
            }
        }
    }
    return matrices;
}

bool IsSkeleton(const Scene& scene, const Path& path) {
    return !path.IsEmpty() && scene.Has(path) &&
           scene.TypeOf(path) == tokens::kSkeleton;
}

}  // namespace

std::optional<SkinBinding> SkinBindingOf(const Scene& scene,
                                         const Path& mesh) {
    const Mesh view(scene, mesh);
    SkinBinding binding;
    binding.skeleton = view.GetSkeleton();
    if (!IsSkeleton(scene, binding.skeleton)) {
        return std::nullopt;
    }
    binding.method =
        view.GetSkinningMethod() == tokens::kDualQuaternion
            ? SkinningMethod::DualQuaternion
            : SkinningMethod::Linear;
    binding.geomBindTransform =
        glm::mat4(view.GetGeomBindTransform());
    const std::vector<std::string> joints =
        Skeleton(scene, binding.skeleton).GetJoints();
    const std::vector<std::string> meshJoints = view.GetSkelJoints();
    if (meshJoints.empty()) {
        for (std::size_t j = 0; j < joints.size(); ++j) {
            binding.joints.push_back(static_cast<int>(j));
        }
    } else {
        binding.joints = NameMapping(meshJoints, joints);
    }
    binding.jointIndices =
        IntsOf(scene, mesh, tokens::kSkelJointIndices);
    binding.jointWeights =
        FloatsOf(scene, mesh, tokens::kSkelJointWeights);
    if (const Tensor* indices =
            scene.Read(mesh, tokens::kSkelJointIndices)) {
        binding.perPoint =
            indices->Rank() == 2
                ? static_cast<std::size_t>(indices->Shape()[1])
                : 1;
        binding.constant = indices->Shape()[0] == 1;
    }
    if (binding.jointWeights.size() != binding.jointIndices.size()) {
        throw std::invalid_argument(
            mesh.String() + " needs a joint weight per joint index");
    }
    for (const std::int32_t j : binding.jointIndices) {
        if (j < 0 ||
            static_cast<std::size_t>(j) >= binding.joints.size()) {
            throw std::invalid_argument(
                mesh.String() + " has a joint index out of range");
        }
    }

    const std::vector<std::string> shapes = view.GetSkelBlendShapes();
    const std::vector<Path> targets = view.GetBlendShapeTargets();
    for (std::size_t i = 0; i < shapes.size() && i < targets.size();
         ++i) {
        const Path& shape = targets[i];
        if (!scene.Has(shape)) {
            continue;
        }
        SkinBinding::Shape entry{
            shapes[i], binding.subShapes.size(),
            FloatsOf(scene, shape, tokens::kInbetweenWeights)};
        const std::vector<float> offsets =
            FloatsOf(scene, shape, tokens::kOffsets);
        const std::vector<std::int32_t> pointIndices =
            IntsOf(scene, shape, tokens::kPointIndices);
        if (!pointIndices.empty() &&
            offsets.size() != 3 * pointIndices.size()) {
            throw std::invalid_argument(
                shape.String() + " needs an offset per point index");
        }
        const std::vector<float> inbetweens =
            FloatsOf(scene, shape, tokens::kInbetweenOffsets);
        if (inbetweens.size() !=
            entry.inbetweenWeights.size() * offsets.size()) {
            throw std::invalid_argument(
                shape.String() + " needs offsets for each inbetween");
        }
        binding.subShapes.push_back({offsets, pointIndices});
        for (std::size_t k = 0; k < entry.inbetweenWeights.size();
             ++k) {
            const auto first =
                static_cast<std::ptrdiff_t>(k * offsets.size());
            binding.subShapes.push_back(
                {std::vector<float>(
                     inbetweens.begin() + first,
                     inbetweens.begin() + first +
                         static_cast<std::ptrdiff_t>(offsets.size())),
                 pointIndices});
        }
        binding.shapes.push_back(std::move(entry));
    }
    return binding;
}

SkeletonPose SkeletonPoseOf(const Scene& scene,
                            const Path& skeletonPath,
                            double timeCode) {
    const Skeleton skeleton(scene, skeletonPath);
    SkeletonPose pose;
    pose.joints = skeleton.GetJoints();
    const std::vector<int> parents = JointParents(pose.joints);
    const std::vector<glm::mat4> binds = MatricesOf(
        FloatsOf(scene, skeletonPath, tokens::kBindTransforms));
    if (binds.size() != pose.joints.size()) {
        throw std::invalid_argument(
            skeletonPath.String() +
            " needs a bind transform per joint");
    }
    // The rest pose, or without one the bind pose taken apart.
    std::vector<glm::mat4> locals = MatricesOf(
        FloatsOf(scene, skeletonPath, tokens::kRestTransforms));
    if (locals.size() != pose.joints.size()) {
        locals.resize(pose.joints.size());
        for (std::size_t j = 0; j < pose.joints.size(); ++j) {
            locals[j] =
                parents[j] < 0
                    ? binds[j]
                    : glm::inverse(binds[static_cast<std::size_t>(
                          parents[j])]) *
                          binds[j];
        }
    }
    const Path animPath = skeleton.GetAnimationSource();
    if (!animPath.IsEmpty() && scene.Has(animPath) &&
        scene.TypeOf(animPath) == tokens::kSkelAnimation) {
        const SkelAnimation anim(scene, animPath);
        const std::vector<std::string> animJoints = anim.GetJoints();
        pose.blendShapes = anim.GetBlendShapes();
        const std::vector<float> times32 =
            FloatsOf(scene, animPath, tokens::kAnimationTimes);
        const std::vector<double> times(times32.begin(),
                                        times32.end());
        const AnimationPose sample = SampleAnimation(
            times, animJoints.size(), pose.blendShapes.size(),
            FloatsOf(scene, animPath, tokens::kTranslations),
            FloatsOf(scene, animPath, tokens::kRotations),
            FloatsOf(scene, animPath, tokens::kScales),
            FloatsOf(scene, animPath, tokens::kBlendShapeWeights),
            timeCode);
        if (!sample.translations.empty() &&
            !sample.rotations.empty() && !sample.scales.empty()) {
            const std::vector<int> into =
                NameMapping(animJoints, pose.joints);
            for (std::size_t a = 0; a < animJoints.size(); ++a) {
                if (into[a] >= 0) {
                    locals[static_cast<std::size_t>(into[a])] =
                        JointTransform(sample.translations[a],
                                       sample.rotations[a],
                                       sample.scales[a]);
                }
            }
        }
        pose.blendShapeWeights = sample.blendShapeWeights;
        pose.blendShapeWeights.resize(pose.blendShapes.size(), 0.0f);
    }
    pose.skinning = SkinningTransforms(
        SkelSpaceTransforms(parents, locals), binds);
    pose.world = scene.WorldTransform(skeletonPath);
    return pose;
}

SkinPose SkinPoseOf(const Scene& scene, const Path& mesh,
                    const SkinBinding& binding,
                    const SkeletonPose& skeleton) {
    SkinPose pose;
    for (const int j : binding.joints) {
        pose.skinning.push_back(
            j < 0 || static_cast<std::size_t>(j) >=
                         skeleton.skinning.size()
                ? glm::mat4(1.0f)
                : skeleton.skinning[static_cast<std::size_t>(j)]);
    }
    pose.skelToMesh =
        glm::inverse(scene.WorldTransform(mesh)) * skeleton.world;
    pose.subShapeWeights.assign(binding.subShapes.size(), 0.0f);
    const std::vector<std::string> names = [&] {
        std::vector<std::string> out;
        for (const SkinBinding::Shape& shape : binding.shapes) {
            out.push_back(shape.name);
        }
        return out;
    }();
    const std::vector<int> weightOf =
        NameMapping(names, skeleton.blendShapes);
    for (std::size_t i = 0; i < binding.shapes.size(); ++i) {
        const int at = weightOf[i];
        const float weight =
            at < 0
                ? 0.0f
                : skeleton.blendShapeWeights[static_cast<std::size_t>(
                      at)];
        if (weight == 0.0f) {
            continue;
        }
        const SkinBinding::Shape& shape = binding.shapes[i];
        for (const auto& [sub, subWeight] :
             SubShapeWeights(weight, shape.inbetweenWeights)) {
            pose.subShapeWeights[shape.first +
                                 static_cast<std::size_t>(sub)] =
                subWeight;
        }
    }
    return pose;
}

std::vector<float> SkinnedPoints(const SkinBinding& binding,
                                 const SkinPose& pose,
                                 std::span<const float> restPoints) {
    std::vector<float> points(restPoints.begin(), restPoints.end());
    for (std::size_t s = 0; s < binding.subShapes.size(); ++s) {
        if (pose.subShapeWeights[s] != 0.0f) {
            ApplyBlendShape(points, binding.subShapes[s].offsets,
                            binding.subShapes[s].pointIndices,
                            pose.subShapeWeights[s]);
        }
    }
    if (binding.perPoint != 0) {
        SkinPoints(binding.method, binding.geomBindTransform,
                   pose.skinning, binding.jointIndices,
                   binding.jointWeights, binding.perPoint,
                   binding.constant, points);
    }
    for (std::size_t p = 0; p + 2 < points.size(); p += 3) {
        const glm::dvec3 at(
            pose.skelToMesh *
            glm::dvec4(points[p], points[p + 1], points[p + 2], 1.0));
        points[p] = static_cast<float>(at.x);
        points[p + 1] = static_cast<float>(at.y);
        points[p + 2] = static_cast<float>(at.z);
    }
    return points;
}

}  // namespace ernie
