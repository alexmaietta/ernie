// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/geometry/skinning.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace ernie {
namespace {

constexpr float kSubShapeEpsilon = 1e-6f;

/// `count` floats from `values` at sample `sample`, or none.
std::span<const float> SampleOf(std::span<const float> values,
                                std::size_t sample,
                                std::size_t count) {
    return values.empty() ? values
                          : values.subspan(sample * count, count);
}

void CheckSamples(std::span<const float> values, std::size_t samples,
                  std::size_t count, const char* what) {
    if (!values.empty() && values.size() != samples * count) {
        throw std::invalid_argument(
            std::string(what) +
            " need one value per sample and joint "
            "or shape");
    }
}

}  // namespace

std::vector<int> JointParents(std::span<const std::string> joints) {
    std::unordered_map<std::string_view, int> index;
    std::vector<int> parents;
    parents.reserve(joints.size());
    for (std::size_t j = 0; j < joints.size(); ++j) {
        const std::string_view joint = joints[j];
        const std::size_t slash = joint.rfind('/');
        int parent = -1;
        if (slash != std::string_view::npos) {
            const auto it = index.find(joint.substr(0, slash));
            if (it == index.end()) {
                throw std::invalid_argument(
                    "Joint " + std::string(joint) +
                    " comes before its parent, or has none");
            }
            parent = it->second;
        }
        if (!index.emplace(joint, static_cast<int>(j)).second) {
            throw std::invalid_argument(
                "Joint " + std::string(joint) + " is named twice");
        }
        parents.push_back(parent);
    }
    return parents;
}

glm::mat4 JointTransform(const glm::vec3& translation,
                         const glm::quat& rotation,
                         const glm::vec3& scale) {
    glm::mat4 m = glm::mat4_cast(rotation);
    m[0] *= scale.x;
    m[1] *= scale.y;
    m[2] *= scale.z;
    m[3] = glm::vec4(translation, 1.0f);
    return m;
}

std::vector<glm::mat4>
SkelSpaceTransforms(std::span<const int> parents,
                    std::span<const glm::mat4> locals) {
    if (parents.size() != locals.size()) {
        throw std::invalid_argument(
            "A skeleton needs a local transform per joint");
    }
    std::vector<glm::mat4> skel(locals.size());
    for (std::size_t j = 0; j < locals.size(); ++j) {
        const int parent = parents[j];
        skel[j] =
            parent < 0
                ? locals[j]
                : skel[static_cast<std::size_t>(parent)] * locals[j];
    }
    return skel;
}

std::vector<glm::mat4>
SkinningTransforms(std::span<const glm::mat4> skelSpace,
                   std::span<const glm::mat4> bindTransforms) {
    if (skelSpace.size() != bindTransforms.size()) {
        throw std::invalid_argument(
            "A skeleton needs a bind transform per joint");
    }
    std::vector<glm::mat4> skinning(skelSpace.size());
    for (std::size_t j = 0; j < skelSpace.size(); ++j) {
        skinning[j] = skelSpace[j] * glm::inverse(bindTransforms[j]);
    }
    return skinning;
}

AnimationPose SampleAnimation(
    std::span<const double> times, std::size_t joints,
    std::size_t shapes, std::span<const float> translations,
    std::span<const float> rotations, std::span<const float> scales,
    std::span<const float> blendShapeWeights, double time) {
    const std::size_t samples = times.empty() ? 1 : times.size();
    CheckSamples(translations, samples, 3 * joints, "Translations");
    CheckSamples(rotations, samples, 4 * joints, "Rotations");
    CheckSamples(scales, samples, 3 * joints, "Scales");
    CheckSamples(blendShapeWeights, samples, shapes,
                 "Blend shape weights");
    // The samples either side of the time, and how far between them.
    std::size_t lo = 0;
    std::size_t hi = 0;
    float alpha = 0.0f;
    if (samples > 1) {
        const auto it =
            std::upper_bound(times.begin(), times.end(), time);
        if (it == times.begin()) {
            lo = hi = 0;
        } else if (it == times.end()) {
            lo = hi = samples - 1;
        } else {
            hi = static_cast<std::size_t>(it - times.begin());
            lo = hi - 1;
            alpha = static_cast<float>((time - times[lo]) /
                                       (times[hi] - times[lo]));
        }
    }
    const auto vec3s = [&](std::span<const float> values) {
        std::vector<glm::vec3> out;
        if (values.empty()) {
            return out;
        }
        const std::span<const float> a =
            SampleOf(values, lo, 3 * joints);
        const std::span<const float> b =
            SampleOf(values, hi, 3 * joints);
        for (std::size_t j = 0; j < joints; ++j) {
            out.push_back(glm::mix(
                glm::vec3(a[3 * j], a[3 * j + 1], a[3 * j + 2]),
                glm::vec3(b[3 * j], b[3 * j + 1], b[3 * j + 2]),
                alpha));
        }
        return out;
    };
    AnimationPose pose;
    pose.translations = vec3s(translations);
    pose.scales = vec3s(scales);
    if (!rotations.empty()) {
        const std::span<const float> a =
            SampleOf(rotations, lo, 4 * joints);
        const std::span<const float> b =
            SampleOf(rotations, hi, 4 * joints);
        for (std::size_t j = 0; j < joints; ++j) {
            const glm::quat qa(a[4 * j + 3], a[4 * j], a[4 * j + 1],
                               a[4 * j + 2]);
            const glm::quat qb(b[4 * j + 3], b[4 * j], b[4 * j + 1],
                               b[4 * j + 2]);
            pose.rotations.push_back(glm::normalize(
                alpha == 0.0f ? qa : glm::slerp(qa, qb, alpha)));
        }
    }
    if (!blendShapeWeights.empty()) {
        const std::span<const float> a =
            SampleOf(blendShapeWeights, lo, shapes);
        const std::span<const float> b =
            SampleOf(blendShapeWeights, hi, shapes);
        for (std::size_t s = 0; s < shapes; ++s) {
            pose.blendShapeWeights.push_back(a[s] +
                                             (b[s] - a[s]) * alpha);
        }
    }
    return pose;
}

std::vector<int> NameMapping(std::span<const std::string> from,
                             std::span<const std::string> to) {
    std::unordered_map<std::string_view, int> index;
    for (std::size_t i = 0; i < to.size(); ++i) {
        index.emplace(to[i], static_cast<int>(i));
    }
    std::vector<int> mapping;
    mapping.reserve(from.size());
    for (const std::string& name : from) {
        const auto it = index.find(name);
        mapping.push_back(it == index.end() ? -1 : it->second);
    }
    return mapping;
}

std::vector<std::pair<int, float>>
SubShapeWeights(float weight,
                std::span<const float> inbetweenWeights) {
    if (inbetweenWeights.empty()) {
        return {{0, weight}};
    }
    // The sub-shapes by weight: the null shape (-1) at 0, the primary
    // (0) at 1, the inbetweens (1 + k) at theirs.
    std::vector<std::pair<float, int>> shapes{{0.0f, -1}, {1.0f, 0}};
    for (std::size_t k = 0; k < inbetweenWeights.size(); ++k) {
        const float w = inbetweenWeights[k];
        if (std::abs(w) > kSubShapeEpsilon &&
            std::abs(w - 1.0f) > kSubShapeEpsilon) {
            shapes.emplace_back(w, static_cast<int>(k) + 1);
        }
    }
    std::stable_sort(shapes.begin(), shapes.end(),
                     [](const auto& a, const auto& b) {
                         return a.first < b.first;
                     });
    const auto it = std::upper_bound(
        shapes.begin(), shapes.end(), weight,
        [](float w, const auto& shape) { return w < shape.first; });
    std::size_t upper = 0;
    if (it == shapes.end()) {
        upper = shapes.size() - 1;
    } else if (it == shapes.begin()) {
        upper = 1;
    } else {
        upper = static_cast<std::size_t>(it - shapes.begin());
    }
    const auto& [loWeight, loShape] = shapes[upper - 1];
    const auto& [hiWeight, hiShape] = shapes[upper];
    std::vector<std::pair<int, float>> out;
    const float delta = hiWeight - loWeight;
    if (delta > kSubShapeEpsilon) {
        const float alpha = (weight - loWeight) / delta;
        if (loShape >= 0 &&
            std::abs(alpha - 1.0f) > kSubShapeEpsilon) {
            out.emplace_back(loShape, 1.0f - alpha);
        }
        if (hiShape >= 0 && std::abs(alpha) > kSubShapeEpsilon) {
            out.emplace_back(hiShape, alpha);
        }
    }
    return out;
}

void ApplyBlendShape(std::span<float> points,
                     std::span<const float> offsets,
                     std::span<const int> pointIndices,
                     float weight) {
    const std::size_t count = points.size() / 3;
    if (pointIndices.empty()) {
        if (offsets.size() != points.size()) {
            throw std::invalid_argument("A blend shape without point "
                                        "indices needs an offset per "
                                        "point");
        }
        for (std::size_t i = 0; i < points.size(); ++i) {
            points[i] += weight * offsets[i];
        }
        return;
    }
    if (offsets.size() != 3 * pointIndices.size()) {
        throw std::invalid_argument(
            "A blend shape needs an offset per point index");
    }
    for (std::size_t k = 0; k < pointIndices.size(); ++k) {
        const int p = pointIndices[k];
        if (p < 0 || static_cast<std::size_t>(p) >= count) {
            throw std::invalid_argument(
                "A blend shape has a point index out of range");
        }
        for (std::size_t c = 0; c < 3; ++c) {
            points[3 * static_cast<std::size_t>(p) + c] +=
                weight * offsets[3 * k + c];
        }
    }
}

void NormalizeWeights(std::span<float> weights,
                      std::size_t perPoint) {
    if (perPoint == 0) {
        return;
    }
    for (std::size_t first = 0; first + perPoint <= weights.size();
         first += perPoint) {
        float sum = 0.0f;
        for (std::size_t k = 0; k < perPoint; ++k) {
            sum += weights[first + k];
        }
        const float scale = std::abs(sum) > 1e-6f ? 1.0f / sum : 0.0f;
        for (std::size_t k = 0; k < perPoint; ++k) {
            weights[first + k] *= scale;
        }
    }
}

JointDualQuaternion FactorJoint(const glm::mat4& transform) {
    const glm::mat3 a(transform);
    // The orthonormal polar factor by Newton's iteration, a
    // reflection taken out first so it converges to a rotation.
    const float det = glm::determinant(a);
    glm::mat3 r = det < 0.0f ? -a : a;
    for (int i = 0; i < 20; ++i) {
        const glm::mat3 next =
            0.5f * (r + glm::transpose(glm::inverse(r)));
        float change = 0.0f;
        for (int c = 0; c < 3; ++c) {
            change = std::max(change, glm::length(next[c] - r[c]));
        }
        r = next;
        if (change < 1e-7f) {
            break;
        }
    }
    JointDualQuaternion dq;
    if (!std::isfinite(r[0][0]) || std::abs(det) < 1e-12f) {
        dq.real = glm::quat(0.0f, 0.0f, 0.0f, 0.0f);
        return dq;
    }
    dq.real = glm::normalize(glm::quat_cast(r));
    const glm::vec3 t(transform[3]);
    dq.dual = 0.5f * (glm::quat(0.0f, t) * dq.real);
    // What the rotation leaves: transform = translate * r * scale.
    dq.scale = glm::transpose(r) * a;
    return dq;
}

void SkinPoints(SkinningMethod method,
                const glm::mat4& geomBindTransform,
                std::span<const glm::mat4> skinning,
                std::span<const int> jointIndices,
                std::span<const float> jointWeights,
                std::size_t perPoint, bool constant,
                std::span<float> points) {
    const std::size_t count = points.size() / 3;
    const std::size_t influences =
        constant ? perPoint : perPoint * count;
    if (perPoint == 0 || jointIndices.size() != influences ||
        jointWeights.size() != influences) {
        throw std::invalid_argument(
            "Joint indices and weights need the same number of "
            "influences "
            "per point, or one set for all points");
    }
    for (const int j : jointIndices) {
        if (j < 0 || static_cast<std::size_t>(j) >= skinning.size()) {
            throw std::invalid_argument(
                "A joint index is out of range");
        }
    }
    std::vector<JointDualQuaternion> dqs;
    bool scaled = false;
    if (method == SkinningMethod::DualQuaternion) {
        for (const glm::mat4& m : skinning) {
            dqs.push_back(FactorJoint(m));
            const glm::mat3 d = dqs.back().scale - glm::mat3(1.0f);
            for (int c = 0; c < 3; ++c) {
                scaled = scaled || glm::length(d[c]) > 1e-6f;
            }
        }
    }
    for (std::size_t p = 0; p < count; ++p) {
        const std::size_t first = constant ? 0 : p * perPoint;
        const glm::vec3 bound(geomBindTransform *
                              glm::vec4(points[3 * p],
                                        points[3 * p + 1],
                                        points[3 * p + 2], 1.0f));
        glm::vec3 skinned(0.0f);
        if (method == SkinningMethod::Linear) {
            for (std::size_t k = 0; k < perPoint; ++k) {
                const float w = jointWeights[first + k];
                if (w != 0.0f) {
                    skinned +=
                        w *
                        glm::vec3(skinning[static_cast<std::size_t>(
                                      jointIndices[first + k])] *
                                  glm::vec4(bound, 1.0f));
                }
            }
        } else {
            // Signs taken against the heaviest joint (antipodality).
            std::size_t pivot = first;
            for (std::size_t k = 1; k < perPoint; ++k) {
                if (jointWeights[first + k] > jointWeights[pivot]) {
                    pivot = first + k;
                }
            }
            const glm::quat pivotReal =
                dqs[static_cast<std::size_t>(jointIndices[pivot])]
                    .real;
            glm::vec3 scaledPoint(0.0f);
            glm::quat real(0.0f, 0.0f, 0.0f, 0.0f);
            glm::quat dual(0.0f, 0.0f, 0.0f, 0.0f);
            for (std::size_t k = 0; k < perPoint; ++k) {
                float w = jointWeights[first + k];
                if (w == 0.0f) {
                    continue;
                }
                const JointDualQuaternion& dq =
                    dqs[static_cast<std::size_t>(
                        jointIndices[first + k])];
                if (scaled) {
                    scaledPoint += w * (dq.scale * bound);
                }
                if (glm::dot(dq.real, pivotReal) < 0.0f) {
                    w = -w;
                }
                real += w * dq.real;
                dual += w * dq.dual;
            }
            if (!scaled) {
                scaledPoint = bound;
            }
            const float length = glm::length(real);
            if (length > 0.0f) {
                real /= length;
                dual /= length;
            }
            // Rotate, then translate by 2 dual conj(real).
            const glm::vec3 t(2.0f * (dual * glm::conjugate(real)).x,
                              2.0f * (dual * glm::conjugate(real)).y,
                              2.0f * (dual * glm::conjugate(real)).z);
            skinned = real * scaledPoint + t;
        }
        points[3 * p] = skinned.x;
        points[3 * p + 1] = skinned.y;
        points[3 * p + 2] = skinned.z;
    }
}

}  // namespace ernie
