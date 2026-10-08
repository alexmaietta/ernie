#include "ernie/session/frame_params.h"

#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ernie {

glm::dmat4 InterpolatePose(const glm::dmat4& from, const glm::dmat4& to, double s) {
    // Rotation and scale apart, so the rotations slerp and the scales lerp.
    const auto split = [](const glm::dmat4& pose, glm::dquat& rotation, glm::dvec3& scale) {
        const glm::dmat3 linear(pose);
        scale = {glm::length(linear[0]), glm::length(linear[1]), glm::length(linear[2])};
        rotation = glm::quat_cast(glm::dmat3(linear[0] / scale.x, linear[1] / scale.y,
                                             linear[2] / scale.z));
    };
    glm::dquat a;
    glm::dquat b;
    glm::dvec3 scaleA;
    glm::dvec3 scaleB;
    split(from, a, scaleA);
    split(to, b, scaleB);
    // The shorter arc.
    if (glm::dot(a, b) < 0.0) {
        b = -b;
    }
    const double angle = std::acos(std::clamp(glm::dot(a, b), -1.0, 1.0));
    glm::dquat rotation;
    if (angle < 1e-9) {
        rotation = a;
    } else {
        rotation = (std::sin((1.0 - s) * angle) * a + std::sin(s * angle) * b) / std::sin(angle);
    }
    const glm::dmat3 r = glm::mat3_cast(glm::normalize(rotation));
    const glm::dvec3 scale = glm::mix(scaleA, scaleB, s);
    glm::dmat4 pose(1.0);
    pose[0] = glm::dvec4(r[0] * scale.x, 0.0);
    pose[1] = glm::dvec4(r[1] * scale.y, 0.0);
    pose[2] = glm::dvec4(r[2] * scale.z, 0.0);
    pose[3] = glm::dvec4(glm::mix(glm::dvec3(from[3]), glm::dvec3(to[3]), s), 1.0);
    return pose;
}

FrameParams FrameParams::Subframes() const {
    FrameParams result = *this;
    const int subframes = camera.Subframes();
    if (subframes == 1) {
        return result;
    }
    const std::size_t views = Views();
    if (!previousPoses.empty() && previousPoses.size() != views) {
        throw std::invalid_argument("A frame's previous poses need one per view");
    }
    result.poses.clear();
    result.worlds.clear();
    for (int k = 0; k < subframes; ++k) {
        // The last frame is at time -1, this one at 0.
        const double s = 1.0 + camera.SubframeTime(k);
        for (std::size_t v = 0; v < views; ++v) {
            const glm::dmat4& pose = poses.empty() ? camera.transform : poses[v];
            result.poses.push_back(previousPoses.empty()
                                       ? pose
                                       : InterpolatePose(previousPoses[v], pose, s));
            result.worlds.push_back(ViewWorld(v));
        }
    }
    return result;
}

}  // namespace ernie
