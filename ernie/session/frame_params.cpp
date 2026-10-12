// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/session/frame_params.h"

#include "ernie/scene/wide_view.h"
#include "ernie/util/spectrum.h"

#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ernie {

LightParams RelativeDistantLight(const glm::vec3& towards,
                                 const SpectralBands& bands) {
    // A white Lambertian surface facing it shows E / pi of radiance,
    // whose luminance is K E / pi times the integral of ybar.
    static const double perBand = [] {
        double integral = 0.0;
        for (double w = 360.5; w < 830.0; w += 1.0) {
            integral += CieXyz(w).y;
        }
        return M_PI / (kLuminousEfficacy * integral);
    }();
    LightParams light;
    light.type = LightType::Distant;
    light.direction = glm::normalize(towards);
    light.angle = 0.53f;
    light.spectrum.assign(static_cast<std::size_t>(bands.count),
                          static_cast<float>(perBand));
    return light;
}

glm::dmat4 LightParams::Pose::Matrix() const {
    const glm::dvec3 z = -glm::normalize(glm::dvec3(direction));
    const glm::dvec3 x = glm::normalize(
        glm::dvec3(tangent) - glm::dot(glm::dvec3(tangent), z) * z);
    glm::dmat4 matrix(1.0);
    matrix[0] = glm::dvec4(x, 0.0);
    matrix[1] = glm::dvec4(glm::cross(z, x), 0.0);
    matrix[2] = glm::dvec4(z, 0.0);
    matrix[3] = glm::dvec4(position, 1.0);
    return matrix;
}

glm::dmat4 InterpolatePose(const glm::dmat4& from,
                           const glm::dmat4& to, double s) {
    // Rotation and scale apart, so the rotations slerp and the scales
    // lerp.
    const auto split = [](const glm::dmat4& pose,
                          glm::dquat& rotation, glm::dvec3& scale) {
        const glm::dmat3 linear(pose);
        scale = {glm::length(linear[0]), glm::length(linear[1]),
                 glm::length(linear[2])};
        rotation = glm::quat_cast(glm::dmat3(linear[0] / scale.x,
                                             linear[1] / scale.y,
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
    const double angle =
        std::acos(std::clamp(glm::dot(a, b), -1.0, 1.0));
    glm::dquat rotation;
    if (angle < 1e-9) {
        rotation = a;
    } else {
        rotation = (std::sin((1.0 - s) * angle) * a +
                    std::sin(s * angle) * b) /
                   std::sin(angle);
    }
    const glm::dmat3 r = glm::mat3_cast(glm::normalize(rotation));
    const glm::dvec3 scale = glm::mix(scaleA, scaleB, s);
    glm::dmat4 pose(1.0);
    pose[0] = glm::dvec4(r[0] * scale.x, 0.0);
    pose[1] = glm::dvec4(r[1] * scale.y, 0.0);
    pose[2] = glm::dvec4(r[2] * scale.z, 0.0);
    pose[3] = glm::dvec4(
        glm::mix(glm::dvec3(from[3]), glm::dvec3(to[3]), s), 1.0);
    return pose;
}

namespace {

/// The rotation of camera-to-world `pose`, its scale left out.
glm::dquat RotationOf(const glm::dmat4& pose) {
    return glm::normalize(glm::quat_cast(
        glm::dmat3(glm::normalize(glm::dvec3(pose[0])),
                   glm::normalize(glm::dvec3(pose[1])),
                   glm::normalize(glm::dvec3(pose[2])))));
}

}  // namespace

glm::mat4 PushbroomStrip::Packed() const {
    glm::mat4 packed;
    packed[0] = glm::vec4(from, nearPlane);
    packed[1] = glm::vec4(to, farPlane);
    packed[2] = glm::vec4(fromRotation.x, fromRotation.y,
                          fromRotation.z, fromRotation.w);
    packed[3] = glm::vec4(toRotation.x, toRotation.y, toRotation.z,
                          toRotation.w);
    return packed;
}

CameraView PushbroomEnvelope(const CameraParams& camera,
                             const glm::dmat4& from,
                             const glm::dmat4& to) {
    const glm::dmat4 middle = InterpolatePose(from, to, 0.5);
    const glm::dmat3 rotation = glm::mat3_cast(RotationOf(middle));
    const glm::dvec3 forward = -rotation[2];
    const double length =
        glm::length(glm::dvec3(to[3]) - glm::dvec3(from[3]));
    glm::dmat4 eyePose(rotation);
    eyePose[3] =
        glm::dvec4(glm::dvec3(middle[3]) - forward * length, 1.0);
    const glm::dmat4 toEye = glm::inverse(eyePose);

    // The detector line's ends at the clipping planes, from poses
    // along the strip: the rays between them are straight, and their
    // image coordinates from the eye monotonic along them.
    const double f = camera.FocalLengthPixels();
    const double cx = camera.PrincipalPoint().x;
    const double line = camera.lineOffsetMm / camera.focalLengthMm;
    // The rays of the line's end pixels, through the lens's
    // distortion.
    const double edges[] = {
        camera.lens.Undistort({-cx / f, line}).x,
        camera.lens.Undistort({(camera.width - cx) / f, line}).x};
    double tanX = 0.0;
    double tanY = 0.0;
    double nearest = std::numeric_limits<double>::max();
    double farthest = 0.0;
    constexpr int kSamples = 16;
    for (int i = 0; i <= kSamples; ++i) {
        const glm::dmat4 sample =
            InterpolatePose(from, to, double(i) / kSamples);
        const glm::dmat4 pose(glm::mat3_cast(RotationOf(sample)));
        const glm::dvec3 position(sample[3]);
        for (const double x : edges) {
            for (const double depth :
                 {camera.nearPlane, camera.farPlane}) {
                const glm::dvec3 world =
                    position +
                    glm::dvec3(pose * glm::dvec4(x * depth,
                                                 line * depth, -depth,
                                                 0.0));
                const glm::dvec3 p(toEye * glm::dvec4(world, 1.0));
                const double z = std::max(-p.z, 1e-6);
                nearest = std::min(nearest, z);
                farthest = std::max(farthest, z);
                tanX = std::max(tanX, std::abs(p.x) / z);
                tanY = std::max(tanY, std::abs(p.y) / z);
            }
        }
    }
    // Padding for the rays between samples, and a sliver of height
    // for a still camera.
    tanX *= 1.02;
    tanY = std::max(tanY * 1.02, 1e-4);
    CameraView view;
    view.verticalFov = static_cast<float>(2.0 * std::atan(tanY));
    view.aspect = static_cast<float>(tanX / tanY);
    view.nearPlane = static_cast<float>(0.98 * nearest);
    view.farPlane = static_cast<float>(1.02 * farthest);
    view.eye = glm::vec3(eyePose[3]);
    view.forward = glm::vec3(forward);
    view.view = glm::mat4(toEye);
    view.projection = glm::perspectiveRH_ZO(
        view.verticalFov, view.aspect, view.nearPlane, view.farPlane);
    return view;
}

glm::dmat4 FrameParams::PreviousPose(std::size_t view) const {
    glm::dmat4 pose =
        previousPoses.empty()
            ? (poses.empty() ? camera.transform : poses.at(view))
            : previousPoses.at(view);
    pose[3] -= glm::dvec4(origin, 0.0);
    return pose;
}

PushbroomStrip FrameParams::Strip(std::size_t view) const {
    glm::dmat4 pose =
        poses.empty() ? camera.transform : poses.at(view);
    pose[3] -= glm::dvec4(origin, 0.0);
    const glm::dmat4 from = PreviousPose(view);
    const glm::dquat a = RotationOf(from);
    glm::dquat b = RotationOf(pose);
    if (glm::dot(a, b) < 0.0) {
        b = -b;  // the shorter arc, as `InterpolatePose`
    }
    PushbroomStrip strip;
    strip.from = glm::vec3(from[3]);
    strip.to = glm::vec3(pose[3]);
    strip.fromRotation = glm::quat(a);
    strip.toRotation = glm::quat(b);
    strip.nearPlane = static_cast<float>(camera.nearPlane);
    strip.farPlane = static_cast<float>(camera.farPlane);
    const double width = camera.width;
    // The raster image is drawn with the focal length over the
    // overscan, so the undistorted line's ends stay in it (the lens
    // head resamples it through the distortion).
    strip.scale = static_cast<float>(
        2.0 * camera.FocalLengthPixels() / overscan / width);
    strip.shift = static_cast<float>(
        2.0 * camera.PrincipalPoint().x / width - 1.0);
    strip.line = static_cast<float>(camera.lineOffsetMm /
                                    camera.focalLengthMm);
    return strip;
}

std::vector<LightParams::Beam::Lit>
LightParams::Beam::LitBetween(double start, double end) const {
    const bool pulsed = pulseRate > 0.0;
    const double period = pulsed ? 1.0 / pulseRate : 0.0;
    const double duty =
        pulsed ? std::min(pulseWidth * pulseRate, 1.0) : 1.0;
    // The pulses' seconds on from `pulseDelay` to `t`.
    const auto onUntil = [&](double t) {
        const double u = t - pulseDelay;
        const double n = std::floor(u / period);
        return n * pulseWidth +
               std::clamp(u - n * period, 0.0, pulseWidth);
    };
    // The light's mean over [a, b] over its average.
    const auto mean = [&](double a, double b) {
        if (!pulsed) {
            return 1.0;
        }
        if (!(duty > 0.0)) {
            return 0.0;
        }
        if (!(b > a)) {
            const double u = a - pulseDelay;
            return u - std::floor(u / period) * period < pulseWidth
                       ? 1.0 / duty
                       : 0.0;
        }
        return (onUntil(b) - onUntil(a)) / (b - a) / duty;
    };
    const int count = std::max(sequenceCount, 1);
    if (count == 1) {
        return {{0u, static_cast<float>(mean(start, end))}};
    }
    // The sequence's slots the span meets, slot n shown from
    // `sequenceDelay + n sequencePeriod`.
    const auto frameOf = [&](double slot) {
        const auto n = static_cast<long long>(slot) % count;
        return static_cast<std::uint32_t>(n < 0 ? n + count : n);
    };
    const double first =
        std::floor((start - sequenceDelay) / sequencePeriod);
    if (!(end > start)) {
        return {
            {frameOf(first), static_cast<float>(mean(start, end))}};
    }
    const double last =
        std::ceil((end - sequenceDelay) / sequencePeriod) - 1.0;
    const double slots = std::max(last - first + 1.0, 1.0);
    std::vector<Lit> lit;
    if (slots <= static_cast<double>(kMaxLit)) {
        for (double slot = first; slot <= last; slot += 1.0) {
            const double a = std::max(
                start, sequenceDelay + slot * sequencePeriod);
            const double b = std::min(
                end, sequenceDelay + (slot + 1.0) * sequencePeriod);
            if (b > a) {
                lit.push_back(
                    {frameOf(slot),
                     static_cast<float>(mean(a, b) * (b - a) /
                                        (end - start))});
            }
        }
        return lit;
    }
    const auto share = static_cast<float>(
        mean(start, end) / static_cast<double>(kMaxLit));
    for (std::size_t i = 0; i < kMaxLit; ++i) {
        lit.push_back(
            {frameOf(
                 first +
                 std::floor((static_cast<double>(i) + 0.5) * slots /
                            static_cast<double>(kMaxLit))),
             share});
    }
    return lit;
}

FrameParams FrameParams::Subframes() const {
    FrameParams result = *this;
    const int subframes = camera.Subframes();
    if (subframes == 1) {
        return result;
    }
    const std::size_t views = Views();
    if (!previousPoses.empty() && previousPoses.size() != views) {
        throw std::invalid_argument(
            "A frame's previous poses need one per view");
    }
    result.poses.clear();
    result.worlds.clear();
    for (int k = 0; k < subframes; ++k) {
        // The last frame is at time -1, this one at 0.
        const double s = 1.0 + camera.SubframeTime(k);
        for (std::size_t v = 0; v < views; ++v) {
            const glm::dmat4& pose =
                poses.empty() ? camera.transform : poses[v];
            result.poses.push_back(
                previousPoses.empty()
                    ? pose
                    : InterpolatePose(previousPoses[v], pose, s));
            result.worlds.push_back(ViewWorld(v));
        }
    }
    return result;
}

glm::dvec3 FrameParams::FaceIntrinsics(std::uint32_t width,
                                       std::uint32_t height) {
    const double side = std::min(width, height);
    const double focal = 0.5 * side / CubeFaceExtent(side);
    return {focal, 0.5 * width, 0.5 * height};
}

FrameParams FrameParams::Faces(int faces) const {
    if (faces != 1 && faces != kCubeFaces && faces != kAllCubeFaces) {
        throw std::invalid_argument(
            "A frame renders on 1, " + std::to_string(kCubeFaces) +
            " or " + std::to_string(kAllCubeFaces) +
            " cube faces, not " + std::to_string(faces));
    }
    FrameParams result = *this;
    if (faces == 1) {
        return result;
    }
    // The faces' camera: a centred pinhole, its focal length in
    // pixels the face's.
    CameraParams& face = result.camera;
    face.lens = LensModel{};
    face.lens.transmission = camera.lens.transmission;
    face.lens.vignetting = camera.lens.vignetting;
    const glm::dvec3 intrinsics =
        FaceIntrinsics(camera.width, camera.height);
    face.focalLengthMm =
        intrinsics.x * camera.verticalApertureMm / camera.height;
    face.horizontalApertureOffsetMm = 0.0;
    face.verticalApertureOffsetMm = 0.0;
    result.overscan = 1.0;
    const std::size_t views = Views();
    result.poses.clear();
    result.worlds.clear();
    result.previousPoses.clear();
    for (std::size_t v = 0; v < views; ++v) {
        const glm::dmat4& pose =
            poses.empty() ? camera.transform : poses[v];
        for (int f = 0; f < faces; ++f) {
            const glm::dmat4 turn(
                CubeFaceAxes(static_cast<CubeFace>(f)));
            result.poses.push_back(pose * turn);
            // A sub-frame's view (`Subframes`) has its output view's
            // last pose.
            if (!previousPoses.empty()) {
                result.previousPoses.push_back(
                    previousPoses[v % previousPoses.size()] * turn);
            }
            result.worlds.push_back(ViewWorld(v));
        }
    }
    return result;
}

}  // namespace ernie
