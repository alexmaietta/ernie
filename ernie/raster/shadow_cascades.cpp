// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/raster/shadow_cascades.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace ernie {

namespace {

std::array<glm::vec3, 8> SliceCorners(const CameraView& camera,
                                      float nearDepth,
                                      float farDepth) {
    const glm::mat4 cameraToWorld = glm::inverse(camera.view);
    std::array<glm::vec3, 8> corners{};
    if (camera.IsOrthographic()) {
        // The view's box: the projection's edges at every depth.
        const glm::mat4 toView = glm::inverse(camera.projection);
        for (int i = 0; i < 8; ++i) {
            const float depth = (i & 4) ? farDepth : nearDepth;
            const glm::vec4 edge =
                toView * glm::vec4(i & 1 ? 1.0f : -1.0f,
                                   i & 2 ? 1.0f : -1.0f, 0.0f, 1.0f);
            corners[static_cast<std::size_t>(i)] =
                glm::vec3(cameraToWorld *
                          glm::vec4(edge.x, edge.y, -depth, 1.0f));
        }
        return corners;
    }
    const float tanY = std::tan(0.5f * camera.verticalFov);
    const float tanX = tanY * camera.aspect;
    for (int i = 0; i < 8; ++i) {
        const float depth = (i & 4) ? farDepth : nearDepth;
        const glm::vec4 inView((i & 1 ? 1.0f : -1.0f) * tanX * depth,
                               (i & 2 ? 1.0f : -1.0f) * tanY * depth,
                               -depth, 1.0f);
        corners[static_cast<std::size_t>(i)] =
            glm::vec3(cameraToWorld * inView);
    }
    return corners;
}

}  // namespace

std::vector<float> CascadeSplits(float nearPlane, float farPlane,
                                 int count, float exponent) {
    // Unreal's ComputeAccumulatedScale: cascade i spans exponent^i
    // parts of the sum of them all.
    std::vector<float> splits(static_cast<std::size_t>(count) + 1);
    float total = 0.0f;
    float scale = 1.0f;
    for (int i = 0; i < count; ++i) {
        total += scale;
        scale *= exponent;
    }
    float reached = 0.0f;
    scale = 1.0f;
    for (int i = 0; i <= count; ++i) {
        splits[static_cast<std::size_t>(i)] =
            nearPlane + reached / total * (farPlane - nearPlane);
        reached += scale;
        scale *= exponent;
    }
    splits.front() = nearPlane;
    splits.back() = farPlane;
    return splits;
}

std::pair<glm::vec3, glm::vec3> WithShadows(const glm::vec3& low,
                                            const glm::vec3& high,
                                            const glm::vec3& toSun,
                                            float maxLength) {
    if (!(toSun.z > 1e-3f)) {
        return {low, high};
    }
    const float reach = maxLength * glm::length(high - low);
    glm::vec3 grownLow = low;
    glm::vec3 grownHigh = high;
    for (int i = 0; i < 8; ++i) {
        const glm::vec3 corner((i & 1) ? high.x : low.x,
                               (i & 2) ? high.y : low.y,
                               (i & 4) ? high.z : low.z);
        const float along =
            std::min((corner.z - low.z) / toSun.z, reach);
        const glm::vec3 landing = corner - toSun * along;
        grownLow = glm::min(grownLow, landing);
        grownHigh = glm::max(grownHigh, landing);
    }
    return {grownLow, grownHigh};
}

std::vector<ShadowCascade>
FitCascades(const CameraView& camera, float shadowDistance,
            const glm::vec3& toSun, const glm::vec3& sceneMin,
            const glm::vec3& sceneMax, int count, float exponent,
            std::uint32_t resolution, float around,
            bool clampCasters) {
    const glm::vec3 up = std::fabs(toSun.z) > 0.99f
                             ? glm::vec3(0.0f, 1.0f, 0.0f)
                             : glm::vec3(0.0f, 0.0f, 1.0f);
    const glm::mat4 lightView =
        glm::lookAtRH(toSun, glm::vec3(0.0f), up);

    float sceneNear = std::numeric_limits<float>::max();
    float sceneFar = std::numeric_limits<float>::lowest();
    for (int i = 0; i < 8; ++i) {
        const glm::vec3 corner((i & 1) ? sceneMax.x : sceneMin.x,
                               (i & 2) ? sceneMax.y : sceneMin.y,
                               (i & 4) ? sceneMax.z : sceneMin.z);
        const float z = (lightView * glm::vec4(corner, 1.0f)).z;
        sceneNear = std::min(sceneNear, -z);
        sceneFar = std::max(sceneFar, -z);
    }
    const float depthPadding = 0.01f * (sceneFar - sceneNear) + 1e-3f;

    const float res = static_cast<float>(resolution);
    // The sphere fills the map inside its border.
    const float inner = std::max(
        res - 2.0f * static_cast<float>(kCascadeBorderTexels), 1.0f);
    // The distance up to a quarter stop (a power of 2^(1/4)): one
    // that drifts as the view turns or what it sees changes keeps the
    // same cascades, whose snapped texels then hold still, until it
    // crosses a step.
    const float stops =
        4.0f * std::log2(std::max(shadowDistance, 1e-6f));
    const float start = camera.nearPlane;
    const std::vector<float> splits = CascadeSplits(
        start,
        std::max(std::exp2(std::ceil(stops) / 4.0f), start * 1.001f),
        count, exponent);
    const float tanY = std::tan(0.5f * camera.verticalFov);
    const float tanX = tanY * camera.aspect;
    const glm::vec3 forward =
        -glm::vec3(glm::inverse(camera.view)[2]);
    std::vector<ShadowCascade> cascades;
    for (int c = 0; c < count; ++c) {
        ShadowCascade cascade;
        cascade.splitNear = splits[static_cast<std::size_t>(c)];
        cascade.splitFar = splits[static_cast<std::size_t>(c) + 1];
        // Into the next cascade's fade, or the last's fade out.
        const float sliceFar =
            cascade.splitFar +
            kCascadeTransition *
                (cascade.splitFar - cascade.splitNear);
        glm::vec3 center(0.0f);
        float radius = 0.0f;
        const auto corners =
            SliceCorners(camera, cascade.splitNear, sliceFar);
        if (around > 0.0f) {
            center = camera.eye;
            radius =
                sliceFar * std::sqrt(1.0f + 2.0f * around * around);
        } else if (camera.IsOrthographic()) {
            for (const glm::vec3& corner : corners) {
                center += corner / 8.0f;
            }
        } else {
            // The centre on the view axis nearest every corner:
            // Fx = (Dn^2 - Df^2) / 2L + L / 2 short of the far plane,
            // Dn and Df the near and far diagonals, L the length.
            const float length = sliceFar - cascade.splitNear;
            const float farSq =
                (tanX * tanX + tanY * tanY) * sliceFar * sliceFar;
            const float nearSq = (tanX * tanX + tanY * tanY) *
                                 cascade.splitNear *
                                 cascade.splitNear;
            const float offset =
                (nearSq - farSq) / (2.0f * length) + 0.5f * length;
            const float along = std::clamp(
                sliceFar - offset, cascade.splitNear, sliceFar);
            center = camera.eye + forward * along;
        }
        if (around <= 0.0f) {
            for (const glm::vec3& corner : corners) {
                radius =
                    std::max(radius, glm::length(corner - center));
            }
        }
        // Whole centimetres, at least one.
        radius = std::max(std::ceil(radius * 100.0f) / 100.0f, 0.01f);
        const float texel = 2.0f * radius / inner;
        const float half = 0.5f * res * texel;

        const glm::vec3 centerInLight =
            glm::vec3(lightView * glm::vec4(center, 1.0f));
        const float snap = kCascadeSnapTexels * texel;
        const float x = std::floor(centerInLight.x / snap) * snap;
        const float y = std::floor(centerInLight.y / snap) * snap;
        // The sphere's depth, at least kCascadeMinDepthHalf either
        // way. Casters nearer the light clamp onto the near plane, or
        // without clamping the range reaches back to hold them.
        const float depthHalf =
            std::max(radius, kCascadeMinDepthHalf);
        float zNear = -centerInLight.z - depthHalf;
        const float zFar = -centerInLight.z + depthHalf;
        if (!clampCasters) {
            zNear = std::min(zNear, sceneNear - depthPadding);
        }
        const glm::mat4 projection = glm::orthoRH_ZO(
            x - half, x + half, y - half, y + half, zNear, zFar);
        cascade.lightViewProjection = projection * lightView;
        cascade.texelWorldSize = texel;
        cascades.push_back(cascade);
    }
    return cascades;
}

}  // namespace ernie
