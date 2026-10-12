// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/scene/camera.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace ernie {

/// One cascade of a cascaded shadow map, or another shadow view (only
/// `lightViewProjection` and `texelWorldSize` apply).
struct ShadowCascade {
    glm::mat4 lightViewProjection{
        1.0f};  ///< World to light clip space, [0, 1] depth.
    float splitNear =
        0.0f;  ///< View-space depth where this cascade starts.
    float splitFar =
        0.0f;  ///< View-space depth where this cascade ends.
    float texelWorldSize =
        0.0f;  ///< World size of one shadow map texel.
    /// Texels along each side of its map, drawn into that corner of
    /// a larger layer; 0 for the whole layer.
    std::uint32_t resolution = 0;
    /// Fitted on the GPU (a light posed there): `lightViewProjection`
    /// is only the CPU's estimate, so passes neither cull nor select
    /// what they draw by it on the CPU; the fitted matrix is
    /// `gpuMatrix` of `RasterRenderer::LightViewMatrices`.
    bool gpuFitted = false;
    std::uint32_t gpuMatrix = 0;
    /// A batch's map fitted to one world (`WorldShadowFit::World`):
    /// the world whose casters it draws besides the shared ones, so
    /// passes consider only theirs; none for every world's.
    std::optional<std::int32_t> world;
    /// Shadow linking (`LightParams::shadowMask`): the link groups of
    /// the geometry its map draws.
    std::uint32_t shadowMask = ~0u;
    /// Of a light moving across the shutter, posed again per
    /// sub-frame (`RasterRenderer::subframeLightViews_`):
    /// `lightViewProjection` is only the frame's pose, so passes cull
    /// nothing by it.
    bool moving = false;
};

/// View-space split depths from `nearPlane` to `farPlane` with the
/// practical split scheme, which blends logarithmic splits (weight
/// `lambda`) with uniform splits (weight 1 - `lambda`).
///
/// Reference: F. Zhang, H. Sun, L. Xu and L. K. Lun, "Parallel-Split
/// Shadow Maps for Large-Scale Virtual Environments", Proc. ACM VRCIA
/// 2006, 311-318. https://doi.org/10.1145/1128923.1128975
///
/// @return `count + 1` increasing depths, starting at `nearPlane` and
/// ending at `farPlane`.
std::vector<float> PracticalSplits(float nearPlane, float farPlane,
                                   int count, float lambda);

/// The box from `low` to `high` grown to hold the shadows its
/// contents cast along `toSun` (unit, towards the light) onto the
/// level of its bottom, such as a ground plane under them: each
/// corner projected along the light onto z = `low.z`, at most
/// `maxLength` times the box's diagonal away, so a grazing sun does
/// not stretch it without bound. A sun at or below the horizon casts
/// no shadows to hold, leaving the box as it is.
std::pair<glm::vec3, glm::vec3> WithShadows(const glm::vec3& low,
                                            const glm::vec3& high,
                                            const glm::vec3& toSun,
                                            float maxLength = 8.0f);

/// Fits one orthographic shadow map per split of the camera frustum
/// up to `shadowDistance`.
///
/// Each cascade bounds its frustum slice with a sphere, so its size
/// does not change as the camera rotates, and snaps its position to
/// whole shadow map texels, so shadow edges do not shimmer as the
/// camera moves (Valient 2008). Light-space depth spans the whole
/// scene box, so casters outside the view still cast into it; with
/// `clampCasters`, for maps whose pipelines clamp depth, only each
/// cascade's own sphere within it, casters nearer the light landing
/// on its near plane ("pancaking"), so a vast scene such as a planet
/// keeps depth precision.
///
/// References:
/// - W. Engel, "Cascaded Shadow Maps", in ShaderX5, Charles River
///   Media, 2006.
/// - M. Valient, "Stable Rendering of Cascaded Shadow Maps", in
///   ShaderX6, Charles River Media, 2008.
///
/// @param toSun Unit vector pointing towards the light.
/// @param splitNear Where the first cascade starts, such as the
/// nearest depth the view sees (sample distribution shadow maps); 0
/// for the camera's near plane.
/// @param around For a camera rendered on cube faces (wide_view.h),
/// the faces' extent (the tangent of their half field,
/// `CubeFaceExtent`), 0 for others: each cascade then bounds every
/// face's slice, which shading picks by the face's own view depth,
/// with a sphere about the eye of radius the split's far depth along
/// a face's corner ray.
/// @param clampCasters Whether the maps' pipelines clamp depth
/// (`GraphicsPipelineDesc::depthClamp`).
std::vector<ShadowCascade>
FitCascades(const CameraView& camera, float shadowDistance,
            const glm::vec3& toSun, const glm::vec3& sceneMin,
            const glm::vec3& sceneMax, int count, float lambda,
            std::uint32_t resolution, float splitNear = 0.0f,
            float around = 0.0f, bool clampCasters = false);

}  // namespace ernie
