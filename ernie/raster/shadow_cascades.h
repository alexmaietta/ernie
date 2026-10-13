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
    /// A distant light's long-range view (`FitLongRangeView`):
    /// casters far towards the light, which shading reads for the
    /// share of the light's disc they leave, not by depth comparison.
    bool longRange = false;
};

/// The share of each cascade's depth span that it reaches past its
/// split into the next, where shading cross-fades between the two
/// (Unreal's `CascadeTransitionFraction`). The last fades out to lit
/// over as much past its split: Unreal fades it short of its fixed
/// shadow distance, but a distance fitted to the farthest depth seen
/// would then lose shadows the view sees.
inline constexpr float kCascadeTransition = 0.1f;
/// Texels round each cascade's map that its sphere leaves free, so
/// filters near its edge stay on it (Unreal's CSM border).
inline constexpr std::uint32_t kCascadeBorderTexels = 4;
/// Texels each cascade snaps its centre to a multiple of (Unreal's
/// `MaxDownsampleFactor`).
inline constexpr float kCascadeSnapTexels = 4.0f;
/// Metres of depth a cascade spans at least on either side of its
/// sphere's centre (Unreal's `DepthRangeClamp`, 5000 cm).
inline constexpr float kCascadeMinDepthHalf = 50.0f;
/// Metres towards the light from a receiver within which casters
/// shadow it through the cascades; farther ones, such as moons, cast
/// through the long-range map (`FitLongRangeView`). As far as
/// cascades reach along the view, too.
inline constexpr float kLongRangeStart = 10e3f;

/// Unreal Engine's cascaded shadow map biases and filtering, its
/// defaults: `r.Shadow.CSMDepthBias`,
/// `r.Shadow.CSMSlopeScaleDepthBias`, a light's `ShadowBias` and
/// `ShadowSlopeBias`, `r.Shadow.ShadowMaxSlopeScaleDepthBias`,
/// `r.Shadow.CSMReceiverBias` and `r.Shadow.MaxSoftKernelSize`
/// (texels). Casters move away from the light by kCsmDepthBias
/// kShadowBias half-texels, plus as much times
/// kCsmSlopeScaleDepthBias kShadowSlopeBias per unit of their slope
/// to it, up to kMaxSlopeDepthBias; shading's comparisons fade in
/// over the constant part (`CascadeCasterBias`). Soft shadows reach
/// 64 texels rather than Unreal's 40, which cuts a wide sun's
/// penumbrae short in fine cascades.
///
/// Reference: Unreal Engine 5, FProjectedShadowInfo::
/// UpdateShaderDepthBias and ComputeTransitionSize
/// (ShadowRendering.cpp).
inline constexpr float kCsmDepthBias = 10.0f;
inline constexpr float kCsmSlopeScaleDepthBias = 3.0f;
inline constexpr float kShadowBias = 0.5f;
inline constexpr float kShadowSlopeBias = 0.5f;
inline constexpr float kMaxSlopeDepthBias = 1.0f;
inline constexpr float kCsmReceiverBias = 0.9f;
inline constexpr float kMaxSoftKernelTexels = 64.0f;

/// A cascade's constant caster bias in world units, for its texels
/// `texelWorld` across: Unreal's CSM depth bias times the light's,
/// in units of half a texel (its sphere's radius over its texels).
constexpr float CascadeCasterBias(float texelWorld) {
    return kCsmDepthBias * kShadowBias * 0.5f * texelWorld;
}

/// View-space split depths from `nearPlane` to `farPlane`, each
/// cascade `exponent` times as deep as the last (Unreal's
/// `CascadeDistributionExponent`; 1 for uniform splits).
///
/// @return `count + 1` increasing depths, starting at `nearPlane` and
/// ending at `farPlane`.
std::vector<float> CascadeSplits(float nearPlane, float farPlane,
                                 int count, float exponent);

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
/// from its near plane to `shadowDistance`, as Unreal Engine's
/// cascaded shadow maps do.
///
/// The splits grow geometrically (`CascadeSplits`). Each cascade
/// reaches `kCascadeTransition` of its span past its split, where
/// shading fades into the next (or out), and bounds that slice with
/// the smallest sphere centred on the view axis, its radius rounded
/// up to the centimetre, so its size does not change as the camera
/// turns. The sphere fills the map but for `kCascadeBorderTexels`
/// round it, and its centre snaps to multiples of
/// `kCascadeSnapTexels` texels, so shadow edges do not shimmer as the
/// camera moves. Light-space depth spans the sphere, at least
/// `kCascadeMinDepthHalf` either way, and towards the light to
/// `kLongRangeStart`; with `clampCasters`, for maps whose pipelines
/// clamp depth, casters partly nearer the light land on its near
/// plane ("pancaking"), else the depth reaches back no farther than
/// the scene box. `shadowDistance` is rounded up to a quarter stop,
/// so a distance fitted to what the view sees keeps its cascades as
/// it drifts.
///
/// References:
/// - Unreal Engine 5, FDirectionalLightSceneProxy::GetSplitDistance,
///   GetShadowSplitBounds and GetShadowSplitBoundsDepthRange
///   (DirectionalLightComponent.cpp), and
///   FProjectedShadowInfo::SetupWholeSceneProjection
///   (ShadowSetup.cpp).
/// - M. Valient, "Stable Rendering of Cascaded Shadow Maps", in
///   ShaderX6, Charles River Media, 2008.
///
/// @param toSun Unit vector pointing towards the light.
/// @param exponent Each cascade's depth over the last's.
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
            const glm::vec3& sceneMax, int count, float exponent,
            std::uint32_t resolution, float around = 0.0f,
            bool clampCasters = false);

/// The long-range shadow view of a distant light towards `toSun`
/// (unit): an orthographic map of the casters more than
/// `kLongRangeStart` towards the light from the receivers in the box
/// `receiverLow` to `receiverHigh`, such as moons eclipsing the
/// ground and the air, for shading to integrate the light's
/// limb-darkened disc over (its angular radius's tangent
/// `tanRadius`).
///
/// The map holds the receivers' box as the light sees it, widened by
/// the widest penumbra a caster in the scene box `sceneMin` to
/// `sceneMax` throws on them, so casters off it still shade them; its
/// centre snaps to whole texels. Its depth reaches from the scene's
/// nearest point to the light to `kLongRangeStart` short of the
/// receivers: nearer casters, the cascades', land on its far plane
/// with depth clamping and read as none.
///
/// @return None without a caster that far towards the light.
std::optional<ShadowCascade>
FitLongRangeView(const glm::vec3& receiverLow,
                 const glm::vec3& receiverHigh,
                 const glm::vec3& toSun, const glm::vec3& sceneMin,
                 const glm::vec3& sceneMax, float tanRadius,
                 std::uint32_t resolution);

}  // namespace ernie
