// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/pipeline.h"
#include "ernie/kernel/embedded_shaders.h"
#include "ernie/raster/render_targets.h"
#include "ernie/raster/shadow_cascades.h"
#include "ernie/session/frame_params.h"
#include "ernie/util/shader_source.h"

#include <glm/glm.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace ernie {

/// Module `projection`, which `frame.slang` imports for
/// `projectWorld`: a pushbroom camera's (projection_pushbroom.slang)
/// or a frame camera's (projection_pinhole.slang). Passes facing the
/// camera compile with the frame's; others, such as shadow passes,
/// with the frame camera's.
inline ShaderModuleSource ProjectionModule(bool pushbroom) {
    return {"projection", pushbroom
                              ? shaders::kProjectionPushbroomSlang
                              : shaders::kProjectionPinholeSlang};
}

/// The per-frame resources every raster pass binds (`frame.slang`):
/// every view's view and light uniforms (a storage buffer, the same
/// for all views), shadow uniforms, cascaded shadow maps, their
/// comparison sampler and a clamping linear sampler, at bindings 0 to
/// 4; and the view drawn, which draws name in their push constants
/// (`DrawConstants::view`).
struct RasterFrame {
    DescriptorResource uniforms;
    std::uint32_t view = 0;
    DescriptorResource shadows;
    DescriptorResource shadowMap;
    DescriptorResource shadowSampler;
    DescriptorResource clampSampler;
    /// Shadow passes: `shadows` is the light view's own block, whose
    /// first view-projection the pass draws with
    /// (`DrawConstants::shadowFromFrame`), as the GPU may have fitted
    /// it; the matrix `RecordShadow` gets is then only the CPU's
    /// estimate.
    bool shadowFromFrame = false;
    /// Shadow passes: the layer of the layered shadow map their draws
    /// go to (`DrawConstants::shadowLayer`), so one pass draws every
    /// world's map.
    std::uint32_t shadowLayer = 0;
    /// Shadow passes: drawing a virtual shadow map's level
    /// (`VirtualShadowMap`), named by `shadows`' `virtualMap`: no
    /// depth attachment, the drawable's virtual-shadow pipeline,
    /// whose fragments write through the map's pages.
    bool virtualShadow = false;
};

/// The shadow casters a shadow pass draws: what every world shares
/// (`tokens::kWorld` -1), and the casters of one world, of every
/// world, or of none.
struct ShadowCasters {
    static constexpr std::int32_t kAllWorlds = -1;
    static constexpr std::int32_t kNoWorld = -2;

    bool shared =
        true;  ///< Draw what every world shares, such as the terrain.
    std::int32_t world =
        kAllWorlds;  ///< The world whose casters to draw.
    /// The light link groups whose geometry casts these shadows
    /// (`LightParams::shadowMask`), a bit each.
    std::uint32_t linkGroups = ~0u;

    /// Whether casters (or a view's draws) in world `of` (-1: shared)
    /// are drawn.
    bool Draws(std::int32_t of) const {
        return of == -1 ? shared
                        : (world == kAllWorlds || world == of);
    }

    /// Whether geometry of light link group `group` casts them.
    bool Casts(std::uint32_t group) const {
        return ((linkGroups >> (group & 31u)) & 1u) != 0;
    }
};

/// Which of a view's draws a G-buffer pass records. With occlusion
/// culling a view draws twice: first the occluders, then, after a
/// depth pyramid of them is built, the instances they do not hide.
enum class DrawPhase {
    All,        ///< Everything: one pass, without occlusion culling.
    Occluders,  ///< What draws before the depth pyramid: all but the
                ///< occlusion-culled instances.
    Occludees,  ///< The occlusion-culled instances, after the depth
                ///< pyramid.
};

/// Something a `RasterRenderer` draws: terrain, meshes, and in future
/// volumes or points. Each frame the renderer asks every drawable for
/// its bounds, fits the shadow views around them (the sun's cascades,
/// then the views of the other lights' shadow maps), lets each
/// drawable prepare for the final view and shadow views, and then has
/// each record its draws into every shadow view and into the main
/// pass. Drawables bind the frame's resources at bindings 0 to 4, a
/// material's at 5 to 9 (`RasterRenderer::Material`) and their own
/// from 10.
class RasterDrawable {
  public:
    virtual ~RasterDrawable() = default;

    /// The name of the drawable's GPU zones in each pass the renderer
    /// records it in (`ERNIE_GPU_ZONE_NAMED`), such as "Meshes".
    virtual const char* ProfileName() const = 0;

    /// Moves the origin the drawable's coordinates are relative to
    /// (`FrameParams::origin`): every position it gives the GPU and
    /// every bound it reports is world minus `origin`. The renderer
    /// calls it at the start of each `Update`, before `Bounds`; it
    /// changes rarely, so drawables recompute what depends on it only
    /// when it does.
    virtual void SetOrigin(const glm::dvec3& /*origin*/) {}

    /// Bounds of the drawable's geometry relative to its origin
    /// (`SetOrigin`), for fitting shadow cascades; none if it has
    /// none.
    virtual std::optional<Bounds3> Bounds() const = 0;

    /// Whether the drawable draws with minimal shading
    /// (`MinimalMode`) into `MinimalTargets`: its colour passes then
    /// use pipelines of their formats. None by default.
    virtual bool DrawsMinimal() const {
        return false;
    }

    /// World bounds of the drawable's geometry in each world it has
    /// some in (`tokens::kWorld`, the shared -1 too), for fitting the
    /// worlds' shadow maps; in one pass, as a batch may have
    /// thousands of worlds.
    virtual std::map<std::int32_t, Bounds3> WorldsBounds() const {
        return {};
    }

    /// The height (relative to the origin) of an unbounded floor the
    /// drawable draws, such as a ground plane, which the sun's
    /// shadows of the other drawables reach down to; none for a
    /// drawable without one.
    virtual std::optional<float> ShadowFloor() const {
        return std::nullopt;
    }

    /// Prepares the next frame once the views and shadow views are
    /// final, for example selecting levels of detail for every view
    /// (`params.View(v)` for each of `params.Views()`) and each
    /// shadow view (orthographic or perspective; only their
    /// `lightViewProjection` applies to every view). Called by
    /// `RasterRenderer::Update` while earlier frames may still be
    /// running: host data for buffers they read goes through
    /// `RasterRenderer::Uploads`, and buffers are replaced only after
    /// `vkDeviceWaitIdle`.
    virtual void Prepare(const FrameParams& params,
                         std::span<const ShadowCascade> cascades) = 0;

    /// Records compute work the views' GPU poses and culling depend
    /// on, such as poses composed on the GPU and the views posed
    /// under them, first of the frame's work after its uploads; the
    /// renderer follows it with a barrier. Nothing by default.
    virtual void RecordPrepass(VkCommandBuffer /*commands*/) const {}

    /// Records compute work the frame's draws depend on, such as
    /// culling instances on the GPU, before any shadow or G-buffer
    /// pass; the renderer follows it with a barrier. Nothing by
    /// default.
    virtual void RecordCull(VkCommandBuffer /*commands*/) const {}

    /// Records the patching of the views' camera fields in the
    /// drawable's per-view buffers from the GPU poses
    /// (`RasterRenderer::PatchViews`) when the frame's poses are on
    /// the GPU (`FrameParams::posesOnGpu`), before `RecordCull`.
    /// Nothing by default.
    virtual void RecordViewPoses(VkCommandBuffer /*commands*/) const {
    }

    /// Records the patching of the drawable's per-view buffers for
    /// the shadow views the GPU fits (`ShadowCascade::gpuFitted`)
    /// from their matrices (`RasterRenderer::LightViewMatrices`),
    /// after the renderer's light pass and before `RecordCull`.
    /// Nothing by default.
    virtual void
    RecordShadowViews(VkCommandBuffer /*commands*/) const {}

    /// Records compute work that reads the frame's shadow maps before
    /// its views draw, such as the terrain's horizon over what stands
    /// on it (`RasterRenderer::Casters`); the renderer follows it
    /// with a barrier. Nothing by default.
    virtual void
    RecordAfterShadows(VkCommandBuffer /*commands*/) const {}

    /// Records depth-only draws of `casters` for shadow view
    /// `cascade` (an index into the views given to `Prepare`) into
    /// the open rendering pass. Drawables without worlds draw only
    /// when `casters.shared` is set.
    virtual void RecordShadow(VkCommandBuffer commands,
                              const RasterFrame& frame,
                              std::size_t cascade,
                              const glm::mat4& lightViewProjection,
                              const ShadowCasters& casters) const = 0;

    /// Whether `RecordShadow` draws a virtual shadow map's levels
    /// (`RasterFrame::virtualShadow`), with a pipeline writing
    /// through its pages; those that do not cast no shadow in it.
    /// None by default.
    virtual bool DrawsVirtualShadows() const {
        return false;
    }

    /// Records colour draws of view `view` in `phase` into its open
    /// G-buffer rendering pass; `frame` holds that view's uniforms.
    /// Drawables without occlusion-culled instances draw in `All` and
    /// `Occluders`.
    virtual void
    RecordColor(VkCommandBuffer commands, const RasterFrame& frame,
                std::size_t view,
                DrawPhase phase = DrawPhase::All) const = 0;

    /// `RecordColor` of views `first` to `first + frames.size()`,
    /// `frames` theirs, one after another into the open rendering
    /// pass: drawables keep what every view binds alike (the frame's
    /// resources are the same for all views but the view drawn) bound
    /// across them.
    virtual void RecordColorViews(
        VkCommandBuffer commands, std::span<const RasterFrame> frames,
        std::size_t first, DrawPhase phase = DrawPhase::All) const {
        for (std::size_t v = 0; v < frames.size(); ++v) {
            RecordColor(commands, frames[v], first + v, phase);
        }
    }

    /// Records, into view `view`'s open motion pass (one RG16 float
    /// attachment of raster-pixel motion, the G-buffer's depth to
    /// test against, not written), the draws of what moved since the
    /// last frame with the motion each of its pixels made; `motion`
    /// is the view's motion_vectors.slang `MotionUniforms` (the last
    /// frame's view-projection). A camera pass has already written
    /// every pixel's motion from the camera's alone, so drawables
    /// that stand still, such as terrain, draw nothing: the default.
    virtual void
    RecordMotion(VkCommandBuffer /*commands*/,
                 const RasterFrame& /*frame*/, std::size_t /*view*/,
                 const DescriptorResource& /*motion*/) const {}

    /// Whether the last `Prepare`'s frame culls some of this
    /// drawable's instances by occlusion, so views draw in two phases
    /// around a depth pyramid.
    virtual bool CullsOcclusion() const {
        return false;
    }

    /// Records occlusion culling of views `firstView` to `firstView +
    /// views` against `hzb`, their depth pyramid
    /// (`RenderTargets::Hzb`, one layer each, in the general layout),
    /// between the two phases; the renderer follows it with a
    /// barrier.
    virtual void RecordOcclusionCull(VkCommandBuffer /*commands*/,
                                     const GpuImage& /*hzb*/,
                                     std::uint32_t /*firstView*/,
                                     std::uint32_t /*views*/) const {}
};

}  // namespace ernie
