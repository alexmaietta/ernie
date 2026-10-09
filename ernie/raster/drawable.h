#pragma once

#include "ernie/device/pipeline.h"
#include "ernie/kernel/embedded_shaders.h"
#include "ernie/util/shader_source.h"
#include "ernie/raster/render_targets.h"
#include "ernie/raster/shadow_cascades.h"
#include "ernie/session/frame_params.h"

#include <glm/glm.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace ernie {

/// Module `projection`, which `frame.slang` imports for `projectWorld`: a pushbroom camera's
/// (projection_pushbroom.slang) or a frame camera's (projection_pinhole.slang). Passes facing the
/// camera compile with the frame's; others, such as shadow passes, with the frame camera's.
inline ShaderModuleSource ProjectionModule(bool pushbroom) {
    return {"projection", pushbroom ? shaders::kProjectionPushbroomSlang
                                    : shaders::kProjectionPinholeSlang};
}

/// The per-frame resources every raster pass binds (`frame.slang`): view and light uniforms,
/// shadow uniforms, cascaded shadow maps, their comparison sampler and a clamping linear sampler,
/// at bindings 0 to 4.
struct RasterFrame {
    DescriptorResource uniforms;
    DescriptorResource shadows;
    DescriptorResource shadowMap;
    DescriptorResource shadowSampler;
    DescriptorResource clampSampler;
};

/// The shadow casters a shadow pass draws: what every world shares (`tokens::kWorld` -1), and
/// the casters of one world, of every world, or of none.
struct ShadowCasters {
    static constexpr std::int32_t kAllWorlds = -1;
    static constexpr std::int32_t kNoWorld = -2;

    bool shared = true;               ///< Draw what every world shares, such as the terrain.
    std::int32_t world = kAllWorlds;  ///< The world whose casters to draw.

    /// Whether casters (or a view's draws) in world `of` (-1: shared) are drawn.
    bool Draws(std::int32_t of) const {
        return of == -1 ? shared : (world == kAllWorlds || world == of);
    }
};

/// Which of a view's draws a G-buffer pass records. With occlusion culling a view draws twice:
/// first the occluders, then, after a depth pyramid of them is built, the instances they do not
/// hide.
enum class DrawPhase {
    All,        ///< Everything: one pass, without occlusion culling.
    Occluders,  ///< What draws before the depth pyramid: all but the occlusion-culled instances.
    Occludees,  ///< The occlusion-culled instances, after the depth pyramid.
};

/// Something a `RasterRenderer` draws: terrain, meshes, and in future volumes or points. Each
/// frame the renderer asks every drawable for its bounds, fits the shadow views around them (the
/// sun's cascades, then the views of the other lights' shadow maps), lets each drawable prepare for
/// the final view and shadow views, and then has each record its draws into every shadow view and
/// into the main pass. Drawables bind the frame's resources at bindings 0 to
/// 4, a material's at 5 to 9 (`RasterRenderer::Material`) and their own from 10.
class RasterDrawable {
  public:
    virtual ~RasterDrawable() = default;

    /// Moves the origin the drawable's coordinates are relative to (`FrameParams::origin`): every
    /// position it gives the GPU and every bound it reports is world minus `origin`. The renderer
    /// calls it at the start of each `Update`, before `Bounds`; it changes rarely, so drawables
    /// recompute what depends on it only when it does.
    virtual void SetOrigin(const glm::dvec3& /*origin*/) {}

    /// Bounds of the drawable's geometry relative to its origin (`SetOrigin`), for fitting shadow
    /// cascades; none if it has none.
    virtual std::optional<Bounds3> Bounds() const = 0;

    /// World bounds of the drawable's geometry in world `world` (`tokens::kWorld`, not the
    /// shared -1), for fitting that world's shadow map; none if it has none there.
    virtual std::optional<Bounds3> WorldBounds(std::int32_t /*world*/) const {
        return std::nullopt;
    }

    /// The height (relative to the origin) of an unbounded floor the drawable draws, such as a
    /// ground plane, which the sun's shadows of the other drawables reach down to; none for a
    /// drawable without one.
    virtual std::optional<float> ShadowFloor() const {
        return std::nullopt;
    }

    /// Prepares the next frame once the views and shadow views are final, for example selecting
    /// levels of detail for every view (`params.View(v)` for each of `params.Views()`) and each
    /// shadow view (orthographic or perspective; only their `lightViewProjection` applies to
    /// every view). Called by `RasterRenderer::Update` while earlier frames may still be running:
    /// host data for buffers they read goes through `RasterRenderer::Uploads`, and buffers are
    /// replaced only after `vkDeviceWaitIdle`.
    virtual void Prepare(const FrameParams& params, std::span<const ShadowCascade> cascades) = 0;

    /// Records compute work the frame's draws depend on, such as culling instances on the GPU,
    /// before any shadow or G-buffer pass; the renderer follows it with a barrier. Nothing by
    /// default.
    virtual void RecordCull(VkCommandBuffer /*commands*/) const {}

    /// Records the patching of the views' camera fields in the drawable's per-view buffers from
    /// the GPU poses (`RasterRenderer::PatchViews`) when the frame's poses are on the GPU
    /// (`FrameParams::posesOnGpu`), before `RecordCull`. Nothing by default.
    virtual void RecordViewPoses(VkCommandBuffer /*commands*/) const {}

    /// Records depth-only draws of `casters` for shadow view `cascade` (an index into the views
    /// given to `Prepare`) into the open rendering pass. Drawables without worlds draw only when
    /// `casters.shared` is set.
    virtual void RecordShadow(VkCommandBuffer commands, const RasterFrame& frame,
                              std::size_t cascade, const glm::mat4& lightViewProjection,
                              const ShadowCasters& casters) const = 0;

    /// Records colour draws of view `view` in `phase` into its open G-buffer rendering pass;
    /// `frame` holds that view's uniforms. Drawables without occlusion-culled instances draw in
    /// `All` and `Occluders`.
    virtual void RecordColor(VkCommandBuffer commands, const RasterFrame& frame, std::size_t view,
                             DrawPhase phase = DrawPhase::All) const = 0;

    /// Records, into view `view`'s open motion pass (one RG16 float attachment of raster-pixel
    /// motion, the G-buffer's depth to test against, not written), the draws of what moved since
    /// the last frame with the motion each of its pixels made; `motion` is the view's
    /// motion_vectors.slang `MotionUniforms` (the last frame's view-projection). A camera pass has
    /// already written every pixel's motion from the camera's alone, so drawables that stand
    /// still, such as terrain, draw nothing: the default.
    virtual void RecordMotion(VkCommandBuffer /*commands*/, const RasterFrame& /*frame*/,
                              std::size_t /*view*/, const DescriptorResource& /*motion*/) const {}

    /// Whether the last `Prepare`'s frame culls some of this drawable's instances by occlusion,
    /// so views draw in two phases around a depth pyramid.
    virtual bool CullsOcclusion() const {
        return false;
    }

    /// Records occlusion culling of views `firstView` to `firstView + views` against `hzb`, their
    /// depth pyramid (`RenderTargets::Hzb`, one layer each, in the general layout), between the
    /// two phases; the renderer follows it with a barrier.
    virtual void RecordOcclusionCull(VkCommandBuffer /*commands*/, const GpuImage& /*hzb*/,
                                     std::uint32_t /*firstView*/, std::uint32_t /*views*/) const {}
};

}  // namespace ernie
