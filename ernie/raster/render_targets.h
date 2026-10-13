// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/image.h"
#include "ernie/device/pipeline.h"
#include "ernie/device/vulkan_context.h"
#include "ernie/scene/sensor_filters.h"
#include "ernie/session/outputs.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// The images one frame renders into: the G-buffer and depth of the
/// raster passes, the spectral radiance of the shading pass, and
/// every render var the sensor filters write. Render var images stay
/// in the general layout between frames; with `shared` set the
/// single-view ones live in memory CUDA can map in place.
///
/// For a batch of `views` views every image has one layer per view
/// (spectral images one group of `bands` layers per view,
/// `GpuImage::GroupView`), so the views of a frame render into one
/// set of images: `LayerView(view)` is a view's G-buffer, depth or
/// render var.
class RenderTargets : public FrameImages {
  public:
    /// G-buffer attachment formats: normal and visibility, albedo
    /// scale and modulation, ids, and the surface's tangent
    /// (gbuffer.slang's `packTangent`), held only where a material
    /// reads it (`SetTangents`).
    static constexpr std::array<VkFormat, 4> kGBufferFormats{
        VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R16G16_UNORM};
    static constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
    /// Camera views' depth is reverse-Z (`CameraView`): cleared to 0,
    /// the far plane, nearer surfaces greater; the motion passes
    /// redraw the colour pass's own depths. Light shadow maps keep
    /// standard depth (cleared to 1, `VK_COMPARE_OP_LESS`).
    static constexpr float kDepthClear = 0.0f;
    static constexpr VkCompareOp kDepthCompare =
        VK_COMPARE_OP_GREATER;
    static constexpr VkCompareOp kDepthCompareEqual =
        VK_COMPARE_OP_GREATER_OR_EQUAL;

    /// @throws std::runtime_error if allocation fails, or
    /// std::invalid_argument for a zero extent, a band count outside
    /// 1 to 64, no views, or more layers than the device allows
    /// (`MaxViews`). Holds the render vars `vars` (`RequiredVars`;
    /// empty for every var); the others get placeholders of one texel
    /// per view that passes bind but do not write. `aovs` names the
    /// AOV slots in order (`AovOutputsOf` of the frame's chain);
    /// `vars` may hold slots by their `kAovSlotNames`, which `Var`
    /// also finds as "aov:<name>". A camera rendered on cube faces
    /// (wide_view.h) draws `faces` raster layers per view: the
    /// G-buffer, depth, raster radiance and the raster images below
    /// hold view `v`'s face `f` at layer `v * faces + f`
    /// (`RasterLayers` of them), while the render vars, AOVs and
    /// sensor images keep a layer per view.
    RenderTargets(const VulkanContext& context, VkExtent2D extent,
                  int bands, bool shared, std::uint32_t views = 1,
                  std::span<const std::string_view> vars = {},
                  std::span<const std::string> aovs = {},
                  std::uint32_t faces = 1);

    /// The most views one set of targets with `bands` bands and
    /// `faces` raster layers per view can hold on `context`'s device:
    /// its `maxImageArrayLayers` over the bands and faces.
    static std::uint32_t MaxViews(const VulkanContext& context,
                                  int bands, std::uint32_t faces = 1);

    /// Raster layers per view: 1, or a cube camera's faces.
    std::uint32_t Faces() const {
        return faces_;
    }

    /// The raster images' layers: the views' faces.
    std::uint32_t RasterLayers() const {
        return views_ * faces_;
    }

    VkExtent2D Extent() const override {
        return extent_;
    }

    int Bands() const {
        return bands_;
    }

    std::uint32_t Views() const override {
        return views_;
    }

    bool Shared() const {
        return shared_;
    }

    /// Records the transitions that make the G-buffer and depth
    /// attachments.
    void PrepareGBuffer(VkCommandBuffer commands);

    /// Every view's G-buffer and depth attachments for one layered
    /// `BeginRendering`, each view drawing into its layer, clearing
    /// the ids to "background", or keeping what an earlier pass drew
    /// with `load`. Without tangents (`SetTangents`), their
    /// attachment is a null view, which takes no writes, where the
    /// device allows (`VulkanContext::SupportsUnusedAttachments`).
    RenderingTarget GBufferTarget(bool load = false) const;

    /// Whether the G-buffer holds the surfaces' tangents, which
    /// anisotropic materials read: their image is made at the
    /// targets' size the first time, a one-texel placeholder until
    /// then, so frames whose materials read none neither store nor
    /// clear them (on devices that cannot leave an attachment
    /// unbound, they are always held and written).
    void SetTangents(bool tangents);

    bool Tangents() const {
        return tangents_;
    }

    /// The views' hierarchical depth (Hi-Z) for occlusion culling,
    /// created on first use: per view, mips of the farthest depth
    /// below each texel, level 0 the largest power-of-two size within
    /// the image, so a texel of level `l` bounds 2^l x 2^l of it at
    /// most twice over.
    GpuImage& Hzb();

    /// The images consecutive sensor kernels pass on for spatial
    /// filters (sensor.slang's bindings 16 to 19), set `set` (0 or 1,
    /// alternating by kernel), in the general layout: the spectrum at
    /// each sensor pixel (a group of `Bands` layers per view) with
    /// `spectrum`, else the four channel domains (four RGBA32F layers
    /// per view). Created on first use with `full`; otherwise a
    /// one-texel placeholder that kernels bind but do not touch. With
    /// `pyramid`, the channel images have every mip level, for
    /// spatial filters reading a pyramid (`SensorPlan::pyramids`).
    GpuImage& SensorImage(bool spectrum, std::uint32_t set, bool full,
                          bool pyramid = false);

    /// Records the transitions that let compute passes read the
    /// G-buffer, depth and raster radiance and write the radiance and
    /// the render vars.
    void PrepareCompute(VkCommandBuffer commands);

    /// Records what the transparent surfaces' pass draws into: their
    /// G-buffer (the opaque one's formats) cleared, and their depth,
    /// the opaque depth copied, so the nearest transparent surface in
    /// front of the opaque ones stays. Made the first time a frame
    /// draws transparent surfaces; their radiance (as the raster
    /// radiance) too.
    void PrepareTransparency(VkCommandBuffer commands);

    /// The transparent pass's attachments, after
    /// `PrepareTransparency`.
    RenderingTarget TransparentTarget() const;

    /// Records the transitions that let compute passes read the
    /// transparent G-buffer and depth and write their radiance.
    void PrepareTransparentCompute(VkCommandBuffer commands);

    GpuImage& TransparentGBuffer(std::size_t index) {
        return *transparentGBuffer_.at(index);
    }

    GpuImage& TransparentDepth() {
        return *transparentDepth_;
    }

    /// The transparent surfaces' own radiance, then, composited over
    /// what lies behind them, what the raster radiance takes there.
    GpuImage& TransparentRadiance() {
        return *transparentRadiance_;
    }

    GpuImage& GBuffer(std::size_t index) {
        return *gbuffer_.at(index);
    }

    GpuImage& Depth() {
        return *depth_;
    }

    /// Spectral radiance of the raster image, before the lens filter:
    /// a group of `Bands` layers per raster layer, and with Stokes
    /// planes (`SetStokes`) two more blocks of as many, its linear
    /// Stokes Q then U, each `StokesOffset` layers on.
    GpuImage& RasterRadiance() {
        return *rasterRadiance_;
    }

    /// The volumes' temperature for the Temperature var, which the
    /// volume march writes and the sensor reads (sensor.slang's
    /// `volumeTemperatureSlot`): RG32F, a layer per raster layer, x
    /// the sum of each step's absorbed share reaching the eye times
    /// its kelvin and y the sum of the shares; a texel per layer
    /// unless the frame holds Temperature (`HoldsMediaTemperature`).
    GpuImage& MediaTemperature() {
        return *mediaTemperature_;
    }
    bool HoldsMediaTemperature() const {
        return holdsMediaTemperature_;
    }

    /// Whether the raster radiance holds Stokes Q and U beside its
    /// intensity, for a camera with a polarizer; changing it waits
    /// for the device and makes the radiance images anew.
    /// @throws std::invalid_argument when the three blocks are more
    /// layers than the device's images hold.
    void SetStokes(bool stokes);
    bool Stokes() const {
        return stokes_;
    }

    /// Whether the raster radiance holds 16-bit floats rather than
    /// 32-bit (`FrameParams::halfRadiance`); changing it waits for
    /// the device and makes the image anew.
    void SetHalfRadiance(bool half);
    bool HalfRadiance() const {
        return halfRadiance_;
    }
    /// The layers from a raster radiance layer to its Stokes Q, and
    /// on to U: one block of every raster layer's bands.
    std::uint32_t StokesOffset() const {
        return RasterLayers() * static_cast<std::uint32_t>(bands_);
    }

    /// With motion blur, the sum of the sub-frames' raster radiance
    /// so far, shaped as `RasterRadiance`; created on first use.
    GpuImage& RadianceSum();

    /// With `full`, per raster pixel (RGBA32 float, a layer per
    /// view): xy its motion since the last frame in raster pixels,
    /// which the motion passes draw for the MotionVector var, and z
    /// the share of its exposure prims covered, which motion blur
    /// accumulates and the sensor blends the clear colour by; created
    /// on first use. Otherwise a one-texel placeholder that the
    /// sensor binds but does not read.
    GpuImage& Raster(bool full);

    /// The image of render var `name`, or null for an unknown name or
    /// one the targets do not hold.
    GpuImage* Var(std::string_view name) override;

    /// The image passes bind for render var `name`: its own, or its
    /// placeholder.
    /// @throws std::invalid_argument for an unknown name.
    GpuImage& Bound(std::string_view name);

    /// The image the sensor binds for AOV slot `slot`: its own when
    /// held, else a placeholder.
    GpuImage& BoundAov(std::size_t slot) {
        return *aovImages_.at(slot);
    }

    /// The AOV slots' names, as the targets were made with.
    const std::vector<std::string>& AovNames() const {
        return aovNames_;
    }

    /// The render vars the targets hold.
    const std::vector<std::string_view>& Vars() const {
        return held_;
    }

    /// Render vars a frame also holds as linear buffers over all its
    /// views, which the sensor writes beside the images (sensor.slang's
    /// `SensorLinear`), so mapping them to a compute API copies
    /// nothing (`FrameImages::Linear`).
    struct LinearVars {
        /// The vars the sensor can write so: `VarNames` up to
        /// BrightnessTemperature (sensor.slang's `kLinearVars`).
        static constexpr std::size_t kVars = 16;
        /// Then the AOV slots.
        static constexpr std::size_t kEntries = kVars + kMaxAovs;
        /// Per var, then AOV slot, its buffer (shared memory), or
        /// none.
        std::array<std::shared_ptr<GpuBuffer>, kEntries> buffers;
        /// The buffers' addresses, as the sensor reads them.
        std::unique_ptr<GpuBuffer> table;
        /// The vars and AOV slots held so, a bit each in `buffers`'
        /// order.
        std::uint32_t held = 0;
        /// Those the frame's sensor wrote: overlays drawn over
        /// LdrColor after it leave that out
        /// (`RasterRenderer::RecordViews`).
        std::uint32_t written = 0;
    };

    /// Render var `name`'s entry in `LinearVars::buffers`: a var's
    /// index in `VarNames`, or an AOV slot's (by slot name or as
    /// "aov:<output>" of `aovs`, the outputs in slot order) after
    /// them; none for another.
    static std::optional<std::size_t>
    LinearEntry(std::string_view name,
                std::span<const std::string> aovs);

    /// Linear buffers of the `LinearVars` entries in `entries` (a bit
    /// each) for `views` views of `extent` and `bands` bands, in
    /// shared memory.
    static std::shared_ptr<LinearVars>
    MakeLinearVars(const VulkanContext& context, VkExtent2D extent,
                   std::uint32_t views, int bands,
                   std::uint32_t entries);

    /// Makes the sensor write the vars `linear` holds into its
    /// buffers too (each chunk of a frame's views shares the
    /// frame's), or no longer with null.
    void SetLinearVars(std::shared_ptr<LinearVars> linear) {
        linear_ = std::move(linear);
    }
    LinearVars* GetLinearVars() const {
        return linear_.get();
    }

    LinearVar Linear(std::string_view name) const override;

    /// Every render var name, in binding order of the filters.
    static std::span<const std::string_view> VarNames();

    std::uint64_t GpuBytes() const;

  private:
    const VulkanContext& context_;
    VkExtent2D extent_;
    int bands_;
    std::uint32_t views_;
    std::uint32_t faces_ = 1;
    bool shared_;
    std::array<std::unique_ptr<GpuImage>, 4> gbuffer_;
    bool tangents_ = false;  ///< See `SetTangents`.
    /// See `PrepareTransparency`; null until a frame draws
    /// transparent surfaces.
    std::array<std::unique_ptr<GpuImage>, 4> transparentGBuffer_;
    std::unique_ptr<GpuImage> transparentDepth_;
    std::unique_ptr<GpuImage> transparentRadiance_;
    std::unique_ptr<GpuImage> depth_;
    std::unique_ptr<GpuImage> rasterRadiance_;
    std::unique_ptr<GpuImage> mediaTemperature_;
    bool holdsMediaTemperature_ = false;
    std::unique_ptr<GpuImage> radianceSum_;
    bool stokes_ = false;        ///< See `SetStokes`.
    bool halfRadiance_ = false;  ///< See `SetHalfRadiance`.
    std::unique_ptr<GpuImage> raster_;
    std::unique_ptr<GpuImage> rasterPlaceholder_;
    std::unique_ptr<GpuImage> hzb_;
    /// `SensorImage`s: spectrum then channels, two sets each, and
    /// their placeholders.
    std::array<std::unique_ptr<GpuImage>, 4> sensorImages_;
    std::array<std::unique_ptr<GpuImage>, 2> sensorPlaceholders_;
    std::vector<std::unique_ptr<GpuImage>> vars_;
    std::vector<bool> holds_;  ///< Per var of `VarNames`, whether
                               ///< `vars_` holds it or a placeholder.
    std::vector<std::string_view> held_;
    std::array<std::unique_ptr<GpuImage>, kMaxAovs> aovImages_;
    std::array<bool, kMaxAovs> holdsAov_{};
    std::vector<std::string> aovNames_;
    std::shared_ptr<LinearVars> linear_;  ///< `SetLinearVars`'.
};

}  // namespace ernie
