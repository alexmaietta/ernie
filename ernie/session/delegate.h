// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/compute_interop.h"
#include "ernie/device/pipeline.h"
#include "ernie/device/vulkan_context.h"
#include "ernie/scene/brdf.h"
#include "ernie/scene/light_pattern.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/temperature_model.h"
#include "ernie/session/frame_params.h"
#include "ernie/session/outputs.h"
#include "ernie/util/tensor.h"

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ernie {

class RenderDelegate;

/// What a render backend is and does, so applications, settings
/// panels and bindings can adapt.
struct RendererInfo {
    std::string name;  ///< The backend's name, as
                       ///< `CreateRenderDelegate` takes it.
    /// Whether consecutive steps of an unchanged scene refine one
    /// image, as a path tracer's samples accumulate, rather than each
    /// step rendering a complete image.
    bool progressive = false;
    /// The render vars `Record` can draw, such as LdrColor, Normal or
    /// PrimId.
    std::vector<std::string> viewVars;
};

/// What `RenderDelegate::WaitForTextures` left: textures still
/// loading when it timed out, and the errors of those that failed.
struct TextureWait {
    std::uint64_t loading = 0;
    std::vector<std::string> failed;

    bool Complete() const {
        return loading == 0 && failed.empty();
    }
};

/// What a delegate holds, for statistics panels.
struct RenderStats {
    std::uint64_t gpuBytes = 0;  ///< GPU memory the delegate holds.
    std::uint64_t triangles =
        0;  ///< Triangles drawn or traced by the last frame.
    /// Backend-specific counts by name, such as the rasterizer's
    /// terrain patches.
    std::vector<std::pair<std::string, std::uint64_t>> counters;
};

/// Where a mesh's writer (`RenderDelegate::SetMeshWriter`) writes
/// this frame: device addresses into the renderer's own buffers,
/// which a shader reads and writes as pointers (Slang's `float*`).
struct MeshWriteTarget {
    /// Points, x, y and z floats each, `pointCapacity` of them.
    VkDeviceAddress points = 0;
    /// Normals as the points; 0 for a mesh without normals.
    VkDeviceAddress normals = 0;
    /// Triangle indices, three uint32 per triangle into the points,
    /// `triangleCapacity` triangles.
    VkDeviceAddress indices = 0;
    /// The points and triangle indices in use, two int32: the rest
    /// are neither drawn nor bounded.
    VkDeviceAddress counts = 0;
    std::uint32_t pointCapacity = 0;
    std::uint32_t triangleCapacity = 0;
};

/// Records GPU work writing a mesh (`MeshWriteTarget`) into a frame's
/// command buffer, such as compute dispatches that build its surface.
using MeshWriter =
    std::function<void(VkCommandBuffer, const MeshWriteTarget&)>;

/// Where a DenseField's writer (`RenderDelegate::SetFieldWriter`)
/// writes this frame: its (x, y, z) float32 values, z fastest, at
/// device address `values`, which is `offset` bytes into `buffer`
/// (for copies), `shape` of them.
struct FieldWriteTarget {
    VkDeviceAddress values = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    std::array<std::uint32_t, 3> shape{};
};

/// Records GPU work writing a DenseField's values
/// (`FieldWriteTarget`) into a frame's command buffer, such as a
/// simulation's compute dispatches.
using FieldWriter =
    std::function<void(VkCommandBuffer, const FieldWriteTarget&)>;

/// Where an image sits in the deterministic random streams of
/// `sampling.h`.
struct FrameId {
    std::uint64_t frame = 0;
    std::uint32_t stream = 0;
};

/// A render backend for a `Scene`, after Hydra's HdRenderDelegate:
/// the rasterizer, and in future ray and path tracers, behind one
/// interface that applications, the GUI viewer and the Python
/// bindings drive without knowing which backend renders.
///
/// The scene is the single source of truth. A delegate pulls its
/// changes on `Sync` (each delegate keeps its own `ChangeCursor`) and
/// uploads only what changed, so there is no separate render index.
/// Every backend reads the scene with the same resolution rules
/// (`resolve.h`) and writes the same render vars through the camera's
/// sensor filters (`sensor_filters.h`).
///
/// There are two ways to render:
/// - headless and in simulation: `StepAsync` (or `Step` and `Render`)
///   renders the UsdRender products of a scene and returns their
///   render vars;
/// - interactively: `Update` then `Record` into the application's own
///   command buffer each frame, submitted after `SubmitWaits`.
///
/// Rendering is deterministic: an image's noise and samples derive
/// from the camera's `ernie:sensor:seed`, its render product's path
/// and its frame number (`NextFrame`), so the same scene renders the
/// same images on any machine and whatever else renders alongside.
class RenderDelegate {
  public:
    virtual ~RenderDelegate() = default;
    RenderDelegate(const RenderDelegate&) = delete;
    RenderDelegate& operator=(const RenderDelegate&) = delete;

    virtual const RendererInfo& Info() const = 0;

    /// The reflectance models materials may name in `ernie:brdf`.
    virtual const BrdfRegistry& Brdfs() const = 0;

    /// Registers custom sensor filter `model` for SensorFilter prims
    /// to name in `ernie:filter:type`.
    /// @throws std::invalid_argument if the model is invalid or fails
    /// to compile, or the backend runs no custom filters.
    virtual void AddSensorFilter(SensorFilterModel model) = 0;

    /// Registers custom light pattern `model` for LaserLights to name
    /// in `ernie:laser:pattern`.
    /// @throws std::invalid_argument if the model is invalid or fails
    /// to compile, or the backend draws no custom patterns (the
    /// default).
    virtual void AddLightPattern(LightPatternModel model) {
        throw std::invalid_argument(
            "This renderer draws no custom light patterns: " +
            model.id);
    }

    /// Whether light pattern `id` is registered, built in or added.
    virtual bool HasLightPattern(std::string_view /*id*/) const {
        return false;
    }

    /// Registers custom temperature model `model` for materials to
    /// name in `ernie:thermal:model`.
    /// @throws std::invalid_argument if the model is invalid or fails
    /// to compile, or the backend runs no custom models (the
    /// default).
    virtual void AddTemperatureModel(TemperatureModel model) {
        throw std::invalid_argument(
            "This renderer runs no custom temperature models: " +
            model.id);
    }

    /// The custom sensor filters registered so far.
    virtual const SensorFilterRegistry& SensorFilters() const = 0;

    /// Uploads what changed in `scene` since the last call, or
    /// everything when `scene` is not the scene of the last call. An
    /// upload that fails is not retried until its attributes change
    /// again.
    /// @throws std::invalid_argument naming the prim and attribute of
    /// data that cannot be uploaded.
    virtual void Sync(const Scene& scene) = 0;

    /// Syncs, then prepares to draw the scene as the RenderSettings
    /// prim `settings` describes into an image of `extent`, for
    /// `Record`.
    /// @throws std::invalid_argument as `Sync` and the scene's
    /// resolution (`resolve.h`).
    virtual void Update(const Scene& scene, const Path& settings,
                        VkExtent2D extent) = 0;

    /// Semaphore waits the submission of commands from `Record`
    /// needs, for CUDA copies into the delegate's buffers that may
    /// still be running; empty without CUDA inputs.
    virtual std::vector<TimelinePoint> SubmitWaits() const = 0;

    /// Maps array `attribute` of the prim at `path` for writing in
    /// place on the compute device: syncs `scene`, writes into it as
    /// that attribute a tensor over the renderer's own memory for it,
    /// holding its values as they are, and returns that tensor. Write
    /// into it on `stream` (which first waits for the frames already
    /// submitted), then `scene.Unmap(path, attribute, stream)`: the
    /// next frame draws the values without copying them. Map again
    /// each step, as the memory behind it may alternate (for motion
    /// blur).
    /// @throws std::invalid_argument if the backend maps no arrays,
    /// has no compute interop, or cannot map that attribute of that
    /// prim.
    virtual Tensor MapArray(Scene& scene, const Path& path,
                            std::string_view attribute,
                            std::uintptr_t stream = 1);

    /// Maps array `attribute` written as rows of one array over
    /// `prims` (`Scene::Write` of a PrimSet), as the single prim's
    /// `MapArray`: writes the tensor as those rows, prim i's row i,
    /// and returns it. Unmap every prim (`Scene::Unmap` of the
    /// PrimSet) after writing.
    /// @throws std::invalid_argument as the single prim's.
    virtual Tensor MapArray(Scene& scene, const PrimSet& prims,
                            std::string_view attribute,
                            std::uintptr_t stream = 1);

    /// Maps array `array` ("points", "values"...,
    /// `Overlay::ReplaceBatchArray`'s names) of batch `batch` of
    /// overlay `overlay` for writing in place on the compute device:
    /// a tensor over the renderer's own memory for it, holding its
    /// values, which the batch then holds. Write into it on `stream`
    /// (which first waits for the frames already submitted); the
    /// next frames draw it without copying. Map again each step (the
    /// same memory while the overlay keeps its batches), so the
    /// stream waits for the frames still reading it.
    /// @throws std::invalid_argument if the backend maps no arrays,
    /// has no compute interop, or has not drawn that batch yet.
    virtual Tensor MapOverlay(Scene& scene, std::string_view overlay,
                              std::size_t batch,
                              std::string_view array,
                              std::uintptr_t stream = 1);

    /// Has `writer` write the Mesh at `path` on the GPU every frame,
    /// recorded into the frame before anything reads the mesh: its
    /// points, normals, triangles and the counts of them in use,
    /// within the capacity its arrays in the scene hold (their values
    /// there are placeholders). Topology may change from frame to
    /// frame, so the mesh has no motion between frames. Without
    /// `bounds` the points in use are measured on the GPU, frames
    /// late; with them, they bound the mesh. A null writer stops
    /// writing. The writer holds across re-uploads of the mesh.
    /// @throws std::invalid_argument if the backend has no writers.
    virtual void SetMeshWriter(const Path& path, MeshWriter writer,
                               std::optional<Bounds3> bounds = {});

    /// Has `writer` write the DenseField at `path` on the GPU every
    /// frame, recorded into the frame before anything samples it: its
    /// `shape` (x, y, z) float32 values, z fastest, in place of its
    /// `ernie:values` (which may be unset). A null writer stops
    /// writing.
    /// @throws std::invalid_argument if the backend has no writers.
    virtual void SetFieldWriter(const Path& path,
                                std::array<std::uint32_t, 3> shape,
                                FieldWriter writer);

    /// The render vars `Record` can draw of the frame the last
    /// `Update` prepared: `Info().viewVars` and what that frame adds,
    /// such as the AOVs its sensor filters write ("aov:<name>").
    virtual std::vector<std::string> ViewVars() const {
        return Info().viewVars;
    }

    /// Records the frame prepared by the last `Update` and draws its
    /// render var `var` (one of `ViewVars`) into `target.color` (in
    /// the colour attachment layout, of the delegate's present
    /// format), as an image for display.
    /// @throws std::invalid_argument for a var not in `ViewVars`.
    virtual void Record(VkCommandBuffer commands,
                        const RenderingTarget& target,
                        std::string_view var) = 0;

    /// Renders `products` of `scene` (see `ResolveRenderProducts`)
    /// and returns at once: wait for or poll the operation, then
    /// fetch the outputs and map their render vars. A product through
    /// several cameras or a camera with `ernie:poses` renders a batch
    /// of views (`ViewPosesOf`), such as one per environment in
    /// reinforcement learning, in one submission; its render vars
    /// gain a leading view dimension. Each product renders into a
    /// frame of its own that later steps reuse once its outputs are
    /// dropped and unmapped. Steps do not wait for earlier ones to
    /// finish: the next step is prepared while the last renders, so a
    /// loop that maps its outputs to CUDA without waiting keeps the
    /// GPU busy (a backend may bound the steps in flight, as the
    /// rasterizer does at two).
    ///
    /// The step first moves the sensor clock on by `deltaTime`
    /// seconds (`AdvanceTime`; none by default), so its frames span
    /// the clock from the last frame's end to the new time, as
    /// ovrtx's `step` simulates [last time, last time + delta].
    /// @throws std::invalid_argument as `ResolveRenderProducts`,
    /// `Update` and `AdvanceTime`.
    virtual StepOperation
    StepAsync(const Scene& scene, std::span<const Path> products = {},
              const Path& settings = {}, double deltaTime = 0.0) = 0;

    /// `StepAsync(...).Fetch()`.
    RenderProductSetOutputs Step(const Scene& scene,
                                 std::span<const Path> products = {},
                                 const Path& settings = {},
                                 double deltaTime = 0.0);

    /// Renders the first product of the settings and returns its
    /// `LdrColor`: a uint8 (rows, cols, 4) RGBA tensor, rows top to
    /// bottom, on the CPU, or with a GPU compute `device` (CUDA or
    /// ROCm) a linear tensor there that the GPU fills without the
    /// host. A GPU result returns without waiting for the render;
    /// work reading it must first wait with
    /// `ComputeInterop::WaitForVulkan` (Python's `__dlpack__` does so
    /// on the consumer's stream; `ComputeInterop::Download` does so
    /// itself). It stays valid while the tensor lives, but not beyond
    /// the delegate's Vulkan context.
    /// @throws std::invalid_argument as `StepAsync` and
    /// `RenderVarOutput::Map`, or for a GPU device other than the
    /// delegate's.
    Tensor Render(const Scene& scene, const Path& settings = {},
                  Device device = {});

    /// The compute interop (CUDA or HIP) of the delegate's device,
    /// connected on first use; null if unavailable, with the reason
    /// in `why` if given.
    virtual ComputeInterop* Compute(std::string* why = nullptr) = 0;

    /// The prim path of a PrimId render var value, or the empty path
    /// for 0 or an unknown id.
    virtual Path PrimPath(std::uint32_t id) const = 0;

    /// The scene's average luminance in cd/m^2 a recent frame
    /// measured with metering on (`tokens::kMeter`), for automatic
    /// exposure and meters; none before one has, or from backends
    /// that do not meter.
    virtual std::optional<double> MeteredLuminance() const {
        return std::nullopt;
    }

    /// Exposes the viewport's frames (`Record`) automatically, as a
    /// viewer's automatic exposure asks: each frame moves `blend` of
    /// the way (1 at once) to the exposure that keys the metered
    /// luminance to middle grey, on the GPU where the backend can;
    /// none for the camera's own exposure.
    virtual void SetViewportAutoExposure(std::optional<float> blend) {
        static_cast<void>(blend);
    }

    /// Which spectral band `Record` shows when it presents
    /// SpectralRadiance: 0 to the camera's band count - 1, or -1 for
    /// the mean over the bands; backends that cannot choose ignore
    /// it.
    virtual void SetViewBand(int band) {
        static_cast<void>(band);
    }

    /// The render var the next `Record` draws, for `Update` to
    /// prepare the vars only drawing it needs (such as the opt-in
    /// MotionVector); call before `Update`. Backends that render
    /// every var regardless ignore it.
    virtual void SetViewVar(std::string_view var) {
        static_cast<void>(var);
    }

    /// Outlines the prims at `paths` and their descendants in
    /// `Record`'s image, as Omniverse Kit outlines its selection, in
    /// display colour `colour` (sRGB-encoded rgb, alpha its opacity);
    /// none for no paths. Backends that cannot outline ignore it.
    virtual void SetHighlight(std::span<const Path> paths,
                              glm::vec4 colour) {
        static_cast<void>(paths);
        static_cast<void>(colour);
    }

    /// The prim the last frame `Record` drew shows at pixel (`x`,
    /// `y`) of its image, from its PrimId; the empty path for the
    /// background, a pixel outside the image, or a backend that
    /// cannot tell. Waits for the device, so for clicks rather than
    /// every frame.
    virtual Path Pick(std::uint32_t x, std::uint32_t y) {
        static_cast<void>(x);
        static_cast<void>(y);
        return {};
    }

    /// The overlay item the view's last frame drew at pixel (`x`,
    /// `y`), among `scene`'s overlays (`OverlayHit`): a marker, a
    /// line, a contact of a batch. None where none is, or for a
    /// backend that cannot tell. Waits for the device, as `Pick`.
    virtual std::optional<OverlayHit> PickOverlay(const Scene& scene,
                                                  std::uint32_t x,
                                                  std::uint32_t y) {
        static_cast<void>(scene);
        static_cast<void>(x);
        static_cast<void>(y);
        return std::nullopt;
    }

    virtual RenderStats Stats() const = 0;

    /// Waits up to `timeoutSeconds` (infinity for no limit) for the
    /// textures `scene`'s materials read to load (they load in the
    /// background, `ernie:textureLoading`), so that a program can
    /// render only once none is missing. Nothing to wait for by
    /// default.
    virtual TextureWait WaitForTextures(const Scene& scene,
                                        double timeoutSeconds) {
        static_cast<void>(scene);
        static_cast<void>(timeoutSeconds);
        return {};
    }

    /// World bounds of what the delegate draws as of its last `Sync`
    /// or `Update`, for framing cameras; none if nothing has finite
    /// bounds (an infinite ground plane does not count).
    virtual std::optional<Bounds3> SceneBounds() const = 0;

    /// The sensor clock, seconds: where the last frame ended. Steps
    /// advance it by their `deltaTime` (`StepAsync`), an interactive
    /// view by `AdvanceTime` before each `Update`, and `ResetTime`
    /// sets it, as ovrtx's `step` and `reset` do. Each frame spans
    /// the clock from the last frame's end to now
    /// (`FrameParams::startTime`, `time`): what its lights, pulses
    /// and rolling shutter run through. The scene's animation has a
    /// clock of its own (`Stage::GetTimeCode`), so it can hold or
    /// loop while sensors run on.
    double Time() const {
        return time_;
    }

    /// Moves the clock on by `deltaTime` seconds, which the next
    /// frame spans.
    /// @throws std::invalid_argument for a negative or non-finite
    /// `deltaTime`.
    void AdvanceTime(double deltaTime);

    /// Sets the clock to `time` seconds; the next frame starts there.
    /// @throws std::invalid_argument for a non-finite `time`.
    void ResetTime(double time);

  protected:
    RenderDelegate() = default;

    /// The span of the clock the next frame covers, from the last
    /// frame's end to now, which the frame after starts from.
    struct TimeWindow {
        double start = 0.0;
        double end = 0.0;
    };
    TimeWindow TakeTimeWindow() {
        const TimeWindow window{frameStart_, time_};
        frameStart_ = time_;
        return window;
    }

    /// The frame of the next image of `product` (the settings' path
    /// for the interactive view): the `ernie:frame` of the resolved
    /// RenderSettings prim `settings` when authored, else how many
    /// images of the product this delegate rendered before; and the
    /// product's stream, `StreamOf` its path.
    FrameId NextFrame(const Scene& scene, const Path& settings,
                      const Path& product);

  private:
    std::map<Path, std::uint64_t> frameCounts_;
    double time_ = 0.0;
    double frameStart_ = 0.0;
};

/// The backends `CreateRenderDelegate` knows: "raster".
std::vector<std::string> RenderBackends();

/// A delegate of `backend` on `context`, drawing `Record`'s LdrColor
/// into colour attachments of `presentFormat`, with the reflectance
/// models in `brdfs`.
/// @throws std::invalid_argument for an unknown backend, or
/// std::runtime_error if the backend cannot start on `context`.
std::unique_ptr<RenderDelegate> CreateRenderDelegate(
    std::string_view backend, const VulkanContext& context,
    VkFormat presentFormat = VK_FORMAT_R8G8B8A8_UNORM,
    BrdfRegistry brdfs = BrdfRegistry::WithBuiltins());

}  // namespace ernie
