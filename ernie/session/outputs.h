// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/compute_interop.h"
#include "ernie/device/image.h"
#include "ernie/scene/path.h"
#include "ernie/util/tensor.h"

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// Where `RenderVarOutput::Map` delivers an output. The GPU kinds are
/// on the delegate's compute device (CUDA or ROCm, see
/// `ComputeInterop`); `Cuda` and `CudaArray` are their earlier names.
enum class MapDevice {
    Cpu,  ///< A host tensor; the texels are copied to the host.
    /// A linear tensor on the compute device; the GPU detiles the
    /// image into it. Its memory goes back to the renderer once every
    /// copy of the tensor is released, after the work queued by then
    /// on the map's sync stream: read it there, or synchronise before
    /// releasing it.
    Gpu,
    GpuArray,  ///< The rendered image itself as a compute array: no
               ///< copy at all.
    Cuda = Gpu,
    CudaArray = GpuArray,
};

/// A mapped render var: a tensor for `Cpu` and `Gpu`, a compute array
/// for `GpuArray`.
struct MappedRenderVar {
    MapDevice device = MapDevice::Cpu;
    Tensor tensor;
    ComputeArrayView array;
};

/// The images a backend renders one frame of a render product into,
/// by render var name: what `RenderVarOutput` reads, whatever the
/// backend.
class FrameImages {
  public:
    virtual ~FrameImages() = default;

    virtual VkExtent2D Extent() const = 0;

    /// The image of render var `name`, or null if the frame has none.
    /// A batch of views has one layer per view, or one group of a
    /// layer per band for SpectralRadiance.
    virtual GpuImage* Var(std::string_view name) = 0;

    /// The number of views the images hold.
    virtual std::uint32_t Views() const {
        return 1;
    }

    /// A render var the frame holds as one linear buffer over every
    /// view rather than an image: written by the frame's own
    /// submission, in memory a compute API views in place
    /// (`BufferMemory::Shared`), texels of `format` view after view.
    struct LinearVar {
        std::shared_ptr<GpuBuffer> buffer;
        VkFormat format = VK_FORMAT_UNDEFINED;
    };

    /// Render var `name` as a linear buffer, or an empty one when the
    /// frame holds it as an image (`Var`) or not at all.
    virtual LinearVar Linear(std::string_view /*name*/) const {
        return {};
    }
};

/// The images of a frame of a batch of views that do not fit one set
/// of images (such as more views than an image has layers):
/// consecutive chunks of the views, each its own `FrameImages`.
class ChunkedFrameImages final : public FrameImages {
  public:
    explicit ChunkedFrameImages(
        std::vector<std::unique_ptr<FrameImages>> chunks);

    VkExtent2D Extent() const override {
        return chunks_.front()->Extent();
    }

    /// The first chunk's image; `Chunks` has the others.
    GpuImage* Var(std::string_view name) override {
        return chunks_.front()->Var(name);
    }

    std::uint32_t Views() const override;

    std::span<const std::unique_ptr<FrameImages>> Chunks() const {
        return chunks_;
    }

  private:
    std::vector<std::unique_ptr<FrameImages>> chunks_;
};

/// Internal state of one rendered frame of a render product.
struct ProductFrame;

/// One render var of one rendered frame, such as `LdrColor`, named
/// after UsdRender's RenderVar.
///
/// Outputs stay valid while any `RenderVarOutput` of the frame lives,
/// and keep its image from being rendered over: drop them (after
/// unmapping) so later steps can reuse the frame. They must not
/// outlive the delegate that rendered them.
class RenderVarOutput {
  public:
    RenderVarOutput() = default;
    RenderVarOutput(std::shared_ptr<ProductFrame> frame, Path path,
                    std::string name);

    /// The RenderVar prim's path, or the product's path plus the
    /// var's name for default vars.
    const Path& GetPath() const {
        return path_;
    }

    /// The var's `sourceName`, such as `LdrColor`.
    const std::string& Name() const {
        return name_;
    }

    std::uint32_t Width() const;
    std::uint32_t Height() const;

    /// Shape of the mapped tensor: (rows, cols, channels), (rows,
    /// cols) for one channel, or (bands, rows, cols) for
    /// SpectralRadiance; a product rendering a batch of views (see
    /// `ViewPosesOf`) adds a leading dimension, the view.
    std::vector<std::int64_t> Shape() const;

    /// The number of views: 1, or the batch's.
    std::uint32_t Views() const;

    /// Element type of the mapped tensor.
    DType Type() const;

    /// Maps the output to `device`. GPU mappings make compute stream
    /// `syncStream` (DLPack numbering for the device kind; for CUDA 1
    /// legacy default, 2 per-thread default) wait on the GPU for the
    /// render, without blocking the host; a CPU mapping waits on the
    /// host. `Gpu` and `Cpu` tensors own their texels; a `GpuArray`
    /// is the frame's image, which must be unmapped before the frame
    /// is dropped, and is not available for the layered
    /// SpectralRadiance or a batch of views.
    /// @throws std::invalid_argument for a GPU device without compute
    /// interop, or a format compute arrays do not support.
    MappedRenderVar Map(MapDevice device,
                        std::uintptr_t syncStream = 1);

    /// Ends a mapping. For a `GpuArray`, Vulkan waits for the work
    /// queued on compute stream `stream` before rendering into the
    /// image again.
    void Unmap(std::uintptr_t stream = 1);

    /// Whether the var is mapped as a compute array.
    bool IsMapped() const;

  private:
    /// The var's images in the frame: one, or one per chunk of a
    /// batch's views.
    /// @throws std::invalid_argument for an empty var or one the
    /// frame lacks.
    std::vector<GpuImage*> Images() const;
    /// `Map` of a var the frame holds as a linear buffer
    /// (`FrameImages::Linear`): on the GPU the buffer itself, which
    /// keeps the frame from rendering again until released.
    MappedRenderVar MapLinear(const FrameImages::LinearVar& linear,
                              MapDevice device,
                              std::uintptr_t syncStream);

    std::shared_ptr<ProductFrame> frame_;
    Path path_;
    std::string name_;
};

/// The outputs of one render product in a step: its render vars by
/// name.
struct RenderProductOutput {
    Path product;
    std::map<std::string, RenderVarOutput> vars;
};

/// Outputs of a step by render product path.
using RenderProductSetOutputs = std::map<Path, RenderProductOutput>;

/// A submitted step: poll or wait for its GPU work, then fetch its
/// outputs.
class StepOperation {
  public:
    StepOperation(const VulkanContext& context,
                  std::uint64_t submission,
                  RenderProductSetOutputs outputs);

    /// Waits up to `timeout` (zero polls) for the step's GPU work;
    /// true once it is complete.
    bool Wait(std::chrono::nanoseconds timeout =
                  std::chrono::nanoseconds::max());

    /// Whether the step's GPU work is complete, without waiting.
    bool Done() {
        return Wait(std::chrono::nanoseconds::zero());
    }

    /// The step's outputs, waiting for its GPU work first.
    RenderProductSetOutputs Fetch();

    /// The step's outputs without waiting: mapping them to the
    /// compute device orders the consumer after the step on the GPU,
    /// and mapping them to the CPU waits on the host.
    const RenderProductSetOutputs& Outputs() const {
        return outputs_;
    }

  private:
    const VulkanContext* context_;
    std::uint64_t submission_;
    RenderProductSetOutputs outputs_;
};

}  // namespace ernie
