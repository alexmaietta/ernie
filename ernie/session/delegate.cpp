#include "ernie/session/delegate.h"

#include "ernie/scene/sensor_filters.h"
#include "ernie/scene/tokens.h"
#include "ernie/session/resolve.h"
#include "ernie/session/sampling.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace ernie {

RenderProductSetOutputs RenderDelegate::Step(const Scene& scene, std::span<const Path> products,
                                             const Path& settings) {
    return StepAsync(scene, products, settings).Fetch();
}

Tensor RenderDelegate::Render(const Scene& scene, const Path& settings, Device device) {
    const bool cuda = device.kind == DeviceKind::Cuda;
    if (cuda) {
        std::string why;
        CudaInterop* interop = Cuda(&why);
        if (interop == nullptr) {
            throw std::invalid_argument("A CUDA render needs CUDA interop: " + why);
        }
        if (interop->Device() != device.id) {
            throw std::invalid_argument("The renderer's Vulkan device is CUDA device " +
                                        std::to_string(interop->Device()));
        }
    }
    StepOperation step = StepAsync(scene, {}, settings);
    RenderProductSetOutputs outputs = cuda ? step.Outputs() : step.Fetch();
    for (auto& [path, output] : outputs) {
        if (const auto it = output.vars.find(std::string(render_vars::kLdrColor));
            it != output.vars.end()) {
            return it->second.Map(cuda ? MapDevice::Cuda : MapDevice::Cpu).tensor;
        }
    }
    throw std::invalid_argument("The render products have no LdrColor render var");
}

FrameId RenderDelegate::NextFrame(const Scene& scene, const Path& settings, const Path& product) {
    std::uint64_t& count = frameCounts_[product];
    FrameId id;
    id.stream = StreamOf(product.String());
    const std::optional<std::uint64_t> authored = FrameOf(scene, settings);
    id.frame = authored.value_or(count);
    count += 1;
    return id;
}

}  // namespace ernie
