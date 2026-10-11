// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/util/tensor.h"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {
class ComputeInterop;
class GpuBuffer;
class GpuImage;
class Sampler;
class VulkanContext;
}  // namespace ernie

namespace ernie::app {

class Viewer;
class ImagePanel;

/// What the viewer's logging calls collect (`Viewer::LogScalar`,
/// `LogArray`, `LogImage`), after Newton's viewers' `log_scalar`,
/// `log_array` and `log_image`: scalar signals and arrays drawn in
/// the Plots panel, and images each in a panel of its own, as
/// textures the GPU samples where they were written.
///
/// An image lives in a linear shared Vulkan image where the device
/// samples those (`VulkanContext::SupportsLinearSampling`): a CUDA
/// array view of it (`Map`) is the texture itself, written in place
/// with no copy, and CUDA images logged from elsewhere arrive by one
/// device-to-device copy. Host images are staged through the frame.
/// Everything logged belongs to the registration scope current when
/// first logged (`Viewer::SetScope`).
///
/// References:
/// - Newton's viewers (`newton.viewer`: `log_scalar`, `log_array`,
///   `log_image`), https://github.com/newton-physics/newton
class ViewerLogs {
  public:
    /// Samples a scalar signal keeps.
    static constexpr std::size_t kHistory = 1000;

    explicit ViewerLogs(VulkanContext& gpu);
    ~ViewerLogs();
    ViewerLogs(const ViewerLogs&) = delete;
    ViewerLogs& operator=(const ViewerLogs&) = delete;

    void LogScalar(std::string_view name, double value, bool clear,
                   int smoothing, std::uint64_t scope);
    std::vector<double> Scalars(std::string_view name) const;

    /// `array` on the host: (n) or (rows, cols) numbers.
    void LogArray(std::string_view name, const Tensor& array,
                  std::uint64_t scope);

    /// Logs `image` (on the host or `compute`'s device) into the
    /// image panel `name`, which `viewer` gets in `scope` if it is
    /// new.
    void LogImage(Viewer& viewer, std::string_view name,
                  const Tensor& image, ComputeInterop* compute,
                  std::uintptr_t stream, std::uint64_t scope);

    /// The image panel `name`'s texture as a tensor on `compute`'s
    /// device to write in place.
    Tensor MapImage(Viewer& viewer, std::string_view name,
                    std::vector<std::int64_t> shape, DType type,
                    ComputeInterop& compute, std::uintptr_t stream,
                    std::uint64_t scope);
    void UnmapImage(std::string_view name, ComputeInterop& compute,
                    std::uintptr_t stream);

    /// Forgets `name`'s plot or image panel.
    void Remove(Viewer& viewer, std::string_view name);

    /// Forgets the plots logged in `scope`; its image panels go with
    /// the scope's panels.
    void RemoveScope(std::uint64_t scope);

    /// Draws the Plots panel's contents.
    void DrawPlots();

    /// Records this frame's host uploads into `commands`, using frame
    /// slot `slot`'s staging.
    void Record(VkCommandBuffer commands, std::size_t slot);

    /// Whether the frame samples an image a compute device writes, so
    /// its submission must signal the compute side
    /// (`ComputeInterop::NextVulkanSignal`) before it writes the
    /// image again.
    bool SamplesComputeImages() const;

    /// Called by an image panel going away.
    void Forget(ImagePanel& panel);

    /// Releases every image panel's texture and image, while ImGui
    /// and the device still exist: for the viewer's shutdown, which
    /// destroys its panels afterwards.
    void ReleaseImages();

    VulkanContext& Gpu() {
        return gpu_;
    }
    const Sampler& NearestSampler() const {
        return *sampler_;
    }

  private:
    struct Scalar {
        std::deque<float> samples;
        double pending = 0.0;
        int pendingCount = 0;
        int smoothing = 1;
        std::uint64_t scope = 0;
    };
    struct Array {
        std::vector<float> values;
        int rows = 0;
        int cols = 0;
        std::uint64_t scope = 0;
    };

    ImagePanel& Panel(Viewer& viewer, std::string_view name,
                      std::uint64_t scope);

    VulkanContext& gpu_;
    std::unique_ptr<Sampler> sampler_;
    std::map<std::string, Scalar, std::less<>> scalars_;
    std::map<std::string, Array, std::less<>> arrays_;
    std::map<std::string, ImagePanel*, std::less<>> images_;
};

}  // namespace ernie::app
