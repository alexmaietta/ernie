// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/logs.h"

#include "ernie/app/panel.h"
#include "ernie/app/viewer.h"
#include "ernie/device/buffer.h"
#include "ernie/device/compute_interop.h"
#include "ernie/device/image.h"
#include "ernie/device/vulkan_context.h"

#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <implot.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace ernie::app {

namespace {

/// An image or batch as the panels hold it: views side by side, each
/// rows by cols.
struct ImageShape {
    std::int64_t views = 1;
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::int64_t channels = 1;  ///< As logged: 1, 3 or 4.

    /// Channels the texture holds: 3 widen to 4.
    std::int64_t Stored() const {
        return channels == 3 ? 4 : channels;
    }
    bool operator==(const ImageShape&) const = default;
};

ImageShape ShapeOf(const std::vector<std::int64_t>& shape) {
    ImageShape image;
    switch (shape.size()) {
    case 2:
        image.rows = shape[0];
        image.cols = shape[1];
        break;
    case 3:
        image.rows = shape[0];
        image.cols = shape[1];
        image.channels = shape[2];
        break;
    case 4:
        image.views = shape[0];
        image.rows = shape[1];
        image.cols = shape[2];
        image.channels = shape[3];
        break;
    default:
        throw std::invalid_argument(
            "An image is (rows, cols), (rows, cols, channels) or "
            "(views, rows, cols, channels)");
    }
    if (image.channels != 1 && image.channels != 3 &&
        image.channels != 4) {
        throw std::invalid_argument(
            "An image has 1, 3 or 4 channels");
    }
    if (image.views < 1 || image.rows < 1 || image.cols < 1) {
        throw std::invalid_argument(
            "An image needs at least one view, row and column");
    }
    return image;
}

VkFormat FormatOf(DType type, std::int64_t channels) {
    if (type == DType::UInt8) {
        return channels == 1 ? VK_FORMAT_R8_UNORM
                             : VK_FORMAT_R8G8B8A8_UNORM;
    }
    if (type == DType::Float32) {
        return channels == 1 ? VK_FORMAT_R32_SFLOAT
                             : VK_FORMAT_R32G32B32A32_SFLOAT;
    }
    throw std::invalid_argument(
        "An image is uint8 (0 to 255) or float32 (0 to 1)");
}

const char* TypeName(DType type) {
    return type == DType::UInt8 ? "uint8" : "float32";
}

}  // namespace

/// A logged image's panel: the texture, its shape, and what the next
/// frame uploads.
class ImagePanel final : public Panel {
  public:
    ImagePanel(ViewerLogs& logs, std::string title)
        : Panel(std::move(title), Dock::RightLower), logs_(&logs) {}

    ~ImagePanel() override {
        // Detached panels released their resources when the logs went
        // (`ReleaseImages`).
        if (logs_ != nullptr) {
            Release();
            logs_->Forget(*this);
        }
    }

    /// The image no longer belongs to the logs, which are going away.
    void Detach() {
        logs_ = nullptr;
    }

    /// Makes the texture `shape` of `type`, recreating it if it was
    /// another.
    void Ensure(const ImageShape& shape, DType type) {
        if (image_ && shape == shape_ && type == type_) {
            return;
        }
        Release();
        VulkanContext& gpu = logs_->Gpu();
        const VkFormat format = FormatOf(type, shape.Stored());
        linear_ = gpu.SupportsExternalMemory() &&
                  gpu.SupportsLinearSampling(format);
        ImageDesc desc;
        desc.width =
            static_cast<std::uint32_t>(shape.views * shape.cols);
        desc.height = static_cast<std::uint32_t>(shape.rows);
        desc.format = format;
        desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        desc.shared = linear_;
        desc.linear = linear_;
        if (shape.Stored() == 1) {
            desc.components = {
                VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R,
                VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_ONE};
        }
        image_ = std::make_shared<GpuImage>(gpu, desc);
        // Sampled from the start: a linear image stays in the general
        // layout, which compute APIs and copies write in and shaders
        // sample from.
        layout_ = linear_ ? VK_IMAGE_LAYOUT_GENERAL
                          : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        gpu.Submit([&](VkCommandBuffer commands) {
            image_->Transition(commands, layout_);
        });
        texture_ = ImGui_ImplVulkan_AddTexture(
            logs_->NearestSampler().Handle(), image_->View(),
            layout_);
        shape_ = shape;
        type_ = type;
        hasContent_ = false;
        computeWritten_ = false;
    }

    /// Logs host texels: uploaded by the next frame.
    void FromHost(const Tensor& image) {
        const ImageShape shape = ShapeOf(image.Shape());
        Ensure(shape, image.Type());
        // Element strides of view, row, column and channel, whatever
        // the tensor's rank.
        const std::vector<std::int64_t>& strides = image.Strides();
        const std::size_t rank = strides.size();
        const std::int64_t viewStride = rank == 4 ? strides[0] : 0;
        const std::int64_t rowStride = strides[rank == 4 ? 1 : 0];
        const std::int64_t colStride = strides[rank == 4 ? 2 : 1];
        const std::int64_t channelStride =
            rank >= 3 ? strides[rank - 1] : 0;
        const std::size_t element = SizeOf(image.Type());
        const std::size_t width =
            static_cast<std::size_t>(shape.views * shape.cols);
        const auto stored = static_cast<std::size_t>(shape.Stored());
        pending_.assign(width * static_cast<std::size_t>(shape.rows) *
                            stored * element,
                        std::byte{0});
        const auto* from =
            static_cast<const std::byte*>(image.Data());
        const auto channels =
            static_cast<std::size_t>(shape.channels);
        for (std::int64_t view = 0; view < shape.views; ++view) {
            for (std::int64_t row = 0; row < shape.rows; ++row) {
                for (std::int64_t col = 0; col < shape.cols; ++col) {
                    const std::int64_t texel = view * viewStride +
                                               row * rowStride +
                                               col * colStride;
                    const std::size_t target =
                        (static_cast<std::size_t>(row) * width +
                         static_cast<std::size_t>(view * shape.cols +
                                                  col)) *
                        stored * element;
                    for (std::size_t c = 0; c < channels; ++c) {
                        const std::int64_t at =
                            texel + static_cast<std::int64_t>(c) *
                                        channelStride;
                        std::memcpy(
                            pending_.data() + target + c * element,
                            from + static_cast<std::size_t>(at) *
                                       element,
                            element);
                    }
                    if (channels == 3) {
                        // Opaque.
                        if (image.Type() == DType::UInt8) {
                            pending_[target + 3] = std::byte{255};
                        } else {
                            const float one = 1.0f;
                            std::memcpy(pending_.data() + target +
                                            3 * element,
                                        &one, element);
                        }
                    }
                }
            }
        }
        upload_ = true;
        how_ = ", uploaded from the host";
        hasContent_ = true;
    }

    /// Logs texels on the compute device: copied into the texture on
    /// the GPU where it is linear, else brought to the host.
    void FromCompute(const Tensor& image, ComputeInterop& compute,
                     std::uintptr_t stream) {
        const ImageShape shape = ShapeOf(image.Shape());
        Ensure(shape, image.Type());
        if (!linear_ || shape.channels == 3 ||
            !image.IsContiguous()) {
            FromHost(compute.Download(image, stream));
            return;
        }
        compute.CopyToImage(image, *image_, stream);
        computeWritten_ = true;
        how_ = ", copied on the GPU";
        hasContent_ = true;
    }

    Tensor Map(std::vector<std::int64_t> shape, DType type,
               ComputeInterop& compute, std::uintptr_t stream) {
        const ImageShape image = ShapeOf(shape);
        if (image.channels == 3) {
            throw std::invalid_argument(
                "A mapped image has 1 or 4 channels");
        }
        Ensure(image, type);
        if (!linear_) {
            throw std::runtime_error(
                "This GPU cannot sample linear shared images of this "
                "format, which a mapped "
                "image needs; log the image instead (log_image)");
        }
        // Rows of the strip, the views side by side; in elements.
        const auto element = static_cast<std::int64_t>(SizeOf(type));
        const auto pitch =
            static_cast<std::int64_t>(image_->RowPitch()) / element;
        std::vector<std::int64_t> strides;
        switch (shape.size()) {
        case 2:
            strides = {pitch, 1};
            break;
        case 3:
            strides = {pitch, image.channels, 1};
            break;
        default:
            strides = {image.cols * image.channels, pitch,
                       image.channels, 1};
            break;
        }
        // The writes follow the frames that sampled the texture.
        compute.WaitForVulkan(stream);
        computeWritten_ = true;
        how_ = ", written in place on the GPU";
        return compute.ViewImage(image_, type, std::move(shape),
                                 std::move(strides));
    }

    void Unmap(ComputeInterop& compute, std::uintptr_t stream) {
        compute.SignalVulkan(stream);
        hasContent_ = true;
    }

    void Record(VkCommandBuffer commands, std::size_t slot) {
        if (!upload_ || !image_) {
            return;
        }
        upload_ = false;
        std::unique_ptr<GpuBuffer>& staging =
            staging_[slot % staging_.size()];
        if (!staging || staging->Size() < pending_.size()) {
            staging = std::make_unique<GpuBuffer>(
                logs_->Gpu(), pending_.size(),
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                BufferMemory::Upload);
        }
        staging->Write(pending_);
        // After the frames that sampled the texture, and before this
        // one samples it.
        Barrier(commands);
        const VkImageLayout copyLayout =
            linear_ ? VK_IMAGE_LAYOUT_GENERAL
                    : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        if (!linear_) {
            image_->Transition(commands, copyLayout);
        }
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0,
                                   1};
        region.imageExtent = {image_->Desc().width,
                              image_->Desc().height, 1};
        vkCmdCopyBufferToImage(commands, staging->Handle(),
                               image_->Handle(), copyLayout, 1,
                               &region);
        if (!linear_) {
            image_->Transition(commands, layout_);
        }
        Barrier(commands);
    }

    bool ComputeWritten() const {
        return computeWritten_;
    }

    void Draw(Viewer& /*viewer*/) override {
        if (!hasContent_ || texture_ == VK_NULL_HANDLE) {
            ImGui::TextDisabled("Nothing logged yet");
            return;
        }
        ImGui::TextDisabled(
            "%lld x %lld, %lld channel%s, %s%s%s",
            static_cast<long long>(shape_.cols),
            static_cast<long long>(shape_.rows),
            static_cast<long long>(shape_.channels),
            shape_.channels == 1 ? "" : "s", TypeName(type_),
            shape_.views > 1
                ? (", " + std::to_string(shape_.views) + " views")
                      .c_str()
                : "",
            how_);
        // The views in a grid as square as it gets, filling the
        // panel's width.
        const auto views = static_cast<int>(shape_.views);
        const int columns = std::max(
            1, static_cast<int>(std::ceil(std::sqrt(views))));
        const float spacing = ImGui::GetStyle().ItemSpacing.x;
        const float width =
            std::max(16.0f, (ImGui::GetContentRegionAvail().x -
                             spacing * (columns - 1)) /
                                static_cast<float>(columns));
        const ImVec2 size(width, width *
                                     static_cast<float>(shape_.rows) /
                                     static_cast<float>(shape_.cols));
        const ImTextureRef texture(
            reinterpret_cast<ImTextureID>(texture_));
        for (int view = 0; view < views; ++view) {
            if (view % columns != 0) {
                ImGui::SameLine();
            }
            const float u0 =
                static_cast<float>(view) / static_cast<float>(views);
            const float u1 = static_cast<float>(view + 1) /
                             static_cast<float>(views);
            ImGui::Image(texture, size, ImVec2(u0, 0.0f),
                         ImVec2(u1, 1.0f));
            if (views > 1 && ImGui::IsItemHovered()) {
                ImGui::SetTooltip("View %d", view);
            }
        }
    }

    void Release() {
        if (texture_ == VK_NULL_HANDLE && !image_) {
            return;
        }
        // Frames in flight may still sample the texture.
        vkDeviceWaitIdle(logs_->Gpu().Device());
        if (texture_ != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RemoveTexture(texture_);
            texture_ = VK_NULL_HANDLE;
        }
        image_.reset();
        for (std::unique_ptr<GpuBuffer>& staging : staging_) {
            staging.reset();
        }
        hasContent_ = false;
    }

  private:
    static void Barrier(VkCommandBuffer commands) {
        VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT |
                                VK_ACCESS_2_MEMORY_READ_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT |
                                VK_ACCESS_2_MEMORY_READ_BIT;
        VkDependencyInfo dependency{
            VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(commands, &dependency);
    }

    ViewerLogs* logs_;
    std::shared_ptr<GpuImage> image_;
    VkDescriptorSet texture_ = VK_NULL_HANDLE;
    VkImageLayout layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    ImageShape shape_;
    DType type_ = DType::UInt8;
    bool linear_ = false;
    bool hasContent_ = false;
    bool computeWritten_ = false;
    /// How the last image arrived, for the panel's caption.
    const char* how_ = "";
    bool upload_ = false;
    std::vector<std::byte> pending_;
    /// A staging buffer per frame in flight, each written once its
    /// frame has finished.
    std::array<std::unique_ptr<GpuBuffer>, 2> staging_;
};

ViewerLogs::ViewerLogs(VulkanContext& gpu)
    : gpu_(gpu),
      sampler_(std::make_unique<Sampler>(
          gpu, SamplerDesc{VK_FILTER_NEAREST,
                           VK_SAMPLER_MIPMAP_MODE_NEAREST})) {}

ViewerLogs::~ViewerLogs() {
    ReleaseImages();
}

void ViewerLogs::ReleaseImages() {
    // The viewer destroys the image panels later, with its other
    // panels.
    for (auto& [name, panel] : images_) {
        panel->Release();
        panel->Detach();
    }
    images_.clear();
}

void ViewerLogs::LogScalar(std::string_view name, double value,
                           bool clear, int smoothing,
                           std::uint64_t scope) {
    auto found = scalars_.find(name);
    if (found == scalars_.end()) {
        found = scalars_.emplace(std::string(name), Scalar{}).first;
        found->second.scope = scope;
    }
    Scalar& scalar = found->second;
    if (clear) {
        scalar.samples.clear();
        scalar.pending = 0.0;
        scalar.pendingCount = 0;
    }
    scalar.smoothing = std::max(smoothing, 1);
    scalar.pending += value;
    if (++scalar.pendingCount < scalar.smoothing) {
        return;
    }
    scalar.samples.push_back(
        static_cast<float>(scalar.pending / scalar.pendingCount));
    scalar.pending = 0.0;
    scalar.pendingCount = 0;
    if (scalar.samples.size() > kHistory) {
        scalar.samples.pop_front();
    }
}

std::vector<double> ViewerLogs::Scalars(std::string_view name) const {
    const auto found = scalars_.find(name);
    if (found == scalars_.end()) {
        return {};
    }
    return {found->second.samples.begin(),
            found->second.samples.end()};
}

void ViewerLogs::LogArray(std::string_view name, const Tensor& array,
                          std::uint64_t scope) {
    if (array.Rank() < 1 || array.Rank() > 2) {
        throw std::invalid_argument(
            "A logged array is (n) or (rows, cols)");
    }
    if (array.Location().kind != DeviceKind::Cpu) {
        throw std::invalid_argument(
            "A logged array reaches ViewerLogs on the host");
    }
    Array logged;
    logged.rows =
        array.Rank() == 2 ? static_cast<int>(array.Shape()[0]) : 1;
    logged.cols = static_cast<int>(array.Shape()[array.Rank() - 1]);
    logged.scope = scope;
    logged.values.resize(static_cast<std::size_t>(logged.rows) *
                         static_cast<std::size_t>(logged.cols));
    const std::vector<std::int64_t>& strides = array.Strides();
    const std::int64_t rowStride = array.Rank() == 2 ? strides[0] : 0;
    const std::int64_t colStride = strides[array.Rank() - 1];
    for (int r = 0; r < logged.rows; ++r) {
        for (int c = 0; c < logged.cols; ++c) {
            const std::int64_t at = r * rowStride + c * colStride;
            float value = 0.0f;
            switch (array.Type()) {
            case DType::Float32:
                value = static_cast<const float*>(array.Data())[at];
                break;
            case DType::Float64:
                value = static_cast<float>(
                    static_cast<const double*>(array.Data())[at]);
                break;
            case DType::Int32:
                value = static_cast<float>(
                    static_cast<const std::int32_t*>(
                        array.Data())[at]);
                break;
            case DType::Int64:
                value = static_cast<float>(
                    static_cast<const std::int64_t*>(
                        array.Data())[at]);
                break;
            case DType::UInt8:
                value = static_cast<const std::uint8_t*>(
                    array.Data())[at];
                break;
            default:
                throw std::invalid_argument(
                    "A logged array holds float or integer numbers");
            }
            logged.values[static_cast<std::size_t>(r * logged.cols +
                                                   c)] = value;
        }
    }
    auto found = arrays_.find(name);
    if (found != arrays_.end()) {
        logged.scope = found->second.scope;
        found->second = std::move(logged);
    } else {
        arrays_.emplace(std::string(name), std::move(logged));
    }
}

ImagePanel& ViewerLogs::Panel(Viewer& viewer, std::string_view name,
                              std::uint64_t scope) {
    if (const auto found = images_.find(name);
        found != images_.end()) {
        return *found->second;
    }
    if (viewer.FindPanel(name) != nullptr) {
        throw std::invalid_argument(
            "A panel titled '" + std::string(name) +
            "' exists: log the image under another name");
    }
    const std::uint64_t previous = viewer.Scope();
    viewer.SetScope(scope);
    auto& panel = static_cast<ImagePanel&>(viewer.AddPanel(
        std::make_unique<ImagePanel>(*this, std::string(name))));
    viewer.SetScope(previous);
    viewer.FocusPanel(name);
    images_.emplace(std::string(name), &panel);
    return panel;
}

void ViewerLogs::LogImage(Viewer& viewer, std::string_view name,
                          const Tensor& image,
                          ComputeInterop* compute,
                          std::uintptr_t stream,
                          std::uint64_t scope) {
    ImagePanel& panel = Panel(viewer, name, scope);
    if (image.Location().kind == DeviceKind::Cpu) {
        panel.FromHost(image);
    } else if (compute != nullptr &&
               image.Location() == compute->Location()) {
        panel.FromCompute(image, *compute, stream);
    } else {
        throw std::invalid_argument(
            "A logged image is on the host or the viewer's GPU");
    }
}

Tensor ViewerLogs::MapImage(Viewer& viewer, std::string_view name,
                            std::vector<std::int64_t> shape,
                            DType type, ComputeInterop& compute,
                            std::uintptr_t stream,
                            std::uint64_t scope) {
    return Panel(viewer, name, scope)
        .Map(std::move(shape), type, compute, stream);
}

void ViewerLogs::UnmapImage(std::string_view name,
                            ComputeInterop& compute,
                            std::uintptr_t stream) {
    const auto found = images_.find(name);
    if (found == images_.end()) {
        throw std::invalid_argument("No image '" + std::string(name) +
                                    "' is mapped");
    }
    found->second->Unmap(compute, stream);
}

void ViewerLogs::Remove(Viewer& viewer, std::string_view name) {
    scalars_.erase(std::string(name));
    arrays_.erase(std::string(name));
    if (images_.contains(name)) {
        viewer.RemovePanel(std::string(name));  // forgets it (Forget)
    }
}

void ViewerLogs::RemoveScope(std::uint64_t scope) {
    std::erase_if(scalars_, [scope](const auto& entry) {
        return entry.second.scope == scope;
    });
    std::erase_if(arrays_, [scope](const auto& entry) {
        return entry.second.scope == scope;
    });
}

void ViewerLogs::Forget(ImagePanel& panel) {
    std::erase_if(images_, [&panel](const auto& entry) {
        return entry.second == &panel;
    });
}

void ViewerLogs::DrawPlots() {
    if (scalars_.empty() && arrays_.empty()) {
        ImGui::TextDisabled(
            "Nothing logged yet: log_scalar and log_array plot here");
        return;
    }
    for (const auto& [name, scalar] : scalars_) {
        if (!ImGui::CollapsingHeader(
                name.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            continue;
        }
        if (ImPlot::BeginPlot(("##" + name).c_str(),
                              ImVec2(-1.0f, 160.0f),
                              ImPlotFlags_NoLegend)) {
            ImPlot::SetupAxes(nullptr, nullptr,
                              ImPlotAxisFlags_AutoFit,
                              ImPlotAxisFlags_AutoFit);
            const std::vector<float> samples(scalar.samples.begin(),
                                             scalar.samples.end());
            ImPlot::PlotLine(name.c_str(), samples.data(),
                             static_cast<int>(samples.size()));
            ImPlot::EndPlot();
        }
        if (!scalar.samples.empty()) {
            ImGui::TextDisabled(
                "latest %.6g",
                static_cast<double>(scalar.samples.back()));
        }
    }
    for (const auto& [name, array] : arrays_) {
        if (!ImGui::CollapsingHeader(
                name.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            continue;
        }
        const auto [low, high] = std::minmax_element(
            array.values.begin(), array.values.end());
        const double lowest = array.values.empty() ? 0.0 : *low;
        const double highest = array.values.empty()
                                   ? 1.0
                                   : std::max(*high, *low + 1e-12f);
        const float height = std::clamp(
            ImGui::GetContentRegionAvail().x *
                static_cast<float>(array.rows) /
                static_cast<float>(std::max(array.cols, 1)),
            40.0f, 320.0f);
        ImPlot::PushColormap(ImPlotColormap_Viridis);
        if (ImPlot::BeginPlot(
                ("##" + name).c_str(), ImVec2(-80.0f, height),
                ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText)) {
            ImPlot::SetupAxes(nullptr, nullptr,
                              ImPlotAxisFlags_NoDecorations,
                              ImPlotAxisFlags_NoDecorations);
            ImPlot::PlotHeatmap(name.c_str(), array.values.data(),
                                array.rows, array.cols, lowest,
                                highest, nullptr);
            ImPlot::EndPlot();
        }
        ImGui::SameLine();
        ImPlot::ColormapScale(("##scale" + name).c_str(), lowest,
                              highest, ImVec2(70.0f, height));
        ImPlot::PopColormap();
    }
}

void ViewerLogs::Record(VkCommandBuffer commands, std::size_t slot) {
    for (auto& [name, panel] : images_) {
        panel->Record(commands, slot);
    }
}

bool ViewerLogs::SamplesComputeImages() const {
    return std::any_of(images_.begin(), images_.end(),
                       [](const auto& entry) {
                           return entry.second->ComputeWritten();
                       });
}

}  // namespace ernie::app
