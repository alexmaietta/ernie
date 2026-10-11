// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/session/outputs.h"

#include "ernie/scene/sensor_filters.h"
#include "ernie/session/product_frame.h"
#include "ernie/util/profile.h"

#include <stdexcept>
#include <utility>

namespace ernie {

RenderVarOutput::RenderVarOutput(std::shared_ptr<ProductFrame> frame,
                                 Path path, std::string name)
    : frame_(std::move(frame)), path_(std::move(path)),
      name_(std::move(name)) {}

namespace {

struct TexelLayout {
    DType type;
    std::int64_t channels;
};

TexelLayout LayoutOf(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8G8B8A8_UNORM:
        return {DType::UInt8, 4};
    case VK_FORMAT_R16G16B16A16_UINT:
        return {DType::UInt16, 4};
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return {DType::Float32, 4};
    case VK_FORMAT_R32_SFLOAT:
        return {DType::Float32, 1};
    case VK_FORMAT_R32G32_SFLOAT:
        return {DType::Float32, 2};
    case VK_FORMAT_R32_UINT:
        return {DType::UInt32, 1};
    default:
        throw std::invalid_argument(
            "Render var with an unsupported format");
    }
}

/// The shape of one view of `image`, which holds `views` views:
/// (bands, rows, cols) for the layered SpectralRadiance, (rows, cols)
/// for one channel, else (rows, cols, channels).
std::vector<std::int64_t> ViewShapeOf(const GpuImage& image,
                                      std::uint32_t views,
                                      std::string_view name) {
    const ImageDesc& desc = image.Desc();
    const std::int64_t channels = LayoutOf(desc.format).channels;
    if (name == render_vars::kSpectralRadiance) {
        return {desc.layers / views, desc.height, desc.width};
    }
    if (channels == 1) {
        return {desc.height, desc.width};
    }
    return {desc.height, desc.width, channels};
}

/// Bytes of every layer of `image`, tightly packed.
VkDeviceSize BytesOf(const GpuImage& image) {
    const ImageDesc& desc = image.Desc();
    return VkDeviceSize{desc.width} * desc.height * desc.layers *
           GpuImage::TexelBytes(desc.format);
}

/// Records a tightly packed copy of every layer of `image` into
/// `buffer` from `offset`, leaving the image in the general layout.
void RecordCopy(VkCommandBuffer commands, GpuImage& image,
                GpuBuffer& buffer, VkDeviceSize offset = 0) {
    const ImageDesc& desc = image.Desc();
    const VkDeviceSize layerBytes = VkDeviceSize{desc.width} *
                                    desc.height *
                                    GpuImage::TexelBytes(desc.format);
    image.Transition(commands, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    std::vector<VkBufferImageCopy> regions;
    for (std::uint32_t layer = 0; layer < desc.layers; ++layer) {
        VkBufferImageCopy region{};
        region.bufferOffset = offset + layer * layerBytes;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                   layer, 1};
        region.imageExtent = {desc.width, desc.height, 1};
        regions.push_back(region);
    }
    vkCmdCopyImageToBuffer(
        commands, image.Handle(),
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer.Handle(),
        static_cast<std::uint32_t>(regions.size()), regions.data());
    image.Transition(commands, VK_IMAGE_LAYOUT_GENERAL);
}

}  // namespace

std::uint32_t RenderVarOutput::Width() const {
    return frame_->images->Extent().width;
}

std::uint32_t RenderVarOutput::Height() const {
    return frame_->images->Extent().height;
}

std::uint32_t RenderVarOutput::Views() const {
    return frame_ == nullptr ? 1 : frame_->images->Views();
}

bool RenderVarOutput::IsMapped() const {
    return frame_ != nullptr && frame_->arrayMaps > 0;
}

std::vector<std::int64_t> RenderVarOutput::Shape() const {
    if (frame_ == nullptr) {
        throw std::invalid_argument("Empty render var");
    }
    if (const FrameImages::LinearVar linear =
            frame_->images->Linear(name_);
        linear.buffer) {
        const VkExtent2D extent = frame_->images->Extent();
        const std::int64_t channels =
            LayoutOf(linear.format).channels;
        std::vector<std::int64_t> shape{extent.height, extent.width};
        if (channels > 1) {
            shape.push_back(channels);
        }
        if (frame_->batched) {
            shape.insert(shape.begin(), static_cast<std::int64_t>(
                                            frame_->images->Views()));
        }
        return shape;
    }
    const std::vector<GpuImage*> images = Images();
    const FrameImages& first = *frame_->images;
    const auto* chunked =
        dynamic_cast<const ChunkedFrameImages*>(&first);
    const std::uint32_t chunkViews =
        chunked != nullptr ? chunked->Chunks().front()->Views()
                           : first.Views();
    std::vector<std::int64_t> shape =
        ViewShapeOf(*images.front(), chunkViews, name_);
    if (frame_->batched) {
        shape.insert(shape.begin(),
                     static_cast<std::int64_t>(first.Views()));
    }
    return shape;
}

DType RenderVarOutput::Type() const {
    if (frame_ != nullptr) {
        if (const FrameImages::LinearVar linear =
                frame_->images->Linear(name_);
            linear.buffer) {
            return LayoutOf(linear.format).type;
        }
    }
    return LayoutOf(Images().front()->Desc().format).type;
}

std::vector<GpuImage*> RenderVarOutput::Images() const {
    if (frame_ == nullptr) {
        throw std::invalid_argument("Empty render var");
    }
    std::vector<GpuImage*> images;
    if (const auto* chunked = dynamic_cast<const ChunkedFrameImages*>(
            frame_->images.get())) {
        for (const std::unique_ptr<FrameImages>& chunk :
             chunked->Chunks()) {
            images.push_back(chunk->Var(name_));
        }
    } else {
        images.push_back(frame_->images->Var(name_));
    }
    for (const GpuImage* image : images) {
        if (image == nullptr) {
            throw std::invalid_argument(
                "The frame has no render var " + name_);
        }
    }
    return images;
}

MappedRenderVar RenderVarOutput::Map(MapDevice device,
                                     std::uintptr_t syncStream) {
    ERNIE_ZONE("Map render var");
    if (frame_ == nullptr) {
        throw std::invalid_argument(
            "RenderVarOutput::Map on an empty render var");
    }
    ProductFrame& frame = *frame_;
    if (const FrameImages::LinearVar linear =
            frame.images->Linear(name_);
        linear.buffer) {
        return MapLinear(linear, device, syncStream);
    }
    const std::vector<GpuImage*> images = Images();
    const std::vector<std::int64_t> shape = Shape();
    const DType type = LayoutOf(images.front()->Desc().format).type;
    VkDeviceSize bytes = 0;
    for (const GpuImage* image : images) {
        bytes += BytesOf(*image);
    }
    const auto recordCopies = [&](VkCommandBuffer commands,
                                  GpuBuffer& buffer) {
        VkDeviceSize offset = 0;
        for (GpuImage* image : images) {
            RecordCopy(commands, *image, buffer, offset);
            offset += BytesOf(*image);
        }
    };
    MappedRenderVar mapped;
    mapped.device = device;
    if (device == MapDevice::Cpu) {
        frame.context->WaitForSubmission(frame.submission);
        GpuBuffer readback(*frame.context, bytes, 0,
                           BufferMemory::Readback);
        frame.context->Submit([&](VkCommandBuffer commands) {
            recordCopies(commands, readback);
        });
        auto owner = std::make_shared<std::vector<std::byte>>(
            static_cast<std::size_t>(bytes));
        readback.Read(*owner);
        mapped.tensor =
            Tensor(owner->data(), type, shape, {}, {}, owner);
        return mapped;
    }
    if (frame.compute == nullptr) {
        throw std::invalid_argument(
            "Mapping a render var to the GPU needs compute interop");
    }
    if (device == MapDevice::GpuArray) {
        GpuImage& image = *images.front();
        if (image.Desc().layers > 1 || images.size() > 1) {
            throw std::invalid_argument(
                "Render var " + name_ +
                " has several layers or views; map it to a linear "
                "GPU "
                "tensor instead");
        }
        mapped.array = frame.compute->MapImage(image);
        frame.compute->WaitForVulkan(syncStream);
        ++frame.arrayMaps;
        return mapped;
    }
    std::shared_ptr<GpuBuffer> buffer;
    for (const std::shared_ptr<GpuBuffer>& candidate :
         *frame.linearPool) {
        if (candidate.use_count() == 1 &&
            candidate->Size() >= bytes) {
            buffer = candidate;
            break;
        }
    }
    if (buffer == nullptr) {
        buffer = std::make_shared<GpuBuffer>(
            *frame.context, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            BufferMemory::Shared);
        frame.linearPool->push_back(buffer);
    }
    // A buffer back in the pool may still be read by work its last
    // consumer queued: the copy waits for the signals its release
    // queued (below).
    std::vector<TimelinePoint> waits;
    if (frame.compute->VulkanWait().semaphore != VK_NULL_HANDLE) {
        waits.push_back(frame.compute->VulkanWait());
    }
    const TimelinePoint signal = frame.compute->NextVulkanSignal();
    frame.context->SubmitAsync(
        [&](VkCommandBuffer commands) {
            recordCopies(commands, *buffer);
        },
        waits, std::span(&signal, 1));
    frame.compute->WaitForVulkan(syncStream);
    const Tensor view = frame.compute->View(buffer, type, shape);
    // Released, the tensor queues on `syncStream` a signal after the
    // reads queued there, and only then returns the buffer to the
    // pool (`use_count`).
    ComputeInterop* compute = frame.compute;
    const std::shared_ptr<const void> owner(
        buffer.get(),
        [buffer, compute, syncStream](const void*) mutable {
            compute->SignalVulkan(syncStream);
            buffer.reset();
        });
    mapped.tensor =
        Tensor(view.Data(), type, shape, {}, view.Location(), owner);
    return mapped;
}

MappedRenderVar
RenderVarOutput::MapLinear(const FrameImages::LinearVar& linear,
                           MapDevice device,
                           std::uintptr_t syncStream) {
    ProductFrame& frame = *frame_;
    const std::vector<std::int64_t> shape = Shape();
    const DType type = LayoutOf(linear.format).type;
    MappedRenderVar mapped;
    mapped.device = device;
    if (device == MapDevice::Cpu) {
        frame.context->WaitForSubmission(frame.submission);
        const VkDeviceSize bytes = linear.buffer->Size();
        GpuBuffer readback(*frame.context, bytes, 0,
                           BufferMemory::Readback);
        frame.context->Submit([&](VkCommandBuffer commands) {
            const VkBufferCopy region{0, 0, bytes};
            vkCmdCopyBuffer(commands, linear.buffer->Handle(),
                            readback.Handle(), 1, &region);
        });
        auto owner = std::make_shared<std::vector<std::byte>>(
            static_cast<std::size_t>(bytes));
        readback.Read(*owner);
        mapped.tensor =
            Tensor(owner->data(), type, shape, {}, {}, owner);
        return mapped;
    }
    if (frame.compute == nullptr) {
        throw std::invalid_argument(
            "Mapping a render var to the GPU needs compute interop");
    }
    if (device == MapDevice::GpuArray) {
        throw std::invalid_argument(
            "Render var " + name_ +
            " is a linear buffer; map it to a linear GPU tensor");
    }
    // The frame's own submission wrote it: nothing to copy, only the
    // render to wait for.
    frame.compute->WaitForVulkan(syncStream);
    const Tensor view =
        frame.compute->View(linear.buffer, type, shape);
    // The frame renders again only once every copy of the tensor is
    // released, after the reads its consumer queued on `syncStream`
    // by then.
    ComputeInterop* compute = frame.compute;
    const std::shared_ptr<const void> owner(
        linear.buffer.get(),
        [held = frame_, compute, syncStream](const void*) mutable {
            compute->SignalVulkan(syncStream);
            held.reset();
        });
    mapped.tensor =
        Tensor(view.Data(), type, shape, {}, view.Location(), owner);
    return mapped;
}

void RenderVarOutput::Unmap(std::uintptr_t stream) {
    if (frame_ == nullptr || frame_->arrayMaps == 0) {
        return;
    }
    frame_->compute->SignalVulkan(stream);
    --frame_->arrayMaps;
}

ChunkedFrameImages::ChunkedFrameImages(
    std::vector<std::unique_ptr<FrameImages>> chunks)
    : chunks_(std::move(chunks)) {
    if (chunks_.empty()) {
        throw std::invalid_argument(
            "A chunked frame needs at least one chunk");
    }
}

std::uint32_t ChunkedFrameImages::Views() const {
    std::uint32_t views = 0;
    for (const std::unique_ptr<FrameImages>& chunk : chunks_) {
        views += chunk->Views();
    }
    return views;
}

StepOperation::StepOperation(const VulkanContext& context,
                             std::uint64_t submission,
                             RenderProductSetOutputs outputs)
    : context_(&context), submission_(submission),
      outputs_(std::move(outputs)) {}

bool StepOperation::Wait(std::chrono::nanoseconds timeout) {
    const auto nanoseconds =
        timeout == std::chrono::nanoseconds::max()
            ? UINT64_MAX
            : static_cast<std::uint64_t>(
                  std::max<std::int64_t>(timeout.count(), 0));
    return context_->WaitForSubmission(submission_, nanoseconds);
}

RenderProductSetOutputs StepOperation::Fetch() {
    Wait();
    return outputs_;
}

}  // namespace ernie
