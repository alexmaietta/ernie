// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/tensor.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <optional>

#include "ernie/device/buffer.h"
#include "ernie/device/device.h"
#include "ernie/device/queue.h"
#include "ernie/util/error.h"

namespace ernie::device {
namespace {

std::vector<int64_t> ContiguousStrides(std::span<const int64_t> shape) {
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

int64_t LastElement(std::span<const int64_t> shape, std::span<const int64_t> strides) {
    int64_t last = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        last += (std::max<int64_t>(shape[i], 1) - 1) * strides[i];
    }
    return last;
}

std::vector<std::byte> Gather(std::span<const std::byte> source, std::span<const int64_t> shape,
                              std::span<const int64_t> strides, uint32_t elementBytes,
                              int64_t elements) {
    std::vector<std::byte> out(uint64_t(elements) * elementBytes);
    std::vector<int64_t> index(shape.size(), 0);
    for (int64_t e = 0; e < elements; ++e) {
        int64_t at = 0;
        for (size_t i = 0; i < shape.size(); ++i) {
            at += index[i] * strides[i];
        }
        std::memcpy(out.data() + e * elementBytes, source.data() + at * elementBytes, elementBytes);
        for (size_t i = shape.size(); i-- > 0;) {
            if (++index[i] < shape[i]) {
                break;
            }
            index[i] = 0;
        }
    }
    return out;
}

}

Tensor::Tensor() = default;
Tensor::~Tensor() = default;
Tensor::Tensor(const Tensor& other) = default;
Tensor::Tensor(Tensor&& other) noexcept = default;
Tensor& Tensor::operator=(const Tensor& other) = default;
Tensor& Tensor::operator=(Tensor&& other) noexcept = default;

Tensor::Tensor(Ref<Buffer> buffer, uint64_t offset, DType dtype, std::vector<int64_t> shape,
               std::vector<int64_t> strides)
    : m_buffer(std::move(buffer)), m_offset(offset), m_dtype(dtype), m_shape(std::move(shape)),
      m_strides(std::move(strides)) {
    if (m_strides.empty()) {
        m_strides = ContiguousStrides(m_shape);
    }
    if (m_strides.size() != m_shape.size()) {
        throw Error("Tensor of {} axes with {} strides", m_shape.size(), m_strides.size());
    }

    const uint64_t end =
        m_offset + (uint64_t(LastElement(m_shape, m_strides)) + 1) * ElementBytes(m_dtype);
    if (Elements() > 0 && end > m_buffer->Bytes()) {
        throw Error("Tensor runs to byte {} of buffer \"{}\", which has {}", end, m_buffer->Label(),
                    m_buffer->Bytes());
    }
}

Device& Tensor::GetDevice() const {
    if (!m_buffer) {
        throw Error("An empty tensor is on no device");
    }
    return m_buffer->GetDevice();
}

uint64_t Tensor::Address() const {
    return m_buffer->Address() + m_offset;
}

int64_t Tensor::Elements() const {
    int64_t elements = 1;
    for (const int64_t size : m_shape) {
        elements *= size;
    }
    return Empty() ? 0 : elements;
}

Tensor Tensor::Select(int64_t index) const {
    if (m_shape.empty() || index < 0 || index >= m_shape[0]) {
        throw Error("Index {} of a tensor of {} along its first axis", index,
                    m_shape.empty() ? 0 : m_shape[0]);
    }
    return Tensor(m_buffer, m_offset + uint64_t(index * m_strides[0]) * ElementBytes(m_dtype),
                  m_dtype, {m_shape.begin() + 1, m_shape.end()},
                  {m_strides.begin() + 1, m_strides.end()});
}

std::vector<std::byte> Tensor::ToHostBytes() const {
    const int64_t elements = Elements();
    if (elements == 0) {
        return {};
    }

    const uint32_t size = ElementBytes(m_dtype);
    std::vector<std::byte> span((uint64_t(LastElement(m_shape, m_strides)) + 1) * size);
    m_buffer->GetDevice().GetQueue().Read(m_buffer->Slice(m_offset, span.size()), span.data());
    return Gather(span, m_shape, m_strides, size, elements);
}

Download Tensor::ToHostAsync() const {
    if (Empty()) {
        throw Error("An empty tensor has nothing to read");
    }
    int64_t packed = 1;
    for (size_t axis = m_shape.size(); axis-- > 0;) {
        if (!m_strides.empty() && m_shape[axis] > 1 && m_strides[axis] != packed) {
            throw UnsupportedError("ToHostAsync reads contiguous tensors only");
        }
        packed *= m_shape[axis];
    }
    return m_buffer->GetDevice().GetQueue().ReadAsync(
        m_buffer->Slice(m_offset, uint64_t(Elements()) * ElementBytes(m_dtype)));
}

DLTensor Tensor::DLPack() const {
    const std::optional<DLDevice> shared =
        m_buffer ? m_buffer->GetDevice().SharedDevice() : std::nullopt;
    if (!shared) {
        throw UnsupportedError("This tensor's memory isn't shared with CUDA or HIP");
    }
    const ExternalAddress external = m_buffer->External(shared->device_type);
    const uint8_t bits = uint8_t(ElementBytes(m_dtype) * 8);
    const auto code = uint8_t(m_dtype == DType::Float32 || m_dtype == DType::Float64 ? kDLFloat
                              : m_dtype == DType::Int32 || m_dtype == DType::Int64   ? kDLInt
                                                                                     : kDLUInt);
    DLTensor tensor{};
    tensor.data = reinterpret_cast<void*>(external.address + m_offset);
    tensor.device = external.device;
    tensor.ndim = int32_t(m_shape.size());
    tensor.dtype = {code, bits, 1};
    tensor.shape = const_cast<int64_t*>(m_shape.data());
    tensor.strides = m_strides.empty() ? nullptr : const_cast<int64_t*>(m_strides.data());
    return tensor;
}

DLManagedTensorVersioned* Tensor::Export() const {
    struct Exported {
        DLManagedTensorVersioned managed{};
        Tensor tensor;
    };

    auto exported = std::make_unique<Exported>();
    exported->tensor = *this;
    DLManagedTensorVersioned& managed = exported->managed;
    managed.dl_tensor = exported->tensor.DLPack();
    managed.version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    managed.manager_ctx = exported.get();
    managed.deleter = [](DLManagedTensorVersioned* self) {
        delete static_cast<Exported*>(self->manager_ctx);
    };
    return &exported.release()->managed;
}

}
