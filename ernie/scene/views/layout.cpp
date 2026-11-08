// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/views/layout.h"

#include <optional>
#include <string>

#include "ernie/device/queue.h"
#include "ernie/util/error.h"

namespace ernie {

namespace {

bool Contiguous(const DLTensor& tensor) {
    if (tensor.strides == nullptr) {
        return true;
    }
    if (tensor.ndim == 1) {
        return tensor.strides[0] == 1;
    }
    return tensor.strides[1] == 1 && tensor.strides[0] == tensor.shape[1];
}

bool Same(DLDevice a, DLDevice b) {
    return a.device_type == b.device_type && a.device_id == b.device_id;
}

}

ColumnLayout Layout(ColumnType type) {
    switch (type) {
    case ColumnType::Bool:
    case ColumnType::Uint:
    case ColumnType::Enum:
    case ColumnType::Token:
    case ColumnType::Asset:
    case ColumnType::PrimArray:
    case ColumnType::TokenArray:
        return {{kDLUInt, 32, 1}, 1, device::DType::UInt32};
    case ColumnType::Uint64:
        return {{kDLUInt, 64, 1}, 1, device::DType::UInt64};
    case ColumnType::Uint2:
        return {{kDLUInt, 32, 1}, 2, device::DType::UInt32};
    case ColumnType::Float2:
    case ColumnType::Float2Array:
        return {{kDLFloat, 32, 1}, 2, device::DType::Float32};
    case ColumnType::Int:
    case ColumnType::IntArray:
        return {{kDLInt, 32, 1}, 1, device::DType::Int32};
    case ColumnType::Int64Array:
        return {{kDLInt, 64, 1}, 1, device::DType::Int64};
    case ColumnType::Float:
    case ColumnType::FloatArray:
        return {{kDLFloat, 32, 1}, 1, device::DType::Float32};
    case ColumnType::Float3:
    case ColumnType::Float3Array:
        return {{kDLFloat, 32, 1}, 3, device::DType::Float32};
    case ColumnType::Float4:
    case ColumnType::Float4Array:
        return {{kDLFloat, 32, 1}, 4, device::DType::Float32};
    case ColumnType::Double3:
        return {{kDLFloat, 64, 1}, 3, device::DType::Float64};
    case ColumnType::Quatd:
        return {{kDLFloat, 64, 1}, 4, device::DType::Float64};
    case ColumnType::WorldTransform:
        return {{kDLFloat, 32, 1}, 15, device::DType::Float32};
    }
    throw Error("No layout for column type {}", int(type));
}

void WaitOnHost(device::Device& device, Stream producer) {
    if (!producer.IsCompute()) {
        return;
    }
    device::Queue& queue = device.GetQueue();
    queue.Follow(producer);
    queue.Mark().Wait();
}

void CheckReadable(device::Device& device, DLDevice tensor) {
    if (device.ReadsDirectly(tensor)) {
        return;
    }
    const std::optional<DLDevice> shared = device.SharedDevice();
    if (!shared || !Same(*shared, tensor)) {
        throw UnsupportedError("{} can't read a tensor on device type {} (device {})",
                               device.Adapter().alias, int(tensor.device_type), tensor.device_id);
    }
}

uint32_t TensorRows(const DLTensor& tensor, const ColumnLayout& layout, std::string_view field) {
    if (tensor.ndim < 1 || tensor.ndim > 2 || (tensor.ndim == 2 && tensor.dtype.lanes != 1)) {
        throw Error("{} is written from [n] or [n, lanes] tensors, not {} dimensions", field,
                    tensor.ndim);
    }
    const uint32_t lanes = tensor.ndim == 1 ? tensor.dtype.lanes : uint32_t(tensor.shape[1]);
    if (tensor.dtype.code != layout.dtype.code || tensor.dtype.bits != layout.dtype.bits ||
        lanes != layout.lanes) {
        throw Error("{} is {} lanes of {}-bit values; the tensor's are {} lanes of {}-bit", field,
                    layout.lanes, layout.dtype.bits, lanes, tensor.dtype.bits);
    }
    if (!Contiguous(tensor)) {
        throw UnsupportedError("{} is written from contiguous tensors only", field);
    }
    return uint32_t(tensor.shape[0]);
}

uint32_t TensorItems(const DLTensor& tensor, const ColumnLayout& layout, uint32_t prims,
                     std::string_view field) {
    const int32_t dimensions = layout.lanes > 1 ? 3 : 2;
    const int32_t first = prims == 1 && tensor.ndim == dimensions - 1 ? 0 : 1;
    if (tensor.ndim != dimensions - 1 + first || tensor.dtype.lanes != 1 ||
        (first == 1 && tensor.shape[0] != prims) ||
        (layout.lanes > 1 && tensor.shape[first + 1] != layout.lanes)) {
        throw Error("{} of {} prims is held in a [{}, items{}] tensor", field, prims, prims,
                    layout.lanes > 1 ? ", " + std::to_string(layout.lanes) : "");
    }
    if (tensor.dtype.code != layout.dtype.code || tensor.dtype.bits != layout.dtype.bits) {
        throw Error("{} holds {}-bit values; the tensor's are {}-bit", field, layout.dtype.bits,
                    tensor.dtype.bits);
    }
    if (!Contiguous(tensor)) {
        throw UnsupportedError("{} is held in contiguous tensors only", field);
    }
    if (tensor.shape[first] <= 0) {
        throw Error("{} is held in a tensor of at least one item per prim", field);
    }
    return uint32_t(tensor.shape[first]);
}

const std::byte* TensorBytes(const DLTensor& tensor) {
    return static_cast<const std::byte*>(tensor.data) + tensor.byte_offset;
}

}
