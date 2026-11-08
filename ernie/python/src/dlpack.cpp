// SPDX-License-Identifier: BSD-3-Clause
#include <Python.h>
#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "bindings.h"
#include "ernie/device/commands.h"
#include "ernie/device/queue.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace ernie::python {

namespace {

constexpr const char* kVersioned = "dltensor_versioned";
constexpr const char* kUnversioned = "dltensor";

struct Export {
    DLManagedTensorVersioned managed{};
    device::Tensor tensor;
    std::vector<std::byte> host;
};

struct Producer {
    DLManagedTensorVersioned wrapper{};
    DLManagedTensorVersioned* versioned = nullptr;
    DLManagedTensor* unversioned = nullptr;
};

void ReleaseProducer(DLManagedTensorVersioned* self) {
    auto* producer = static_cast<Producer*>(self->manager_ctx);
    if (Py_IsInitialized()) {
        nb::gil_scoped_acquire gil;
        if (producer->versioned != nullptr && producer->versioned->deleter != nullptr) {
            producer->versioned->deleter(producer->versioned);
        }
        if (producer->unversioned != nullptr && producer->unversioned->deleter != nullptr) {
            producer->unversioned->deleter(producer->unversioned);
        }
    }
    delete producer;
}

DLManagedTensorVersioned* CreateExport(const device::Tensor& tensor, bool toHost) {
    if (!toHost) {
        return tensor.Export();
    }
    auto held = std::make_unique<Export>();
    held->tensor = tensor;
    held->host = tensor.ToHostBytes();
    DLManagedTensorVersioned& managed = held->managed;
    DLTensor& exported = managed.dl_tensor;
    exported.data = held->host.data();
    exported.device = {kDLCPU, 0};
    exported.ndim = int32_t(tensor.Shape().size());
    exported.dtype = DataType(tensor.Type());
    exported.shape = const_cast<int64_t*>(held->tensor.Shape().data());
    managed.flags = DLPACK_FLAG_BITMASK_IS_COPIED;
    managed.version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    managed.manager_ctx = held.get();
    managed.deleter = [](DLManagedTensorVersioned* self) {
        delete static_cast<Export*>(self->manager_ctx);
    };
    return &held.release()->managed;
}

device::Tensor CopyOnDevice(const device::Tensor& tensor) {
    const Span<const int64_t> shape = tensor.Shape();
    const Span<const int64_t> strides = tensor.Strides();
    int64_t last = 0;
    int64_t packed = 1;
    for (size_t axis = shape.size(); axis-- > 0;) {
        last += (shape[axis] - 1) * (strides.empty() ? packed : strides[axis]);
        packed *= shape[axis];
    }
    const uint64_t bytes =
        tensor.Elements() == 0 ? 0 : uint64_t(last + 1) * device::ElementBytes(tensor.Type());
    device::Device& device = tensor.GetBuffer().GetDevice();
    const Ref<device::Buffer> copied = device.CreateBuffer({.bytes = std::max<uint64_t>(bytes, 4),
                                                            .usage = device::BufferUsage::Interop,
                                                            .label = "dlpack copy"});
    if (bytes > 0) {
        device::Queue& queue = device.GetQueue();
        device::CommandEncoder encoder = queue.CreateEncoder();
        encoder.Copy(tensor.GetBuffer().Slice(tensor.Offset(), bytes), copied->Slice(0, bytes));
        queue.Submit(encoder.Finish());
    }
    return device::Tensor(copied, 0, tensor.Type(), {shape.begin(), shape.end()},
                          {strides.begin(), strides.end()});
}

void CheckStream(int type, int64_t stream) {
    if (type == kDLCUDA && stream == 0) {
        throw nb::value_error("CUDA consumers name the default stream as 1 or 2, never 0");
    }
    if (type == kDLROCM && (stream == 1 || stream == 2)) {
        throw nb::value_error("ROCm consumers name the default stream as 0, never 1 or 2");
    }
}

PyObject* VersionedCapsule(DLManagedTensorVersioned* managed) {
    return PyCapsule_New(managed, kVersioned, [](PyObject* self) {
        if (PyCapsule_IsValid(self, kVersioned)) {
            auto* tensor =
                static_cast<DLManagedTensorVersioned*>(PyCapsule_GetPointer(self, kVersioned));
            tensor->deleter(tensor);
        }
    });
}

PyObject* UnversionedCapsule(DLManagedTensorVersioned* managed) {
    auto* legacy = new DLManagedTensor();
    legacy->dl_tensor = managed->dl_tensor;
    legacy->manager_ctx = managed;
    legacy->deleter = [](DLManagedTensor* self) {
        auto* inner = static_cast<DLManagedTensorVersioned*>(self->manager_ctx);
        inner->deleter(inner);
        delete self;
    };
    return PyCapsule_New(legacy, kUnversioned, [](PyObject* self) {
        if (PyCapsule_IsValid(self, kUnversioned)) {
            auto* tensor = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(self, kUnversioned));
            tensor->deleter(tensor);
        }
    });
}

std::optional<int64_t> RequestedStream(nb::handle source, std::optional<uintptr_t> stream) {
    if (!nb::hasattr(source, "__dlpack_device__")) {
        return std::nullopt;
    }
    const nb::tuple device = nb::cast<nb::tuple>(source.attr("__dlpack_device__")());
    const int type = nb::cast<int>(device[0]);
    const bool cuda = type == kDLCUDA || type == kDLCUDAHost || type == kDLCUDAManaged;
    const bool rocm = type == kDLROCM || type == kDLROCMHost;
    if (!cuda && !rocm) {
        return std::nullopt;
    }
    if (stream && (*stream != 0 || !cuda)) {
        return int64_t(*stream);
    }
    return cuda ? 1 : 0;
}

nb::object RequestCapsule(nb::handle source, std::optional<int64_t> stream) {
    const nb::object request = source.attr("__dlpack__");
    try {
        return stream ? request("stream"_a = *stream, "max_version"_a = nb::make_tuple(1, 0))
                      : request("max_version"_a = nb::make_tuple(1, 0));
    } catch (nb::python_error& error) {
        if (!error.matches(PyExc_TypeError)) {
            throw;
        }
        return stream ? request("stream"_a = *stream) : request();
    }
}

}

DLDataType DataType(device::DType type) {
    const uint8_t bits = uint8_t(device::ElementBytes(type) * 8);
    switch (type) {
    case device::DType::Float32:
    case device::DType::Float64:
        return {kDLFloat, bits, 1};
    case device::DType::Int32:
    case device::DType::Int64:
        return {kDLInt, bits, 1};
    default:
        return {kDLUInt, bits, 1};
    }
}

nb::tuple TensorDevice(const device::Tensor& tensor) {
    if (tensor.Empty()) {
        throw nb::buffer_error("An empty tensor has no memory to share");
    }
    if (const std::optional<DLDevice> shared = tensor.GetBuffer().GetDevice().SharedDevice()) {
        return nb::make_tuple(int(shared->device_type), shared->device_id);
    }
    return nb::make_tuple(int(kDLVulkan), 0);
}

nb::object ExportTensor(const device::Tensor& tensor, std::optional<int64_t> stream,
                        std::optional<nb::tuple> maxVersion, std::optional<nb::tuple> dlDevice,
                        std::optional<bool> copy) {
    const nb::tuple own = TensorDevice(tensor);
    const int type = nb::cast<int>(own[0]);
    const int id = nb::cast<int>(own[1]);
    bool toHost = false;
    if (dlDevice) {
        const int wanted = nb::cast<int>((*dlDevice)[0]);
        const int wantedId = nb::cast<int>((*dlDevice)[1]);
        toHost = wanted == kDLCPU;
        if (toHost && copy == false) {
            throw nb::buffer_error("This tensor reaches the host only as a copy, and copy=False "
                                   "forbids one");
        }
        if (!toHost && (wanted != type || wantedId != id)) {
            throw nb::buffer_error(fmt::format("This tensor is on device ({}, {}); it's exported "
                                               "there or copied to the host, not to ({}, {})",
                                               type, id, wanted, wantedId)
                                       .c_str());
        }
    }
    if (!toHost && type == kDLVulkan) {
        throw nb::buffer_error("This tensor's memory isn't shared with CUDA or HIP; ask for "
                               "dl_device=(1, 0) to copy it to the host");
    }
    if (!toHost && stream) {
        CheckStream(type, *stream);
    }

    DLManagedTensorVersioned* managed = nullptr;
    try {
        nb::gil_scoped_release release;
        managed = CreateExport(copy == true && !toHost ? CopyOnDevice(tensor) : tensor, toHost);
        if (copy == true) {
            managed->flags |= DLPACK_FLAG_BITMASK_IS_COPIED;
        }
        if (!toHost && stream != -1) {
            const int64_t handle = stream.value_or(type == kDLCUDA ? 1 : 0);
            tensor.GetBuffer().GetDevice().GetQueue().Mark().Wait(
                Stream::Compute(reinterpret_cast<void*>(uintptr_t(handle))));
        }
    } catch (const UnsupportedError& error) {
        if (managed != nullptr) {
            managed->deleter(managed);
        }
        throw nb::buffer_error(error.what());
    } catch (...) {
        if (managed != nullptr) {
            managed->deleter(managed);
        }
        throw;
    }
    const bool versioned = maxVersion && nb::cast<int>((*maxVersion)[0]) >= 1;
    PyObject* capsule = versioned ? VersionedCapsule(managed) : UnversionedCapsule(managed);
    if (capsule == nullptr) {
        managed->deleter(managed);
        throw nb::python_error();
    }
    return nb::steal(capsule);
}

TakenTensor TakeTensor(nb::handle source, std::optional<uintptr_t> stream) {
    const std::optional<int64_t> requested = RequestedStream(source, stream);
    const nb::object capsule = RequestCapsule(source, requested);
    auto* producer = new Producer();
    if (PyCapsule_IsValid(capsule.ptr(), kVersioned)) {
        producer->versioned =
            static_cast<DLManagedTensorVersioned*>(PyCapsule_GetPointer(capsule.ptr(), kVersioned));
        PyCapsule_SetName(capsule.ptr(), "used_dltensor_versioned");
        producer->wrapper.flags = producer->versioned->flags;
        producer->wrapper.dl_tensor = producer->versioned->dl_tensor;
    } else if (PyCapsule_IsValid(capsule.ptr(), kUnversioned)) {
        producer->unversioned =
            static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule.ptr(), kUnversioned));
        PyCapsule_SetName(capsule.ptr(), "used_dltensor");
        producer->wrapper.dl_tensor = producer->unversioned->dl_tensor;
    } else {
        delete producer;
        throw nb::type_error("__dlpack__ returned no DLPack capsule");
    }

    producer->wrapper.version = {DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    producer->wrapper.manager_ctx = producer;
    producer->wrapper.deleter = ReleaseProducer;
    const Stream followed = requested
                                ? Stream::Compute(reinterpret_cast<void*>(uintptr_t(*requested)))
                                : ToStream(stream);
    return {&producer->wrapper, followed};
}

}
