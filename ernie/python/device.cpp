// SPDX-License-Identifier: BSD-3-Clause
/// @file device.cpp
/// @brief Python bindings of device columns, shared with PyTorch, Warp and any DLPack library in
/// place: `torch.from_dlpack(column)` and `wp.from_dlpack(column)` view Ernie's memory.
#include "ernie/python/device.h"

#include "ernie/device/cuda.h"
#include "ernie/device/staging.h"
#include "ernie/python/dlpack.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace nb = nanobind;

namespace ernie::python {
namespace {

constexpr std::array<std::pair<std::string_view, DType>, 8> kTypes{{
    {"uint8", DType::UInt8},
    {"uint16", DType::UInt16},
    {"uint32", DType::UInt32},
    {"int32", DType::Int32},
    {"int64", DType::Int64},
    {"float16", DType::Float16},
    {"float32", DType::Float32},
    {"float64", DType::Float64},
}};

DType ParseType(std::string_view name) {
    for (const auto& [text, type] : kTypes) {
        if (text == name) {
            return type;
        }
    }
    throw nb::value_error(("No element type '" + std::string(name) + "'").c_str());
}

std::string_view TypeName(DType type) {
    for (const auto& [text, value] : kTypes) {
        if (value == type) {
            return text;
        }
    }
    return "";
}

/// A DLPack stream argument as a CUDA stream: None and 1 are the legacy default stream, 2 the
/// per-thread one, anything else a `cudaStream_t`.
CUstream ToStream(const nb::object& stream) {
    if (stream.is_none()) {
        return nullptr;
    }
    const auto value = nb::cast<std::intptr_t>(stream);
    return reinterpret_cast<CUstream>(value == 1 ? 0 : value);
}

/// A device, with the shared arenas Python's columns live in and its CUDA interop.
struct Context {
    std::unique_ptr<DeviceContext> context;
    std::unique_ptr<Arenas> shared;
    std::unique_ptr<CudaInterop> cuda;

    explicit Context(const std::string& device)
        : context(std::make_unique<DeviceContext>(DeviceContextOptions{.device = device})) {
        if (context->Capabilities().externalMemory) {
            shared = std::make_unique<Arenas>(*context, MemoryUse::Shared, "python columns");
        }
        cuda = CudaInterop::Create(*context);
    }
    ~Context() {
        cuda.reset();
        context->WaitIdle();
    }

    CudaInterop& Cuda() const {
        if (!cuda) {
            throw nb::buffer_error(("No CUDA device pairs with " + context->Info().name).c_str());
        }
        return *cuda;
    }
    Arenas& Shared() const {
        if (!shared) {
            throw nb::buffer_error((context->Info().name + " cannot share memory").c_str());
        }
        return *shared;
    }
};

/// A column, and the ticket of CUDA's writes Ernie has yet to see.
struct Column {
    Context& owner;
    std::unique_ptr<DeviceColumn> column;
    /// The tensor a wrapped column shares, which `__dlpack__` hands back as it is.
    std::optional<Tensor> wrapped;
    Ticket written;

    Column(Context& context, std::unique_ptr<DeviceColumn> device,
           std::optional<Tensor> tensor = std::nullopt)
        : owner(context), column(std::move(device)), wrapped(std::move(tensor)) {}

    nb::object Dlpack(const nb::object& stream, const nb::object& maxVersion) {
        CudaInterop& cuda = owner.Cuda();
        // -1 asks for no synchronization.
        if (stream.is_none() || nb::cast<std::intptr_t>(stream) != -1) {
            for (const Queue* queue : owner.context->Queues()) {
                cuda.Wait(ToStream(stream), queue->Last());
            }
        }
        const bool versioned =
            !maxVersion.is_none() && nb::cast<std::tuple<int, int>>(maxVersion) >=
                                         std::tuple<int, int>(DLPACK_MAJOR_VERSION, 0);
        return Capsule((wrapped ? *wrapped : cuda.View(*column)).ToDlpack(), versioned);
    }

    void Assign(const nb::object& source, const nb::object& stream) {
        written = owner.Cuda().Assign(*column, TakeTensor(source, stream), ToStream(stream));
    }

    nb::bytes Read() {
        Readback readback(*owner.context, owner.context->Main(), column->Storage(),
                          column->Range().offset, column->Bytes(), std::span(&written, 1));
        const std::span<const std::byte> bytes = readback.Bytes();
        return nb::bytes(reinterpret_cast<const char*>(bytes.data()), column->Bytes());
    }
};

/// @return The allocations `ernie_cuda_malloc` made, or none without a CUDA driver: PyTorch's
/// allocator hook is process-wide, so they are too.
ShareableAllocations* TorchAllocations() {
    static const std::unique_ptr<ShareableAllocations> allocations =
        CudaDriver::Get() != nullptr ? std::make_unique<ShareableAllocations>(*CudaDriver::Get())
                                     : nullptr;
    return allocations.get();
}

}  // namespace

void BindDevice(nb::module_& module) {
    nb::class_<Context>(module, "DeviceContext",
                        "A device: where columns live, and the CUDA device paired with it.")
        .def(nb::init<const std::string&>(), nb::arg("device") = "",
             "Opens the best device, or the one whose UUID or name matches `device`.")
        .def_prop_ro("name", [](const Context& self) { return self.context->Info().name; })
        .def_prop_ro("uuid", [](const Context& self) { return self.context->Info().uuid; })
        .def_prop_ro("cuda_ordinal",
                     [](const Context& self) -> nb::object {
                         if (!self.cuda) {
                             return nb::none();
                         }
                         return nb::int_(self.cuda->Ordinal());
                     })
        .def_prop_ro("transfers",
                     [](const Context& self) {
                         const Transfers transfers = self.context->Transfers();
                         nb::dict result;
                         result["uploaded"] = transfers.uploaded;
                         result["read_back"] = transfers.readBack;
                         result["copied"] = transfers.copied;
                         return result;
                     })
        .def(
            "column",
            [](Context& self, const std::string& name, const std::string& dtype,
               const std::vector<std::int64_t>& shape) {
                if (shape.empty() || shape.front() < 0) {
                    throw nb::value_error("A column's shape starts with its row count");
                }
                return new Column(self,
                                  std::make_unique<DeviceColumn>(
                                      self.Shared(), name, ParseType(dtype),
                                      std::vector<std::int64_t>(shape.begin() + 1, shape.end()),
                                      static_cast<std::uint64_t>(shape.front())));
            },
            nb::arg("name"), nb::arg("dtype"), nb::arg("shape"), nb::keep_alive<0, 1>(),
            "A column of `shape[0]` rows on the device, shared in place through DLPack.")
        .def(
            "wrap",
            [](Context& self, const nb::object& tensor, const std::string& name,
               const nb::object& stream) {
                CudaInterop& cuda = self.Cuda();
                Tensor source = TakeTensor(tensor, stream);
                std::unique_ptr<DeviceColumn> column = cuda.Wrap(source, name);
                if (!column) {
                    throw nb::buffer_error(
                        "The tensor's memory cannot be shared: allocate it with "
                        "ernie.cuda_allocator(), or copy it into a column with assign()");
                }
                auto* wrapped = new Column(self, std::move(column), std::move(source));
                wrapped->written = cuda.Signal(ToStream(stream));
                return wrapped;
            },
            nb::arg("tensor"), nb::arg("name") = "", nb::arg("stream") = nb::none(),
            nb::keep_alive<0, 1>(),
            "A column over a CUDA tensor's own memory, in place, when it is exportable.");

    nb::class_<Column>(module, "DeviceColumn",
                       "Rows on the device that CUDA, PyTorch and Warp read and write in place.")
        .def_prop_ro("name", [](const Column& self) { return self.column->Name(); })
        .def_prop_ro("shape", [](const Column& self) { return self.column->Shape(); })
        .def_prop_ro("dtype", [](const Column& self) { return TypeName(self.column->Type()); })
        .def(
            "__dlpack__",
            [](Column& self, const nb::object& stream, const nb::object& maxVersion,
               const nb::object&, const nb::object& copy) {
                if (!copy.is_none() && nb::cast<bool>(copy)) {
                    throw nb::buffer_error("Device columns are shared in place, never copied");
                }
                return self.Dlpack(stream, maxVersion);
            },
            nb::kw_only(), nb::arg("stream") = nb::none(), nb::arg("max_version") = nb::none(),
            nb::arg("dl_device") = nb::none(), nb::arg("copy") = nb::none(),
            "A CUDA view of the rows; `stream` waits for Ernie's work on them first.")
        .def("__dlpack_device__",
             [](const Column& self) {
                 return std::make_tuple(static_cast<int>(kDLCUDA), self.owner.Cuda().Ordinal());
             })
        .def(
            "ready",
            [](Column& self, const nb::object& stream) {
                self.written = self.owner.Cuda().Signal(ToStream(stream));
            },
            nb::arg("stream") = nb::none(),
            "Hands the rows back to Ernie once the work enqueued on `stream` so far is done.")
        .def("assign", &Column::Assign, nb::arg("tensor"), nb::arg("stream") = nb::none(),
             "Takes a DLPack tensor's values: in place for this column's own view, else by one "
             "copy on the device.")
        .def("read", &Column::Read, "Copies the rows to the host, after CUDA's writes.");
}

}  // namespace ernie::python

#if defined(_WIN32)
#define ERNIE_ALLOCATOR extern "C" __declspec(dllexport)
#else
#define ERNIE_ALLOCATOR extern "C" __attribute__((visibility("default")))
#endif

/// @brief PyTorch's pluggable allocator, allocating each tensor as exportable memory that
/// `DeviceContext.wrap` shares in place (`ernie.cuda_allocator()` installs it).
ERNIE_ALLOCATOR void* ernie_cuda_malloc(std::size_t size, int device, void* stream) {
    static_cast<void>(stream);
    ernie::ShareableAllocations* allocations = ernie::python::TorchAllocations();
    return allocations != nullptr ? reinterpret_cast<void*>(allocations->Allocate(device, size))
                                  : nullptr;
}

ERNIE_ALLOCATOR void ernie_cuda_free(void* pointer, std::size_t size, int device, void* stream) {
    static_cast<void>(size);
    static_cast<void>(device);
    // Work enqueued on the stream may still use the memory: unmapping waits for none.
    ernie::CudaDriver::Get()->streamSynchronize(static_cast<ernie::CUstream>(stream));
    ernie::python::TorchAllocations()->Free(reinterpret_cast<ernie::CUdeviceptr>(pointer));
}
