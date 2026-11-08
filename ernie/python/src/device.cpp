// SPDX-License-Identifier: BSD-3-Clause
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "bindings.h"

namespace nb = nanobind;

namespace ernie::python {

using device::AdapterInfo;
using device::AdapterType;
using device::Backend;
using device::Device;
using device::DType;
using device::Sampling;
using device::Tensor;
using device::Traversal;

nb::object Numpy(std::vector<std::byte> bytes, const std::vector<size_t>& shape, DLDataType dtype) {
    auto* owned = new std::vector<std::byte>(std::move(bytes));
    nb::capsule owner(
        owned, [](void* held) noexcept { delete static_cast<std::vector<std::byte>*>(held); });
    return nb::cast(nb::ndarray<nb::numpy>(owned->data(), shape.size(), shape.data(), owner,
                                           nullptr, {dtype.code, dtype.bits, dtype.lanes}));
}

namespace {

void RegisterEnums(nb::module_& module) {
    nb::enum_<Backend>(module, "Backend")
        .value("VULKAN", Backend::Vulkan)
        .value("CUDA", Backend::Cuda)
        .value("HIP", Backend::Hip)
        .value("METAL", Backend::Metal)
        .value("CPU", Backend::Cpu);

    nb::enum_<Traversal>(module, "Traversal")
        .value("AUTO", Traversal::Auto)
        .value("HARDWARE", Traversal::Hardware)
        .value("SOFTWARE", Traversal::Software);

    nb::enum_<Sampling>(module, "Sampling")
        .value("AUTO", Sampling::Auto)
        .value("HARDWARE", Sampling::Hardware)
        .value("SOFTWARE", Sampling::Software);

    nb::enum_<AdapterType>(module, "AdapterType")
        .value("DISCRETE", AdapterType::Discrete)
        .value("INTEGRATED", AdapterType::Integrated)
        .value("VIRTUAL", AdapterType::Virtual)
        .value("CPU", AdapterType::Cpu)
        .value("OTHER", AdapterType::Other);

    nb::enum_<DType>(module, "DType")
        .value("UINT8", DType::UInt8)
        .value("UINT32", DType::UInt32)
        .value("FLOAT32", DType::Float32)
        .value("FLOAT64", DType::Float64)
        .value("INT32", DType::Int32)
        .value("UINT64", DType::UInt64)
        .value("INT64", DType::Int64);
}

void RegisterAdapters(nb::module_& module) {
    nb::class_<AdapterInfo>(module, "AdapterInfo")
        .def_ro("backend", &AdapterInfo::backend)
        .def_ro("name", &AdapterInfo::name)
        .def_ro("alias", &AdapterInfo::alias)
        .def_ro("ordinal", &AdapterInfo::ordinal)
        .def_ro("type", &AdapterInfo::type)
        .def_ro("architecture", &AdapterInfo::architecture)
        .def_ro("memory_bytes", &AdapterInfo::memoryBytes)
        .def_ro("unsupported", &AdapterInfo::unsupported)
        .def_prop_ro("api_version",
                     [](const AdapterInfo& adapter) { return adapter.apiVersion.ToString(); })
        .def_prop_ro("driver_version",
                     [](const AdapterInfo& adapter) { return adapter.driverVersion.ToString(); })
        .def_prop_ro("hardware_traversal",
                     [](const AdapterInfo& adapter) { return adapter.features.hardwareTraversal; })
        .def("__repr__", [](const AdapterInfo& adapter) {
            return "AdapterInfo(" + adapter.alias + ": " + adapter.name + ")";
        });

    module.def("list_adapters", &device::EnumerateAdapters);
    module.def("backend_name",
               [](Backend backend) { return std::string(device::BackendName(backend)); });
}

void RegisterDevice(nb::module_& module) {
    nb::class_<Device>(module, "Device")
        .def(nb::new_([](std::optional<Backend> backend, std::string adapter, Traversal traversal,
                         Sampling sampling, bool validation, bool synchronous, bool presentation) {
                 nb::gil_scoped_release release;
                 return Device::Create({.backend = backend,
                                        .adapter = std::move(adapter),
                                        .traversal = traversal,
                                        .sampling = sampling,
                                        .validation = validation,
                                        .synchronous = synchronous,
                                        .presentation = presentation});
             }),
             nb::kw_only(), nb::arg("backend") = nb::none(), nb::arg("adapter") = "",
             nb::arg("traversal") = Traversal::Auto, nb::arg("sampling") = Sampling::Auto,
             nb::arg("validation") = false, nb::arg("synchronous") = false,
             nb::arg("presentation") = false)
        .def_prop_ro("adapter", &Device::Adapter)
        .def_prop_ro("traversal", &Device::GetTraversal)
        .def_prop_ro("sampling", &Device::GetSampling)
        .def_prop_ro("shared_device",
                     [](Device& device) -> std::optional<nb::tuple> {
                         if (const std::optional<DLDevice> shared = device.SharedDevice()) {
                             return nb::make_tuple(int(shared->device_type), shared->device_id);
                         }
                         return std::nullopt;
                     })
        .def(
            "create_tensor",
            [](Device& device, DType dtype, std::vector<int64_t> shape, std::string label) {
                nb::gil_scoped_release release;
                return device.CreateTensor(dtype, std::move(shape), std::move(label));
            },
            nb::arg("dtype"), nb::arg("shape"), nb::arg("label") = "tensor")
        .def(
            "copy",
            [](Device& device, const Tensor& src, const Tensor& dst,
               std::optional<device::Operation> after) {
                nb::gil_scoped_release release;
                return device.Copy(src, dst, after.value_or(device::Operation{}));
            },
            nb::arg("src"), nb::arg("dst"), nb::arg("after") = nb::none(), nb::keep_alive<0, 1>())
        .def(
            "import_tensor",
            [](Device& device, nb::handle source, std::optional<uintptr_t> stream) {
                const TakenTensor taken = TakeTensor(source, stream);
                const std::shared_ptr<void> held = Hold(taken.managed);
                nb::gil_scoped_release release;
                return device.Import(taken.managed->dl_tensor, taken.producer);
            },
            nb::arg("source"), nb::arg("stream") = nb::none(), nb::keep_alive<0, 2>())
        .def(
            "follow",
            [](Device& device, std::optional<uintptr_t> stream) {
                return device.Follow(ToStream(stream));
            },
            nb::arg("stream") = nb::none(), nb::call_guard<nb::gil_scoped_release>(),
            nb::keep_alive<0, 1>())
        .def("__repr__",
             [](const Device& device) { return "Device(" + device.Adapter().name + ")"; });

    nb::class_<device::Operation>(module, "Operation")
        .def_prop_ro("ready", &device::Operation::Ready, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "wait",
            [](const device::Operation& operation, std::optional<uintptr_t> stream) {
                if (stream) {
                    operation.Wait(ToStream(stream));
                } else {
                    operation.Wait();
                }
            },
            nb::arg("stream") = nb::none(), nb::call_guard<nb::gil_scoped_release>());
}

void RegisterTensor(nb::module_& module) {
    nb::class_<Tensor>(module, "Tensor")
        .def_prop_ro("dtype", &Tensor::Type)
        .def_prop_ro("shape",
                     [](const Tensor& tensor) {
                         return std::vector<int64_t>(tensor.Shape().begin(), tensor.Shape().end());
                     })
        .def_prop_ro("address", &Tensor::Address)
        .def_prop_ro("device",
                     [](const Tensor& tensor) { return Ref<Device>(&tensor.GetDevice()); })
        .def("__len__",
             [](const Tensor& tensor) { return tensor.Shape().empty() ? 0 : tensor.Shape()[0]; })
        .def("__getitem__",
             [](const Tensor& tensor, int64_t index) {
                 const int64_t size = tensor.Shape().empty() ? 0 : tensor.Shape()[0];
                 if (index < -size || index >= size) {
                     throw nb::index_error();
                 }
                 return tensor.Select(index < 0 ? index + size : index);
             })
        .def(
            "__dlpack__",
            [](const Tensor& tensor, std::optional<int64_t> stream,
               std::optional<nb::tuple> maxVersion, std::optional<nb::tuple> dlDevice,
               std::optional<bool> copy) {
                return ExportTensor(tensor, stream, maxVersion, dlDevice, copy);
            },
            nb::kw_only(), nb::arg("stream") = nb::none(), nb::arg("max_version") = nb::none(),
            nb::arg("dl_device") = nb::none(), nb::arg("copy") = nb::none())
        .def("__dlpack_device__", &TensorDevice)
        .def("numpy", [](const Tensor& tensor) {
            std::vector<std::byte> bytes;
            {
                nb::gil_scoped_release release;
                bytes = tensor.ToHostBytes();
            }
            return Numpy(std::move(bytes), {tensor.Shape().begin(), tensor.Shape().end()},
                         DataType(tensor.Type()));
        });
}

void RegisterBuild(nb::module_& module) {
    nb::class_<BuildInfo>(module, "BuildInfo")
        .def_prop_ro("version",
                     [](const BuildInfo& build) { return std::string(build.fullVersion); })
        .def_prop_ro("commit", [](const BuildInfo& build) { return std::string(build.commit); })
        .def_ro("modified", &BuildInfo::modified)
        .def_prop_ro("commit_date",
                     [](const BuildInfo& build) { return std::string(build.commitDate); })
        .def_prop_ro("build_date",
                     [](const BuildInfo& build) { return std::string(build.buildDate); });
    module.def("build_info", &GetBuildInfo, nb::rv_policy::reference);

    nb::class_<device::BuildProgress>(module, "BuildProgress")
        .def_ro("queued", &device::BuildProgress::queued)
        .def_ro("finished", &device::BuildProgress::finished)
        .def_ro("cached", &device::BuildProgress::cached)
        .def_ro("building", &device::BuildProgress::building)
        .def_prop_ro("done", &device::BuildProgress::Done);
    module.def("kernel_builds", &device::CurrentBuilds);
}

}

void BindDevice(nb::module_& module) {
    RegisterBuild(module);
    RegisterEnums(module);
    RegisterAdapters(module);
    RegisterDevice(module);
    RegisterTensor(module);
}

}
