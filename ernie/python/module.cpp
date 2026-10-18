// SPDX-License-Identifier: BSD-3-Clause
/// @file module.cpp
/// @brief The _ernie Python extension module.
#include "ernie/device/device.h"
#include "ernie/util/version.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

NB_MODULE(_ernie, module) {
    module.doc() = "Ernie's native module.";
    module.attr("__version__") = std::string(ernie::Version());

    nb::enum_<ernie::VulkanDeviceType>(module, "VulkanDeviceType")
        .value("OTHER", ernie::VulkanDeviceType::Other)
        .value("INTEGRATED_GPU", ernie::VulkanDeviceType::IntegratedGpu)
        .value("DISCRETE_GPU", ernie::VulkanDeviceType::DiscreteGpu)
        .value("VIRTUAL_GPU", ernie::VulkanDeviceType::VirtualGpu)
        .value("CPU", ernie::VulkanDeviceType::Cpu);

    nb::class_<ernie::VulkanDeviceInfo>(module, "VulkanDeviceInfo")
        .def_ro("name", &ernie::VulkanDeviceInfo::name)
        .def_ro("type", &ernie::VulkanDeviceInfo::type)
        .def_ro("api_major", &ernie::VulkanDeviceInfo::apiMajor)
        .def_ro("api_minor", &ernie::VulkanDeviceInfo::apiMinor)
        .def_prop_ro("ray_query",
                     [](const ernie::VulkanDeviceInfo& info) { return info.capabilities.rayQuery; })
        .def_ro("uuid", &ernie::VulkanDeviceInfo::uuid)
        .def_ro("supported", &ernie::VulkanDeviceInfo::supported);

    module.def("vulkan_devices", &ernie::ListVulkanDevices,
               "Lists the Vulkan devices of the installed drivers.");
}
