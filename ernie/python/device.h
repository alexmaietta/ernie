// SPDX-License-Identifier: BSD-3-Clause
/// @file device.h
/// @brief Binding the device to Python.
#pragma once

#include <nanobind/nanobind.h>

namespace ernie::python {

/// @brief Adds `DeviceContext` and `DeviceColumn` to the module.
void BindDevice(nanobind::module_& module);

}  // namespace ernie::python
