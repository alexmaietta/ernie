// SPDX-License-Identifier: BSD-3-Clause
/// @file render.h
/// @brief Binding the renderer to Python.
#pragma once

#include <nanobind/nanobind.h>

namespace ernie::python {

/// @brief Adds `Renderer` to the module.
void BindRender(nanobind::module_& module);

}  // namespace ernie::python
