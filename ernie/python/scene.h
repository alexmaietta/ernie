// SPDX-License-Identifier: BSD-3-Clause
/// @file scene.h
/// @brief Binding the scene to Python.
#pragma once

#include <nanobind/nanobind.h>

namespace ernie::python {

/// @brief Adds `Scene`, `Prim` and `SceneArray` to the module.
void BindScene(nanobind::module_& module);

}  // namespace ernie::python
