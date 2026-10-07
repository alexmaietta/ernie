#pragma once

// Python.h, through nanobind, must come before any system header.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "ernie/scene/value.h"
#include "ernie/util/tensor.h"

namespace ernie::python {

namespace nb = nanobind;

/// Any array a DLPack-capable framework hands over (NumPy, PyTorch, JAX, CuPy, Warp), read-only
/// arrays included, since Ernie itself never writes into them.
using AnyArray = nb::ndarray<nb::ro>;

/// Adopts `array` without copying; the tensor keeps the Python array alive and releases it with
/// the GIL held.
/// @throws std::invalid_argument for an element type Ernie does not exchange.
Tensor TensorFromArray(const AnyArray& array);

/// The DLPack element type of `type`.
nb::dlpack::dtype DlpackOf(DType type);

/// A view of `tensor` that keeps it alive, read-only unless `writable`: a NumPy array for CPU
/// tensors, an array-API object with `__dlpack__` (for torch.from_dlpack, CuPy, JAX) otherwise.
nb::object ArrayFromTensor(const Tensor& tensor, bool writable = false);

/// A CUDA stream argument in DLPack numbering: None for the legacy default stream (1), an int,
/// or an object with a `cuda_stream` attribute such as a `torch.cuda.Stream`.
std::uintptr_t StreamOf(nb::handle stream);

/// Converts a Python value to a scene `Value`; see the module documentation for the mapping.
/// @throws nb::type_error for values with no `Value` equivalent.
Value ValueFromPython(nb::handle object);

nb::object ValueToPython(const Value& value);

void BindTensor(nb::module_& m);
void BindScene(nb::module_& m);
void BindRender(nb::module_& m);

/// The `app` submodule; built only with Ernie's app (`ERNIE_HAS_APP`).
void BindApp(nb::module_& m);

}  // namespace ernie::python
