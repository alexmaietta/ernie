// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Python.h, through nanobind, must come before any system header.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "ernie/scene/light_pattern.h"
#include "ernie/scene/value.h"
#include "ernie/util/tensor.h"

#include <string>

namespace ernie::python {

namespace nb = nanobind;

/// Any array a DLPack-capable framework hands over (NumPy, PyTorch,
/// JAX, CuPy, Warp), read-only arrays included, since Ernie itself
/// never writes into them.
using AnyArray = nb::ndarray<nb::ro>;

/// Adopts `array` without copying; the tensor keeps the Python array
/// alive and releases it with the GIL held.
/// @throws std::invalid_argument for an element type Ernie does not
/// exchange.
Tensor TensorFromArray(const AnyArray& array);

/// The DLPack device type of memory of `kind` (`kDLCPU`, `kDLCUDA` or
/// `kDLROCM`).
int DlpackDeviceType(DeviceKind kind);

/// The DLPack element type of `type`.
nb::dlpack::dtype DlpackOf(DType type);

/// A view of `tensor` that keeps it alive, read-only unless
/// `writable`: a NumPy array for CPU tensors, an array-API object
/// with `__dlpack__` (for torch.from_dlpack, CuPy, JAX) otherwise.
nb::object ArrayFromTensor(const Tensor& tensor,
                           bool writable = false);

/// `tensor`, also keeping the Python object `keep` alive, so arrays
/// in memory of the renderer's Vulkan device or CUDA context cannot
/// outlive the renderer.
Tensor KeepingAlive(const Tensor& tensor, nb::handle keep);

/// A CUDA stream argument in DLPack numbering: None for the legacy
/// default stream (1), an int, or an object with a `cuda_stream`
/// attribute such as a `torch.cuda.Stream`.
std::uintptr_t StreamOf(nb::handle stream);

/// Converts a Python value to a scene `Value`; see the module
/// documentation for the mapping.
/// @throws nb::type_error for values with no `Value` equivalent.
Value ValueFromPython(nb::handle object);

nb::object ValueToPython(const Value& value);

/// Whether `object` names a prim as `PathFromPython` reads one.
bool IsPrimPath(nb::handle object) noexcept;

/// The prim path `object` names: a str, a Path, a Prim, or a typed
/// prim (`ernie.Camera`, any object whose `prim` is a Prim).
/// @throws nb::type_error for anything else; std::invalid_argument
/// for a str that is no path.
Path PathFromPython(nb::handle object);

/// A light pattern from `add_light_pattern`'s arguments: `params` as
/// {name: default} in order, a list of {"name", "default",
/// "description"}, or None.
LightPatternModel LightPatternFromPython(
    const std::string& id, const std::string& typeName,
    const std::string& slang, nb::handle params, double extent,
    bool normalized, const std::string& name,
    const std::string& description);

/// A prim path argument, as `PathFromPython` reads it: bind
/// `PrimPath` where a function takes a prim's path, and use it as a
/// `Path`. Converted in the call, so a malformed path raises its own
/// error rather than a mismatched signature.
struct PrimPath {
    nb::handle object;

    operator Path() const {
        return PathFromPython(object);
    }
};

void BindTensor(nb::module_& m);
void BindScene(nb::module_& m);
/// `save_scene`, the JSON description of a scene
/// (ernie/scene/scene_json.h).
void BindSceneJson(nb::module_& m);
/// The prim type registry (`PrimType`, `AttributeDef`), which
/// `ernie.schema` builds classes from.
void BindSchema(nb::module_& m);
void BindRender(nb::module_& m);
/// `Scene.import_usd` (ernie/scene/usd_import.h).
void BindUsd(nb::module_& m);

/// The `app` submodule; built only with Ernie's app
/// (`ERNIE_HAS_APP`).
void BindApp(nb::module_& m);

/// The demo base class and runner, in the `app` submodule `m`.
void BindDemo(nb::module_& m);

}  // namespace ernie::python

namespace nanobind::detail {

/// Accepts what `ernie::python::IsPrimPath` does, named as such in
/// signatures and stubs.
template <> struct type_caster<ernie::python::PrimPath> {
    NB_TYPE_CASTER(
        ernie::python::PrimPath,
        const_name("str | Path | Prim | ernie.schema.Schema"))

    bool from_python(handle src, uint8_t, cleanup_list*) noexcept {
        if (!ernie::python::IsPrimPath(src)) {
            return false;
        }
        value.object = src;
        return true;
    }

    static handle from_cpp(const ernie::python::PrimPath& path,
                           rv_policy, cleanup_list*) noexcept {
        return path.object.inc_ref();
    }
};

}  // namespace nanobind::detail
