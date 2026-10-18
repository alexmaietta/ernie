// SPDX-License-Identifier: BSD-3-Clause
/// @file dlpack.cpp
/// @brief DLPack capsules, versioned and not, per the DLPack Python specification.
#include "ernie/python/dlpack.h"

#include <nanobind/stl/tuple.h>

#include <Python.h>

#include <cstdint>
#include <tuple>

namespace nb = nanobind;

namespace ernie::python {
namespace {

/// Names a capsule takes, per the DLPack Python specification.
constexpr const char* kVersioned = "dltensor_versioned";
constexpr const char* kUnversioned = "dltensor";

/// An unversioned DLPack tensor over a versioned one, for consumers that ask for no version.
DLManagedTensor* Unversioned(DLManagedTensorVersioned* versioned) {
    auto* managed = new DLManagedTensor{.dl_tensor = versioned->dl_tensor,
                                        .manager_ctx = versioned,
                                        .deleter = [](DLManagedTensor* self) {
                                            auto* inner = static_cast<DLManagedTensorVersioned*>(
                                                self->manager_ctx);
                                            inner->deleter(inner);
                                            delete self;
                                        }};
    return managed;
}

/// Takes the tensor from a DLPack capsule, marking the capsule used.
Tensor FromCapsule(const nb::object& capsule) {
    PyObject* object = capsule.ptr();
    if (PyCapsule_IsValid(object, kVersioned)) {
        auto* managed =
            static_cast<DLManagedTensorVersioned*>(PyCapsule_GetPointer(object, kVersioned));
        PyCapsule_SetName(object, "used_dltensor_versioned");
        return Tensor::FromDlpack(managed);
    }
    if (PyCapsule_IsValid(object, kUnversioned)) {
        auto* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(object, kUnversioned));
        PyCapsule_SetName(object, "used_dltensor");
        return Tensor::FromDlpack(managed);
    }
    throw nb::type_error("__dlpack__ returned no DLPack capsule");
}

}  // namespace

nb::object Capsule(DLManagedTensorVersioned* managed, bool versioned) {
    PyObject* capsule = nullptr;
    if (versioned) {
        capsule = PyCapsule_New(managed, kVersioned, [](PyObject* self) {
            // A consumer renames the capsule when it takes the tensor.
            if (PyCapsule_IsValid(self, kVersioned)) {
                auto* tensor =
                    static_cast<DLManagedTensorVersioned*>(PyCapsule_GetPointer(self, kVersioned));
                tensor->deleter(tensor);
            }
        });
    } else {
        capsule = PyCapsule_New(Unversioned(managed), kUnversioned, [](PyObject* self) {
            if (PyCapsule_IsValid(self, kUnversioned)) {
                auto* tensor =
                    static_cast<DLManagedTensor*>(PyCapsule_GetPointer(self, kUnversioned));
                tensor->deleter(tensor);
            }
        });
    }
    if (capsule == nullptr) {
        throw nb::python_error();
    }
    return nb::steal(capsule);
}

Tensor TakeTensor(const nb::object& source, const nb::object& given) {
    // A stream handle of 0, as PyTorch gives its default stream, is spelled 1 in DLPack.
    const nb::object stream =
        !given.is_none() && nb::cast<std::intptr_t>(given) == 0 ? nb::int_(1) : given;
    nb::object capsule;
    try {
        capsule = source.attr("__dlpack__")(nb::arg("stream") = stream,
                                            nb::arg("max_version") =
                                                std::make_tuple(DLPACK_MAJOR_VERSION, 0));
    } catch (const nb::python_error& error) {
        if (!error.matches(PyExc_TypeError)) {
            throw;
        }
        capsule = source.attr("__dlpack__")(nb::arg("stream") = stream);
    }
    return FromCapsule(capsule);
}

}  // namespace ernie::python
