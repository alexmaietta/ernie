// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <nanobind/nanobind.h>

#include <functional>
#include <memory>

namespace ernie::python {

namespace nb = nanobind;

/// Python objects that Ernie's C++ objects hold, such as a viewer's
/// callbacks and a demo's viewer, recorded by their owner so the
/// owner's Python type can show them to Python's cyclic garbage
/// collector (`HeldSlots`). Without that, a callback that reaches
/// back to what calls it (an `on_frame` lambda using the demo that
/// holds the viewer) is a cycle through C++ the collector cannot see,
/// never freed, which nanobind reports as leaks at exit.
///
/// Everything here runs under the GIL.
///
/// Reference: nanobind, "Cyclic garbage collection",
/// https://nanobind.readthedocs.io/en/latest/typeslots.html
using Held = std::shared_ptr<nb::object>;

/// Holds `object` for `owner`: the shared slot to call it through,
/// which the collector may empty (`ClearHeld`) when the owner is part
/// of an unreachable cycle, so users check it before use. Dropping
/// the last copy releases the object, taking the GIL.
Held Hold(const void* owner, nb::object object);

/// Counts what `child` holds as `owner`'s too while `alive` says
/// `owner` still has it: a viewer's or demo's widget panels. Once per
/// child, as each held object must be visited once.
void Adopt(const void* owner, const void* child,
           std::function<bool()> alive);

/// The collector's visit of, and release of, what `owner` holds.
int TraverseHeld(const void* owner, visitproc visit, void* arg);
void ClearHeld(const void* owner);

/// `tp_traverse` and `tp_clear` slots for a class whose instances
/// are `T`s holding Python objects, for `nb::type_slots`.
template <typename T> PyType_Slot* HeldSlots() {
    static PyType_Slot slots[] = {
        {Py_tp_traverse,
         reinterpret_cast<void*>(+[](PyObject* self, visitproc visit,
                                     void* arg) -> int {
             Py_VISIT(Py_TYPE(self));
             if (!nb::inst_ready(self)) {
                 return 0;
             }
             return TraverseHeld(nb::inst_ptr<T>(self), visit, arg);
         })},
        {Py_tp_clear,
         reinterpret_cast<void*>(+[](PyObject* self) -> int {
             if (nb::inst_ready(self)) {
                 ClearHeld(nb::inst_ptr<T>(self));
             }
             return 0;
         })},
        {0, nullptr}};
    return slots;
}

}  // namespace ernie::python
