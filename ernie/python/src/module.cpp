// SPDX-License-Identifier: BSD-3-Clause
#include <nanobind/nanobind.h>

#include "bindings.h"

NB_MODULE(_ernie, module) {
    module.attr("__version__") = ERNIE_VERSION;
    ernie::python::BindDevice(module);
    ernie::python::BindScene(module);
    ernie::python::BindRender(module);
    ernie::python::BindImport(module);
    ernie::python::BindIo(module);
    ernie::python::BindRotation(module);
}
