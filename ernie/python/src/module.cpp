// SPDX-License-Identifier: BSD-3-Clause
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "bindings.h"
#include "ernie/app/cli.h"

NB_MODULE(_ernie, module) {
    module.attr("__version__") = ERNIE_VERSION;
    ernie::python::BindDevice(module);
    ernie::python::BindScene(module);
    ernie::python::BindRender(module);
    ernie::python::BindPipeline(module);
    ernie::python::BindImport(module);
    ernie::python::BindIo(module);
    ernie::python::BindRotation(module);
    module.def(
        "run_cli",
        [](const std::vector<std::string>& args) {
            const nanobind::gil_scoped_release release;
            return ernie::app::RunCli(args);
        },
        nanobind::arg("args"));
}
