// SPDX-License-Identifier: BSD-3-Clause
#include "bindings.h"

#include "ernie/util/log.h"

NB_MODULE(_ernie, m) {
    m.doc() =
        "Ernie: a standalone Vulkan renderer with a USD-style scene.";
    // ernie.config's quiet and verbose (ernie/util/log.h).
    m.def("_verbosity", [] {
        return static_cast<int>(ernie::log::GetVerbosity());
    });
    m.def("_set_verbosity", [](int verbosity) {
        ernie::log::SetVerbosity(
            static_cast<ernie::log::Verbosity>(verbosity));
    });
    ernie::python::BindTensor(m);
    ernie::python::BindScene(m);
    ernie::python::BindSceneJson(m);
    ernie::python::BindSchema(m);
    ernie::python::BindRender(m);
#ifdef ERNIE_HAS_APP
    ernie::python::BindApp(m);
#endif
}
