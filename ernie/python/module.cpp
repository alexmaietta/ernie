// SPDX-License-Identifier: BSD-3-Clause
#include "bindings.h"

NB_MODULE(_ernie, m) {
    m.doc() = "Ernie: a standalone Vulkan renderer with a USD-style scene.";
    ernie::python::BindTensor(m);
    ernie::python::BindScene(m);
    ernie::python::BindSchema(m);
    ernie::python::BindRender(m);
#ifdef ERNIE_HAS_APP
    ernie::python::BindApp(m);
#endif
}
