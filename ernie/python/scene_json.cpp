// SPDX-License-Identifier: BSD-3-Clause
#include "bindings.h"

#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include "ernie/scene/scene.h"
#include "ernie/scene/scene_json.h"

#include <filesystem>
#include <fstream>
#include <ios>
#include <optional>
#include <stdexcept>
#include <string>

namespace ernie::python {

void BindSceneJson(nb::module_& m) {
    m.def(
        "save_scene",
        [](const Scene& scene, const std::optional<std::filesystem::path>& path) {
            std::string text = SaveScene(scene);
            if (path) {
                std::ofstream file(*path, std::ios::binary | std::ios::trunc);
                if (!file || !file.write(text.data(), static_cast<std::streamsize>(text.size())) ||
                    !file.flush()) {
                    throw std::invalid_argument("Cannot write the scene file " + path->string());
                }
            }
            return text;
        },
        nb::arg("scene"), nb::arg("path") = nb::none(),
        "Describes scene as JSON in the format ernie.load_scene reads (C++'s SaveScene, "
        "ernie/scene/scene_json.h) and returns the text, also writing it to path when given: "
        "every prim but the viewer's (under /__session), roots by path and the others nested "
        "under their parents, \"type\" first, then the attributes by USD name in name order, "
        "then the children. Values are written so that loading the text rebuilds the scene: "
        "paths as strings, matrices as 4 rows, quaternions as [w, x, y, z], arrays nested by "
        "their shape. Declared attributes come back as their types; undeclared ones as the "
        "JSON reads (2 to 4 numbers as a vector, paths as strings), arrays as int32 or float32. "
        "Raises ValueError for a GPU array (copy it to the host first), a NaN or infinite "
        "number, or a name the loader would read as something else, naming the prim and "
        "attribute.");
}

}  // namespace ernie::python
