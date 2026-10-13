// SPDX-License-Identifier: BSD-3-Clause
#include "bindings.h"

#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include "ernie/scene/scene.h"
#include "ernie/scene/usd_import.h"

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace ernie::python {

namespace {

/// What `Scene.import_usd` returns: the root prim, and the Ernie prim
/// each USD prim became.
struct PyUsdImport {
    Prim root;
    std::unordered_map<std::string, Path> prims;
    std::vector<std::string> warnings;
    std::optional<std::pair<double, double>> timeCodeRange;
    double timeCodesPerSecond = 24.0;
    double framesPerSecond = 24.0;
};

}  // namespace

void BindUsd(nb::module_& m) {
    nb::class_<PyUsdImport>(
        m, "UsdImport",
        "What Scene.import_usd made: the prim the file became "
        "(root), "
        "the Ernie path of each translated USD prim by its USD path "
        "(prims), what was skipped or approximated (warnings), and "
        "the "
        "file's authored animation range (time_code_range, (start, "
        "end) "
        "or None) and rates. The range becomes the scene's Stage "
        "range "
        "when the scene has none.")
        .def_ro("root", &PyUsdImport::root)
        .def_ro("prims", &PyUsdImport::prims)
        .def_ro("warnings", &PyUsdImport::warnings)
        .def_ro("time_code_range", &PyUsdImport::timeCodeRange)
        .def_ro("time_codes_per_second",
                &PyUsdImport::timeCodesPerSecond)
        .def_ro("frames_per_second", &PyUsdImport::framesPerSecond)
        .def("__repr__", [](const PyUsdImport& result) {
            return "UsdImport(" + result.root.GetPath().String() +
                   ", " + std::to_string(result.prims.size()) +
                   " prims, " +
                   std::to_string(result.warnings.size()) +
                   " warnings)";
        });

    m.def(
        "usd_import_available",
        [] {
            std::string reason;
            const bool available = UsdImportAvailable(&reason);
            return std::make_pair(available, reason);
        },
        "Whether Scene.import_usd can load its OpenUSD plugin, and "
        "why "
        "not when it cannot.");

    nb::type<Scene>().attr("import_usd") = nb::cpp_function(
        [](Scene& scene, nb::handle root,
           const std::filesystem::path& file,
           const std::optional<std::unordered_map<
               std::string,
               std::unordered_map<std::string, std::string>>>&
               variants,
           std::optional<double> time, bool proxy, bool guide) {
            UsdImportOptions options;
            if (variants) {
                for (const auto& [prim, sets] : *variants) {
                    for (const auto& [set, selection] : sets) {
                        options.variants.push_back(
                            {prim, set, selection});
                    }
                }
            }
            options.time = time;
            options.proxy = proxy;
            options.guide = guide;
            const Path path = PathFromPython(root);
            UsdImportResult result;
            {
                nb::gil_scoped_release release;
                result = ImportUsd(scene, path, file, options);
            }
            PyUsdImport out{Prim(&scene, result.root),
                            std::move(result.prims),
                            std::move(result.warnings)};
            if (result.timeCodeRange) {
                out.timeCodeRange = std::pair(
                    result.timeCodeRange->x, result.timeCodeRange->y);
            }
            out.timeCodesPerSecond = result.timeCodesPerSecond;
            out.framesPerSecond = result.framesPerSecond;
            return out;
        },
        nb::is_method(), nb::arg("root"), nb::arg("file"),
        nb::arg("variants") = nb::none(),
        nb::arg("time") = nb::none(), nb::arg("proxy") = false,
        nb::arg("guide") = false,
        "Imports a USD file (.usd, .usda, .usdc, .usdz) at root, a "
        "prim "
        "path, through OpenUSD: composed (sublayers, references, "
        "payloads, inherits, variants), then translated prim by "
        "prim. "
        "The file's default prim becomes root, as a USD reference "
        "places it, with its upAxis and metersPerUnit converted into "
        "the scene's. variants selects before composing, as "
        "{usd_prim_path: {variant_set: selection}}; time is the time "
        "code read (the earliest otherwise); proxy and guide also "
        "translate prims of those purposes. Returns a UsdImport. "
        "Raises "
        "RuntimeError when the plugin (libernie_usd, built with "
        "-DERNIE_USD=ON) is missing or the file cannot be opened.");
}

}  // namespace ernie::python
