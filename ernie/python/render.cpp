// SPDX-License-Identifier: BSD-3-Clause
/// @file render.cpp
/// @brief Python bindings of the renderer: a scene's render settings rendered to a NumPy image.
#include "ernie/python/render.h"

#include "ernie/render/renderer.h"

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstring>
#include <string>

namespace nb = nanobind;

namespace ernie::python {

void BindRender(nb::module_& module) {
    nb::class_<Renderer>(module, "Renderer",
                         "A device and what it keeps of the scene it last rendered, so rendering "
                         "the same scene again costs only what changed.")
        .def(
            "__init__",
            [](Renderer* self, const std::string& device, bool validation) {
                new (self) Renderer(RendererOptions{.device = device, .validation = validation});
            },
            nb::kw_only(), nb::arg("device") = "", nb::arg("validation") = false,
            "Opens a device that ray traces: by UUID or part of its name, or the best one.")
        .def(
            "render",
            [](Renderer& self, const Scene& scene) {
                const Tensor image = self.Render(scene);
                const std::vector<std::int64_t>& shape = image.Shape();
                const std::size_t bytes = static_cast<std::size_t>(shape[0] * shape[1] * shape[2]);
                auto* copy = new std::uint8_t[bytes];
                std::memcpy(copy, image.Data(), bytes);
                const nb::capsule owner(
                    copy, [](void* data) noexcept { delete[] static_cast<std::uint8_t*>(data); });
                const std::array<std::size_t, 3> dimensions{static_cast<std::size_t>(shape[0]),
                                                            static_cast<std::size_t>(shape[1]),
                                                            static_cast<std::size_t>(shape[2])};
                return nb::ndarray<nb::numpy, std::uint8_t, nb::ndim<3>>(copy, 3, dimensions.data(),
                                                                         owner);
            },
            nb::arg("scene"),
            "Renders the camera of the scene's first RenderSettings at its resolution: LdrColor "
            "as a (height, width, 4) uint8 NumPy array.");
}

}  // namespace ernie::python
