// SPDX-License-Identifier: BSD-3-Clause
/// @file render.cpp
/// @brief Python bindings of the renderer: a scene's products rendered to NumPy images, or stepped
/// into device columns PyTorch and Warp read in place.
#include "ernie/python/render.h"

#include "ernie/device/context.h"
#include "ernie/python/device.h"
#include "ernie/python/dlpack.h"
#include "ernie/render/renderer.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace ernie::python {
namespace {

/// A renderer, the CUDA device paired with its device, and the columns bound to it, whose writes
/// each step waits for. Members go in reverse: CUDA first.
struct PythonRenderer {
    Renderer renderer;
    std::unique_ptr<CudaInterop> cuda;
    std::vector<nb::weakref> bound;

    explicit PythonRenderer(const RendererOptions& options)
        : renderer(options), cuda(CudaInterop::Create(renderer.Context())) {}

    Interop Pair() {
        return {&renderer.Context(), cuda.get()};
    }

    /// Makes the next step wait for the bound columns' writes, forgetting columns gone.
    void WaitForBound() {
        std::erase_if(bound, [&](const nb::weakref& reference) {
            const nb::object column = reference();
            if (column.is_none()) {
                return true;
            }
            renderer.WaitFor(nb::cast<const Column&>(column).written);
            return false;
        });
    }
};

/// A batch of a step: its products and each var's columns, valid until the next step.
struct PythonBatch {
    std::vector<std::string> products;
    std::tuple<std::uint32_t, std::uint32_t> size;
    std::vector<std::pair<std::string, nb::object>> vars;

    nb::object Get(const std::string& name) const {
        for (const auto& [var, column] : vars) {
            if (var == name) {
                return column;
            }
        }
        throw nb::key_error(("The batch holds no '" + name + "'").c_str());
    }
};

std::vector<PythonBatch> Step(PythonRenderer& self, const nb::object& owner, Scene& scene) {
    self.WaitForBound();
    const std::span<const RenderBatch> batches = self.renderer.Step(scene);
    const Interop interop = self.Pair();
    const Ticket ready = self.renderer.Context().Main().Last();
    std::vector<PythonBatch> result;
    for (const RenderBatch& batch : batches) {
        PythonBatch& next = result.emplace_back();
        next.size = {batch.size.x, batch.size.y};
        for (const Path& product : batch.products) {
            next.products.push_back(product.String());
        }
        for (const RenderVarOutput& var : batch.vars) {
            auto* column = new Column(interop, *var.column, var.shape, ready);
            column->keeper = owner;
            next.vars.emplace_back(var.name, nb::cast(column, nb::rv_policy::take_ownership));
        }
    }
    return result;
}

}  // namespace

void BindRender(nb::module_& module) {
    nb::class_<PythonBatch>(module, "RenderBatch",
                            "Products of one size and vars rendered in one dispatch: each var a "
                            "DeviceColumn of shape (views, height, width, channels...), a view a "
                            "product, valid until the renderer's next step.")
        .def_ro("products", &PythonBatch::products)
        .def_ro("size", &PythonBatch::size, "(width, height) in pixels.")
        .def("__getitem__", &PythonBatch::Get, nb::arg("name"))
        .def("__contains__",
             [](const PythonBatch& self, const std::string& name) {
                 return std::ranges::any_of(self.vars,
                                            [&](const auto& var) { return var.first == name; });
             })
        .def("keys", [](const PythonBatch& self) {
            std::vector<std::string> names;
            for (const auto& var : self.vars) {
                names.push_back(var.first);
            }
            return names;
        });

    nb::class_<PythonRenderer>(module, "Renderer",
                               "A device and what it keeps of the scene it last rendered, so "
                               "rendering the same scene again costs only what changed.")
        .def(
            "__init__",
            [](PythonRenderer* self, const std::string& device, bool validation,
               std::uint32_t worlds, std::uint32_t instancesPerWorld, std::uint32_t views) {
                new (self) PythonRenderer(RendererOptions{
                    .device = device,
                    .validation = validation,
                    .capacities = {
                        .worlds = worlds, .instancesPerWorld = instancesPerWorld, .views = views}});
            },
            nb::kw_only(), nb::arg("device") = "", nb::arg("validation") = false,
            nb::arg("worlds") = 0, nb::arg("instances_per_world") = 0, nb::arg("views") = 0,
            "Opens a device that ray traces: by UUID or part of its name, or the best one. "
            "worlds, instances_per_world and views declare the most the scene holds, laid out "
            "once; a scene beyond them raises. 0 is as many as there are.")
        .def(
            "render",
            [](PythonRenderer& self, Scene& scene) { return ToNumpy(self.renderer.Render(scene)); },
            nb::arg("scene"),
            "Renders the scene's first product: LdrColor as a (height, width, 4) uint8 NumPy "
            "array.")
        .def(
            "step",
            [](nb::object self, Scene& scene) {
                return Step(nb::cast<PythonRenderer&>(self), self, scene);
            },
            nb::arg("scene"),
            "Renders every product of the scene's first RenderSettings, products alike in size "
            "and vars in one dispatch, and returns the RenderBatches without waiting: their vars "
            "are device columns that torch.from_dlpack views in place, or read() copies.")
        .def(
            "bind_column",
            [](nb::object self, Scene& scene, const std::string& prim,
               const std::string& attribute) {
                auto& renderer = nb::cast<PythonRenderer&>(self);
                const DeviceColumn& bound =
                    renderer.renderer.BindColumn(scene, Path(prim), attribute);
                auto* column = new Column(renderer.Pair(), bound, {}, Ticket{});
                column->keeper = self;
                nb::object handle = nb::cast(column, nb::rv_policy::take_ownership);
                renderer.bound.emplace_back(handle);
                return handle;
            },
            nb::arg("scene"), nb::arg("prim"), nb::arg("attribute"),
            "Hands a transform op (such as xformOp:translate) or primvars:displayColor of every "
            "prim in the prim's bucket to the device: returns its DeviceColumn, a row a prim in "
            "the bucket's order, which PyTorch or Warp write in place; call ready(stream) after "
            "writing, and the next step renders the new values with no copy through the host.");
}

}  // namespace ernie::python
