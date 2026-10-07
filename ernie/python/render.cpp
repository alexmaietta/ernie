#include "bindings.h"

#include "ernie/device/vulkan_context.h"
#include "ernie/session/delegate.h"
#include "ernie/session/renderer.h"

#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ernie::python {

namespace {

Device ParseDevice(const std::string& device) {
    if (device == "cpu") {
        return {};
    }
    if (device == "cuda") {
        return {DeviceKind::Cuda, 0};
    }
    if (device.rfind("cuda:", 0) == 0) {
        try {
            return {DeviceKind::Cuda, std::stoi(device.substr(5))};
        } catch (const std::exception&) {
        }
    }
    throw std::invalid_argument("device must be \"cpu\", \"cuda\" or \"cuda:<index>\", not \"" +
                                device + "\"");
}

/// Holds a Python object, releasing it with the GIL held.
struct PythonReference {
    nb::object object;

    ~PythonReference() {
        if (Py_IsInitialized() != 0) {
            nb::gil_scoped_acquire gil;
            object = nb::object();
        }
    }
};

/// `tensor`, also keeping the Python object `keep` alive, so arrays in memory of the renderer's
/// Vulkan device or CUDA context cannot outlive the renderer.
Tensor KeepingAlive(const Tensor& tensor, nb::handle keep) {
    // The tensor goes before the reference, so what it keeps alive outlives it.
    struct Owner {
        PythonReference keep;
        Tensor tensor;
    };
    auto owner = std::make_shared<Owner>();
    owner->keep.object = nb::borrow(keep);
    owner->tensor = tensor;
    return Tensor(tensor.Data(), tensor.Type(), tensor.Shape(), tensor.Strides(), tensor.Location(),
                  std::move(owner));
}

/// A CUDA render result: implements the DLPack protocol so a consumer's stream waits on the GPU
/// for the render before reading it.
struct CudaArray {
    Tensor tensor;
    CudaInterop* cuda = nullptr;

    nb::object Dlpack(nb::handle stream) const {
        const std::intptr_t handle = stream.is_none() ? 1 : nb::cast<std::intptr_t>(stream);
        if (handle != -1) {
            cuda->WaitForVulkan(static_cast<std::uintptr_t>(handle));
        }
        auto* kept = new Tensor(tensor);
        nb::capsule owner(kept, [](void* p) noexcept { delete static_cast<Tensor*>(p); });
        std::vector<std::size_t> shape(tensor.Shape().begin(), tensor.Shape().end());
        nb::ndarray<nb::ro> array(tensor.Data(), shape.size(), shape.data(), owner, nullptr,
                                  DlpackOf(tensor.Type()), nb::device::cuda::value,
                                  tensor.Location().id);
        return nb::cast(array, nb::rv_policy::reference);
    }
};

/// A render var of a step's outputs, keeping its renderer alive.
struct PyRenderVar {
    nb::object renderer;  // first, so it is released after the var's GPU memory
    RenderVar var;
};

/// A render var mapped as a CUDA array: the frame's own image, read in place.
struct PyCudaArrayMapping {
    CudaArrayView view;
    PyRenderVar var;
};

/// One frame of a render product: its render vars by path.
struct PyFrame {
    std::map<std::string, PyRenderVar> renderVars;
};

/// The outputs of one render product of a step.
struct PyProductOutput {
    std::string product;
    std::vector<PyFrame> frames;
};

using PyOutputs = std::map<std::string, PyProductOutput>;

PyOutputs Wrap(const RenderProductSetOutputs& outputs, nb::handle renderer) {
    PyOutputs wrapped;
    for (const auto& [path, output] : outputs) {
        PyProductOutput product;
        product.product = path.String();
        PyFrame frame;
        for (const auto& [name, var] : output.vars) {
            frame.renderVars.emplace(var.GetPath().String(),
                                     PyRenderVar{nb::borrow(renderer), var});
        }
        product.frames.push_back(std::move(frame));
        wrapped.emplace(path.String(), std::move(product));
    }
    return wrapped;
}

/// A submitted step.
struct PyOperation {
    nb::object renderer;  // first, so it is released after the step's GPU memory
    std::optional<StepOperation> operation;
};

std::vector<Path> PathsOf(nb::handle products) {
    std::vector<Path> paths;
    if (products.is_none()) {
        return paths;
    }
    for (nb::handle product : products) {
        paths.push_back(nb::isinstance<Path>(product) ? nb::cast<Path>(product)
                                                      : Path(nb::cast<std::string>(product)));
    }
    return paths;
}

CudaInterop& RequireCuda(Renderer& r) {
    std::string why;
    CudaInterop* cuda = r.Delegate().Cuda(&why);
    if (cuda == nullptr) {
        throw std::invalid_argument("CUDA interop is unavailable: " + why);
    }
    return *cuda;
}

}  // namespace

void BindRender(nb::module_& m) {
    nb::class_<CudaArray>(m, "CudaArray",
                          "A render in CUDA memory. Pass it to torch.from_dlpack, cupy.from_dlpack "
                          "or jax.dlpack.from_dlpack: its __dlpack__ makes the consumer's stream "
                          "wait for the render on the GPU, without blocking the host.")
        .def(
            "__dlpack__",
            [](const CudaArray& a, nb::handle stream, nb::handle, nb::handle, nb::handle) {
                return a.Dlpack(stream);
            },
            nb::kw_only(), nb::arg("stream").none() = nb::none(),
            nb::arg("max_version").none() = nb::none(), nb::arg("dl_device").none() = nb::none(),
            nb::arg("copy").none() = nb::none())
        .def("__dlpack_device__",
             [](const CudaArray& a) {
                 return nb::make_tuple(static_cast<int>(nb::device::cuda::value),
                                       a.tensor.Location().id);
             })
        .def_prop_ro("shape", [](const CudaArray& a) {
            nb::list shape;
            for (const std::int64_t extent : a.tensor.Shape()) {
                shape.append(extent);
            }
            return nb::tuple(shape);
        });

    nb::enum_<Wireframe>(m, "Wireframe",
                         "How the rasterizer draws triangle edges (tokens.WIREFRAME).")
        .value("OFF", Wireframe::Off, "Shaded surfaces only.")
        .value("OVERLAY", Wireframe::Overlay, "Triangle edges over the shaded surfaces.")
        .value("HIDDEN_LINE", Wireframe::HiddenLine,
               "Triangle edges on flat, unlit surfaces that hide the edges behind them.");

    nb::enum_<MapDevice>(m, "Device", "Where RenderVar.map delivers an output.")
        .value("CPU", MapDevice::Cpu, "A NumPy array; the texels are copied to the host.")
        .value("CUDA", MapDevice::Cuda,
               "A linear CUDA array the GPU fills from the render, without the host.")
        .value("CUDA_ARRAY", MapDevice::CudaArray,
               "The rendered image itself as a CUDA array (CUarray): no copy at all.");

    nb::class_<PyCudaArrayMapping>(
        m, "CudaArrayMapping",
        "A render var mapped as a CUDA array: the frame's image, read in place with CUDA texture "
        "or surface APIs or wp.Texture2D(cuda_array=mapping.cuda_array). Unmap the render var "
        "when done.")
        .def_prop_ro(
            "cuda_array", [](const PyCudaArrayMapping& mapping) { return mapping.view.array; },
            "The CUarray handle.")
        .def_prop_ro("shape",
                     [](const PyCudaArrayMapping& mapping) {
                         return nb::make_tuple(mapping.view.height, mapping.view.width,
                                               mapping.view.channels);
                     })
        .def_prop_ro("device_id",
                     [](const PyCudaArrayMapping& mapping) { return mapping.view.device; });

    nb::class_<PyRenderVar>(m, "RenderVar",
                            "One output of a rendered frame, such as LdrColor, as UsdRender's "
                            "RenderVar.")
        .def_prop_ro("path", [](const PyRenderVar& v) { return v.var.GetPath(); })
        .def_prop_ro("name", [](const PyRenderVar& v) { return v.var.Name(); })
        .def_prop_ro(
            "shape",
            [](const PyRenderVar& v) {
                nb::list shape;
                for (const std::int64_t extent : v.var.Shape()) {
                    shape.append(extent);
                }
                return nb::tuple(shape);
            },
            "(height, width, channels), (height, width) for single-channel vars, or (bands, "
            "height, width) for SpectralRadiance.")
        .def_prop_ro(
            "dtype",
            [](const PyRenderVar& v) {
                const nb::dlpack::dtype type = DlpackOf(v.var.Type());
                const char* kind =
                    type.code == static_cast<std::uint8_t>(nb::dlpack::dtype_code::Float) ? "float"
                                                                                          : "uint";
                return std::string(kind) + std::to_string(type.bits);
            },
            "The NumPy name of the element type: uint8, uint16, uint32 or float32.")
        .def(
            "map",
            [](PyRenderVar& v, MapDevice device, nb::handle syncStream) -> nb::object {
                MappedRenderVar mapped;
                const std::uintptr_t stream = StreamOf(syncStream);  // needs the GIL
                {
                    nb::gil_scoped_release release;
                    mapped = v.var.Map(device, stream);
                }
                if (device == MapDevice::CudaArray) {
                    return nb::cast(PyCudaArrayMapping{mapped.array, v});
                }
                return ArrayFromTensor(KeepingAlive(mapped.tensor, v.renderer));
            },
            nb::arg("device") = MapDevice::Cpu, nb::arg("sync_stream").none() = nb::none(),
            "Maps the output: Device.CPU waits for the render and returns a NumPy array; "
            "Device.CUDA and Device.CUDA_ARRAY make CUDA stream `sync_stream` (a CUstream handle "
            "or DLPack stream number, default the legacy stream) wait for the render on the GPU "
            "and return at once.")
        .def(
            "unmap", [](PyRenderVar& v, nb::handle stream) { v.var.Unmap(StreamOf(stream)); },
            nb::arg("stream").none() = nb::none(),
            "Ends a CUDA_ARRAY mapping; the renderer waits for the work queued on `stream` before "
            "rendering into the frame again.")
        .def_prop_ro("mapped", [](const PyRenderVar& v) { return v.var.IsMapped(); });

    nb::class_<PyFrame>(m, "Frame", "A rendered frame of a render product.")
        .def_ro("render_vars", &PyFrame::renderVars, "Render vars by RenderVar path.");

    nb::class_<PyProductOutput>(m, "RenderProductOutput", "The outputs of one render product.")
        .def_ro("product", &PyProductOutput::product)
        .def_ro("frames", &PyProductOutput::frames);

    nb::class_<PyOperation>(m, "Operation", "A submitted step: wait or poll, then fetch.")
        .def(
            "wait",
            [](nb::handle self, std::optional<std::int64_t> timeoutNs) -> nb::object {
                PyOperation& op = nb::cast<PyOperation&>(self);
                bool done = false;
                {
                    nb::gil_scoped_release release;
                    done = op.operation->Wait(timeoutNs ? std::chrono::nanoseconds(*timeoutNs)
                                                        : std::chrono::nanoseconds::max());
                }
                return done ? nb::borrow(self) : nb::none();
            },
            nb::arg("timeout_ns").none() = nb::none(),
            "Waits up to `timeout_ns` nanoseconds (None: forever, 0: poll) for the step; returns "
            "the operation once it is done, for .fetch(), or None while it is pending.")
        .def_prop_ro("done", [](PyOperation& op) { return op.operation->Done(); })
        .def(
            "fetch",
            [](PyOperation& op) {
                RenderProductSetOutputs outputs;
                {
                    nb::gil_scoped_release release;
                    outputs = op.operation->Fetch();
                }
                return Wrap(outputs, op.renderer);
            },
            "The step's outputs by render product path, waiting for the step first.")
        .def(
            "outputs", [](PyOperation& op) { return Wrap(op.operation->Outputs(), op.renderer); },
            "The step's outputs without waiting, for CUDA mappings that order on the GPU.");

    nb::class_<Renderer>(
        m, "Renderer",
        "Renders a Scene's terrain, meshes, materials, lights, cameras and render settings on a "
        "GPU, with one of Ernie's render backends.")
        .def(
            "__init__",
            [](Renderer* self, const std::string& backend, bool validation,
               const std::string& device) {
                new (self) Renderer(RendererOptions{backend, device, validation});
            },
            nb::arg("backend") = "raster", nb::arg("validation") = false, nb::arg("device") = "",
            "Opens a GPU, preferring one whose name contains `device`, with the Vulkan "
            "validation layer when `validation` is set and it is installed, and the render "
            "`backend` (see `backends`) on it. Raises RuntimeError without a Vulkan device and "
            "ValueError for an unknown backend.")
        .def_static("backends", &RenderBackends, "The names of the render backends.")
        .def_prop_ro("backend", [](const Renderer& r) { return r.Delegate().Info().name; })
        .def_prop_ro(
            "progressive", [](const Renderer& r) { return r.Delegate().Info().progressive; },
            "Whether consecutive steps of an unchanged scene refine one image.")
        .def_prop_ro("device_name", [](const Renderer& r) { return r.Context().DeviceName(); })
        .def(
            "add_sensor_filter",
            [](Renderer& r, const std::string& id, const std::string& typeName,
               const std::string& slang, const std::string& kind, nb::object domain,
               nb::object params, const std::string& name, const std::string& description) {
                SensorFilterModel model;
                model.id = id;
                model.typeName = typeName;
                model.slang = slang;
                model.name = name.empty() ? id : name;
                model.description = description;
                if (kind == "band") {
                    model.kind = FilterKind::Band;
                } else if (kind == "pixel") {
                    model.kind = FilterKind::Pixel;
                } else {
                    throw std::invalid_argument("kind must be \"band\" or \"pixel\", not \"" +
                                                kind + "\"");
                }
                const std::string on = domain.is_none()
                                           ? (model.kind == FilterKind::Band ? "spectrum" : "")
                                           : nb::cast<std::string>(domain);
                const std::pair<const char*, FilterDomain> domains[] = {
                    {"spectrum", FilterDomain::Spectrum},
                    {"electrons", FilterDomain::Electrons},
                    {"digital", FilterDomain::Digital},
                    {"rgb", FilterDomain::Rgb},
                    {"display", FilterDomain::Display}};
                const auto found =
                    std::find_if(std::begin(domains), std::end(domains),
                                 [&on](const auto& entry) { return on == entry.first; });
                if (found == std::end(domains)) {
                    throw std::invalid_argument(
                        "domain must be \"spectrum\", \"electrons\", \"digital\", \"rgb\" or "
                        "\"display\", not \"" +
                        on + "\"");
                }
                model.domain = found->second;
                // Parameters as {name: default} in order, or [{"name", "default", "description"}].
                if (nb::isinstance<nb::dict>(params)) {
                    for (const auto& [key, value] : nb::cast<nb::dict>(params)) {
                        model.params.push_back(
                            {nb::cast<std::string>(key), nb::cast<float>(value), {}});
                    }
                } else if (!params.is_none()) {
                    for (const nb::handle entry : params) {
                        const nb::dict param = nb::cast<nb::dict>(entry);
                        model.params.push_back(
                            {nb::cast<std::string>(param["name"]),
                             param.contains("default") ? nb::cast<float>(param["default"]) : 0.0f,
                             param.contains("description")
                                 ? nb::cast<std::string>(param["description"])
                                 : std::string()});
                    }
                }
                nb::gil_scoped_release release;
                r.Delegate().AddSensorFilter(std::move(model));
            },
            nb::arg("id"), nb::arg("type_name"), nb::arg("slang"), nb::arg("kind") = "pixel",
            nb::arg("domain").none() = nb::none(), nb::arg("params").none() = nb::none(),
            nb::arg("name") = "", nb::arg("description") = "",
            "Registers a sensor filter written in Slang, for SensorFilter prims to name in "
            "ernie:filter:type: struct `type_name` in module `slang` implementing IBandFilter "
            "(kind=\"band\", on the spectrum) or IPixelFilter (kind=\"pixel\", domain "
            "\"electrons\", \"digital\", \"rgb\" or \"display\"), with a nested Params "
            "struct of one float per parameter. `params` is {name: default} in order, or a list "
            "of {\"name\", \"default\", \"description\"}; prims set them as inputs:<name>. "
            "Raises ValueError for an invalid filter or one that fails to compile, with the "
            "compiler's diagnostics.")
        .def_prop_ro(
            "sensor_filters",
            [](const Renderer& r) {
                std::vector<std::string> ids;
                for (const std::string_view name :
                     {"lens", "qe", "noise", "adc", "cie", "exposure"}) {
                    ids.emplace_back(name);
                }
                for (const SensorFilterModel& model : r.Delegate().SensorFilters().Models()) {
                    ids.push_back(model.id);
                }
                return ids;
            },
            "The sensor filter types render products may use: the built-in ones, then the "
            "registered ones.")
        .def_prop_ro("brdfs",
                     [](const Renderer& r) {
                         std::vector<std::string> ids;
                         for (const BrdfModel& model : r.Delegate().Brdfs().Models()) {
                             ids.push_back(model.id);
                         }
                         return ids;
                     })
        .def_prop_ro(
            "stats",
            [](const Renderer& r) {
                const RenderStats stats = r.Delegate().Stats();
                nb::dict result;
                result["gpu_bytes"] = stats.gpuBytes;
                result["triangles"] = stats.triangles;
                for (const auto& [name, value] : stats.counters) {
                    result[nb::str(name.c_str())] = value;
                }
                return result;
            },
            "GPU memory, triangles and the backend's own counts, by name.")
        .def(
            "sync",
            [](Renderer& r, const Scene& scene) {
                nb::gil_scoped_release release;
                r.Delegate().Sync(scene);
            },
            nb::arg("scene"), "Uploads the parts of the scene that changed since the last sync.")
        .def(
            "render",
            [](nb::pointer_and_handle<Renderer> self, const Scene& scene, const Path& settings,
               const std::string& device) {
                const Device target = ParseDevice(device);
                Tensor image;
                {
                    nb::gil_scoped_release release;
                    image = self.p->Delegate().Render(scene, settings, target);
                }
                if (target.kind == DeviceKind::Cuda) {
                    return nb::cast(
                        CudaArray{KeepingAlive(image, self.h), self.p->Delegate().Cuda()});
                }
                return ArrayFromTensor(KeepingAlive(image, self.h));
            },
            nb::arg("scene"), nb::arg("settings") = Path(), nb::arg("device") = "cpu",
            "Renders `scene` as the RenderSettings prim `settings` (or the first one) describes "
            "and returns the LdrColor render var: a read-only uint8 array (rows, cols, 4). With "
            "device=\"cuda\" the array stays in GPU memory shared with Vulkan, for "
            "torch.from_dlpack and friends; it keeps the renderer alive.")
        .def(
            "step_async",
            [](nb::pointer_and_handle<Renderer> self, const Scene& scene, nb::handle renderProducts,
               const Path& settings) {
                const std::vector<Path> products = PathsOf(renderProducts);
                PyOperation op;
                op.renderer = nb::borrow(self.h);
                {
                    nb::gil_scoped_release release;
                    op.operation = self.p->Delegate().StepAsync(scene, products, settings);
                }
                return op;
            },
            nb::arg("scene"), nb::arg("render_products").none() = nb::none(),
            nb::arg("settings") = Path(),
            "Renders the RenderProduct prims `render_products` (or the settings' products, or a "
            "default one) and returns an Operation at once.")
        .def(
            "step",
            [](nb::pointer_and_handle<Renderer> self, const Scene& scene, nb::handle renderProducts,
               const Path& settings) {
                const std::vector<Path> products = PathsOf(renderProducts);
                RenderProductSetOutputs outputs;
                {
                    nb::gil_scoped_release release;
                    outputs = self.p->Delegate().Step(scene, products, settings);
                }
                return Wrap(outputs, self.h);
            },
            nb::arg("scene"), nb::arg("render_products").none() = nb::none(),
            nb::arg("settings") = Path(),
            "step_async(...).wait().fetch(): the outputs by render product path, each with "
            ".frames[0].render_vars by RenderVar path.")
        .def(
            "cuda_copy",
            [](Renderer& r, const AnyArray& source, const AnyArray& destination,
               nb::handle stream) {
                RequireCuda(r).Copy(TensorFromArray(source), TensorFromArray(destination),
                                    StreamOf(stream));
            },
            nb::arg("source"), nb::arg("destination"), nb::arg("stream").none() = nb::none(),
            "Queues a device-to-device copy between CUDA arrays of the same size on `stream`.")
        .def_prop_ro(
            "cuda_available", [](Renderer& r) { return r.Delegate().Cuda() != nullptr; },
            "Whether CUDA arrays can be written to scenes this renderer draws, and rendered to.")
        .def(
            "cuda_upload",
            [](nb::pointer_and_handle<Renderer> self, const AnyArray& array) {
                return ArrayFromTensor(
                    KeepingAlive(RequireCuda(*self.p).Upload(TensorFromArray(array)), self.h));
            },
            nb::arg("array"),
            "A copy of a host array in CUDA memory, for frameworks without CUDA arrays of their "
            "own.")
        .def(
            "cuda_download",
            [](Renderer& r, const AnyArray& array) {
                return ArrayFromTensor(RequireCuda(r).Download(TensorFromArray(array)));
            },
            nb::arg("array"), "A host copy of a CUDA array.")
        .def(
            "prim_path", [](Renderer& r, std::uint32_t id) { return r.Delegate().PrimPath(id); },
            nb::arg("id"),
            "The prim path of a PrimId render var value; the empty path for 0 (background) or an "
            "unknown id.")
        .def(
            "take_validation_messages",
            [](Renderer& r) { return r.Context().TakeValidationMessages(); },
            "Vulkan validation messages since the last call.");
}

}  // namespace ernie::python
