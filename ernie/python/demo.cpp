// SPDX-License-Identifier: BSD-3-Clause
#include "bindings.h"
#include "held.h"

#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include "ernie/app/cli.h"
#include "ernie/app/demo.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ernie::python {

namespace {

using app::Demo;

/// `demo` as Python sees it: the Python object it is, for a demo made
/// in Python, rather than nanobind's cast of the shared_ptr, which
/// would keep that object alive from itself (the shared_ptr of a
/// Python object holds a reference to it).
nb::object DemoObject(const std::shared_ptr<app::Demo>& demo) {
    if (!demo) {
        return nb::none();
    }
    nb::object found = nb::find(demo.get());
    return found.is_valid() ? found : nb::cast(demo);
}
using app::DemoInfo;
using app::DemoOptions;
using app::DemoRunner;

/// A Python subclass of Demo: its methods override the virtuals, and
/// its class attributes (num_frames, fixed_dt, max_frame_dt) the
/// settings, as a Python demo declares them.
class PyDemo : public Demo {
  public:
    NB_TRAMPOLINE(Demo);

    /// The viewer, held where the collector sees it (held.h).
    Held viewer;

    void Step() override {
        NB_OVERRIDE_NAME("step", Step);
    }
    void Render() override {
        NB_OVERRIDE_NAME("render", Render);
    }
    void SaveOutput(const std::filesystem::path& path) override {
        NB_OVERRIDE_NAME("save_output", SaveOutput, path);
    }
    void TestPostStep() override {
        NB_OVERRIDE_NAME("test_post_step", TestPostStep);
    }
    void TestFinal() override {
        NB_OVERRIDE_NAME("test_final", TestFinal);
    }
    void Close() override {
        NB_OVERRIDE_NAME("close", Close);
    }
    std::string Description() const override {
        NB_OVERRIDE_NAME("description", Description);
    }
    std::uint64_t NumFrames() const override {
        return Attribute("num_frames", Demo::NumFrames());
    }
    double FixedDt() const override {
        return Attribute("fixed_dt", Demo::FixedDt());
    }
    double MaxFrameDt() const override {
        return Attribute("max_frame_dt", Demo::MaxFrameDt());
    }

  private:
    /// The Python attribute `name` of the demo, or `fallback` when it
    /// has none.
    template <typename T>
    T Attribute(const char* name, T fallback) const {
        nb::gil_scoped_acquire gil;
        nb::object value =
            nb::getattr(nb_trampoline.base(), name, nb::none());
        return value.is_none() ? fallback : nb::cast<T>(value);
    }
};

}  // namespace

void BindDemo(nb::module_& m) {
    nb::class_<DemoOptions>(m, "DemoOptions",
                            "The options every demo takes "
                            "(parse_demo_options), as Newton's "
                            "examples share theirs.")
        .def(nb::init<>())
        .def_rw("viewer", &DemoOptions::viewer,
                "Shows the viewer (--viewer window); False renders "
                "without one (--viewer null).")
        .def_rw("headless", &DemoOptions::headless,
                "The viewer without a window.")
        .def_rw("num_frames", &DemoOptions::numFrames,
                nb::arg("value").none(),
                "Frames to run, or None: a window runs until closed, "
                "the other modes the demo's "
                "num_frames.")
        .def_rw("benchmark", &DemoOptions::benchmark,
                "Reports frames per second after a warm-up, without "
                "a viewer.")
        .def_rw("benchmark_seconds", &DemoOptions::benchmarkSeconds,
                nb::arg("value").none(),
                "Stops a benchmark after this many seconds, or None.")
        .def_rw("profile", &DemoOptions::profile,
                "Prints the profiler's zones at the end (--profile).")
        .def_rw("profile_json", &DemoOptions::profileJson,
                "Where to also write them as JSON, or ''.")
        .def_rw("profile_trace", &DemoOptions::profileTrace,
                "Where to also write a Chrome trace, or ''.")
        .def_rw("test", &DemoOptions::test,
                "Runs test_post_step and test_final.")
        .def_rw("output", &DemoOptions::output,
                "Where to write the last frame as PNG, or ''.")
        .def_rw("paused", &DemoOptions::paused)
        .def_rw("device", &DemoOptions::device,
                "GPU to prefer, by part of its name.")
        .def_rw("renderer", &DemoOptions::renderer, "Render backend.")
        .def_rw("validation", &DemoOptions::validation)
        .def_rw("vsync", &DemoOptions::vsync)
        .def_rw("demo", &DemoOptions::demo,
                "The demo to start with (--demo NAME), or ''.")
        .def_rw("extra", &DemoOptions::extra,
                "The arguments that are none of the shared options, "
                "for the demo's own.")
        .def_prop_ro("windowed", &DemoOptions::Windowed,
                     "A viewer with a window.");
    m.def(
        "main",
        [](const std::vector<std::string>& args,
           const std::string& version) {
            return app::ErnieMain(args, version);
        },
        nb::arg("args"), nb::arg("version"),
        "The `ernie` program (ernie.cli): the viewer on a scene file "
        "or the default scene with "
        "a Demos panel of Ernie's C++ demos, or a render without a "
        "window; args without the "
        "program name. Returns the exit status.");
    m.def("registered_demos", &app::RegisteredDemos,
          "The C++ demos compiled in (ERNIE_REGISTER_DEMO), by "
          "category and name.");
    m.def(
        "parse_demo_options",
        [](const std::vector<std::string>& args) {
            return app::ParseDemoOptions(args);
        },
        nb::arg("args"),
        "Parses the shared demo options from a command line without "
        "the program name, leaving "
        "the others in extra; raises ValueError for a bad value.");

    // Shows the collector its viewer and widgets' callbacks (held.h).
    nb::class_<Demo, PyDemo>(
        m, "Demo", nb::type_slots(HeldSlots<Demo>()),
        "A demo: a scene built in __init__, advanced by step, drawn "
        "by render and checked by "
        "test_final, run by DemoRunner in a window, headless or "
        "without a viewer. Subclass "
        "ernie.demo.Demo in Python, which adds argparse options and "
        "NumPy images.")
        .def(
            "__init__",
            [](Demo* self, app::Viewer* viewer,
               const DemoOptions& options) {
                // The trampoline, which Python subclasses need.
                auto* demo = new (self) PyDemo(viewer, options);
                // Keeps the viewer alive where the collector sees it
                // (held.h), unlike keep_alive: a demo whose viewer's
                // callbacks reach it is then a cycle it can free.
                if (viewer != nullptr) {
                    demo->viewer = Hold(
                        static_cast<const Demo*>(demo),
                        nb::cast(viewer, nb::rv_policy::reference));
                }
            },
            nb::arg("viewer").none(), nb::arg("options"))
        .def("step", &Demo::Step,
             "Advances the scene by frame_dt; nothing by default.")
        .def("render", &Demo::Render,
             "Draws a frame: the viewer's next, or without one the "
             "scene through renderer.")
        .def("save_output", &Demo::SaveOutput, nb::arg("path"),
             "Writes --output: the viewport's next frame, or the "
             "scene's LdrColor as PNG.")
        .def("test_post_step", &Demo::TestPostStep,
             "Checks after every step under --test.")
        .def("test_final", &Demo::TestFinal,
             "Checks at the end of a --test run.")
        .def("close", &Demo::Close,
             "Releases what the demo holds, when it ends or is "
             "replaced.")
        .def("description", &Demo::Description,
             "One line on what the demo shows.")
        // Tied to its owner, the viewer (whose owner is the runner)
        // or the demo's own shared scene, not to the demo: prims keep
        // their scene alive, and the demo keeps its prims.
        .def_prop_ro(
            "scene",
            [](Demo& demo) -> nb::object {
                if (app::Viewer* viewer = demo.GetViewer()) {
                    nb::object owner = nb::find(viewer);
                    return nb::cast(
                        &viewer->GetScene(),
                        nb::rv_policy::reference_internal,
                        owner.is_valid() ? owner : nb::find(&demo));
                }
                return nb::cast(demo.OwnScene());
            },
            nb::for_getter(
                nb::sig("def scene(self) -> ernie._ernie.Scene")),
            "The scene to build into: the viewer's, or the demo's "
            "own without one.")
        // Shared, not tied to the demo: images and batch cameras keep
        // the renderer alive, and the demo keeps those, which would
        // make a cycle through C++ that Python cannot collect.
        .def_prop_ro("renderer", &Demo::SharedRenderer,
                     "A Renderer of the demo's own, opened on first "
                     "use with the run's device, "
                     "backend and validation.")
        .def_prop_ro(
            "widgets",
            [](Demo& demo) {
                std::shared_ptr<app::WidgetPanel> panel =
                    demo.SharedWidgets();
                // Its callbacks count as the demo's (held.h).
                Adopt(&demo, panel.get(),
                      [weak = std::weak_ptr<app::WidgetPanel>(
                           panel)] { return !weak.expired(); });
                return panel;
            },
            "The demo's controls in the Demos panel's "
            "Options section: a WidgetPanel "
            "to add sliders, checkboxes and combos to, with "
            "or without a viewer.")
        .def("log_scalar", &Demo::LogScalar, nb::arg("name"),
             nb::arg("value"), nb::arg("clear") = false,
             nb::arg("smoothing") = 1,
             "Logs value to the viewer's scalar plot name "
             "(Viewer.log_scalar) when there is a "
             "viewer, and keeps it either way (logged_scalars).")
        .def(
            "log_array",
            [](Demo& demo, std::string_view name,
               const AnyArray& array, nb::handle stream) {
                demo.LogArray(name, TensorFromArray(array),
                              StreamOf(stream));
            },
            nb::arg("name"), nb::arg("array"),
            nb::arg("stream").none() = nb::none(),
            // Any NumPy or DLPack array, not only NumPy's.
            nb::sig("def log_array(self, name: str, array: object, "
                    "stream: object | None = None) -> None"),
            "Logs array to the viewer's heatmap name "
            "(Viewer.log_array) when there is a viewer, "
            "and keeps it either way (logged_array).")
        .def(
            "log_image",
            [](Demo& demo, std::string_view name,
               const AnyArray& image, nb::handle stream) {
                demo.LogImage(name, TensorFromArray(image),
                              StreamOf(stream));
            },
            nb::arg("name"), nb::arg("image"),
            nb::arg("stream").none() = nb::none(),
            nb::sig("def log_image(self, name: str, image: object, "
                    "stream: object | None = None) -> None"),
            "Logs image to the viewer's image panel name "
            "(Viewer.log_image) when there is a "
            "viewer, and keeps it either way (logged_image).")
        .def("logged_scalars", &Demo::LoggedScalars, nb::arg("name"),
             "Every value logged to scalar name, oldest first.")
        .def(
            "logged_array",
            [](const Demo& demo,
               std::string_view name) -> nb::object {
                const Tensor array = demo.LoggedArray(name);
                return array.Empty() ? nb::none()
                                     : ArrayFromTensor(array);
            },
            nb::arg("name"),
            nb::sig("def logged_array(self, name: str) -> "
                    "numpy.typing.NDArray | None"),
            "The last array logged as name, or None.")
        .def(
            "logged_image",
            [](const Demo& demo,
               std::string_view name) -> nb::object {
                const Tensor image = demo.LoggedImage(name);
                return image.Empty() ? nb::none()
                                     : ArrayFromTensor(image);
            },
            nb::arg("name"),
            nb::sig("def logged_image(self, name: str) -> "
                    "numpy.typing.NDArray | None"),
            "The last image logged as name, as logged, or None.")
        .def_prop_ro("options", &Demo::Options,
                     "The run's shared options.")
        .def_rw("frame", &Demo::frame, "Steps taken so far.")
        .def_rw("time", &Demo::time,
                "Demo seconds at the start of this step.")
        .def_rw("frame_dt", &Demo::frameDt,
                "Seconds this step advances: in a window the time "
                "since the last frame (at most "
                "max_frame_dt), else fixed_dt.");

    nb::class_<DemoInfo>(
        m, "DemoInfo",
        "A demo the runner can build, for the Demos panel.")
        .def(
            "__init__",
            [](DemoInfo* self, std::string category, std::string name,
               std::string description,
               std::function<std::shared_ptr<Demo>(
                   app::Viewer*, const DemoOptions&)>
                   create) {
                new (self) DemoInfo{
                    std::move(category), std::move(name),
                    std::move(description), std::move(create)};
            },
            nb::arg("category"), nb::arg("name"),
            nb::arg("description"), nb::arg("create"),
            "create(viewer, options) builds the demo; viewer is None "
            "without one.")
        .def_ro("category", &DemoInfo::category)
        .def_ro("name", &DemoInfo::name)
        .def_ro("description", &DemoInfo::description)
        .def_prop_ro("label", &DemoInfo::Label,
                     "\"category: name\", as the panel lists it.")
        .def_prop_ro("id", &DemoInfo::Id,
                     "\"category/name\", as `ernie --demo` and "
                     "`--list` name it.");

    nb::class_<DemoRunner>(
        m, "DemoRunner",
        "Runs demos as Newton's examples run: in a window, headless "
        "or without a viewer, writing "
        "--output, running --test checks and reporting --benchmark; "
        "in a window a Demos panel "
        "switches between the catalogue's demos.")
        .def(nb::init<DemoOptions, const std::string&>(),
             nb::arg("options"), nb::arg("title") = "Ernie demos",
             "Opens the viewer the options ask for (none for "
             "--viewer null or --benchmark); demos "
             "built from now on register in a scope of their own.")
        .def_prop_ro("viewer", &DemoRunner::GetViewer,
                     nb::rv_policy::reference_internal,
                     "The viewer, or None without one.")
        .def_prop_ro("options", &DemoRunner::Options)
        .def("set_catalog", &DemoRunner::SetCatalog, nb::arg("demos"),
             nb::arg("current") = 0,
             "The demos the Demos panel lists, and which of them "
             "runs first.")
        .def(
            "create",
            [](DemoRunner& runner, std::size_t index) {
                return DemoObject(runner.Create(index));
            },
            nb::arg("index"),
            "Builds the catalogue's demo index for this runner.")
        .def("request_switch", &DemoRunner::RequestSwitch,
             nb::arg("index"),
             "Switches to the catalogue's demo index before the next "
             "frame, as the Demos panel's "
             "menu does. With a viewer only.")
        .def("request_reset", &DemoRunner::RequestReset,
             "Builds the running demo afresh before the next frame, "
             "as the panel's Reset does.")
        .def(
            "run",
            [](DemoRunner& runner, std::shared_ptr<Demo> demo) {
                return DemoObject(runner.Run(std::move(demo)));
            },
            nb::arg("demo"),
            "Runs demo until its window closes or it has run its "
            "frames, then writes --output, "
            "runs test_final and reports the benchmark; returns the "
            "demo running at the end "
            "(the Demos panel may have switched), or None. Closes "
            "the viewer.");
}

}  // namespace ernie::python
