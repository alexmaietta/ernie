// SPDX-License-Identifier: BSD-3-Clause
#include "bindings.h"

#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "ernie/app/create.h"
#include "ernie/app/viewer.h"
#include "ernie/device/vulkan_context.h"

#include <array>
#include <memory>
#include <span>

namespace ernie::python {

namespace {

using app::FrameInfo;
using app::Lighting;
using app::LogEntry;
using app::LogLevel;
using app::Theme;
using app::Viewer;
using app::ViewerOptions;
using app::WidgetPanel;

std::array<float, 4> ToArray(const glm::vec4& colour) {
    return {colour.r, colour.g, colour.b, colour.a};
}

glm::vec4 FromArray(const std::array<float, 4>& colour) {
    return {colour[0], colour[1], colour[2], colour[3]};
}

template <glm::vec4 Theme::* Field>
void ColourProperty(nb::class_<Theme>& theme, const char* name,
                    const char* doc) {
    theme.def_prop_rw(
        name, [](const Theme& t) { return ToArray(t.*Field); },
        [](Theme& t, const std::array<float, 4>& colour) {
            t.*Field = FromArray(colour);
        },
        doc);
}

/// A Python callable that C++ may copy and drop without the GIL;
/// calls take the GIL.
class Callable {
  public:
    explicit Callable(nb::callable callable)
        : held_(new nb::callable(std::move(callable)),
                [](nb::callable* held) {
                    nb::gil_scoped_acquire gil;
                    delete held;
                }) {}

    /// Calls the callable with the viewer itself (not a copy)
    /// followed by `args`.
    template <typename... Args>
    void operator()(Viewer& viewer, Args&&... args) const {
        nb::gil_scoped_acquire gil;
        (*held_)(nb::cast(&viewer, nb::rv_policy::reference),
                 std::forward<Args>(args)...);
    }

    /// Calls the callable with the viewer and converts what it
    /// returns to `Result`.
    template <typename Result>
    Result Call(const Viewer& viewer) const {
        nb::gil_scoped_acquire gil;
        return nb::cast<Result>(
            (*held_)(nb::cast(&viewer, nb::rv_policy::reference)));
    }

  private:
    std::shared_ptr<nb::callable> held_;
};

}  // namespace

void BindApp(nb::module_& parent) {
    nb::module_ m = parent.def_submodule(
        "app", "Ernie's interactive viewer: a window onto a Scene, "
               "driven by run() or step().");

    nb::enum_<app::Dock>(m, "Dock",
                         "Where a panel docks in the default layout.")
        .value("LEFT", app::Dock::Left)
        .value("RIGHT", app::Dock::Right)
        .value("BOTTOM", app::Dock::Bottom)
        .value("RIGHT_LOWER", app::Dock::RightLower,
               "The right column's lower part.")
        .value("FLOATING", app::Dock::Floating,
               "Not docked: a free window.");

    nb::enum_<Lighting>(m, "Lighting",
                        "How the viewport lights the scene.")
        .value("STAGE", Lighting::Stage,
               "The scene's own lights; black without any.")
        .value("CAMERA", Lighting::Camera,
               "A distant light along the view direction.")
        .value("DEFAULT", Lighting::Default,
               "A white sun from above with shadows.");

    nb::enum_<Viewer::Exposure>(m, "Exposure",
                                "How the viewport exposes its image.")
        .value("DEFAULT", Viewer::Exposure::Default,
               "Automatic through the viewer's camera, the camera's "
               "own through scene cameras.")
        .value("AUTO", Viewer::Exposure::Auto,
               "Keys the metered luminance to middle grey.")
        .value("CAMERA", Viewer::Exposure::Camera,
               "The camera's own exposure attributes.");

    nb::enum_<Viewer::Gizmo>(
        m, "Gizmo",
        "What the transform gizmo on a selected prim does.")
        .value("NONE", Viewer::Gizmo::None)
        .value("TRANSLATE", Viewer::Gizmo::Translate)
        .value("ROTATE", Viewer::Gizmo::Rotate)
        .value("SCALE", Viewer::Gizmo::Scale);

    nb::class_<Theme> themeClass(
        m, "Theme",
        "Colours (sRGB RGBA tuples) and metrics of the viewer's UI.");
    themeClass.def(nb::init<>());
    ColourProperty<&Theme::background>(
        themeClass, "background", "Behind the viewport and docks.");
    ColourProperty<&Theme::panel>(themeClass, "panel",
                                  "Panel bodies.");
    ColourProperty<&Theme::header>(
        themeClass, "header", "Collapsing headers and tab bars.");
    ColourProperty<&Theme::field>(
        themeClass, "field", "Input fields, combos and sliders.");
    ColourProperty<&Theme::fieldHovered>(themeClass, "field_hovered",
                                         "Hovered fields.");
    ColourProperty<&Theme::separator>(themeClass, "separator",
                                      "Lines between areas.");
    ColourProperty<&Theme::text>(themeClass, "text", "Text.");
    ColourProperty<&Theme::textMuted>(themeClass, "text_muted",
                                      "Labels and disabled text.");
    ColourProperty<&Theme::accent>(
        themeClass, "accent", "Selection, active tab, slider fill.");
    ColourProperty<&Theme::outline>(
        themeClass, "outline", "The viewport's selection outline.");
    ColourProperty<&Theme::warning>(themeClass, "warning",
                                    "Warnings.");
    ColourProperty<&Theme::error>(themeClass, "error", "Errors.");
    ColourProperty<&Theme::overlay>(themeClass, "overlay",
                                    "Viewport toolbar and HUD.");
    themeClass
        .def_rw("font_size", &Theme::fontSize,
                "Text size in pixels at a content scale of 1.")
        .def_rw("icon_size", &Theme::iconSize,
                "Icon size in pixels at a content scale of 1.")
        .def_rw("rounding", &Theme::rounding,
                "Corner radius of frames, tabs and popups.")
        .def_rw("property_label_width", &Theme::propertyLabelWidth,
                "Width of the label column in the Property panel.");
    m.def("dark_theme", &app::DarkTheme, "The default dark theme.");
    m.def("light_theme", &app::LightTheme,
          "A light theme with the same metrics and accent.");

    nb::class_<FrameInfo>(m, "FrameInfo",
                          "Passed to on_frame callbacks.")
        .def_ro("frame", &FrameInfo::frame,
                "Frames drawn before this one.")
        .def_ro("time", &FrameInfo::time,
                "Seconds since the viewer opened.")
        .def_ro("delta", &FrameInfo::delta,
                "Seconds since the previous frame.");

    using app::ViewportDisplay;
    nb::class_<ViewportDisplay>(
        m, "ViewportDisplay",
        "What the viewport draws over the scene (the eye menu) and "
        "its "
        "heads-up display lists. Assign a changed copy to "
        "Viewer.viewport_display.")
        .def(nb::init<>())
        .def_rw("lights", &ViewportDisplay::lights,
                "Every light's gizmo.")
        .def_rw("cameras", &ViewportDisplay::cameras,
                "Every scene camera's frustum.")
        .def_rw("grid", &ViewportDisplay::grid,
                "A grid on the ground (z = 0).")
        .def_rw("selection", &ViewportDisplay::selection,
                "The selection's outline.")
        .def_rw("overlays", &ViewportDisplay::overlays,
                "The scene's overlays.")
        .def_rw("hud", &ViewportDisplay::hud, "The heads-up display.")
        .def_rw("hud_fps", &ViewportDisplay::hudFps,
                "Frames per second and frame time.")
        .def_rw("hud_device_memory",
                &ViewportDisplay::hudDeviceMemory, "GPU memory.")
        .def_rw("hud_host_memory", &ViewportDisplay::hudHostMemory,
                "Resident memory.")
        .def_rw("hud_resolution", &ViewportDisplay::hudResolution,
                "The image's size.")
        .def_rw("hud_exposure", &ViewportDisplay::hudExposure,
                "The metered exposure value.")
        .def_rw("hud_triangles", &ViewportDisplay::hudTriangles,
                "Triangles drawn.")
        .def_rw("toolbar_renderer", &ViewportDisplay::toolbarRenderer,
                "The toolbar's renderer menu (renderer, rendering "
                "mode, render settings).")
        .def_rw("toolbar_var", &ViewportDisplay::toolbarVar,
                "The toolbar's menu of the render var the viewport "
                "shows.")
        .def_rw("toolbar_camera", &ViewportDisplay::toolbarCamera,
                "The toolbar's menu of the camera the viewport looks "
                "through.")
        .def_rw("toolbar_lighting", &ViewportDisplay::toolbarLighting,
                "The toolbar's menu of how the viewport lights the "
                "scene.")
        .def(nb::self == nb::self);

    nb::class_<WidgetPanel>(m, "WidgetPanel",
                            "A dockable panel of simple widgets; "
                            "each calls its callback when the "
                            "user changes it.")
        .def_prop_ro("title", &WidgetPanel::Title)
        .def_prop_ro("dock", &WidgetPanel::GetDock)
        .def_prop_rw("visible", &WidgetPanel::IsVisible,
                     &WidgetPanel::SetVisible)
        .def("text", &WidgetPanel::Text, nb::arg("text"),
             "Adds a line of text; returns its id.")
        .def("separator", &WidgetPanel::Separator)
        .def("button", &WidgetPanel::Button, nb::arg("label"),
             nb::arg("on_click"))
        .def("checkbox", &WidgetPanel::Checkbox, nb::arg("label"),
             nb::arg("value"),
             nb::arg("on_change").none() = nb::none())
        .def("slider", &WidgetPanel::Slider, nb::arg("label"),
             nb::arg("min"), nb::arg("max"), nb::arg("value"),
             nb::arg("on_change").none() = nb::none())
        .def("combo", &WidgetPanel::Combo, nb::arg("label"),
             nb::arg("items"), nb::arg("selected") = 0,
             nb::arg("on_change").none() = nb::none())
        .def("input_text", &WidgetPanel::InputText, nb::arg("label"),
             nb::arg("value") = "",
             nb::arg("on_change").none() = nb::none(),
             "A field of editable text; on_change gets the text "
             "after each edit.")
        .def("color", &WidgetPanel::Color, nb::arg("label"),
             nb::arg("value"),
             nb::arg("on_change").none() = nb::none(),
             "An RGB colour swatch with a picker; on_change gets "
             "(r, g, b) in [0, 1].")
        .def("plot", &WidgetPanel::Plot, nb::arg("label"),
             nb::arg("source"),
             "A line plot of the floats `source()` returns, read "
             "each frame.")
        .def(
            "set",
            [](WidgetPanel& panel, WidgetPanel::Id widget,
               nb::handle value) {
                if (nb::isinstance<nb::bool_>(value)) {
                    panel.Set(widget, nb::cast<bool>(value));
                } else if (nb::isinstance<nb::int_>(value)) {
                    panel.Set(widget, nb::cast<std::size_t>(value));
                } else if (nb::isinstance<nb::float_>(value)) {
                    panel.Set(widget, nb::cast<double>(value));
                } else if (nb::isinstance<nb::str>(value)) {
                    panel.Set(widget, nb::cast<std::string>(value));
                } else {
                    panel.Set(widget,
                              nb::cast<std::array<double, 3>>(value));
                }
            },
            nb::arg("widget"), nb::arg("value"),
            nb::sig("def set(self, widget: int, value: bool | int | "
                    "float | str | tuple[float, float, float]) -> "
                    "None"),
            "Sets a checkbox (bool), combo (int index), slider "
            "(float), text or text field (str) or colour (r, g, b) "
            "without calling its callback.")
        .def("remove", &WidgetPanel::Remove, nb::arg("widget"))
        .def("clear", &WidgetPanel::Clear);

    nb::class_<Viewer>(m, "Viewer",
                       "A window onto a Scene: a docked viewport "
                       "plus panels. Edits from code and "
                       "from the GUI go through the same scene. "
                       "Drive it with run(), or call "
                       "step() from your own loop.")
        .def(
            "__init__",
            [](Viewer* self, Scene& scene, const std::string& title,
               std::uint32_t width, std::uint32_t height, bool hidden,
               bool validation, const std::string& device,
               const Theme* theme,
               const std::optional<std::filesystem::path>& layout,
               const Path& settings, const std::string& renderer,
               bool vsync) {
                ViewerOptions options;
                options.vsync = vsync;
                options.title = title;
                options.width = width;
                options.height = height;
                options.hidden = hidden;
                options.validation = validation;
                options.device = device;
                if (theme != nullptr) {
                    options.theme = *theme;
                }
                options.layout =
                    layout.value_or(std::filesystem::path());
                options.settings = settings;
                options.renderer = renderer;
                nb::gil_scoped_release release;
                new (self) Viewer(scene, options);
            },
            nb::arg("scene"), nb::arg("title") = "Ernie",
            nb::arg("width") = 1600, nb::arg("height") = 900,
            nb::arg("hidden") = false, nb::arg("validation") = false,
            nb::arg("device") = "",
            nb::arg("theme").none() = nb::none(),
            nb::arg("layout").none() = nb::none(),
            nb::arg("settings") = Path(),
            nb::arg("renderer") = "raster", nb::arg("vsync") = false,
            nb::keep_alive<1, 2>(),
            "Opens the window (none when `hidden`) on a Vulkan "
            "device. Raises RuntimeError "
            "without a window system or Vulkan device.")
        .def("run", &Viewer::Run,
             nb::call_guard<nb::gil_scoped_release>(),
             "Draws frames until the window closes or close() is "
             "called.")
        .def("step", &Viewer::Step,
             nb::call_guard<nb::gil_scoped_release>(),
             "Draws one frame; returns whether the window is still "
             "open.")
        .def_prop_ro("is_open", &Viewer::IsOpen)
        .def("close", &Viewer::Close)
        .def(
            "on_frame",
            [](Viewer& viewer, nb::callable callback) {
                const Callable call(std::move(callback));
                return viewer.OnFrame(
                    [call](Viewer& v, const FrameInfo& info) {
                        call(v, info);
                    });
            },
            nb::arg("callback"),
            "Calls callback(viewer, frame_info) at the start of "
            "every frame, before the UI; "
            "returns an id for remove_callback.")
        .def(
            "on_pre_render",
            [](Viewer& viewer, nb::callable callback) {
                const Callable call(std::move(callback));
                return viewer.OnPreRender(
                    [call](Viewer& v, const FrameInfo& info) {
                        call(v, info);
                    });
            },
            nb::arg("callback"),
            "Calls callback(viewer, frame_info) after the UI, just "
            "before the scene is synced and "
            "rendered, so its edits show in this frame; returns an "
            "id.")
        .def(
            "on_post_render",
            [](Viewer& viewer, nb::callable callback) {
                const Callable call(std::move(callback));
                return viewer.OnPostRender(
                    [call](Viewer& v, const FrameInfo& info) {
                        call(v, info);
                    });
            },
            nb::arg("callback"),
            "Calls callback(viewer, frame_info) after the frame is "
            "submitted and presented; its "
            "edits show in the next frame. Returns an id.")
        .def(
            "on_selection",
            [](Viewer& viewer, nb::callable callback) {
                const Callable call(std::move(callback));
                return viewer.OnSelection(
                    [call](Viewer& v, std::span<const Path> paths) {
                        call(v, std::vector<Path>(paths.begin(),
                                                  paths.end()));
                    });
            },
            nb::arg("callback"),
            "Calls callback(viewer, paths) when the selection "
            "changes; returns an id.")
        .def("remove_callback", &Viewer::RemoveCallback,
             nb::arg("id"))
        .def_prop_ro("scene", &Viewer::GetScene,
                     nb::rv_policy::reference)
        .def_prop_ro(
            "device_name",
            [](Viewer& v) { return v.Context().DeviceName(); })
        .def_prop_ro_static(
            "renderers",
            [](nb::handle) { return Viewer::Renderers(); },
            "The render backends the viewer can switch between.")
        .def_prop_rw("renderer", &Viewer::Renderer,
                     &Viewer::SetRenderer,
                     "The render backend the viewport draws with; "
                     "assign to switch.")
        .def("sync", &Viewer::Sync,
             "Syncs the scene into the renderer now; raises "
             "ValueError for data it cannot use.")
        .def(
            "add_light_pattern",
            [](Viewer& viewer, const std::string& id,
               const std::string& typeName, const std::string& slang,
               nb::object params, double extent, bool normalized,
               const std::string& name,
               const std::string& description) {
                viewer.Delegate().AddLightPattern(
                    LightPatternFromPython(id, typeName, slang,
                                           params, extent, normalized,
                                           name, description));
            },
            nb::arg("id"), nb::arg("type_name"), nb::arg("slang"),
            nb::arg("params").none() = nb::none(),
            nb::arg("extent") = 1.0, nb::arg("normalized") = false,
            nb::arg("name") = "", nb::arg("description") = "",
            "Registers a laser pattern with the viewport's renderer, "
            "as Renderer.add_light_pattern does.")
        .def(
            "has_light_pattern",
            [](Viewer& viewer, const std::string& id) {
                return viewer.Delegate().HasLightPattern(id);
            },
            nb::arg("id"),
            "Whether the viewport's renderer has light pattern `id`.")
        .def("reset_session", &Viewer::ResetSession,
             "Starts over on the scene's present content once it has "
             "been replaced, as "
             "open_scene and the demo runner do: clears the "
             "selection and undo history, and picks "
             "the RenderSettings, the camera (perspective_camera, "
             "framing the scene) and the "
             "lighting as on opening, except those set (settings, "
             "active_camera, lighting) since "
             "clear_scene or opening.")
        .def("clear_scene", &Viewer::ClearScene,
             "Empties the stage for new content: removes every prim "
             "but the viewer's own and "
             "every overlay but its own, clears the selection and "
             "undo history, and forgets the "
             "settings, camera and lighting set. Call reset_session "
             "once the new content is in.")
        .def("new_scope", &Viewer::NewScope,
             "A registration scope not used before (see scope), "
             "without making it current.")
        .def_prop_rw("scope", &Viewer::Scope, &Viewer::SetScope,
                     "The registration scope: callbacks, panels, "
                     "menu items and viewport items "
                     "added belong to it, and remove_scope removes "
                     "them all at once, as the demo "
                     "runner does switching demos. 0, the default, "
                     "is the application's own.")
        .def("remove_scope", &Viewer::RemoveScope, nb::arg("scope"),
             "Removes everything added in scope; built-in panels are "
             "only hidden.")
        .def("open_scene", &Viewer::OpenScene, nb::arg("path"),
             "Replaces the stage with the scene a JSON file "
             "describes (the format of "
             "ernie.load_scene and ernie.save_scene), as File > Open "
             "Scene does: removes every "
             "prim but the viewer's own and every overlay but its "
             "own, loads the file and starts "
             "the session over (clear_scene, then reset_session). A "
             "file that does not load "
             "raises ValueError and "
             "leaves the stage as it was.")
        .def("save_scene", &Viewer::SaveScene, nb::arg("path"),
             "Writes the stage, every prim but the viewer's own, to "
             "path as JSON "
             "(ernie.save_scene), as File > Save Scene As does.")
        .def_prop_ro_static(
            "session_root",
            [](nb::handle) { return Viewer::SessionRoot(); })
        .def_prop_ro_static(
            "perspective_camera",
            [](nb::handle) { return Viewer::PerspectiveCamera(); })
        .def_prop_rw("selection", &Viewer::Selection,
                     &Viewer::SetSelection)
        .def(
            "edit",
            [](Viewer& viewer, const Path& path,
               std::string_view attribute, nb::handle value) {
                viewer.Edit(path, attribute, ValueFromPython(value));
            },
            nb::arg("path"), nb::arg("attribute"), nb::arg("value"),
            "Sets an attribute as an undoable GUI edit.")
        .def("clear_attribute", &Viewer::ClearAttribute,
             nb::arg("path"), nb::arg("attribute"),
             "Clears an attribute back to its fallback, as an "
             "undoable GUI edit.")
        .def("create", &Viewer::Create, nb::arg("type"),
             "Creates a prim of `type` (any of `creatable_types`) "
             "with defaults that make it "
             "show, as one undo step; selects it and returns its "
             "path.")
        .def(
            "delete",
            [](Viewer& viewer, const std::vector<Path>& paths) {
                viewer.Delete(paths);
            },
            nb::arg("paths"),
            "Deletes prims and their descendants as one undo step.")
        .def(
            "duplicate",
            [](Viewer& viewer, const std::vector<Path>& paths) {
                return viewer.Duplicate(paths);
            },
            nb::arg("paths"),
            "Copies prims with their descendants beside themselves "
            "(<name>_01, ...) as one undo "
            "step; selects and returns the copies.")
        .def("rename", &Viewer::Rename, nb::arg("path"),
             nb::arg("name"),
             "Renames a prim, moving its descendants, as one undo "
             "step; returns its new path.")
        .def_static(
            "creatable_types",
            [] {
                std::vector<std::string> types;
                for (const app::CreatableType& creatable :
                     app::CreatableTypes()) {
                    types.emplace_back(creatable.type);
                }
                return types;
            },
            "The prim types `create` and the Create menu make.")
        .def(
            "log",
            [](Viewer& viewer, std::string text,
               std::string_view level) {
                const LogLevel parsed =
                    level == "error"     ? LogLevel::Error
                    : level == "warning" ? LogLevel::Warning
                    : level == "info"    ? LogLevel::Info
                                         : throw nb::value_error(
                                               "level must be \"info\", "
                                               "\"warning\" or "
                                               "\"error\"");
                viewer.Log(parsed, std::move(text));
            },
            nb::arg("text"), nb::arg("level") = "info",
            "Adds a message to the Log panel.")
        .def_prop_ro(
            "log_entries",
            [](const Viewer& viewer) {
                const char* names[] = {"info", "warning", "error"};
                std::vector<
                    std::tuple<double, std::string, std::string>>
                    entries;
                for (const LogEntry& entry : viewer.LogEntries()) {
                    entries.emplace_back(
                        entry.time,
                        names[static_cast<std::size_t>(entry.level)],
                        entry.text);
                }
                return entries;
            },
            "(seconds, level, text) of every message in the log.")
        .def("clear_log", &Viewer::ClearLog)
        .def("capture_window", &Viewer::CaptureWindow,
             nb::arg("path"),
             "Writes the whole window, UI and all, as PNG; hidden "
             "viewers only.")
        .def("undo", &Viewer::Undo,
             "Undoes the last GUI edit; returns whether there was "
             "one.")
        .def("redo", &Viewer::Redo,
             "Redoes the last undone GUI edit.")
        .def_prop_rw("gizmo", &Viewer::GetGizmo, &Viewer::SetGizmo)
        .def_prop_rw("gizmo_local", &Viewer::GizmoLocal,
                     &Viewer::SetGizmoLocal)
        .def("pick", &Viewer::Pick, nb::arg("x"), nb::arg("y"),
             "The prim the viewport shows at pixel (x, y) of its "
             "last frame; an empty Path for "
             "the background.")
        .def_prop_ro(
            "cameras", &Viewer::Cameras,
            "The viewer's cameras, then the scene's, by path.")
        .def_prop_rw("active_camera", &Viewer::ActiveCamera,
                     &Viewer::SetActiveCamera)
        .def(
            "frame",
            [](Viewer& viewer, const std::vector<Path>& paths) {
                viewer.Frame(paths);
            },
            nb::arg("paths") = std::vector<Path>{},
            "Frames `paths` (the whole scene when empty) with the "
            "active camera.")
        .def("camera_from_view", &Viewer::CameraFromView,
             nb::arg("path"),
             "Defines a Camera prim at `path` with the active "
             "camera's pose and lens.")
        .def_prop_rw(
            "view_var", &Viewer::ViewVar, &Viewer::SetViewVar,
            "The render var the viewport shows, one of `view_vars`.")
        .def_prop_rw("view_band", &Viewer::ViewBand,
                     &Viewer::SetViewBand,
                     "The spectral band shown of SpectralRadiance, "
                     "or -1 for their mean.")
        .def_prop_ro(
            "view_vars",
            [](Viewer& viewer) {
                return viewer.Delegate().ViewVars();
            },
            "The render vars the current renderer can show in the "
            "viewport, with the AOVs of "
            "the viewport's sensor filter chain as 'aov:<name>'.")
        .def_prop_rw("lighting", &Viewer::GetLighting,
                     &Viewer::SetLighting)
        .def_prop_rw("exposure", &Viewer::GetExposure,
                     &Viewer::SetExposure)
        .def_prop_rw("exposure_compensation",
                     &Viewer::ExposureCompensation,
                     &Viewer::SetExposureCompensation,
                     "Stops on top of either exposure mode.")
        .def_prop_ro("metered_ev", &Viewer::MeteredEv,
                     "EV100 of the scene's average luminance as last "
                     "metered, or None.")
        .def_prop_ro("auto_exposing", &Viewer::AutoExposing)
        .def_prop_rw("vsync", &Viewer::GetVsync, &Viewer::SetVsync,
                     "Whether frames wait for the display's refresh; "
                     "off draws as fast as it "
                     "renders. Always False for a hidden viewer.")
        .def_prop_rw("wireframe", &Viewer::GetWireframe,
                     &Viewer::SetWireframe,
                     nb::for_setter(nb::arg("mode").none()),
                     "The viewport's Wireframe mode, overriding the "
                     "render settings' own; None "
                     "follows them.")
        .def_prop_rw("settings", &Viewer::Settings,
                     &Viewer::SetSettings)
        .def(
            "map",
            [](nb::pointer_and_handle<Viewer> self, const Path& path,
               std::string_view attribute, nb::handle stream) {
                Viewer& viewer = *self.p;
                return ArrayFromTensor(
                    KeepingAlive(viewer.Delegate().MapArray(
                                     viewer.GetScene(), path,
                                     attribute, StreamOf(stream)),
                                 self.h),
                    true);
            },
            nb::arg("path"), nb::arg("attribute"),
            nb::arg("stream").none() = nb::none(),
            "As Renderer.map, for the viewer's scene and renderer: a "
            "CUDA array over the "
            "viewer's own memory for the attribute, to write in "
            "place and then "
            "`scene.unmap(path, attribute, stream)`; map again every "
            "frame.")
        .def("capture", &Viewer::Capture, nb::arg("path"),
             nb::arg("var") = "",
             "Writes the next viewport frame to `path` as PNG.")
        .def("add_panel", &Viewer::AddWidgetPanel, nb::arg("title"),
             nb::arg("dock") = app::Dock::Right,
             nb::rv_policy::reference_internal,
             "Adds a WidgetPanel titled `title`, docked at `dock` in "
             "the default layout.")
        .def("remove_panel", &Viewer::RemovePanel, nb::arg("title"))
        .def(
            "panel_visible",
            [](Viewer& viewer, std::string_view title) {
                const app::Panel* panel = viewer.FindPanel(title);
                if (panel == nullptr) {
                    throw nb::value_error(
                        ("No panel is titled " + std::string(title))
                            .c_str());
                }
                return panel->IsVisible();
            },
            nb::arg("title"),
            "Whether the panel titled `title` (such as \"Profiler\") "
            "shows.")
        .def(
            "set_panel_visible",
            [](Viewer& viewer, std::string_view title, bool visible) {
                app::Panel* panel = viewer.FindPanel(title);
                if (panel == nullptr) {
                    throw nb::value_error(
                        ("No panel is titled " + std::string(title))
                            .c_str());
                }
                panel->SetVisible(visible);
            },
            nb::arg("title"), nb::arg("visible"),
            "Shows or hides the panel titled `title`, as the Window "
            "menu does.")
        .def("focus_panel", &Viewer::FocusPanel, nb::arg("title"),
             "Shows the panel titled `title` and brings it to the "
             "front of its dock.")
        .def(
            "add_menu_item",
            [](Viewer& viewer, const std::string& menu,
               const std::string& label, nb::callable action) {
                const Callable call(std::move(action));
                viewer.AddMenuItem(menu, label,
                                   [call](Viewer& v) { call(v); });
            },
            nb::arg("menu"), nb::arg("label"), nb::arg("action"),
            "Adds `label` to the menu bar's `menu`; calls "
            "action(viewer).")
        .def_prop_rw("viewport_display", &Viewer::GetViewportDisplay,
                     &Viewer::SetViewportDisplay,
                     "What the viewport draws over the scene and its "
                     "heads-up display lists.")
        .def(
            "add_viewport_toggle",
            [](Viewer& viewer, std::string label, bool shown,
               std::string overlay,
               std::optional<nb::callable> changed) {
                std::function<void(Viewer&, bool)> callback;
                if (changed) {
                    const Callable call(std::move(*changed));
                    callback = [call](Viewer& v, bool on) {
                        call(v, on);
                    };
                }
                return viewer.AddViewportToggle(
                    std::move(label), shown, std::move(overlay),
                    std::move(callback));
            },
            nb::arg("label"), nb::arg("shown") = true,
            nb::arg("overlay") = "",
            nb::arg("changed").none() = nb::none(),
            "Adds a toggle to the viewport's eye menu, such as for "
            "physics contacts or labels. "
            "With `overlay`, it shows and hides that scene overlay "
            "in the viewport; "
            "changed(viewer, shown) is told when the user switches "
            "it. Returns an id.")
        .def("viewport_toggle_shown", &Viewer::ViewportToggleShown,
             nb::arg("id"))
        .def("set_viewport_toggle_shown",
             &Viewer::SetViewportToggleShown, nb::arg("id"),
             nb::arg("shown"))
        .def(
            "add_hud_item",
            [](Viewer& viewer, std::string label, nb::callable value,
               bool shown) {
                const Callable call(std::move(value));
                return viewer.AddHudItem(
                    std::move(label),
                    [call](Viewer& v) {
                        return call.Call<std::string>(v);
                    },
                    shown);
            },
            nb::arg("label"), nb::arg("value"),
            nb::arg("shown") = true,
            "Adds a line \"label: value(viewer)\" to the viewport's "
            "heads-up display, refreshed a "
            "few times a second; value returns a str. Returns an id.")
        .def("hud_item_shown", &Viewer::HudItemShown, nb::arg("id"))
        .def("set_hud_item_shown", &Viewer::SetHudItemShown,
             nb::arg("id"), nb::arg("shown"))
        .def(
            "add_viewport_menu_item",
            [](Viewer& viewer, std::string label, nb::callable action,
               std::optional<nb::callable> checked) {
                const Callable call(std::move(action));
                std::function<bool(const Viewer&)> isChecked;
                if (checked) {
                    const Callable check(std::move(*checked));
                    isChecked = [check](const Viewer& v) {
                        return check.Call<bool>(v);
                    };
                }
                return viewer.AddViewportMenuItem(
                    std::move(label), [call](Viewer& v) { call(v); },
                    std::move(isChecked));
            },
            nb::arg("label"), nb::arg("action"),
            nb::arg("checked").none() = nb::none(),
            "Adds `label` to the viewport's settings menu; calls "
            "action(viewer), and shows a "
            "check while checked(viewer) returns True. Returns an "
            "id.")
        .def("remove_viewport_item", &Viewer::RemoveViewportItem,
             nb::arg("id"),
             "Removes a viewport toggle, heads-up display item or "
             "viewport menu item.")
        .def("log_scalar", &Viewer::LogScalar, nb::arg("name"),
             nb::arg("value"), nb::arg("clear") = false,
             nb::arg("smoothing") = 1,
             "Logs value as the next sample of the scalar signal "
             "name, drawn as a line of its "
             "last thousand samples in the Plots panel (Newton's "
             "log_scalar). clear drops the "
             "samples before; smoothing averages that many values "
             "into each sample.")
        .def(
            "logged_scalars", &Viewer::LoggedScalars, nb::arg("name"),
            "The samples of scalar signal name so far, oldest first.")
        .def(
            "log_array",
            [](Viewer& viewer, std::string_view name,
               const AnyArray& array, nb::handle stream) {
                viewer.LogArray(name, TensorFromArray(array),
                                StreamOf(stream));
            },
            nb::arg("name"), nb::arg("array"),
            nb::arg("stream").none() = nb::none(),
            // Any NumPy or DLPack array, not only NumPy's.
            nb::sig("def log_array(self, name: str, array: object, "
                    "stream: object | None = None) -> None"),
            "Logs array, (n) or (rows, cols) numbers from any "
            "framework, on the CPU or the GPU, "
            "as a heatmap in the Plots panel, replacing the last "
            "(Newton's log_array).")
        .def(
            "log_image",
            [](Viewer& viewer, std::string_view name,
               const AnyArray& image, nb::handle stream) {
                viewer.LogImage(name, TensorFromArray(image),
                                StreamOf(stream));
            },
            nb::arg("name"), nb::arg("image"),
            nb::arg("stream").none() = nb::none(),
            nb::sig("def log_image(self, name: str, image: object, "
                    "stream: object | None = None) -> None"),
            "Logs image in a panel of its own titled name, replacing "
            "the last (Newton's "
            "log_image): (rows, cols), (rows, cols, channels) or a "
            "batch (views, rows, cols, "
            "channels), with 1, 3 or 4 channels, uint8 (0 to 255, "
            "shown as they are) or float32 "
            "(0 to 1), from any DLPack framework (NumPy, PyTorch, "
            "JAX, Warp, CuPy). A CUDA image "
            "is copied device to device into the panel's texture "
            "after the work queued on "
            "stream, without the host; map_image writes the texture "
            "itself, with no copy.")
        .def(
            "map_image",
            [](nb::pointer_and_handle<Viewer> self,
               std::string_view name, std::vector<std::int64_t> shape,
               nb::handle dtype, nb::handle stream) {
                const std::string type = nb::cast<std::string>(
                    nb::module_::import_("numpy")
                        .attr("dtype")(dtype)
                        .attr("name"));
                if (type != "uint8" && type != "float32") {
                    throw nb::value_error(
                        "map_image takes dtype uint8 or float32");
                }
                const DType parsed =
                    type == "uint8" ? DType::UInt8 : DType::Float32;
                return ArrayFromTensor(
                    KeepingAlive(
                        self.p->MapImage(name, std::move(shape),
                                         parsed, StreamOf(stream)),
                        self.h),
                    true);
            },
            nb::arg("name"), nb::arg("shape"),
            nb::arg("dtype") = "uint8",
            nb::arg("stream").none() = nb::none(),
            "The texture of image panel name itself, as a writable "
            "CUDA array of shape (as "
            "log_image takes, with 1 or 4 channels) and dtype: write "
            "it on stream (with Warp, "
            "PyTorch, CuPy...), then unmap_image(name, stream), and "
            "the panel shows it with no "
            "copy at all. Its rows are the texture's pitch apart "
            "(see its strides). Map again "
            "for each new image.")
        .def(
            "unmap_image",
            [](Viewer& viewer, std::string_view name,
               nb::handle stream) {
                viewer.UnmapImage(name, StreamOf(stream));
            },
            nb::arg("name"), nb::arg("stream").none() = nb::none(),
            "Ends writing image name (map_image): frames show it "
            "after the work queued on "
            "stream.")
        .def("remove_logged", &Viewer::RemoveLogged, nb::arg("name"),
             "Forgets the scalar signal, array or image name.")
        .def("save_layout", &Viewer::SaveLayout, nb::arg("path"))
        .def("load_layout", &Viewer::LoadLayout, nb::arg("path"))
        .def_prop_rw("theme", &Viewer::GetTheme, &Viewer::SetTheme);

    BindDemo(m);
}

}  // namespace ernie::python
