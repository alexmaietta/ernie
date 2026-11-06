// SPDX-License-Identifier: BSD-3-Clause
#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

#include "bindings.h"
#include "ernie/app/viewer/viewer.h"

namespace nb = nanobind;

namespace ernie::app {

void BindImGui(nb::module_& parent);

namespace {

class PythonDelegate : public RenderDelegate {
  public:
    NB_TRAMPOLINE(RenderDelegate);

    std::optional<RenderFrame> Render(const RenderRequest& request) override {
        NB_OVERRIDE_PURE_NAME("render", Render, request);
    }

    std::string_view Name() const override {
        nb::gil_scoped_acquire acquire;
        const nb::handle self = nb::find(this);
        m_name = self.is_valid() ? nb::cast<std::string>(self.attr("name")) : "Python";
        return m_name;
    }

  private:
    mutable std::string m_name;
};

template <class Result, class... Args> auto Locked(nb::object callable) {
    return [held = std::make_shared<nb::object>(std::move(callable))](Args... args) -> Result {
        nb::gil_scoped_acquire acquire;
        if constexpr (std::is_void_v<Result>) {
            (*held)(args...);
        } else {
            return nb::cast<Result>((*held)(args...));
        }
    };
}

template <class Event> std::function<bool(const Event&)> Handler(nb::object callable) {
    return [held = std::make_shared<nb::object>(std::move(callable))](const Event& event) {
        nb::gil_scoped_acquire acquire;
        const nb::object result = (*held)(event);
        return result.is_valid() && !result.is_none() && nb::cast<bool>(result);
    };
}

template <class... Args> std::function<void(Args...)> Drawn(nb::object callable) {
    return [held = std::make_shared<nb::object>(std::move(callable))](Args... args) {
        nb::gil_scoped_acquire acquire;
        try {
            (*held)(args...);
        } catch (nb::python_error& error) {
            error.discard_as_unraisable(held->ptr());
        }
    };
}

Placement MakePlacement(Anchor anchor, float x, float y, float width, float height) {
    return {anchor, x, y, width, height};
}

}

}

NB_MODULE(_viewer, module) {
    using namespace ernie;
    using namespace ernie::app;
    nb::module_::import_("ernie._ernie");
    BindImGui(module);

    nb::class_<ViewportRect>(module, "ViewportRect")
        .def_rw("x", &ViewportRect::x)
        .def_rw("y", &ViewportRect::y)
        .def_rw("width", &ViewportRect::width)
        .def_rw("height", &ViewportRect::height);

    nb::enum_<Dock>(module, "Dock")
        .value("LEFT", Dock::Left)
        .value("RIGHT", Dock::Right)
        .value("RIGHT_BOTTOM", Dock::RightBottom)
        .value("BOTTOM", Dock::Bottom)
        .value("FLOATING", Dock::Floating);

    nb::enum_<LightingMode>(module, "LightingMode")
        .value("STAGE", LightingMode::Stage)
        .value("CAMERA", LightingMode::Camera)
        .value("DEFAULT", LightingMode::Default);

    nb::enum_<WorldPick>(module, "WorldPick")
        .value("ALL", WorldPick::All)
        .value("FIRST", WorldPick::First)
        .value("RANDOM", WorldPick::Random)
        .value("LISTED", WorldPick::Listed)
        .value("SELECTED", WorldPick::Selected);

    nb::enum_<WorldLayout>(module, "WorldLayout")
        .value("GRID", WorldLayout::Grid)
        .value("OVERLAP", WorldLayout::Overlap);

    nb::enum_<InputAction>(module, "InputAction")
        .value("PRESS", InputAction::Press)
        .value("RELEASE", InputAction::Release)
        .value("REPEAT", InputAction::Repeat)
        .value("MOVE", InputAction::Move)
        .value("SCROLL", InputAction::Scroll);

    nb::class_<Modifiers>(module, "Modifiers")
        .def_ro("ctrl", &Modifiers::ctrl)
        .def_ro("shift", &Modifiers::shift)
        .def_ro("alt", &Modifiers::alt)
        .def_ro("super", &Modifiers::super);

    nb::class_<KeyEvent>(module, "KeyEvent")
        .def_ro("key", &KeyEvent::key)
        .def_ro("action", &KeyEvent::action)
        .def_ro("modifiers", &KeyEvent::modifiers);

    nb::class_<MouseEvent>(module, "MouseEvent")
        .def_ro("action", &MouseEvent::action)
        .def_ro("button", &MouseEvent::button)
        .def_ro("x", &MouseEvent::x)
        .def_ro("y", &MouseEvent::y)
        .def_ro("scroll_x", &MouseEvent::scrollX)
        .def_ro("scroll_y", &MouseEvent::scrollY)
        .def_ro("in_viewport", &MouseEvent::inViewport)
        .def_ro("viewport", &MouseEvent::viewport)
        .def_ro("modifiers", &MouseEvent::modifiers);

    nb::enum_<OverlayGesture>(module, "OverlayGesture")
        .value("HOVER", OverlayGesture::Hover)
        .value("LEAVE", OverlayGesture::Leave)
        .value("PRESS", OverlayGesture::Press)
        .value("DRAG", OverlayGesture::Drag)
        .value("RELEASE", OverlayGesture::Release)
        .value("CLICK", OverlayGesture::Click);

    nb::class_<OverlayEvent>(module, "OverlayEvent")
        .def_ro("gesture", &OverlayEvent::gesture)
        .def_ro("viewport", &OverlayEvent::viewport)
        .def_ro("hit", &OverlayEvent::hit)
        .def_ro("ray", &OverlayEvent::ray)
        .def_ro("x", &OverlayEvent::x)
        .def_ro("y", &OverlayEvent::y)
        .def_ro("modifiers", &OverlayEvent::modifiers);

    nb::class_<Screen>(module, "Screen")
        .def_ro("name", &Screen::name)
        .def_ro("x", &Screen::x)
        .def_ro("y", &Screen::y)
        .def_ro("width", &Screen::width)
        .def_ro("height", &Screen::height)
        .def_ro("refresh_rate", &Screen::refreshRate)
        .def_ro("scale", &Screen::scale)
        .def_ro("primary", &Screen::primary);

    nb::enum_<HudItem>(module, "HudItem")
        .value("FPS", HudItem::Fps)
        .value("UI_FPS", HudItem::UiFps)
        .value("DEVICE_MEMORY", HudItem::DeviceMemory)
        .value("HOST_MEMORY", HudItem::HostMemory)
        .value("RESOLUTION", HudItem::Resolution)
        .value("CAMERA", HudItem::Camera)
        .value("RENDERER", HudItem::Renderer)
        .value("SELECTION", HudItem::Selection);

    nb::enum_<Anchor>(module, "Anchor")
        .value("TOP_LEFT", Anchor::TopLeft)
        .value("TOP", Anchor::Top)
        .value("TOP_RIGHT", Anchor::TopRight)
        .value("LEFT", Anchor::Left)
        .value("CENTER", Anchor::Center)
        .value("RIGHT", Anchor::Right)
        .value("BOTTOM_LEFT", Anchor::BottomLeft)
        .value("BOTTOM", Anchor::Bottom)
        .value("BOTTOM_RIGHT", Anchor::BottomRight);

    nb::class_<Placement>(module, "Placement")
        .def(nb::new_(&MakePlacement), nb::arg("anchor") = Anchor::BottomLeft, nb::arg("x") = 0.0f,
             nb::arg("y") = 0.0f, nb::arg("width") = 0.0f, nb::arg("height") = 0.0f)
        .def_rw("anchor", &Placement::anchor)
        .def_rw("x", &Placement::x)
        .def_rw("y", &Placement::y)
        .def_rw("width", &Placement::width)
        .def_rw("height", &Placement::height);

    nb::class_<TextStyle>(module, "TextStyle")
        .def(
            "__init__",
            [](TextStyle* style, const Placement& placement, std::vector<float> color, float size,
               bool shadow, bool background) {
                if (color.size() != 4) {
                    throw nb::value_error("A text color is four floats, red, green, blue, alpha");
                }
                new (style) TextStyle{
                    placement, {color[0], color[1], color[2], color[3]}, size, shadow, background};
            },
            nb::arg("placement") = Placement{}, nb::arg("color") = std::vector<float>{1, 1, 1, 1},
            nb::arg("size") = 0.0f, nb::arg("shadow") = true, nb::arg("background") = false)
        .def_rw("placement", &TextStyle::placement)
        .def_rw("size", &TextStyle::size)
        .def_rw("shadow", &TextStyle::shadow)
        .def_rw("background", &TextStyle::background);

    nb::class_<ColorbarDesc>(module, "ColorbarDesc")
        .def(
            "__init__",
            [](ColorbarDesc* desc, std::optional<Placement> placement, Colormap colormap,
               float minimum, float maximum, bool logarithmic, bool reversed, std::string label,
               std::string units, uint32_t ticks) {
                new (desc) ColorbarDesc{};
                if (placement) {
                    desc->placement = *placement;
                }
                desc->colormap = colormap;
                desc->minimum = minimum;
                desc->maximum = maximum;
                desc->logarithmic = logarithmic;
                desc->reversed = reversed;
                desc->label = std::move(label);
                desc->units = std::move(units);
                desc->ticks = ticks;
            },
            nb::arg("placement") = nb::none(), nb::arg("colormap") = Colormap::Grayscale,
            nb::arg("minimum") = 0.0f, nb::arg("maximum") = 1.0f, nb::arg("logarithmic") = false,
            nb::arg("reversed") = false, nb::arg("label") = "", nb::arg("units") = "",
            nb::arg("ticks") = 5)
        .def_rw("placement", &ColorbarDesc::placement)
        .def_rw("colormap", &ColorbarDesc::colormap)
        .def_rw("minimum", &ColorbarDesc::minimum)
        .def_rw("maximum", &ColorbarDesc::maximum)
        .def_rw("logarithmic", &ColorbarDesc::logarithmic)
        .def_rw("reversed", &ColorbarDesc::reversed)
        .def_rw("label", &ColorbarDesc::label)
        .def_rw("units", &ColorbarDesc::units)
        .def_rw("ticks", &ColorbarDesc::ticks);

    nb::class_<RenderRequest>(module, "RenderRequest")
        .def_prop_ro("product",
                     [](const RenderRequest& request) { return std::string(request.product); })
        .def_ro("width", &RenderRequest::width)
        .def_ro("height", &RenderRequest::height);

    nb::class_<RenderFrame>(module, "RenderFrame")
        .def(
            "__init__",
            [](RenderFrame* frame, device::Tensor image, device::Operation ready,
               std::optional<device::Tensor> overlay) {
                new (frame) RenderFrame{std::move(image), std::move(ready),
                                        overlay.value_or(device::Tensor())};
            },
            nb::arg("image"), nb::arg("ready") = device::Operation{},
            nb::arg("overlay") = nb::none())
        .def_rw("image", &RenderFrame::image)
        .def_rw("ready", &RenderFrame::ready)
        .def_rw("overlay", &RenderFrame::overlay);

    nb::class_<RenderDelegate, PythonDelegate>(module, "RenderDelegate")
        .def(nb::init<>())
        .def("render", &RenderDelegate::Render, nb::arg("request"))
        .def_prop_ro("name", [](const RenderDelegate&) { return std::string("Python"); });

    nb::class_<ViewportDesc>(module, "ViewportDesc")
        .def(
            "__init__",
            [](ViewportDesc* desc, std::string name, std::string product, std::string aov,
               std::shared_ptr<RenderDelegate> delegate, std::shared_ptr<Renderer> renderer,
               bool hud, std::optional<std::vector<HudItem>> hudItems, bool toolbar,
               double targetRenderFps, bool eco, uint32_t ecoFrames) {
                new (desc) ViewportDesc{};
                desc->name = std::move(name);
                desc->product = std::move(product);
                desc->aov = std::move(aov);
                desc->delegate = std::move(delegate);
                desc->renderer = std::move(renderer);
                desc->hud = hud;
                if (hudItems) {
                    desc->hudItems = std::move(*hudItems);
                }
                desc->toolbar = toolbar;
                desc->targetRenderFps = targetRenderFps;
                desc->eco = eco;
                desc->ecoFrames = ecoFrames;
            },
            nb::arg("name"), nb::kw_only(), nb::arg("product") = "", nb::arg("aov") = "LdrColor",
            nb::arg("delegate") = nb::none(), nb::arg("renderer") = nb::none(),
            nb::arg("hud") = true, nb::arg("hud_items") = nb::none(), nb::arg("toolbar") = true,
            nb::arg("target_render_fps") = 0.0, nb::arg("eco") = true, nb::arg("eco_frames") = 500)
        .def_ro("name", &ViewportDesc::name)
        .def_ro("product", &ViewportDesc::product)
        .def_ro("eco", &ViewportDesc::eco)
        .def_ro("eco_frames", &ViewportDesc::ecoFrames);

    nb::class_<Viewer>(module, "Viewer")
        .def(
            "__init__",
            [](Viewer* viewer, std::optional<Ref<Scene>> scene,
               std::optional<Ref<device::Device>> device, std::string product, std::string aov,
               std::string title, uint32_t width, uint32_t height, bool offscreen, bool ui,
               bool vsync, std::shared_ptr<RenderDelegate> delegate,
               std::shared_ptr<Renderer> renderer, bool hud,
               std::optional<std::vector<HudItem>> hudItems, bool toolbar, bool menuBar,
               std::optional<std::pair<int, int>> position, int screen, bool fullscreen,
               double targetRenderFps, bool eco, uint32_t ecoFrames, float fontSize,
               bool builtinPanels, std::optional<std::filesystem::path> layout) {
                ViewerDesc desc;
                desc.product = std::move(product);
                desc.aov = std::move(aov);
                desc.title = std::move(title);
                desc.width = width;
                desc.height = height;
                desc.offscreen = offscreen;
                desc.ui = ui;
                desc.vsync = vsync;
                desc.delegate = std::move(delegate);
                desc.renderer = std::move(renderer);
                desc.hud = hud;
                if (hudItems) {
                    desc.hudItems = std::move(*hudItems);
                }
                desc.toolbar = toolbar;
                desc.menuBar = menuBar;
                if (position) {
                    desc.position = WindowPosition{position->first, position->second};
                }
                desc.screen = screen;
                desc.fullscreen = fullscreen;
                desc.targetRenderFps = targetRenderFps;
                desc.eco = eco;
                desc.ecoFrames = ecoFrames;
                desc.fontSize = fontSize;
                desc.builtinPanels = builtinPanels;
                if (layout) {
                    desc.layout = std::move(*layout);
                }
                if (scene) {
                    new (viewer) Viewer(std::move(*scene), desc);
                } else if (device) {
                    new (viewer) Viewer(std::move(*device), desc);
                } else {
                    throw nb::value_error("A viewer needs a scene or a device");
                }
            },
            nb::arg("scene") = nb::none(), nb::kw_only(), nb::arg("device") = nb::none(),
            nb::arg("product") = "", nb::arg("aov") = "LdrColor", nb::arg("title") = "Ernie",
            nb::arg("width") = 1280, nb::arg("height") = 720, nb::arg("offscreen") = false,
            nb::arg("ui") = true, nb::arg("vsync") = false, nb::arg("delegate") = nb::none(),
            nb::arg("renderer") = nb::none(), nb::arg("hud") = true,
            nb::arg("hud_items") = nb::none(), nb::arg("toolbar") = true,
            nb::arg("menu_bar") = true, nb::arg("position") = nb::none(), nb::arg("screen") = -1,
            nb::arg("fullscreen") = false, nb::arg("target_render_fps") = 0.0,
            nb::arg("eco") = true, nb::arg("eco_frames") = 500, nb::arg("font_size") = 0.0f,
            nb::arg("builtin_panels") = true, nb::arg("layout") = nb::none())
        .def_prop_ro("screens", &Viewer::Screens)
        .def("renderer", &Viewer::GetRenderer, nb::arg("viewport") = "")
        .def(
            "set_world_display",
            [](Viewer& viewer, WorldPick pick, uint32_t count, uint32_t seed,
               std::vector<std::string> worlds, bool globalPrims, WorldLayout layout,
               std::optional<std::tuple<double, double, double>> spacing,
               std::string_view viewport) {
                WorldDisplay display{.pick = pick,
                                     .count = count,
                                     .seed = seed,
                                     .worlds = std::move(worlds),
                                     .global = globalPrims,
                                     .layout = layout};
                if (spacing) {
                    display.spacing = Double3{std::get<0>(*spacing), std::get<1>(*spacing),
                                              std::get<2>(*spacing)};
                }
                viewer.SetWorldDisplay(display, viewport);
            },
            nb::kw_only(), nb::arg("pick") = WorldPick::All, nb::arg("count") = 0,
            nb::arg("seed") = 0, nb::arg("worlds") = std::vector<std::string>(),
            nb::arg("global_prims") = true, nb::arg("layout") = WorldLayout::Grid,
            nb::arg("spacing") = nb::none(), nb::arg("viewport") = "")
        .def("set_lighting", &Viewer::SetLighting, nb::arg("lighting"), nb::arg("viewport") = "")
        .def("lighting", &Viewer::Lighting, nb::arg("viewport") = "")
        .def("save_layout", &Viewer::SaveLayout, nb::arg("path"))
        .def("load_layout", &Viewer::LoadLayout, nb::arg("path"))
        .def("set_aov", &Viewer::SetAov, nb::arg("aov"), nb::arg("viewport") = "")
        .def("aov", &Viewer::Aov, nb::arg("viewport") = "")
        .def(
            "set_aov_colors",
            [](Viewer& viewer, Colormap colormap, std::optional<float> minimum,
               std::optional<float> maximum, bool logarithmic, bool reversed, bool centered,
               uint32_t channel, std::string_view viewport) {
                viewer.SetAovColors(
                    {colormap, minimum, maximum, logarithmic, reversed, centered, channel},
                    viewport);
            },
            nb::kw_only(), nb::arg("colormap") = Colormap::Grayscale,
            nb::arg("minimum") = nb::none(), nb::arg("maximum") = nb::none(),
            nb::arg("logarithmic") = false, nb::arg("reversed") = false,
            nb::arg("centered") = false, nb::arg("channel") = 0, nb::arg("viewport") = "")
        .def("set_aov_exposure", &Viewer::SetAovExposure, nb::arg("stops"),
             nb::arg("viewport") = "")
        .def("set_window_position", &Viewer::SetWindowPosition, nb::arg("x"), nb::arg("y"))
        .def("set_window_size", &Viewer::SetWindowSize, nb::arg("width"), nb::arg("height"))
        .def("set_output_screen", &Viewer::SetOutputScreen, nb::arg("screen"))
        .def("enable_fullscreen", &Viewer::EnableFullscreen, nb::arg("fullscreen") = true)
        .def_prop_ro("fullscreen", &Viewer::Fullscreen)
        .def("render_fps", &Viewer::RenderFps, nb::arg("viewport") = "")
        .def_prop_ro("ui_fps", &Viewer::UiFps)
        .def("set_target_render_fps", &Viewer::SetTargetRenderFps, nb::arg("fps"),
             nb::arg("viewport") = "")
        .def("enable_eco", &Viewer::EnableEco, nb::arg("eco") = true, nb::arg("viewport") = "")
        .def("eco", &Viewer::Eco, nb::arg("viewport") = "")
        .def("set_eco_frames", &Viewer::SetEcoFrames, nb::arg("frames"), nb::arg("viewport") = "")
        .def("eco_frames", &Viewer::EcoFrames, nb::arg("viewport") = "")
        .def("eco_paused", &Viewer::EcoPaused, nb::arg("viewport") = "")
        .def("add_colorbar", &Viewer::AddColorbar, nb::arg("desc") = ColorbarDesc{},
             nb::arg("viewport") = "")
        .def("set_colorbar", &Viewer::SetColorbar, nb::arg("id"), nb::arg("desc"))
        .def("show_gui", &Viewer::ShowGui, nb::arg("shown") = true)
        .def("toggle_gui", &Viewer::ToggleGui)
        .def_prop_ro("gui_shown", &Viewer::GuiShown)
        .def("show_base_gui", &Viewer::ShowBaseGui, nb::arg("shown") = true)
        .def("set_font_size", &Viewer::SetFontSize, nb::arg("pixels"))
        .def(
            "add_key_handler",
            [](Viewer& viewer, nb::object handler) {
                return viewer.AddKeyHandler(Handler<KeyEvent>(std::move(handler)));
            },
            nb::arg("handler"))
        .def(
            "add_mouse_handler",
            [](Viewer& viewer, nb::object handler) {
                return viewer.AddMouseHandler(Handler<MouseEvent>(std::move(handler)));
            },
            nb::arg("handler"))
        .def(
            "add_overlay_handler",
            [](Viewer& viewer, nb::object handler) {
                return viewer.AddOverlayHandler(Handler<OverlayEvent>(std::move(handler)));
            },
            nb::arg("handler"))
        .def(
            "world_to_screen",
            [](const Viewer& viewer, std::array<double, 3> point, std::string_view viewport) {
                const std::optional<Float2> at =
                    viewer.WorldToScreen({point[0], point[1], point[2]}, viewport);
                return at ? nb::object(nb::make_tuple(at->x, at->y)) : nb::none();
            },
            nb::arg("point"), nb::arg("viewport") = "")
        .def(
            "screen_ray",
            [](const Viewer& viewer, std::array<float, 2> at, std::string_view viewport) {
                return viewer.ScreenRay({at[0], at[1]}, viewport);
            },
            nb::arg("at"), nb::arg("viewport") = "")
        .def("remove_handler", &Viewer::RemoveHandler, nb::arg("id"))
        .def_prop_ro("open", &Viewer::Open)
        .def("frame", &Viewer::Frame, nb::call_guard<nb::gil_scoped_release>())
        .def("run", &Viewer::Run, nb::call_guard<nb::gil_scoped_release>())
        .def("show", &Viewer::Show, nb::arg("image"), nb::arg("ready") = device::Operation{},
             nb::arg("viewport") = "")
        .def("add_viewport", &Viewer::AddViewport, nb::arg("desc"))
        .def("remove_viewport", &Viewer::RemoveViewport, nb::arg("name"))
        .def_prop_ro("viewports", &Viewer::Viewports)
        .def("width", &Viewer::Width, nb::arg("viewport") = "")
        .def("height", &Viewer::Height, nb::arg("viewport") = "")
        .def(
            "shown",
            [](const Viewer& viewer, std::string_view viewport) -> std::optional<device::Tensor> {
                const device::Tensor* shown = viewer.Shown(viewport);
                return shown ? std::optional(*shown) : std::nullopt;
            },
            nb::arg("viewport") = "")
        .def_prop_rw(
            "selected", [](const Viewer& viewer) { return viewer.Selected(); },
            [](Viewer& viewer, std::vector<std::string> paths) { viewer.Select(std::move(paths)); })
        .def("read_pixels", &Viewer::ReadPixels, nb::call_guard<nb::gil_scoped_release>())
        .def(
            "record_edit",
            [](Viewer& viewer, std::string name, nb::object undo, nb::object redo) {
                viewer.RecordEdit(std::move(name), Locked<void>(std::move(undo)),
                                  Locked<void>(std::move(redo)));
            },
            nb::arg("name"), nb::arg("undo"), nb::arg("redo"))
        .def("undo", &Viewer::Undo)
        .def("redo", &Viewer::Redo)
        .def(
            "add_panel",
            [](Viewer& viewer, std::string title, nb::object draw, Dock dock, bool visible) {
                viewer.AddPanel(std::move(title), dock, Drawn<>(std::move(draw)), visible);
            },
            nb::arg("title"), nb::arg("draw"), nb::arg("dock") = Dock::Right,
            nb::arg("visible") = true)
        .def(
            "add_overlay",
            [](Viewer& viewer, nb::object draw, std::string_view viewport) {
                viewer.AddOverlay(Drawn<const ViewportRect&>(std::move(draw)), viewport);
            },
            nb::arg("draw"), nb::arg("viewport") = "")
        .def("show_panel", &Viewer::ShowPanel, nb::arg("title"), nb::arg("shown") = true)
        .def(
            "add_menu_item",
            [](Viewer& viewer, std::string menu, std::string label, nb::object action,
               std::string shortcut, nb::object checked, nb::object enabled) {
                MenuItemDesc item{std::move(menu), std::move(label),
                                  Locked<void>(std::move(action)), std::move(shortcut)};
                if (!checked.is_none()) {
                    item.checked = Locked<bool>(std::move(checked));
                }
                if (!enabled.is_none()) {
                    item.enabled = Locked<bool>(std::move(enabled));
                }
                return viewer.AddMenuItem(std::move(item));
            },
            nb::arg("menu"), nb::arg("label"), nb::arg("action"), nb::arg("shortcut") = "",
            nb::arg("checked") = nb::none(), nb::arg("enabled") = nb::none())
        .def("remove_menu_item", &Viewer::RemoveMenuItem, nb::arg("id"))
        .def("show_menu_bar", &Viewer::ShowMenuBar, nb::arg("shown") = true)
        .def("show_menu", &Viewer::ShowMenu, nb::arg("menu"), nb::arg("shown") = true)
        .def("show_menu_item", &Viewer::ShowMenuItem, nb::arg("menu"), nb::arg("label"),
             nb::arg("shown") = true)
        .def("show_hud", &Viewer::ShowHud, nb::arg("shown") = true, nb::arg("viewport") = "")
        .def("show_hud_item", &Viewer::ShowHudItem, nb::arg("item"), nb::arg("shown") = true,
             nb::arg("viewport") = "")
        .def(
            "add_hud_item",
            [](Viewer& viewer, std::string label, nb::object value, std::string_view viewport) {
                viewer.AddHudItem(std::move(label), Locked<std::string>(std::move(value)),
                                  viewport);
            },
            nb::arg("label"), nb::arg("value"), nb::arg("viewport") = "")
        .def("add_text", &Viewer::AddText, nb::arg("text"), nb::arg("style") = TextStyle{},
             nb::arg("viewport") = "")
        .def("set_text", &Viewer::SetText, nb::arg("id"), nb::arg("text"))
        .def("add_image",
             nb::overload_cast<const device::Tensor&, const Placement&, float, std::string_view>(
                 &Viewer::AddImage),
             nb::arg("image"), nb::arg("placement") = Placement{}, nb::arg("opacity") = 1.0f,
             nb::arg("viewport") = "")
        .def("add_image",
             nb::overload_cast<const std::filesystem::path&, const Placement&, float,
                               std::string_view>(&Viewer::AddImage),
             nb::arg("file"), nb::arg("placement") = Placement{}, nb::arg("opacity") = 1.0f,
             nb::arg("viewport") = "")
        .def("remove_overlay", &Viewer::RemoveOverlay, nb::arg("id"));
}
