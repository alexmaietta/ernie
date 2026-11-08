// SPDX-License-Identifier: BSD-3-Clause
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include "bindings.h"
#include "values.h"

namespace nb = nanobind;

namespace ernie::python {

namespace {

void RegisterOutputs(nb::module_& module) {
    nb::class_<IdLabel>(module, "IdLabel")
        .def_ro("path", &IdLabel::path)
        .def_ro("labels", &IdLabel::labels)
        .def("__eq__", [](const IdLabel& a,
                          const IdLabel& b) { return a.path == b.path && a.labels == b.labels; })
        .def("__repr__", [](const IdLabel& label) {
            return nb::str("IdLabel(path={!r}, labels={!r})")
                .format(label.path, nb::cast(label.labels));
        });

    nb::class_<RenderVarOutput>(module, "RenderVarOutput")
        .def_ro("render_var_path", &RenderVarOutput::renderVarPath)
        .def_ro("source_name", &RenderVarOutput::sourceName)
        .def_ro("tensor", &RenderVarOutput::tensor)
        .def_prop_ro("channels",
                     [](const RenderVarOutput& output) {
                         nb::dict channels;
                         for (const auto& [name, tensor] : output.channels) {
                             channels[nb::str(name.data(), name.size())] = nb::cast(tensor);
                         }
                         return channels;
                     })
        .def_prop_ro("id_map",
                     [](const RenderVarOutput& output) -> nb::object {
                         if (!output.idMap) {
                             return nb::none();
                         }
                         const bool single = output.sourceName == "SemanticSegmentation" ||
                                             output.sourceName == "SemanticIdMap" ||
                                             output.sourceName == "NonStableInstanceSegmentation";
                         nb::dict entries;
                         for (const auto& [id, label] : *output.idMap) {
                             if (single) {
                                 entries[nb::int_(id[0])] = label;
                             } else {
                                 entries[nb::make_tuple(id[0], id[1], id[2], id[3])] = label;
                             }
                         }
                         return entries;
                     })
        .def("numpy",
             [](const RenderVarOutput& output) { return nb::cast(output.tensor).attr("numpy")(); });

    nb::class_<FrameOutput>(module, "FrameOutput")
        .def_ro("start_time", &FrameOutput::startTime)
        .def_ro("end_time", &FrameOutput::endTime)
        .def_ro("render_vars", &FrameOutput::renderVars)
        .def_prop_ro("overlay",
                     [](const FrameOutput& frame) -> std::optional<device::Tensor> {
                         if (frame.overlay.Empty()) {
                             return std::nullopt;
                         }
                         return frame.overlay;
                     })
        .def("render_var", &FrameOutput::RenderVar, nb::arg("path"))
        .def("source", &FrameOutput::Source, nb::arg("source_name"));

    module.def(
        "decode_id_map",
        [](const device::Tensor& map) {
            nb::dict entries;
            for (const auto& [id, text] : DecodeIdMap(map)) {
                entries[nb::make_tuple(id[0], id[1], id[2], id[3])] = text;
            }
            return entries;
        },
        nb::arg("map"));

    nb::class_<ProductOutput>(module, "ProductOutput")
        .def_ro("name", &ProductOutput::name)
        .def_ro("frames", &ProductOutput::frames);

    nb::class_<RenderOutput>(module, "RenderOutput")
        .def_ro("simulation_start_time", &RenderOutput::simulationStartTime)
        .def_ro("simulation_end_time", &RenderOutput::simulationEndTime)
        .def_prop_ro("ready", &RenderOutput::Ready)
        .def_prop_ro(
            "done", [](const RenderOutput& output) { return output.Done(); },
            nb::keep_alive<0, 1>())
        .def(
            "wait",
            [](const RenderOutput& output, std::optional<uintptr_t> stream) {
                if (stream) {
                    output.Done().Wait(ToStream(stream));
                } else {
                    output.Wait();
                }
            },
            nb::arg("stream") = nb::none(), nb::call_guard<nb::gil_scoped_release>())
        .def("__getitem__", &RenderOutput::operator[])
        .def("__contains__", &RenderOutput::Contains)
        .def("__len__", &RenderOutput::Size)
        .def("products", [](const RenderOutput& output) {
            std::vector<std::string> names;
            for (const auto& [name, product] : output) {
                names.push_back(name);
            }
            return names;
        });
}

void RegisterRenderer(nb::module_& module) {
    nb::class_<Overlays>(module, "Overlays")
        .def(nb::new_([](Ref<device::Device> device) { return Overlays::Create(*device); }),
             nb::arg("device"))
        .def("set", &Overlays::Set, nb::arg("name"), nb::arg("overlay"),
             nb::call_guard<nb::gil_scoped_release>())
        .def("remove", &Overlays::Remove, nb::arg("name"), nb::call_guard<nb::gil_scoped_release>())
        .def("names", &Overlays::Names)
        .def(
            "pick",
            [](Overlays& overlays, std::string_view product, nb::handle pixel, float radius,
               uint32_t layer) {
                Float2 at;
                Assign(at, pixel);
                return overlays.Pick(product, at, radius, layer);
            },
            nb::arg("product"), nb::arg("pixel"), nb::kw_only(), nb::arg("radius") = 6.0f,
            nb::arg("layer") = 0);

    nb::class_<Renderer>(module, "Renderer")
        .def(
            "__init__",
            [](Renderer* renderer, Ref<Scene> scene, uint64_t maxPixelsPerLaunch,
               std::optional<Ref<Overlays>> overlays) {
                nb::gil_scoped_release release;
                new (renderer) Renderer(std::move(scene),
                                        {.maxPixelsPerLaunch = maxPixelsPerLaunch,
                                         .overlays = overlays ? *overlays : Ref<Overlays>()});
            },
            nb::arg("scene"), nb::kw_only(),
            nb::arg("max_pixels_per_launch") = RendererDesc{}.maxPixelsPerLaunch,
            nb::arg("overlays") = nb::none())
        .def(
            "step",
            [](Renderer& renderer, const std::vector<std::string>& products, double dt,
               std::optional<uintptr_t> stream) {
                return renderer.Step(products, dt, ToStream(stream));
            },
            nb::arg("products"), nb::arg("dt") = 0.0, nb::arg("stream") = nb::none(),
            nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("overlays", &Renderer::GetOverlays);
}

struct OverlaySources {
    std::optional<Ref<device::Device>> device;
    std::optional<uintptr_t> stream;
    std::shared_ptr<std::vector<std::shared_ptr<void>>> held =
        std::make_shared<std::vector<std::shared_ptr<void>>>();

    device::Tensor Take(nb::handle value, const char* label) {
        if (value.is_none()) {
            return {};
        }
        if (nb::isinstance<device::Tensor>(value)) {
            return nb::cast<device::Tensor>(value);
        }
        if (!device) {
            throw nb::value_error("Overlay arrays that aren't ernie tensors need a device to read "
                                  "them on; pass device=, such as the scene's");
        }
        const TakenTensor taken = TakeTensor(value, stream);
        held->push_back(Hold(taken.managed));
        return (*device)->Import(taken.managed->dl_tensor, taken.producer, label);
    }
};

Float3 Point(nb::handle value) {
    Float3 point;
    Assign(point, value);
    return point;
}

Quatd Orientation(nb::handle value) {
    Quatd orientation;
    if (!value.is_none()) {
        Assign(orientation, value);
    }
    return orientation;
}

OverlayStyle Style(std::optional<OverlayStyle> style) {
    return style.value_or(OverlayStyle{});
}

void RegisterOverlays(nb::module_& module) {
    nb::enum_<OverlayDepth>(module, "OverlayDepth")
        .value("TEST", OverlayDepth::Test)
        .value("XRAY", OverlayDepth::XRay)
        .value("ALWAYS", OverlayDepth::Always);

    nb::enum_<MarkerShape>(module, "MarkerShape")
        .value("CIRCLE", MarkerShape::Circle)
        .value("RING", MarkerShape::Ring)
        .value("SQUARE", MarkerShape::Square)
        .value("DIAMOND", MarkerShape::Diamond)
        .value("CROSS", MarkerShape::Cross)
        .value("PLUS", MarkerShape::Plus);

    nb::enum_<OverlayUnits>(module, "OverlayUnits")
        .value("PIXELS", OverlayUnits::Pixels)
        .value("METERS", OverlayUnits::Meters);

    nb::enum_<OverlayPrimShape>(module, "OverlayPrimShape")
        .value("FRAMES", OverlayPrimShape::Frames)
        .value("BOUNDS", OverlayPrimShape::Bounds)
        .value("FRUSTUMS", OverlayPrimShape::Frustums)
        .value("LIGHTS", OverlayPrimShape::Lights)
        .value("CAMERA_IMAGES", OverlayPrimShape::CameraImages);

    nb::class_<OverlayStyle>(module, "OverlayStyle")
        .def(
            "__init__",
            [](OverlayStyle* style, nb::handle color, float width, float size, MarkerShape marker,
               OverlayDepth depth, std::optional<uint32_t> world, std::optional<uint32_t> anchor,
               nb::handle background, nb::handle halo, nb::handle align, float dash, float gap,
               OverlayUnits units, bool screenSized, bool facing, bool pickable, uint32_t id,
               nb::handle screen) {
                new (style) OverlayStyle{};
                if (!screen.is_none()) {
                    Float2 at;
                    Assign(at, screen);
                    style->screen = at;
                }
                if (!color.is_none()) {
                    Assign(style->color, color);
                }
                if (!background.is_none()) {
                    Assign(style->background, background);
                }
                if (!halo.is_none()) {
                    Assign(style->halo, halo);
                }
                if (!align.is_none()) {
                    Assign(style->align, align);
                }
                style->dash = dash;
                style->gap = gap;
                style->units = units;
                style->screenSized = screenSized;
                style->facing = facing;
                style->pickable = pickable;
                style->id = id;
                style->width = width;
                style->size = size;
                style->marker = marker;
                style->depth = depth;
                style->world = world.value_or(kGlobalWorld);
                style->anchor = anchor.value_or(kNoPrim);
            },
            nb::kw_only(), nb::arg("color") = nb::none(), nb::arg("width") = OverlayStyle{}.width,
            nb::arg("size") = OverlayStyle{}.size, nb::arg("marker") = MarkerShape::Circle,
            nb::arg("depth") = OverlayDepth::Test, nb::arg("world") = nb::none(),
            nb::arg("anchor") = nb::none(), nb::arg("background") = nb::none(),
            nb::arg("halo") = nb::none(), nb::arg("align") = nb::none(), nb::arg("dash") = 0.0f,
            nb::arg("gap") = 0.0f, nb::arg("units") = OverlayUnits::Pixels,
            nb::arg("screen_sized") = false, nb::arg("facing") = false, nb::arg("pickable") = true,
            nb::arg("id") = 0, nb::arg("screen") = nb::none())
        .def_prop_ro("color",
                     [](const OverlayStyle& style) {
                         return nb::make_tuple(style.color.x, style.color.y, style.color.z,
                                               style.color.w);
                     })
        .def_rw("width", &OverlayStyle::width)
        .def_rw("size", &OverlayStyle::size)
        .def_rw("marker", &OverlayStyle::marker)
        .def_rw("depth", &OverlayStyle::depth)
        .def_rw("world", &OverlayStyle::world)
        .def_rw("anchor", &OverlayStyle::anchor)
        .def_rw("dash", &OverlayStyle::dash)
        .def_rw("gap", &OverlayStyle::gap)
        .def_rw("units", &OverlayStyle::units)
        .def_rw("screen_sized", &OverlayStyle::screenSized)
        .def_rw("facing", &OverlayStyle::facing)
        .def_rw("pickable", &OverlayStyle::pickable)
        .def_rw("id", &OverlayStyle::id)
        .def_prop_ro("background",
                     [](const OverlayStyle& style) {
                         return nb::make_tuple(style.background.x, style.background.y,
                                               style.background.z, style.background.w);
                     })
        .def_prop_ro("halo",
                     [](const OverlayStyle& style) {
                         return nb::make_tuple(style.halo.x, style.halo.y, style.halo.z,
                                               style.halo.w);
                     })
        .def_prop_ro(
            "align",
            [](const OverlayStyle& style) { return nb::make_tuple(style.align.x, style.align.y); })
        .def_prop_ro("screen", [](const OverlayStyle& style) -> nb::object {
            if (!style.screen) {
                return nb::none();
            }
            return nb::make_tuple(style.screen->x, style.screen->y);
        });

    nb::class_<OverlayHit>(module, "OverlayHit")
        .def_ro("overlay", &OverlayHit::overlay)
        .def_ro("id", &OverlayHit::id)
        .def_ro("batch", &OverlayHit::batch)
        .def_ro("item", &OverlayHit::item)
        .def_ro("piece", &OverlayHit::piece)
        .def_ro("world", &OverlayHit::world)
        .def_ro("prim", &OverlayHit::prim)
        .def_ro("distance", &OverlayHit::distance)
        .def_ro("depth", &OverlayHit::depth)
        .def_prop_ro("position", [](const OverlayHit& hit) {
            return nb::make_tuple(hit.position.x, hit.position.y, hit.position.z);
        });

    nb::class_<OverlayRay>(module, "OverlayRay")
        .def_prop_ro("origin",
                     [](const OverlayRay& ray) {
                         return nb::make_tuple(ray.origin.x, ray.origin.y, ray.origin.z);
                     })
        .def_prop_ro("direction", [](const OverlayRay& ray) {
            return nb::make_tuple(ray.direction.x, ray.direction.y, ray.direction.z);
        });

    nb::class_<OverlayPick>(module, "OverlayPick")
        .def_prop_ro("ready", &OverlayPick::Ready)
        .def("wait", &OverlayPick::Wait, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("hit", &OverlayPick::Hit, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("ray", &OverlayPick::Ray, nb::call_guard<nb::gil_scoped_release>());

    nb::class_<ScreenPlacement>(module, "ScreenPlacement")
        .def(
            "__init__",
            [](ScreenPlacement* placement, nb::handle at, nb::handle offset, nb::handle align) {
                new (placement) ScreenPlacement{};
                if (!at.is_none()) {
                    Assign(placement->at, at);
                }
                if (!offset.is_none()) {
                    Assign(placement->offset, offset);
                }
                if (!align.is_none()) {
                    Float2 value;
                    Assign(value, align);
                    placement->align = value;
                }
            },
            nb::arg("at") = nb::none(), nb::kw_only(), nb::arg("offset") = nb::none(),
            nb::arg("align") = nb::none())
        .def_prop_ro("alignment", [](const ScreenPlacement& placement) {
            const Float2 align = placement.Alignment();
            return nb::make_tuple(align.x, align.y);
        });

    nb::class_<OverlayNumberFormat>(module, "OverlayNumberFormat")
        .def(
            "__init__",
            [](OverlayNumberFormat* format, std::string prefix, std::string suffix,
               uint32_t decimals) {
                new (format) OverlayNumberFormat{std::move(prefix), std::move(suffix), decimals};
            },
            nb::kw_only(), nb::arg("prefix") = "", nb::arg("suffix") = "",
            nb::arg("decimals") = OverlayNumberFormat{}.decimals)
        .def_rw("prefix", &OverlayNumberFormat::prefix)
        .def_rw("suffix", &OverlayNumberFormat::suffix)
        .def_rw("decimals", &OverlayNumberFormat::decimals);

    nb::
        class_<Overlay>(module, "Overlay")
            .def(nb::init<>())
            .def(
                "line",
                [](Overlay& overlay, nb::handle a, nb::handle b,
                   std::optional<OverlayStyle> style) {
                    Float3 from;
                    Float3 to;
                    Assign(from, a);
                    Assign(to, b);
                    overlay.Line(from, to, Style(style));
                },
                nb::arg("a"), nb::arg("b"), nb::arg("style") = nb::none())
            .def(
                "lines",
                [](Overlay& overlay, nb::handle endpoints, std::optional<OverlayStyle> style,
                   nb::handle colors) {
                    std::vector<Float3> points;
                    std::vector<Float4> tints;
                    Assign(points, endpoints);
                    if (!colors.is_none()) {
                        Assign(tints, colors);
                    }
                    overlay.Lines(points, Style(style), tints);
                },
                nb::arg("endpoints"), nb::arg("style") = nb::none(), nb::arg("colors") = nb::none())
            .def(
                "polyline",
                [](Overlay& overlay, nb::handle points, bool closed,
                   std::optional<OverlayStyle> style) {
                    std::vector<Float3> values;
                    Assign(values, points);
                    overlay.Polyline(values, closed, Style(style));
                },
                nb::arg("points"), nb::arg("closed") = false, nb::arg("style") = nb::none())
            .def(
                "marker",
                [](Overlay& overlay, nb::handle point, std::optional<OverlayStyle> style) {
                    Float3 value;
                    Assign(value, point);
                    overlay.Marker(value, Style(style));
                },
                nb::arg("point"), nb::arg("style") = nb::none())
            .def(
                "markers",
                [](Overlay& overlay, nb::handle points, std::optional<OverlayStyle> style,
                   nb::handle colors) {
                    std::vector<Float3> values;
                    std::vector<Float4> tints;
                    Assign(values, points);
                    if (!colors.is_none()) {
                        Assign(tints, colors);
                    }
                    overlay.Markers(values, Style(style), tints);
                },
                nb::arg("points"), nb::arg("style") = nb::none(), nb::arg("colors") = nb::none())
            .def(
                "add_lines",
                [](Overlay& overlay, nb::handle points, nb::handle ends, nb::handle colors,
                   nb::handle widths, nb::handle worlds, nb::handle itemCount,
                   std::optional<OverlayStyle> style, std::optional<Ref<device::Device>> device,
                   std::optional<uintptr_t> stream) {
                    OverlaySources sources{device, stream};
                    OverlayLines lines{.points = sources.Take(points, "overlay points"),
                                       .ends = sources.Take(ends, "overlay ends"),
                                       .colors = sources.Take(colors, "overlay colors"),
                                       .widths = sources.Take(widths, "overlay widths"),
                                       .worlds = sources.Take(worlds, "overlay worlds"),
                                       .itemCount = sources.Take(itemCount, "overlay item counts"),
                                       .style = Style(style),
                                       .owner = sources.held};
                    overlay.Add(std::move(lines));
                },
                nb::arg("points"), nb::arg("ends"), nb::kw_only(), nb::arg("colors") = nb::none(),
                nb::arg("widths") = nb::none(), nb::arg("worlds") = nb::none(),
                nb::arg("item_count") = nb::none(), nb::arg("style") = nb::none(),
                nb::arg("device") = nb::none(), nb::arg("stream") = nb::none())
            .def(
                "add_markers",
                [](Overlay& overlay, nb::handle points, nb::handle colors, nb::handle sizes,
                   nb::handle worlds, nb::handle itemCount, std::optional<OverlayStyle> style,
                   std::optional<Ref<device::Device>> device, std::optional<uintptr_t> stream) {
                    OverlaySources sources{device, stream};
                    OverlayMarkers markers{.points = sources.Take(points, "overlay points"),
                                           .colors = sources.Take(colors, "overlay colors"),
                                           .sizes = sources.Take(sizes, "overlay sizes"),
                                           .worlds = sources.Take(worlds, "overlay worlds"),
                                           .itemCount =
                                               sources.Take(itemCount, "overlay item counts"),
                                           .style = Style(style),
                                           .owner = sources.held};
                    overlay.Add(std::move(markers));
                },
                nb::arg("points"), nb::kw_only(), nb::arg("colors") = nb::none(),
                nb::arg("sizes") = nb::none(), nb::arg("worlds") = nb::none(),
                nb::arg("item_count") = nb::none(), nb::arg("style") = nb::none(),
                nb::arg("device") = nb::none(), nb::arg("stream") = nb::none())

            .def(
                "box",
                [](Overlay& overlay, nb::handle center, nb::handle halfExtents,
                   std::optional<OverlayStyle> style, nb::handle orientation) {
                    overlay.Box(Point(center), Point(halfExtents), Style(style),
                                Orientation(orientation));
                },
                nb::arg("center"), nb::arg("half_extents"), nb::arg("style") = nb::none(),
                nb::arg("orientation") = nb::none())
            .def(
                "sphere",
                [](Overlay& overlay, nb::handle center, float radius,
                   std::optional<OverlayStyle> style) {
                    overlay.Sphere(Point(center), radius, Style(style));
                },
                nb::arg("center"), nb::arg("radius"), nb::arg("style") = nb::none())
            .def(
                "circle",
                [](Overlay& overlay, nb::handle center, nb::handle normal, float radius,
                   std::optional<OverlayStyle> style) {
                    overlay.Circle(Point(center), Point(normal), radius, Style(style));
                },
                nb::arg("center"), nb::arg("normal"), nb::arg("radius"),
                nb::arg("style") = nb::none())
            .def(
                "frame",
                [](Overlay& overlay, nb::handle origin, nb::handle orientation, float length,
                   std::optional<OverlayStyle> style) {
                    overlay.Frame(Point(origin), Orientation(orientation), length, Style(style));
                },
                nb::arg("origin"), nb::arg("orientation") = nb::none(), nb::arg("length") = 1.0f,
                nb::arg("style") = nb::none())
            .def(
                "arrow",
                [](Overlay& overlay, nb::handle from, nb::handle to,
                   std::optional<OverlayStyle> style, float headLength) {
                    overlay.Arrow(Point(from), Point(to), Style(style), headLength);
                },
                nb::arg("start"), nb::arg("end"), nb::arg("style") = nb::none(),
                nb::arg("head_length") = 0.0f)
            .def(
                "capsule",
                [](Overlay& overlay, nb::handle center, float radius, float halfHeight,
                   nb::handle orientation, std::optional<OverlayStyle> style) {
                    overlay.Capsule(Point(center), Orientation(orientation), radius, halfHeight,
                                    Style(style));
                },
                nb::arg("center"), nb::arg("radius"), nb::arg("half_height"),
                nb::arg("orientation") = nb::none(), nb::arg("style") = nb::none())
            .def(
                "add_boxes",
                [](Overlay& overlay, nb::handle points, nb::handle halfExtents,
                   nb::handle orientations, nb::handle colors, nb::handle widths, nb::handle worlds,
                   nb::handle itemCount, std::optional<OverlayStyle> style,
                   std::optional<Ref<device::Device>> device, std::optional<uintptr_t> stream) {
                    OverlaySources sources{device, stream};
                    overlay.Add(OverlayBoxes{
                        .points = sources.Take(points, "overlay points"),
                        .halfExtents = sources.Take(halfExtents, "overlay half extents"),
                        .orientations = sources.Take(orientations, "overlay orientations"),
                        .colors = sources.Take(colors, "overlay colors"),
                        .widths = sources.Take(widths, "overlay widths"),
                        .worlds = sources.Take(worlds, "overlay worlds"),
                        .itemCount = sources.Take(itemCount, "overlay item counts"),
                        .style = Style(style),
                        .owner = sources.held});
                },
                nb::arg("points"), nb::arg("half_extents"), nb::kw_only(),
                nb::arg("orientations") = nb::none(), nb::arg("colors") = nb::none(),
                nb::arg("widths") = nb::none(), nb::arg("worlds") = nb::none(),
                nb::arg("item_count") = nb::none(), nb::arg("style") = nb::none(),
                nb::arg("device") = nb::none(), nb::arg("stream") = nb::none())
            .def(
                "add_spheres",
                [](Overlay& overlay, nb::handle points, nb::handle radii, nb::handle colors,
                   nb::handle widths, nb::handle worlds, nb::handle itemCount,
                   std::optional<OverlayStyle> style, std::optional<Ref<device::Device>> device,
                   std::optional<uintptr_t> stream) {
                    OverlaySources sources{device, stream};
                    overlay.Add(
                        OverlaySpheres{.points = sources.Take(points, "overlay points"),
                                       .radii = sources.Take(radii, "overlay radii"),
                                       .colors = sources.Take(colors, "overlay colors"),
                                       .widths = sources.Take(widths, "overlay widths"),
                                       .worlds = sources.Take(worlds, "overlay worlds"),
                                       .itemCount = sources.Take(itemCount, "overlay item counts"),
                                       .style = Style(style),
                                       .owner = sources.held});
                },
                nb::arg("points"), nb::arg("radii"), nb::kw_only(), nb::arg("colors") = nb::none(),
                nb::arg("widths") = nb::none(), nb::arg("worlds") = nb::none(),
                nb::arg("item_count") = nb::none(), nb::arg("style") = nb::none(),
                nb::arg("device") = nb::none(), nb::arg("stream") = nb::none())
            .def(
                "add_circles",
                [](Overlay& overlay, nb::handle points, nb::handle normals, nb::handle radii,
                   nb::handle colors, nb::handle widths, nb::handle worlds, nb::handle itemCount,
                   std::optional<OverlayStyle> style, std::optional<Ref<device::Device>> device,
                   std::optional<uintptr_t> stream) {
                    OverlaySources sources{device, stream};
                    overlay.Add(
                        OverlayCircles{.points = sources.Take(points, "overlay points"),
                                       .normals = sources.Take(normals, "overlay normals"),
                                       .radii = sources.Take(radii, "overlay radii"),
                                       .colors = sources.Take(colors, "overlay colors"),
                                       .widths = sources.Take(widths, "overlay widths"),
                                       .worlds = sources.Take(worlds, "overlay worlds"),
                                       .itemCount = sources.Take(itemCount, "overlay item counts"),
                                       .style = Style(style),
                                       .owner = sources.held});
                },
                nb::arg("points"), nb::arg("normals"), nb::arg("radii"), nb::kw_only(),
                nb::arg("colors") = nb::none(), nb::arg("widths") = nb::none(),
                nb::arg("worlds") = nb::none(), nb::arg("item_count") = nb::none(),
                nb::arg("style") = nb::none(), nb::arg("device") = nb::none(),
                nb::arg("stream") = nb::none())
            .def(
                "add_frames",
                [](Overlay& overlay, nb::handle points, nb::handle orientations, nb::handle lengths,
                   float length, nb::handle widths, nb::handle worlds, nb::handle itemCount,
                   std::optional<OverlayStyle> style, std::optional<Ref<device::Device>> device,
                   std::optional<uintptr_t> stream) {
                    OverlaySources sources{device, stream};
                    overlay.Add(OverlayFrames{
                        .points = sources.Take(points, "overlay points"),
                        .orientations = sources.Take(orientations, "overlay orientations"),
                        .lengths = sources.Take(lengths, "overlay lengths"),
                        .widths = sources.Take(widths, "overlay widths"),
                        .worlds = sources.Take(worlds, "overlay worlds"),
                        .itemCount = sources.Take(itemCount, "overlay item counts"),
                        .length = length,
                        .style = Style(style),
                        .owner = sources.held});
                },
                nb::arg("points"), nb::kw_only(), nb::arg("orientations") = nb::none(),
                nb::arg("lengths") = nb::none(), nb::arg("length") = 1.0f,
                nb::arg("widths") = nb::none(), nb::arg("worlds") = nb::none(),
                nb::arg("item_count") = nb::none(), nb::arg("style") = nb::none(),
                nb::arg("device") = nb::none(), nb::arg("stream") = nb::none())
            .def(
                "add_arrows",
                [](Overlay& overlay, nb::handle points, nb::handle ends, nb::handle headLengths,
                   float headLength, nb::handle colors, nb::handle widths, nb::handle worlds,
                   nb::handle itemCount, std::optional<OverlayStyle> style,
                   std::optional<Ref<device::Device>> device, std::optional<uintptr_t> stream) {
                    OverlaySources sources{device, stream};
                    overlay.Add(OverlayArrows{
                        .points = sources.Take(points, "overlay points"),
                        .ends = sources.Take(ends, "overlay ends"),
                        .headLengths = sources.Take(headLengths, "overlay head lengths"),
                        .colors = sources.Take(colors, "overlay colors"),
                        .widths = sources.Take(widths, "overlay widths"),
                        .worlds = sources.Take(worlds, "overlay worlds"),
                        .itemCount = sources.Take(itemCount, "overlay item counts"),
                        .headLength = headLength,
                        .style = Style(style),
                        .owner = sources.held});
                },
                nb::arg("points"), nb::arg("ends"), nb::kw_only(),
                nb::arg("head_lengths") = nb::none(), nb::arg("head_length") = 0.0f,
                nb::arg("colors") = nb::none(), nb::arg("widths") = nb::none(),
                nb::arg("worlds") = nb::none(), nb::arg("item_count") = nb::none(),
                nb::arg("style") = nb::none(), nb::arg("device") = nb::none(),
                nb::arg("stream") = nb::none())
            .def(
                "add_capsules",
                [](Overlay& overlay, nb::handle points, nb::handle radii, nb::handle halfHeights,
                   nb::handle orientations, nb::handle colors, nb::handle widths, nb::handle worlds,
                   nb::handle itemCount, std::optional<OverlayStyle> style,
                   std::optional<Ref<device::Device>> device, std::optional<uintptr_t> stream) {
                    OverlaySources sources{device, stream};
                    overlay.Add(OverlayCapsules{
                        .points = sources.Take(points, "overlay points"),
                        .orientations = sources.Take(orientations, "overlay orientations"),
                        .radii = sources.Take(radii, "overlay radii"),
                        .halfHeights = sources.Take(halfHeights, "overlay half heights"),
                        .colors = sources.Take(colors, "overlay colors"),
                        .widths = sources.Take(widths, "overlay widths"),
                        .worlds = sources.Take(worlds, "overlay worlds"),
                        .itemCount = sources.Take(itemCount, "overlay item counts"),
                        .style = Style(style),
                        .owner = sources.held});
                },
                nb::arg("points"), nb::arg("radii"), nb::arg("half_heights"), nb::kw_only(),
                nb::arg("orientations") = nb::none(), nb::arg("colors") = nb::none(),
                nb::arg("widths") = nb::none(), nb::arg("worlds") = nb::none(),
                nb::arg("item_count") = nb::none(), nb::arg("style") = nb::none(),
                nb::arg("device") = nb::none(), nb::arg("stream") = nb::none())
            .def(
                "add_trails",
                [](Overlay& overlay, nb::handle points, nb::handle heads, float fade,
                   nb::handle colors, nb::handle widths, nb::handle worlds, nb::handle itemCount,
                   std::optional<OverlayStyle> style, std::optional<Ref<device::Device>> device,
                   std::optional<uintptr_t> stream) {
                    OverlaySources sources{device, stream};
                    overlay.Add(
                        OverlayTrails{.points = sources.Take(points, "overlay points"),
                                      .heads = sources.Take(heads, "overlay heads"),
                                      .colors = sources.Take(colors, "overlay colors"),
                                      .widths = sources.Take(widths, "overlay widths"),
                                      .worlds = sources.Take(worlds, "overlay worlds"),
                                      .itemCount = sources.Take(itemCount, "overlay item counts"),
                                      .fade = fade,
                                      .style = Style(style),
                                      .owner = sources.held});
                },
                nb::arg("points"), nb::kw_only(), nb::arg("heads") = nb::none(),
                nb::arg("fade") = 0.0f, nb::arg("colors") = nb::none(),
                nb::arg("widths") = nb::none(), nb::arg("worlds") = nb::none(),
                nb::arg("item_count") = nb::none(), nb::arg("style") = nb::none(),
                nb::arg("device") = nb::none(), nb::arg("stream") = nb::none())
            .def(
                "label",
                [](Overlay& overlay, nb::handle point, std::string_view text,
                   std::optional<OverlayStyle> style) {
                    overlay.Label(Point(point), text, Style(style));
                },
                nb::arg("point"), nb::arg("text"), nb::arg("style") = nb::none())
            .def(
                "text",
                [](Overlay& overlay, const ScreenPlacement& placement, std::string_view text,
                   std::optional<OverlayStyle> style) {
                    overlay.Text(placement, text, Style(style));
                },
                nb::arg("placement"), nb::arg("text"), nb::arg("style") = nb::none())
            .def(
                "add_texts",
                [](Overlay& overlay, nb::handle chars, nb::handle points, nb::handle colors,
                   nb::handle sizes, nb::handle worlds, nb::handle itemCount,
                   std::optional<ScreenPlacement> placement, std::optional<OverlayStyle> style,
                   std::optional<Ref<device::Device>> device, std::optional<uintptr_t> stream) {
                    OverlaySources sources{device, stream};
                    overlay.Add(
                        OverlayTexts{.points = sources.Take(points, "overlay points"),
                                     .chars = sources.Take(chars, "overlay chars"),
                                     .colors = sources.Take(colors, "overlay colors"),
                                     .sizes = sources.Take(sizes, "overlay sizes"),
                                     .worlds = sources.Take(worlds, "overlay worlds"),
                                     .itemCount = sources.Take(itemCount, "overlay item counts"),
                                     .placement = placement,
                                     .style = Style(style),
                                     .owner = sources.held});
                },
                nb::arg("chars"), nb::kw_only(), nb::arg("points") = nb::none(),
                nb::arg("colors") = nb::none(), nb::arg("sizes") = nb::none(),
                nb::arg("worlds") = nb::none(), nb::arg("item_count") = nb::none(),
                nb::arg("placement") = nb::none(), nb::arg("style") = nb::none(),
                nb::arg("device") = nb::none(), nb::arg("stream") = nb::none())
            .def(
                "add_numbers",
                [](Overlay& overlay, nb::handle values, nb::handle points,
                   std::optional<OverlayNumberFormat> format, nb::handle colors, nb::handle sizes,
                   nb::handle worlds, nb::handle itemCount,
                   std::optional<ScreenPlacement> placement, std::optional<OverlayStyle> style,
                   std::optional<Ref<device::Device>> device, std::optional<uintptr_t> stream) {
                    OverlaySources sources{device, stream};
                    overlay.Add(
                        OverlayNumbers{.points = sources.Take(points, "overlay points"),
                                       .values = sources.Take(values, "overlay values"),
                                       .format = format.value_or(OverlayNumberFormat{}),
                                       .colors = sources.Take(colors, "overlay colors"),
                                       .sizes = sources.Take(sizes, "overlay sizes"),
                                       .worlds = sources.Take(worlds, "overlay worlds"),
                                       .itemCount = sources.Take(itemCount, "overlay item counts"),
                                       .placement = placement,
                                       .style = Style(style),
                                       .owner = sources.held});
                },
                nb::arg("values"), nb::kw_only(), nb::arg("points") = nb::none(),
                nb::arg("format") = nb::none(), nb::arg("colors") = nb::none(),
                nb::arg("sizes") = nb::none(), nb::arg("worlds") = nb::none(),
                nb::arg("item_count") = nb::none(), nb::arg("placement") = nb::none(),
                nb::arg("style") = nb::none(), nb::arg("device") = nb::none(),
                nb::arg("stream") = nb::none())
            .def(
                "triangle",
                [](Overlay& overlay, nb::handle a, nb::handle b, nb::handle c,
                   std::optional<OverlayStyle> style) {
                    overlay.Triangle(Point(a), Point(b), Point(c), Style(style));
                },
                nb::arg("a"), nb::arg("b"), nb::arg("c"), nb::arg("style") = nb::none())
            .def(
                "polygon",
                [](Overlay& overlay, nb::handle points, std::optional<OverlayStyle> style) {
                    std::vector<Float3> values;
                    Assign(values, points);
                    overlay.Polygon(values, Style(style));
                },
                nb::arg("points"), nb::arg("style") = nb::none())
            .def(
                "curve",
                [](Overlay& overlay, nb::handle controls, std::optional<OverlayStyle> style,
                   uint32_t tessellation) {
                    std::vector<Float3> values;
                    Assign(values, controls);
                    overlay.Curve(values, Style(style), tessellation);
                },
                nb::arg("controls"), nb::arg("style") = nb::none(), nb::arg("tessellation") = 16)
            .def(
                "arc",
                [](Overlay& overlay, nb::handle center, nb::handle normal, float radius,
                   float begin, float end, std::optional<OverlayStyle> style, bool filled) {
                    overlay.Arc(Point(center), Point(normal), radius, begin, end, Style(style),
                                filled);
                },
                nb::arg("center"), nb::arg("normal"), nb::arg("radius"), nb::arg("begin"),
                nb::arg("end"), nb::arg("style") = nb::none(), nb::kw_only(),
                nb::arg("filled") = false)
            .def(
                "rectangle",
                [](Overlay& overlay, nb::handle center, nb::handle halfRight, nb::handle halfUp,
                   std::optional<OverlayStyle> style, bool filled) {
                    overlay.Rectangle(Point(center), Point(halfRight), Point(halfUp), Style(style),
                                      filled);
                },
                nb::arg("center"), nb::arg("half_right"), nb::arg("half_up"),
                nb::arg("style") = nb::none(), nb::kw_only(), nb::arg("filled") = false)
            .def(
                "mesh",
                [](Overlay& overlay, nb::handle points, std::vector<uint32_t> faceCounts,
                   std::vector<uint32_t> indices, std::optional<OverlayStyle> style,
                   bool wireframe) {
                    std::vector<Float3> values;
                    Assign(values, points);
                    overlay.Mesh(values, faceCounts, indices, Style(style), wireframe);
                },
                nb::arg("points"), nb::arg("face_counts"), nb::arg("indices"),
                nb::arg("style") = nb::none(), nb::kw_only(), nb::arg("wireframe") = false)
            .def(
                "image",
                [](Overlay& overlay, nb::handle topLeft, nb::handle right, nb::handle down,
                   nb::handle pixels, std::optional<OverlayStyle> style,
                   std::optional<Ref<device::Device>> device, std::optional<uintptr_t> stream) {
                    OverlaySources sources{device, stream};
                    overlay.Image(Point(topLeft), Point(right), Point(down),
                                  sources.Take(pixels, "overlay image"), Style(style));
                },
                nb::arg("top_left"), nb::arg("right"), nb::arg("down"), nb::arg("pixels"),
                nb::kw_only(), nb::arg("style") = nb::none(), nb::arg("device") = nb::none(),
                nb::arg("stream") = nb::none())
            .def(
                "prims",
                [](Overlay& overlay, std::vector<uint32_t> prims, OverlayPrimShape shape,
                   std::optional<OverlayStyle> style, float length, std::vector<uint32_t> groups,
                   nb::handle colors) {
                    std::vector<Float4> tints;
                    if (!colors.is_none()) {
                        Assign(tints, colors);
                    }
                    overlay.Prims(prims, shape, Style(style), length, groups, tints);
                },
                nb::arg("prims"), nb::arg("shape"), nb::kw_only(), nb::arg("style") = nb::none(),
                nb::arg("length") = 1.0f, nb::arg("groups") = std::vector<uint32_t>(),
                nb::arg("colors") = nb::none())
            .def(
                "outline",
                [](Overlay& overlay, std::vector<uint32_t> prims,
                   std::optional<OverlayStyle> style) { overlay.Outline(prims, Style(style)); },
                nb::arg("prims"), nb::arg("style") = nb::none())
            .def(
                "add_triangles",
                [](Overlay& overlay, nb::handle points, nb::handle colors, nb::handle worlds,
                   nb::handle itemCount, std::optional<OverlayStyle> style,
                   std::optional<Ref<device::Device>> device, std::optional<uintptr_t> stream) {
                    OverlaySources sources{device, stream};
                    overlay.Add(OverlayTriangles{.points = sources.Take(points, "overlay points"),
                                                 .colors = sources.Take(colors, "overlay colors"),
                                                 .worlds = sources.Take(worlds, "overlay worlds"),
                                                 .itemCount =
                                                     sources.Take(itemCount, "overlay item counts"),
                                                 .style = Style(style),
                                                 .owner = sources.held});
                },
                nb::arg("points"), nb::kw_only(), nb::arg("colors") = nb::none(),
                nb::arg("worlds") = nb::none(), nb::arg("item_count") = nb::none(),
                nb::arg("style") = nb::none(), nb::arg("device") = nb::none(),
                nb::arg("stream") = nb::none())
            .def(
                "add_images",
                [](Overlay& overlay, nb::handle points, nb::handle rights, nb::handle downs,
                   nb::handle pixels, nb::handle colors, nb::handle worlds, nb::handle itemCount,
                   std::optional<OverlayStyle> style, std::optional<Ref<device::Device>> device,
                   std::optional<uintptr_t> stream) {
                    OverlaySources sources{device, stream};
                    overlay.Add(
                        OverlayImages{.points = sources.Take(points, "overlay points"),
                                      .rights = sources.Take(rights, "overlay rights"),
                                      .downs = sources.Take(downs, "overlay downs"),
                                      .pixels = sources.Take(pixels, "overlay images"),
                                      .colors = sources.Take(colors, "overlay colors"),
                                      .worlds = sources.Take(worlds, "overlay worlds"),
                                      .itemCount = sources.Take(itemCount, "overlay item counts"),
                                      .style = Style(style),
                                      .owner = sources.held});
                },
                nb::arg("points"), nb::arg("rights"), nb::arg("downs"), nb::arg("pixels"),
                nb::kw_only(), nb::arg("colors") = nb::none(), nb::arg("worlds") = nb::none(),
                nb::arg("item_count") = nb::none(), nb::arg("style") = nb::none(),
                nb::arg("device") = nb::none(), nb::arg("stream") = nb::none())
            .def(
                "add_prims",
                [](Overlay& overlay, nb::handle prims, OverlayPrimShape shape, nb::handle groups,
                   uint32_t groupCount, float length, nb::handle pixels, nb::handle colors,
                   nb::handle widths, std::optional<OverlayStyle> style,
                   std::optional<Ref<device::Device>> device, std::optional<uintptr_t> stream) {
                    OverlaySources sources{device, stream};
                    overlay.Add(OverlayPrims{.prims = sources.Take(prims, "overlay prims"),
                                             .groups = sources.Take(groups, "overlay groups"),
                                             .groupCount = groupCount,
                                             .shape = shape,
                                             .length = length,
                                             .pixels = sources.Take(pixels, "overlay images"),
                                             .colors = sources.Take(colors, "overlay colors"),
                                             .widths = sources.Take(widths, "overlay widths"),
                                             .style = Style(style),
                                             .owner = sources.held});
                },
                nb::arg("prims"), nb::arg("shape"), nb::kw_only(), nb::arg("groups") = nb::none(),
                nb::arg("group_count") = 0, nb::arg("length") = 1.0f,
                nb::arg("pixels") = nb::none(), nb::arg("colors") = nb::none(),
                nb::arg("widths") = nb::none(), nb::arg("style") = nb::none(),
                nb::arg("device") = nb::none(), nb::arg("stream") = nb::none())
            .def_prop_ro("empty", &Overlay::Empty);
}

void RegisterRayCaster(nb::module_& module) {
    nb::enum_<RayCastMode>(module, "RayCastMode")
        .value("CLOSEST", RayCastMode::Closest)
        .value("ANY", RayCastMode::Any);

    nb::enum_<RayOutput>(module, "RayOutput", nb::is_flag())
        .value("DISTANCE", RayOutput::Distance)
        .value("PRIM", RayOutput::Prim)
        .value("NORMAL", RayOutput::Normal)
        .value("POSITION", RayOutput::Position);

    nb::class_<RayHits>(module, "RayHits")
        .def(nb::init<>())
        .def_ro("hit", &RayHits::hit)
        .def_ro("distance", &RayHits::distance)
        .def_ro("prim", &RayHits::prim)
        .def_ro("normal", &RayHits::normal)
        .def_ro("position", &RayHits::position)
        .def_ro("done", &RayHits::done);

    nb::class_<RayCaster>(module, "RayCaster")
        .def(
            "__init__",
            [](RayCaster* caster, Ref<Scene> scene) {
                nb::gil_scoped_release release;
                new (caster) RayCaster(std::move(scene));
            },
            nb::arg("scene"))
        .def(
            "cast",
            [](RayCaster& caster, nb::handle origins, nb::handle directions, nb::handle tMins,
               nb::handle tMaxes, nb::handle worlds, float minDistance, float maxDistance,
               std::optional<uint32_t> world, RayCastMode mode, RayOutput outputs,
               std::string label, std::optional<RayHits> kept, std::optional<uintptr_t> stream) {
                device::Device& device = caster.GetScene().GetDevice();
                HeldTensors held;
                const auto input = [&](nb::handle value, const char* what) {
                    if (value.is_none()) {
                        return device::Tensor();
                    }
                    if (nb::isinstance<device::Tensor>(value)) {
                        return nb::cast<device::Tensor>(value);
                    }
                    return held.Import(device, value, stream, what);
                };
                Rays rays;
                rays.origins = input(origins, "ray origins");
                rays.directions = input(directions, "ray directions");
                rays.tMins = input(tMins, "ray minimum distances");
                rays.tMaxes = input(tMaxes, "ray maximum distances");
                rays.worlds = input(worlds, "ray worlds");
                rays.minDistance = minDistance;
                rays.maxDistance = maxDistance;
                rays.world = world.value_or(kGlobalWorld);
                RayHits hits = kept ? std::move(*kept) : RayHits{};
                nb::gil_scoped_release release;
                held.Retire(caster.GetScene(),
                            caster.Cast(rays, hits,
                                        {.mode = mode, .outputs = outputs, .label = label},
                                        stream ? ToStream(stream) : Stream::Host()));
                return hits;
            },
            nb::arg("origins"), nb::arg("directions"), nb::kw_only(), nb::arg("t_min") = nb::none(),
            nb::arg("t_max") = nb::none(), nb::arg("worlds") = nb::none(),
            nb::arg("min_distance") = Rays{}.minDistance,
            nb::arg("max_distance") = Rays{}.maxDistance, nb::arg("world") = nb::none(),
            nb::arg("mode") = RayCastMode::Closest,
            nb::arg("outputs").sig("RayOutput.Distance | RayOutput.Prim") = RayCastDesc{}.outputs,
            nb::arg("label") = RayCastDesc{}.label, nb::arg("hits") = nb::none(),
            nb::arg("stream") = nb::none());
}

}

void BindRender(nb::module_& module) {
    RegisterOutputs(module);
    RegisterOverlays(module);
    RegisterRenderer(module);
    module.def(
        "composite_overlay",
        [](const device::Tensor& color, const device::Tensor& overlay) {
            return CompositeOverlay(color, overlay);
        },
        nb::arg("color"), nb::arg("overlay"), nb::call_guard<nb::gil_scoped_release>());
    RegisterRayCaster(module);
}

}
