// SPDX-License-Identifier: BSD-3-Clause
#include "bindings.h"

#include "ernie/scene/cahvore.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/dem.h"
#include "ernie/scene/isd.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/util/astronomy.h"
#include "ernie/util/image.h"
#include "ernie/util/nanovdb.h"
#include "ernie/util/noise.h"
#include "ernie/util/reflectance_library.h"
#include "ernie/util/tensor.h"

#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace ernie::python {

using namespace nb::literals;

namespace {

/// An array mapped for writing in place by `Scene.map`, unmapped
/// once.
struct AttributeMapping {
    nb::object scene;
    Path path;
    std::string attribute;
    nb::object stream;
    nb::object array;
    bool unmapped = false;

    void Unmap(nb::handle override) {
        if (unmapped) {
            return;
        }
        unmapped = true;
        nb::cast<Scene&>(scene).Unmap(
            path, attribute,
            StreamOf(override.is_valid() && !override.is_none()
                         ? override
                         : nb::handle(stream)));
    }
};

/// A 4 x 4 NumPy array of `matrix` in math convention: column
/// vectors, translation in the last column.
nb::object MatrixToPython(const glm::dmat4& matrix) {
    auto* values = new std::array<double, 16>();
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            (*values)[static_cast<std::size_t>(row * 4 + col)] =
                matrix[col][row];
        }
    }
    nb::capsule owner(values, [](void* p) noexcept {
        delete static_cast<std::array<double, 16>*>(p);
    });
    const std::size_t shape[2] = {4, 4};
    return nb::cast(nb::ndarray<nb::numpy, double>(values->data(), 2,
                                                   shape, owner),
                    nb::rv_policy::reference);
}

glm::dvec3 Vec3FromPython(nb::handle object) {
    const Value value = ValueFromPython(object);
    if (const auto* d = value.Get<glm::dvec3>()) {
        return *d;
    }
    throw nb::type_error("expected a sequence of three numbers");
}

bool IsNumber(nb::handle item) {
    return nb::isinstance<nb::float_>(item) ||
           nb::isinstance<nb::int_>(item);
}

/// `object` as a C-contiguous NumPy array of `dtype` and rows of
/// `columns` (any leading shape).
nb::ndarray<nb::numpy, nb::c_contig>
RowsOf(nb::handle object, const char* dtype, std::size_t columns) {
    nb::module_ numpy = nb::module_::import_("numpy");
    nb::object array =
        numpy.attr("ascontiguousarray")(object, "dtype"_a = dtype);
    if (nb::cast<std::size_t>(array.attr("size")) % columns != 0) {
        throw nb::value_error(("expected rows of " +
                               std::to_string(columns) + " values")
                                  .c_str());
    }
    array = array.attr("reshape")(-1, columns);
    return nb::cast<nb::ndarray<nb::numpy, nb::c_contig>>(array);
}

/// Points as rows of three numbers: an (N, 3) array or a sequence of
/// 3-sequences.
std::vector<glm::dvec3> PointsFromPython(nb::handle object) {
    const auto rows = RowsOf(object, "float64", 3);
    const auto* values = static_cast<const double*>(rows.data());
    std::vector<glm::dvec3> points(rows.shape(0));
    for (std::size_t i = 0; i < points.size(); ++i) {
        points[i] = {values[3 * i], values[3 * i + 1],
                     values[3 * i + 2]};
    }
    return points;
}

/// Colours as rows of RGBA, or of RGB with opacity 1; none for None.
std::vector<glm::vec4> ColorsFromPython(nb::handle object) {
    std::vector<glm::vec4> colors;
    if (object.is_none()) {
        return colors;
    }
    nb::module_ numpy = nb::module_::import_("numpy");
    nb::object array =
        numpy.attr("asarray")(object, "dtype"_a = "float32");
    const std::size_t width =
        nb::cast<std::size_t>(array.attr("shape")[nb::int_(-1)]);
    if (width != 3 && width != 4) {
        throw nb::value_error(
            "expected colours as rows of RGB or RGBA");
    }
    const auto rows = RowsOf(array, "float32", width);
    const auto* values = static_cast<const float*>(rows.data());
    colors.resize(rows.shape(0));
    for (std::size_t i = 0; i < colors.size(); ++i) {
        const float* row = values + width * i;
        colors[i] = {row[0], row[1], row[2],
                     width == 4 ? row[3] : 1.0f};
    }
    return colors;
}

std::vector<std::uint32_t> IndicesFromPython(nb::handle object) {
    const auto rows = RowsOf(object, "uint32", 1);
    const auto* values =
        static_cast<const std::uint32_t*>(rows.data());
    return {values, values + rows.shape(0)};
}

glm::vec4 ColorFromPython(nb::handle object) {
    const std::vector<glm::vec4> colors = ColorsFromPython(object);
    if (colors.size() != 1) {
        throw nb::value_error("expected one RGB or RGBA colour");
    }
    return colors.front();
}

glm::dmat4 TransformFromPython(nb::handle object) {
    if (object.is_none()) {
        return glm::dmat4(1.0);
    }
    const Value value = ValueFromPython(object);
    if (const auto* matrix = value.Get<glm::dmat4>()) {
        return *matrix;
    }
    throw nb::type_error("expected a 4 x 4 transform");
}

OverlayDepth DepthFromPython(std::string_view name) {
    if (name == "test") {
        return OverlayDepth::Test;
    }
    if (name == "xray") {
        return OverlayDepth::XRay;
    }
    if (name == "always") {
        return OverlayDepth::Always;
    }
    throw nb::value_error(
        "depth must be \"test\", \"xray\" or \"always\"");
}

MarkerShape ShapeFromPython(std::string_view name) {
    constexpr std::pair<std::string_view, MarkerShape> kShapes[] = {
        {"circle", MarkerShape::Circle},
        {"ring", MarkerShape::Ring},
        {"square", MarkerShape::Square},
        {"diamond", MarkerShape::Diamond},
        {"cross", MarkerShape::Cross},
        {"plus", MarkerShape::Plus}};
    for (const auto& [token, shape] : kShapes) {
        if (token == name) {
            return shape;
        }
    }
    throw nb::value_error(
        "shape must be \"circle\", \"ring\", \"square\", "
        "\"diamond\", \"cross\" or \"plus\"");
}

/// An `OverlayStyle` from keyword arguments: color, width, size,
/// depth ("test", "xray" or "always"), world and background.
OverlayStyle StyleFromPython(const nb::kwargs& kwargs) {
    OverlayStyle style;
    for (const auto& [key, value] : kwargs) {
        const std::string name = nb::cast<std::string>(key);
        if (name == "color") {
            style.color = ColorFromPython(value);
        } else if (name == "width") {
            style.width = nb::cast<float>(value);
        } else if (name == "size") {
            style.size = nb::cast<float>(value);
        } else if (name == "depth") {
            style.depth =
                DepthFromPython(nb::cast<std::string>(value));
        } else if (name == "world") {
            style.world = nb::cast<std::int32_t>(value);
        } else if (name == "background") {
            style.background = ColorFromPython(value);
        } else if (name == "anchor") {
            style.anchor =
                value.is_none() ? Path() : PathFromPython(value);
        } else if (name == "dash") {
            style.dash = nb::cast<float>(value);
        } else if (name == "metric") {
            style.metric = nb::cast<bool>(value);
        } else if (name == "halo") {
            style.halo = nb::cast<float>(value);
        } else if (name == "id") {
            style.id = nb::cast<std::uint32_t>(value);
        } else if (name == "hover") {
            if (!value.is_none()) {
                style.hover = ColorFromPython(value);
            }
        } else {
            throw nb::type_error(("unknown overlay style '" + name +
                                  "'; expected color, width, dash, "
                                  "size, depth, world, background, "
                                  "halo, anchor, metric, id or hover")
                                     .c_str());
        }
    }
    return style;
}

constexpr const char* kStyleDoc =
    " Style keywords: color (RGB or RGBA display colour), width "
    "(line pixels), dash (dashes' and gaps' pixels, 0 solid), size "
    "(marker or "
    "text pixels, or metres with metric=True), depth (\"test\", "
    "\"xray\" or \"always\"), world "
    "(-1 for every view), "
    "background (labels' backing colour), halo (pixels of "
    "background's colour outlining text's characters in place of "
    "its box), anchor (a prim the "
    "coordinates are in the space of, followed as it moves, on the "
    "GPU too), id (a number hit tests report, such as a button's) "
    "and "
    "hover (the colour items with that id take while hovered). "
    "Returns the overlay, for chaining.";

glm::vec2 Vec2FromPython(nb::handle object) {
    const auto values = nb::cast<std::vector<float>>(object);
    if (values.size() != 2) {
        throw nb::value_error("expected two numbers");
    }
    return {values[0], values[1]};
}

/// A `ScreenPlacement` of `at`, `offset` and `align` (None for `at`).
ScreenPlacement PlacementFromPython(nb::handle at, nb::handle offset,
                                    nb::handle align) {
    ScreenPlacement placement;
    placement.at = Vec2FromPython(at);
    placement.offset = Vec2FromPython(offset);
    if (!align.is_none()) {
        placement.align = Vec2FromPython(align);
    }
    return placement;
}

constexpr const char* kPlacementDoc =
    " Placed on the image at `at` (fractions of its width and height "
    "from its top left), moved `offset` pixels (y down), with its "
    "own point `align` (fractions of its size) there. Without "
    "`align`, each axis snaps `at` to 0, 0.5 or 1, keeping the item "
    "away from the nearer edge: at=(1, 0) is the top right corner, "
    "at=(0, 0.3) hangs it down the left side from there. Drawn over "
    "everything, in every view (or `world`'s).";

/// An overlay batch's array (`OverlayBatch`): one on a GPU adopted as
/// it is, without copying; a host one as a contiguous NumPy array of
/// `dtype` (copied only if it is not one already); none for None.
Tensor BatchArray(nb::handle object, const char* dtype) {
    if (object.is_none()) {
        return Tensor();
    }
    try {
        const auto array = nb::cast<AnyArray>(object);
        if (array.device_type() != nb::device::cpu::value) {
            return TensorFromArray(array);
        }
    } catch (const nb::cast_error&) {
        // Lists and the like: through NumPy.
    }
    nb::module_ numpy = nb::module_::import_("numpy");
    return TensorFromArray(nb::cast<AnyArray>(
        numpy.attr("ascontiguousarray")(object, "dtype"_a = dtype)));
}

/// The arrays a bulk overlay method takes beside its geometry.
struct BatchOptions {
    nb::handle colors;
    nb::handle sizes;
    nb::handle worlds;
    nb::handle origin;
    nb::handle stream;
};

Overlay& AddBatch(Overlay& overlay, OverlayBatch batch,
                  const BatchOptions& options,
                  const nb::kwargs& style) {
    batch.colors = BatchArray(options.colors, "float32");
    batch.sizes = BatchArray(options.sizes, "float32");
    batch.worlds = BatchArray(options.worlds, "int32");
    if (!options.origin.is_none()) {
        batch.origin = Vec3FromPython(options.origin);
    }
    batch.stream = StreamOf(options.stream);
    // Per-item anchors, poses, counts and directions beside the
    // style's keywords.
    nb::dict rest;
    for (const auto& [key, value] : style) {
        const auto name = nb::cast<std::string_view>(key);
        if (value.is_none() &&
            (name == "anchors" || name == "count" ||
             name == "poses" || name == "pose_indices" ||
             name == "pose_frame" || name == "pose_worlds")) {
            continue;
        }
        if (name == "anchors") {
            for (nb::handle path : value) {
                batch.anchors.push_back(PathFromPython(path));
            }
        } else if (name == "count") {
            batch.count = BatchArray(value, "int32");
        } else if (name == "poses") {
            batch.poses = BatchArray(value, "float32");
        } else if (name == "pose_indices") {
            batch.poseIndices = BatchArray(value, "int32");
        } else if (name == "pose_frame") {
            batch.poseFrame = TransformFromPython(value);
        } else if (name == "pose_worlds") {
            batch.poseWorlds = BatchArray(value, "int32");
        } else if (name == "directions") {
            batch.directions = nb::cast<bool>(value);
        } else if (name == "scale") {
            batch.length = nb::cast<double>(value);
        } else {
            rest[key] = value;
        }
    }
    batch.style = StyleFromPython(nb::borrow<nb::kwargs>(rest));
    return overlay.Batch(std::move(batch));
}

constexpr const char* kBatchDoc =
    " Arrays have a row per item and may live on the host or on the "
    "renderer's CUDA device (NumPy, PyTorch, JAX, CuPy, Warp): GPU "
    "ones are read as they are every frame, without copies to the "
    "host, written on `stream` (or written in place in the "
    "renderer's own memory, `Renderer.map_overlay`); `touch` the "
    "overlay after writing into host ones. "
    "Optional per-item arrays: `colors` (N, 3) or (N, 4) display "
    "colours, line widths or marker sizes in pixels, and `worlds` "
    "(N,) int32, the world whose views show each (batched cameras). "
    "Coordinates are float32, relative to `origin` (a point far "
    "from the world's origin keeps them precise), or in the space of "
    "the style's `anchor`, or of `anchors`, one prim per item (each "
    "world's robot link, followed on the GPU); without `worlds` or a "
    "style's `world`, items take their anchors' worlds. "
    "For physics engines: `count`, a (1,) int32 array of the rows in "
    "use (a contact count), the GPU drawing only those; `poses`, (M, "
    "7) float32 rows of position then quaternion x, y, z, w (Warp "
    "transforms, Newton's body_q) the items are posed under, item i "
    "under pose `pose_indices[i]` (or i), turned on the GPU every "
    "frame and placed by the 4 x 4 `pose_frame`, each in its world "
    "of (M,) int32 `pose_worlds` (Newton's body_world); and for "
    "segments and arrows `directions=True`, `ends` then vectors from "
    "the starts times `scale` (velocities, forces).";

void BindOverlay(nb::module_& m) {
    nb::class_<OverlayHit>(
        m, "OverlayHit",
        "What a pick of a rendered image found "
        "(`Renderer.pick_overlay`): the overlay drawing it, and a "
        "batch's item (`batch` the batch, `item` its index) or a "
        "host item (`batch` -1, `item` its style's `id`).")
        .def_ro("overlay", &OverlayHit::overlay)
        .def_ro("batch", &OverlayHit::batch)
        .def_ro("item", &OverlayHit::item)
        .def("__repr__", [](const OverlayHit& hit) {
            return "OverlayHit(" + hit.overlay +
                   ", batch=" + std::to_string(hit.batch) +
                   ", item=" + std::to_string(hit.item) + ")";
        });
    m.def(
        "camera_image_corners",
        [](const Scene& scene, nb::handle camera, double distance) {
            const std::array<glm::dvec3, 4> corners =
                CameraImageCorners(scene, PathFromPython(camera),
                                   distance);
            nb::list rows;
            for (const glm::dvec3& corner : corners) {
                rows.append(
                    nb::make_tuple(corner.x, corner.y, corner.z));
            }
            return rows;
        },
        nb::arg("scene"), nb::arg("camera"), nb::arg("distance"),
        "The corners of the Camera at `camera`'s image `distance` "
        "metres in front of it, in the world, top left first as "
        "`Overlay.image` takes them: its image drawn there sits "
        "where it was seen from.");
    nb::class_<Overlay>(
        m, "Overlay",
        "A named layer of debug drawing in world coordinates "
        "(`Scene.overlay`): lines, shapes, "
        "markers and labels drawn over the LdrColor of render "
        "products whose `ernie:overlays` "
        "names it (\"*\" for all; the viewer shows all). Items stay "
        "until `clear`.")
        .def("clear", &Overlay::Clear,
             nb::rv_policy::reference_internal)
        .def("touch", &Overlay::Touch,
             nb::rv_policy::reference_internal,
             "Marks the overlay changed, so the renderer reads its "
             "batches' host arrays again: after writing into them in "
             "place. GPU arrays are read every frame.")
        .def_prop_rw("visible", &Overlay::IsVisible,
                     [](Overlay& overlay, bool visible) {
                         overlay.SetVisible(visible);
                     })
        .def_prop_ro("version", &Overlay::Version)
        .def_prop_ro("empty", &Overlay::Empty)
        .def(
            "line",
            [](Overlay& o, nb::handle a, nb::handle b,
               const nb::kwargs& style) -> Overlay& {
                return o.Line(Vec3FromPython(a), Vec3FromPython(b),
                              StyleFromPython(style));
            },
            nb::arg("a"), nb::arg("b"), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string("A segment from `a` to `b`.") + kStyleDoc)
                .c_str())
        .def(
            "lines",
            [](Overlay& o, nb::handle endpoints, nb::handle colors,
               const nb::kwargs& style) -> Overlay& {
                return o.Lines(PointsFromPython(endpoints),
                               StyleFromPython(style),
                               ColorsFromPython(colors));
            },
            nb::arg("endpoints"), nb::arg("colors") = nb::none(),
            nb::arg("style"), nb::rv_policy::reference_internal,
            (std::string(
                 "Many segments: `endpoints` as (N, 2, 3) or (2N, "
                 "3), coloured by "
                 "`colors` (N rows of RGB or RGBA) when given.") +
             kStyleDoc)
                .c_str())
        .def(
            "polyline",
            [](Overlay& o, nb::handle points, bool closed,
               nb::handle colors,
               const nb::kwargs& style) -> Overlay& {
                return o.Polyline(PointsFromPython(points),
                                  StyleFromPython(style), closed,
                                  ColorsFromPython(colors));
            },
            nb::arg("points"), nb::arg("closed") = false,
            nb::arg("colors") = nb::none(), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string("Segments through (N, 3) `points`, back to "
                         "the first when `closed`; with (N, 3 or 4) "
                         "`colors`, one per point, each segment "
                         "blending between its ends' (a gradient).") +
             kStyleDoc)
                .c_str())
        .def(
            "arrow",
            [](Overlay& o, nb::handle start, nb::handle end,
               double head, const nb::kwargs& style) -> Overlay& {
                return o.Arrow(Vec3FromPython(start),
                               Vec3FromPython(end),
                               StyleFromPython(style), head);
            },
            nb::arg("start"), nb::arg("end"), nb::arg("head") = 0.0,
            nb::arg("style"), nb::rv_policy::reference_internal,
            (std::string("A line with a cone head `head` long (a "
                         "fifth of it when 0).") +
             kStyleDoc)
                .c_str())
        .def(
            "box",
            [](Overlay& o, nb::handle low, nb::handle high,
               nb::handle transform,
               const nb::kwargs& style) -> Overlay& {
                return o.Box(Vec3FromPython(low),
                             Vec3FromPython(high),
                             StyleFromPython(style),
                             TransformFromPython(transform));
            },
            nb::arg("low"), nb::arg("high"),
            nb::arg("transform") = nb::none(), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string("The edges of a box, placed by a 4 x 4 "
                         "`transform`.") +
             kStyleDoc)
                .c_str())
        .def(
            "circle",
            [](Overlay& o, nb::handle centre, nb::handle normal,
               double radius, int segments,
               const nb::kwargs& style) -> Overlay& {
                return o.Circle(Vec3FromPython(centre),
                                Vec3FromPython(normal), radius,
                                StyleFromPython(style), segments);
            },
            nb::arg("centre"), nb::arg("normal"), nb::arg("radius"),
            nb::arg("segments") = 48, nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string("A circle in the plane facing `normal`.") +
             kStyleDoc)
                .c_str())
        .def(
            "sphere",
            [](Overlay& o, nb::handle centre, double radius,
               const nb::kwargs& style) -> Overlay& {
                return o.Sphere(Vec3FromPython(centre), radius,
                                StyleFromPython(style));
            },
            nb::arg("centre"), nb::arg("radius"), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string("Three great circles of a sphere.") +
             kStyleDoc)
                .c_str())
        .def(
            "arc",
            [](Overlay& o, nb::handle centre, nb::handle normal,
               nb::handle start, double radius, double angle,
               bool filled, int segments,
               const nb::kwargs& style) -> Overlay& {
                return o.Arc(
                    Vec3FromPython(centre), Vec3FromPython(normal),
                    Vec3FromPython(start), radius, angle,
                    StyleFromPython(style), filled, segments);
            },
            nb::arg("centre"), nb::arg("normal"), nb::arg("start"),
            nb::arg("radius"), nb::arg("angle"),
            nb::arg("filled") = false, nb::arg("segments") = 32,
            nb::arg("style"), nb::rv_policy::reference_internal,
            (std::string(
                 "An arc of the circle of `radius` around `centre` "
                 "in the plane facing `normal`, from direction "
                 "`start` turning `angle` radians counter-clockwise "
                 "about `normal`; with `filled`, the sector it "
                 "sweeps.") +
             kStyleDoc)
                .c_str())
        .def(
            "curve",
            [](Overlay& o, nb::handle p0, nb::handle p1,
               nb::handle p2, nb::handle p3, int segments,
               const nb::kwargs& style) -> Overlay& {
                return o.Curve(Vec3FromPython(p0), Vec3FromPython(p1),
                               Vec3FromPython(p2), Vec3FromPython(p3),
                               StyleFromPython(style), segments);
            },
            nb::arg("p0"), nb::arg("p1"), nb::arg("p2"),
            nb::arg("p3"), nb::arg("segments") = 32, nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string("The cubic Bezier curve from `p0` to `p3`, "
                         "pulled toward `p1` and `p2`.") +
             kStyleDoc)
                .c_str())
        .def(
            "axes",
            [](Overlay& o, nb::handle transform, double length,
               const nb::kwargs& style) -> Overlay& {
                return o.Axes(TransformFromPython(transform), length,
                              StyleFromPython(style));
            },
            nb::arg("transform"), nb::arg("length") = 1.0,
            nb::arg("style"), nb::rv_policy::reference_internal,
            (std::string("A transform's x, y and z axes in red, "
                         "green and blue.") +
             kStyleDoc)
                .c_str())
        .def(
            "wire_mesh",
            [](Overlay& o, nb::handle points, nb::handle indices,
               nb::handle transform,
               const nb::kwargs& style) -> Overlay& {
                return o.WireMesh(PointsFromPython(points),
                                  IndicesFromPython(indices),
                                  StyleFromPython(style),
                                  TransformFromPython(transform));
            },
            nb::arg("points"), nb::arg("indices"),
            nb::arg("transform") = nb::none(), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string("The edges of triangles (`indices` in "
                         "threes over `points`).") +
             kStyleDoc)
                .c_str())
        .def(
            "triangles",
            [](Overlay& o, nb::handle points, nb::handle indices,
               nb::handle transform,
               const nb::kwargs& style) -> Overlay& {
                return o.Triangles(PointsFromPython(points),
                                   IndicesFromPython(indices),
                                   StyleFromPython(style),
                                   TransformFromPython(transform));
            },
            nb::arg("points"), nb::arg("indices"),
            nb::arg("transform") = nb::none(), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string("Filled triangles (`indices` in threes over "
                         "`points`).") +
             kStyleDoc)
                .c_str())
        .def(
            "marker",
            [](Overlay& o, nb::handle point, std::string_view shape,
               const nb::kwargs& style) -> Overlay& {
                return o.Marker(Vec3FromPython(point),
                                StyleFromPython(style),
                                ShapeFromPython(shape));
            },
            nb::arg("point"), nb::arg("shape") = "circle",
            nb::arg("style"), nb::rv_policy::reference_internal,
            (std::string("A marker of `size` pixels: circle, ring, "
                         "square, diamond, cross or "
                         "plus.") +
             kStyleDoc)
                .c_str())
        .def(
            "markers",
            [](Overlay& o, nb::handle points, nb::handle colors,
               std::string_view shape,
               const nb::kwargs& style) -> Overlay& {
                return o.Markers(
                    PointsFromPython(points), StyleFromPython(style),
                    ShapeFromPython(shape), ColorsFromPython(colors));
            },
            nb::arg("points"), nb::arg("colors") = nb::none(),
            nb::arg("shape") = "circle", nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string("Markers at (N, 3) `points`, coloured by "
                         "`colors` when given.") +
             kStyleDoc)
                .c_str())
        .def(
            "segments",
            [](Overlay& o, nb::handle starts, nb::handle ends,
               nb::handle colors, nb::handle widths,
               nb::handle worlds, nb::handle origin,
               nb::handle stream,
               const nb::kwargs& style) -> Overlay& {
                OverlayBatch batch;
                batch.kind = OverlayBatchKind::Lines;
                batch.points = BatchArray(starts, "float32");
                batch.ends = BatchArray(ends, "float32");
                return AddBatch(
                    o, std::move(batch),
                    {colors, widths, worlds, origin, stream}, style);
            },
            nb::arg("starts"), nb::arg("ends"),
            nb::arg("colors") = nb::none(),
            nb::arg("widths") = nb::none(),
            nb::arg("worlds") = nb::none(),
            nb::arg("origin") = nb::none(),
            nb::arg("stream") = nb::none(), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string("Segments from (N, 3) `starts` to `ends`.") +
             kBatchDoc + kStyleDoc)
                .c_str())
        .def(
            "arrows",
            [](Overlay& o, nb::handle starts, nb::handle ends,
               double head, nb::handle colors, nb::handle widths,
               nb::handle worlds, nb::handle origin,
               nb::handle stream,
               const nb::kwargs& style) -> Overlay& {
                OverlayBatch batch;
                batch.kind = OverlayBatchKind::Arrows;
                batch.points = BatchArray(starts, "float32");
                batch.ends = BatchArray(ends, "float32");
                batch.head = head;
                return AddBatch(
                    o, std::move(batch),
                    {colors, widths, worlds, origin, stream}, style);
            },
            nb::arg("starts"), nb::arg("ends"), nb::arg("head") = 0.0,
            nb::arg("colors") = nb::none(),
            nb::arg("widths") = nb::none(),
            nb::arg("worlds") = nb::none(),
            nb::arg("origin") = nb::none(),
            nb::arg("stream") = nb::none(), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string(
                 "Arrows from (N, 3) `starts` to `ends`, with "
                 "cone heads `head` long (a fifth of each "
                 "when 0).") +
             kBatchDoc + kStyleDoc)
                .c_str())
        .def(
            "points",
            [](Overlay& o, nb::handle points, std::string_view shape,
               nb::handle colors, nb::handle sizes, nb::handle worlds,
               nb::handle origin, nb::handle stream,
               const nb::kwargs& style) -> Overlay& {
                OverlayBatch batch;
                batch.kind = OverlayBatchKind::Markers;
                batch.points = BatchArray(points, "float32");
                batch.shape = ShapeFromPython(shape);
                return AddBatch(
                    o, std::move(batch),
                    {colors, sizes, worlds, origin, stream}, style);
            },
            nb::arg("points"), nb::arg("shape") = "circle",
            nb::arg("colors") = nb::none(),
            nb::arg("sizes") = nb::none(),
            nb::arg("worlds") = nb::none(),
            nb::arg("origin") = nb::none(),
            nb::arg("stream") = nb::none(), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string("Markers of `shape` at (N, 3) `points`, "
                         "`sizes` pixels across.") +
             kBatchDoc + kStyleDoc)
                .c_str())
        .def(
            "boxes",
            [](Overlay& o, nb::handle centres, nb::handle halfExtents,
               nb::handle orientations, nb::handle colors,
               nb::handle widths, nb::handle worlds,
               nb::handle origin, nb::handle stream,
               const nb::kwargs& style) -> Overlay& {
                OverlayBatch batch;
                batch.kind = OverlayBatchKind::Boxes;
                batch.points = BatchArray(centres, "float32");
                batch.halfExtents =
                    BatchArray(halfExtents, "float32");
                batch.orientations =
                    BatchArray(orientations, "float32");
                return AddBatch(
                    o, std::move(batch),
                    {colors, widths, worlds, origin, stream}, style);
            },
            nb::arg("centres"), nb::arg("half_extents"),
            nb::arg("orientations") = nb::none(),
            nb::arg("colors") = nb::none(),
            nb::arg("widths") = nb::none(),
            nb::arg("worlds") = nb::none(),
            nb::arg("origin") = nb::none(),
            nb::arg("stream") = nb::none(), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string(
                 "The edges of boxes around (N, 3) `centres`, "
                 "`half_extents` out along their axes, turned "
                 "by (N, 4) `orientations` (x, y, z, w).") +
             kBatchDoc + kStyleDoc)
                .c_str())
        .def(
            "spheres",
            [](Overlay& o, nb::handle centres, nb::handle radii,
               nb::handle colors, nb::handle widths,
               nb::handle worlds, nb::handle origin,
               nb::handle stream,
               const nb::kwargs& style) -> Overlay& {
                OverlayBatch batch;
                batch.kind = OverlayBatchKind::Spheres;
                batch.points = BatchArray(centres, "float32");
                batch.radii = BatchArray(radii, "float32");
                return AddBatch(
                    o, std::move(batch),
                    {colors, widths, worlds, origin, stream}, style);
            },
            nb::arg("centres"), nb::arg("radii"),
            nb::arg("colors") = nb::none(),
            nb::arg("widths") = nb::none(),
            nb::arg("worlds") = nb::none(),
            nb::arg("origin") = nb::none(),
            nb::arg("stream") = nb::none(), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string("Spheres around (N, 3) `centres` of (N,) "
                         "`radii`, as three great circles each.") +
             kBatchDoc + kStyleDoc)
                .c_str())
        .def(
            "frames",
            [](Overlay& o, nb::handle origins,
               nb::handle orientations, double length,
               nb::handle widths, nb::handle worlds,
               nb::handle origin, nb::handle stream,
               const nb::kwargs& style) -> Overlay& {
                OverlayBatch batch;
                batch.kind = OverlayBatchKind::Frames;
                batch.points = BatchArray(origins, "float32");
                batch.orientations =
                    BatchArray(orientations, "float32");
                batch.length = length;
                return AddBatch(
                    o, std::move(batch),
                    {nb::none(), widths, worlds, origin, stream},
                    style);
            },
            nb::arg("origins"), nb::arg("orientations") = nb::none(),
            nb::arg("length") = 1.0, nb::arg("widths") = nb::none(),
            nb::arg("worlds") = nb::none(),
            nb::arg("origin") = nb::none(),
            nb::arg("stream") = nb::none(), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string(
                 "The x, y and z axes, `length` long in red, "
                 "green and blue, at (N, 3) `origins`, turned "
                 "by (N, 4) `orientations` (x, y, z, w).") +
             kBatchDoc + kStyleDoc)
                .c_str())
        .def(
            "label",
            [](Overlay& o, nb::handle point, std::string_view text,
               const nb::kwargs& style) -> Overlay& {
                return o.Label(Vec3FromPython(point), text,
                               StyleFromPython(style));
            },
            nb::arg("point"), nb::arg("text"), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string("ASCII `text`, `size` pixels tall, beside "
                         "`point`.") +
             kStyleDoc)
                .c_str())
        .def(
            "text",
            [](Overlay& o, std::string_view text, nb::handle at,
               nb::handle offset, nb::handle align,
               const nb::kwargs& style) -> Overlay& {
                return o.Text(PlacementFromPython(at, offset, align),
                              text, StyleFromPython(style));
            },
            nb::arg("text"), nb::arg("at") = nb::make_tuple(0, 0),
            nb::arg("offset") = nb::make_tuple(0, 0),
            nb::arg("align") = nb::none(), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string("ASCII `text` (lines split at newlines) "
                         "`size` pixels a line on the image itself, "
                         "a heads-up display, over a `background` "
                         "box.") +
             kPlacementDoc + kStyleDoc)
                .c_str())
        .def(
            "rect",
            [](Overlay& o, nb::handle size, nb::handle at,
               nb::handle offset, nb::handle align,
               const nb::kwargs& style) -> Overlay& {
                return o.Rect(PlacementFromPython(at, offset, align),
                              Vec2FromPython(size),
                              StyleFromPython(style));
            },
            nb::arg("size"), nb::arg("at") = nb::make_tuple(0, 0),
            nb::arg("offset") = nb::make_tuple(0, 0),
            nb::arg("align") = nb::none(), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string("A filled rectangle `size` (width, height) "
                         "pixels on the image: a panel or a bar.") +
             kPlacementDoc + kStyleDoc)
                .c_str())
        .def(
            "button",
            [](Overlay& o, std::string_view label, nb::handle at,
               nb::handle offset, nb::handle align, nb::handle box,
               const nb::kwargs& style) -> Overlay& {
                return o.Button(
                    PlacementFromPython(at, offset, align), label,
                    StyleFromPython(style),
                    box.is_none() ? glm::vec2(0.0f)
                                  : Vec2FromPython(box));
            },
            nb::arg("label"), nb::arg("at") = nb::make_tuple(0, 0),
            nb::arg("offset") = nb::make_tuple(0, 0),
            nb::arg("align") = nb::none(),
            nb::arg("box") = nb::none(), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string(
                 "A button on the image: a box `box` (width, height) "
                 "pixels (fitted around `label` when None) in the "
                 "style's "
                 "`background`, or its `hover` colour while hovered, "
                 "with `label` centred in its `color`. Give it an "
                 "`id`: the viewer reports its clicks "
                 "(`Viewer.add_overlay_click_handler`), and "
                 "`screen_item_at` finds it under a pointer.") +
             kPlacementDoc + kStyleDoc)
                .c_str())
        .def(
            "screen_item_at",
            [](const Overlay& o, nb::handle pixel,
               nb::handle extent) {
                return o.ScreenItemAt(Vec2FromPython(pixel),
                                      Vec2FromPython(extent));
            },
            nb::arg("pixel"), nb::arg("extent"),
            "The id of the topmost screen item with one covering "
            "`pixel` (x, y down) of an image `extent` (width, "
            "height) "
            "pixels, or 0: what a click or pointer there hits.")
        .def_prop_rw(
            "hovered", &Overlay::Hovered,
            [](Overlay& o, std::uint32_t id) { o.SetHovered(id); },
            "The id whose items show their hover colour, 0 for none; "
            "the viewer sets it under its pointer, and anything else "
            "hit testing (a VR controller's ray) may.")
        .def(
            "numbers",
            [](Overlay& o, nb::handle values, nb::handle points,
               std::string prefix, std::string suffix, int decimals,
               bool screen, nb::handle at, nb::handle offset,
               nb::handle align, nb::handle colors, nb::handle sizes,
               nb::handle worlds, nb::handle origin,
               nb::handle stream,
               const nb::kwargs& style) -> Overlay& {
                OverlayBatch batch;
                batch.kind = OverlayBatchKind::Numbers;
                batch.values = BatchArray(values, "float32");
                batch.points = BatchArray(points, "float32");
                batch.format.prefix = std::move(prefix);
                batch.format.suffix = std::move(suffix);
                batch.format.decimals = decimals;
                batch.screen = screen;
                batch.placement =
                    PlacementFromPython(at, offset, align);
                if (!align.is_none()) {
                    batch.format.align = Vec2FromPython(align);
                }
                return AddBatch(
                    o, std::move(batch),
                    {colors, sizes, worlds, origin, stream}, style);
            },
            nb::arg("values"), nb::arg("points") = nb::none(),
            nb::arg("prefix") = "", nb::arg("suffix") = "",
            nb::arg("decimals") = 2, nb::arg("screen") = false,
            nb::arg("at") = nb::make_tuple(0, 0),
            nb::arg("offset") = nb::make_tuple(0, 0),
            nb::arg("align") = nb::none(),
            nb::arg("colors") = nb::none(),
            nb::arg("sizes") = nb::none(),
            nb::arg("worlds") = nb::none(),
            nb::arg("origin") = nb::none(),
            nb::arg("stream") = nb::none(), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string(
                 "(N,) `values` printed as text on the GPU, each "
                 "`prefix`, the value with `decimals` digits after "
                 "the point, then `suffix`: beside (N, 3) `points` "
                 "as labels, or with `screen` on the image, at "
                 "`points`' x and y (fractions of the image) or, "
                 "without points, from `at` and `offset` a line "
                 "apart, its box's point `align` there. A sim's "
                 "time or each world's reward, read on the GPU "
                 "every frame with no copy to the host.") +
             kBatchDoc + kStyleDoc)
                .c_str())
        .def(
            "image",
            [](Overlay& o, nb::handle corners, nb::handle image,
               nb::handle stream,
               const nb::kwargs& style) -> Overlay& {
                OverlayBatch batch;
                batch.kind = OverlayBatchKind::Image;
                batch.points = BatchArray(corners, "float32");
                batch.image = BatchArray(image, "uint8");
                return AddBatch(o, std::move(batch),
                                {nb::none(), nb::none(), nb::none(),
                                 nb::none(), stream},
                                style);
            },
            nb::arg("corners"), nb::arg("image"),
            nb::arg("stream") = nb::none(), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string(
                 "A (rows, cols, 3 or 4) uint8 `image` (display "
                 "colours, row 0 at its top) on the quad between "
                 "four world `corners`, top left, top right, bottom "
                 "right, bottom left (`camera_image_corners` gives a "
                 "camera's): such as another camera's LdrColor, a "
                 "CUDA array read again every frame. Filtered "
                 "between texels, times the style's colour; it may "
                 "follow an "
                 "`anchor` or `poses`.") +
             kStyleDoc)
                .c_str())
        .def(
            "screen_image",
            [](Overlay& o, nb::handle image, nb::handle at,
               nb::handle offset, nb::handle align, nb::handle size,
               nb::handle stream,
               const nb::kwargs& style) -> Overlay& {
                OverlayBatch batch;
                batch.kind = OverlayBatchKind::Image;
                batch.screen = true;
                batch.placement =
                    PlacementFromPython(at, offset, align);
                if (!size.is_none()) {
                    batch.imageSize = Vec2FromPython(size);
                }
                batch.image = BatchArray(image, "uint8");
                return AddBatch(o, std::move(batch),
                                {nb::none(), nb::none(), nb::none(),
                                 nb::none(), stream},
                                style);
            },
            nb::arg("image"), nb::arg("at") = nb::make_tuple(0, 0),
            nb::arg("offset") = nb::make_tuple(0, 0),
            nb::arg("align") = nb::none(),
            nb::arg("size") = nb::none(),
            nb::arg("stream") = nb::none(), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string("`image` on the screen, `size` (width, "
                         "height) pixels (its own when None): a "
                         "picture in the picture.") +
             kPlacementDoc + kStyleDoc)
                .c_str())
        .def(
            "texts",
            [](Overlay& o, nb::handle texts, nb::handle points,
               bool screen, nb::handle at, nb::handle offset,
               nb::handle align, nb::handle colors, nb::handle sizes,
               nb::handle worlds, nb::handle origin,
               nb::handle stream,
               const nb::kwargs& style) -> Overlay& {
                OverlayBatch batch;
                batch.kind = OverlayBatchKind::Texts;
                // Strings become rows of codes, padded with 0s.
                if (nb::isinstance<nb::list>(texts) ||
                    nb::isinstance<nb::tuple>(texts)) {
                    // Latin-1 codes, '?' for what the font lacks.
                    std::vector<std::vector<std::uint8_t>> rows;
                    std::size_t longest = 1;
                    for (nb::handle text : texts) {
                        rows.push_back(OverlayCodes(
                            nb::cast<std::string>(text)));
                        longest =
                            std::max(longest, rows.back().size());
                    }
                    std::vector<std::int32_t> codes(
                        rows.size() * longest, 0);
                    for (std::size_t r = 0; r < rows.size(); ++r) {
                        for (std::size_t c = 0; c < rows[r].size();
                             ++c) {
                            codes[r * longest + c] = rows[r][c];
                        }
                    }
                    batch.chars = Tensor::Copy<std::int32_t>(
                        codes,
                        {static_cast<std::int64_t>(rows.size()),
                         static_cast<std::int64_t>(longest)});
                } else {
                    batch.chars = BatchArray(texts, "int32");
                }
                batch.points = BatchArray(points, "float32");
                batch.screen = screen;
                batch.placement =
                    PlacementFromPython(at, offset, align);
                if (!align.is_none()) {
                    batch.format.align = Vec2FromPython(align);
                }
                return AddBatch(
                    o, std::move(batch),
                    {colors, sizes, worlds, origin, stream}, style);
            },
            nb::arg("texts"), nb::arg("points") = nb::none(),
            nb::arg("screen") = false,
            nb::arg("at") = nb::make_tuple(0, 0),
            nb::arg("offset") = nb::make_tuple(0, 0),
            nb::arg("align") = nb::none(),
            nb::arg("colors") = nb::none(),
            nb::arg("sizes") = nb::none(),
            nb::arg("worlds") = nb::none(),
            nb::arg("origin") = nb::none(),
            nb::arg("stream") = nb::none(), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string(
                 "Text per item, drawn on the GPU: `texts` a list of "
                 "strings, or an (N, K) int32 array of character "
                 "codes (printable ASCII, each row ending at its "
                 "first 0) that a program may rewrite in place. "
                 "Beside (N, 3) `points` as labels, or with `screen` "
                 "on the image as `numbers`.") +
             kBatchDoc + kStyleDoc)
                .c_str())
        .def(
            "trails",
            [](Overlay& o, nb::handle points, nb::handle heads,
               bool fade, nb::handle colors, nb::handle widths,
               nb::handle worlds, nb::handle origin,
               nb::handle stream,
               const nb::kwargs& style) -> Overlay& {
                OverlayBatch batch;
                batch.kind = OverlayBatchKind::Trails;
                batch.points = BatchArray(points, "float32");
                batch.heads = BatchArray(heads, "int32");
                batch.fade = fade;
                return AddBatch(
                    o, std::move(batch),
                    {colors, widths, worlds, origin, stream}, style);
            },
            nb::arg("points"), nb::arg("heads") = nb::none(),
            nb::arg("fade") = true, nb::arg("colors") = nb::none(),
            nb::arg("widths") = nb::none(),
            nb::arg("worlds") = nb::none(),
            nb::arg("origin") = nb::none(),
            nb::arg("stream") = nb::none(), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string(
                 "A polyline per item through its row of (N, L, 3) "
                 "`points`, a ring buffer a simulation writes in "
                 "place: its newest point at (N,) int32 `heads` (the "
                 "last without), fading to clear at its oldest with "
                 "`fade`; a segment with a NaN end is left out, so "
                 "NaN rows start a trail afresh. Per-item arrays "
                 "have a row per trail; `colors` may instead be one "
                 "per point, (N, L, 3 or 4), each segment in its "
                 "newer point's colour (a speed gradient).") +
             kBatchDoc + kStyleDoc)
                .c_str())
        .def(
            "capsules",
            [](Overlay& o, nb::handle centres, nb::handle radii,
               nb::handle halfHeights, nb::handle orientations,
               nb::handle colors, nb::handle widths,
               nb::handle worlds, nb::handle origin,
               nb::handle stream,
               const nb::kwargs& style) -> Overlay& {
                OverlayBatch batch;
                batch.kind = OverlayBatchKind::Capsules;
                batch.points = BatchArray(centres, "float32");
                batch.radii = BatchArray(radii, "float32");
                batch.halfHeights =
                    BatchArray(halfHeights, "float32");
                batch.orientations =
                    BatchArray(orientations, "float32");
                return AddBatch(
                    o, std::move(batch),
                    {colors, widths, worlds, origin, stream}, style);
            },
            nb::arg("centres"), nb::arg("radii"),
            nb::arg("half_heights"),
            nb::arg("orientations") = nb::none(),
            nb::arg("colors") = nb::none(),
            nb::arg("widths") = nb::none(),
            nb::arg("worlds") = nb::none(),
            nb::arg("origin") = nb::none(),
            nb::arg("stream") = nb::none(), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string(
                 "Wireframe capsules around (N, 3) `centres`, along "
                 "their local z axis turned by (N, 4) "
                 "`orientations` (x, y, z, w): cylinders of (N,) "
                 "`radii` reaching `half_heights` each way, capped "
                 "by hemispheres.") +
             kBatchDoc + kStyleDoc)
                .c_str())
        .def_prop_ro(
            "batch_count",
            [](const Overlay& o) { return o.Batches().size(); },
            "Batches added since the last clear: the last one's "
            "index for `Renderer.map_overlay` is one less.")
        .def("__repr__", [](const Overlay& o) {
            return "Overlay(" + std::to_string(o.Segments().size()) +
                   " segments, " + std::to_string(o.Fills().size()) +
                   " triangles, " +
                   std::to_string(o.Sprites().size()) + " sprites)";
        });
}

/// The Prim a typed prim wraps (its `prim` attribute), or null.
nb::object WrappedPrim(nb::handle object) noexcept {
    PyObject* prim = PyObject_GetAttrString(object.ptr(), "prim");
    if (prim == nullptr) {
        PyErr_Clear();
        return {};
    }
    nb::object owned = nb::steal(prim);
    return nb::isinstance<Prim>(owned) ? owned : nb::object();
}

}  // namespace

bool IsPrimPath(nb::handle object) noexcept {
    return nb::isinstance<nb::str>(object) ||
           nb::isinstance<Path>(object) ||
           nb::isinstance<Prim>(object) ||
           WrappedPrim(object).is_valid();
}

Path PathFromPython(nb::handle object) {
    if (nb::isinstance<nb::str>(object)) {
        return Path(nb::cast<std::string_view>(object));
    }
    if (nb::isinstance<Path>(object)) {
        return nb::cast<const Path&>(object);
    }
    if (nb::isinstance<Prim>(object)) {
        return nb::cast<const Prim&>(object).GetPath();
    }
    if (nb::object prim = WrappedPrim(object); prim.is_valid()) {
        return nb::cast<const Prim&>(prim).GetPath();
    }
    throw nb::type_error(
        ("A prim path is a str, Path, Prim or typed prim, not " +
         std::string(nb::type_name(object.type()).c_str()))
            .c_str());
}

Value ValueFromPython(nb::handle object) {
    if (object.is_none()) {
        return {};
    }
    if (nb::isinstance<nb::bool_>(object)) {
        return nb::cast<bool>(object);
    }
    if (nb::isinstance<nb::int_>(object)) {
        int value = 0;
        if (!nb::try_cast<int>(object, value)) {
            throw nb::value_error(
                "Integer scene values must fit in 32 bits");
        }
        return value;
    }
    if (nb::isinstance<nb::float_>(object)) {
        return nb::cast<double>(object);
    }
    if (nb::isinstance<nb::str>(object)) {
        return nb::cast<std::string>(object);
    }
    if (nb::isinstance<Path>(object)) {
        return nb::cast<Path>(object);
    }
    if (nb::isinstance<glm::dquat>(object)) {
        return nb::cast<glm::dquat>(object);
    }
    nb::object sequence = nb::borrow(object);
    if (nb::hasattr(object, "tolist")) {
        sequence = object.attr("tolist")();
        if (IsNumber(sequence)) {
            return ValueFromPython(sequence);
        }
    }
    if (!nb::isinstance<nb::list>(sequence) &&
        !nb::isinstance<nb::tuple>(sequence)) {
        throw nb::type_error(
            "Scene values are bool, int, float, str, Path, Quat, a "
            "sequence of "
            "numbers, str or Path, or a 4 x 4 matrix");
    }
    std::vector<nb::object> items;
    for (nb::handle item : sequence) {
        items.push_back(nb::borrow(item));
    }
    if (!items.empty() && nb::isinstance<Path>(items[0])) {
        std::vector<Path> paths;
        for (const nb::object& item : items) {
            if (!nb::isinstance<Path>(item)) {
                throw nb::type_error(
                    "A list of targets must hold only Path values");
            }
            paths.push_back(nb::cast<Path>(item));
        }
        return paths;
    }
    if (!items.empty() && nb::isinstance<nb::str>(items[0])) {
        std::vector<std::string> tokens;
        for (const nb::object& item : items) {
            if (!nb::isinstance<nb::str>(item)) {
                throw nb::type_error(
                    "A list of tokens must hold only str values");
            }
            tokens.push_back(nb::cast<std::string>(item));
        }
        return tokens;
    }
    if (items.size() == 4 && !IsNumber(items[0])) {
        glm::dmat4 matrix(1.0);
        for (int row = 0; row < 4; ++row) {
            const std::vector<double> values =
                nb::cast<std::vector<double>>(
                    items[static_cast<std::size_t>(row)]);
            if (values.size() != 4) {
                throw nb::type_error(
                    "A matrix needs 4 rows of 4 numbers");
            }
            for (int col = 0; col < 4; ++col) {
                matrix[col][row] =
                    values[static_cast<std::size_t>(col)];
            }
        }
        return matrix;
    }
    bool integers = true;
    std::vector<double> numbers;
    for (const nb::object& item : items) {
        if (!IsNumber(item) || nb::isinstance<nb::bool_>(item)) {
            throw nb::type_error(
                "Sequences in scene values must hold numbers");
        }
        integers = integers && nb::isinstance<nb::int_>(item);
        numbers.push_back(nb::cast<double>(item));
    }
    if (numbers.size() == 2) {
        if (integers) {
            return glm::ivec2(static_cast<int>(numbers[0]),
                              static_cast<int>(numbers[1]));
        }
        return glm::dvec2(numbers[0], numbers[1]);
    }
    if (numbers.size() == 3) {
        return glm::dvec3(numbers[0], numbers[1], numbers[2]);
    }
    if (numbers.size() == 4) {
        return glm::vec4(numbers[0], numbers[1], numbers[2],
                         numbers[3]);
    }
    return std::vector<float>(numbers.begin(), numbers.end());
}

RotationOrder RotationOrderFromPython(std::string_view name) {
    constexpr std::pair<std::string_view, RotationOrder> kOrders[] = {
        {"XYZ", RotationOrder::XYZ}, {"XZY", RotationOrder::XZY},
        {"YXZ", RotationOrder::YXZ}, {"YZX", RotationOrder::YZX},
        {"ZXY", RotationOrder::ZXY}, {"ZYX", RotationOrder::ZYX}};
    for (const auto& [token, order] : kOrders) {
        if (token == name) {
            return order;
        }
    }
    throw nb::value_error("rotation_order must be \"XYZ\", \"XZY\", "
                          "\"YXZ\", \"YZX\", \"ZXY\" or "
                          "\"ZYX\"");
}

/// An `XformOps` from `set_xform`'s keyword arguments; a single
/// number scales uniformly.
XformOps XformFromPython(nb::handle translate, nb::handle rotate,
                         std::string_view rotationOrder,
                         nb::handle orient, nb::handle scale) {
    XformOps xform;
    if (!translate.is_none()) {
        xform.translate = Vec3FromPython(translate);
    }
    if (!rotate.is_none()) {
        xform.rotate = Vec3FromPython(rotate);
    }
    xform.rotationOrder = RotationOrderFromPython(rotationOrder);
    if (!orient.is_none()) {
        if (!nb::isinstance<glm::dquat>(orient)) {
            throw nb::type_error("orient must be a Quat (Quat(w, x, "
                                 "y, z) or Quat.from_xyzw)");
        }
        xform.orient = nb::cast<glm::dquat>(orient);
    }
    if (!scale.is_none()) {
        xform.scale = IsNumber(scale)
                          ? glm::dvec3(nb::cast<double>(scale))
                          : Vec3FromPython(scale);
    }
    return xform;
}

nb::object ValueToPython(const Value& value) {
    return std::visit(
        [](const auto& v) -> nb::object {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                return nb::none();
            } else if constexpr (std::is_same_v<T, glm::ivec2>) {
                return nb::make_tuple(v.x, v.y);
            } else if constexpr (std::is_same_v<T, glm::vec2> ||
                                 std::is_same_v<T, glm::dvec2>) {
                return nb::make_tuple(double(v.x), double(v.y));
            } else if constexpr (std::is_same_v<T, glm::vec3> ||
                                 std::is_same_v<T, glm::dvec3>) {
                return nb::make_tuple(double(v.x), double(v.y),
                                      double(v.z));
            } else if constexpr (std::is_same_v<T, glm::vec4> ||
                                 std::is_same_v<T, glm::dvec4>) {
                return nb::make_tuple(double(v.x), double(v.y),
                                      double(v.z), double(v.w));
            } else if constexpr (std::is_same_v<T, glm::dmat4>) {
                return MatrixToPython(v);
            } else if constexpr (std::is_same_v<T,
                                                std::vector<Path>>) {
                nb::list paths;
                for (const Path& path : v) {
                    paths.append(nb::cast(path));
                }
                return paths;
            } else if constexpr (std::is_same_v<T, float>) {
                return nb::cast(double(v));
            } else {
                return nb::cast(v);
            }
        },
        value.Raw());
}

void BindScene(nb::module_& m) {
    BindOverlay(m);
    nb::class_<Path>(m, "Path",
                     "Absolute path of a prim, as in USD: `/` is the "
                     "root and `/World/Sun` the "
                     "child `Sun` of `/World`. Functions taking a "
                     "path also accept a str, a "
                     "Prim or a typed prim (its path).")
        .def(nb::init<>())
        .def(nb::init_implicit<std::string>(), nb::arg("text"))
        .def(nb::init_implicit<PrimPath>(), nb::arg("prim"))
        .def_static("root", &Path::Root)
        .def_static("is_valid_name", &Path::IsValidName,
                    nb::arg("name"))
        .def_prop_ro(
            "name",
            [](const Path& p) { return std::string(p.Name()); })
        .def_prop_ro("parent", &Path::Parent)
        .def_prop_ro("is_root", &Path::IsRoot)
        .def_prop_ro("is_empty", &Path::IsEmpty)
        .def("child", &Path::Child, nb::arg("name"))
        .def("has_prefix", &Path::HasPrefix, nb::arg("ancestor"))
        .def("__str__", &Path::String)
        .def("__repr__",
             [](const Path& p) {
                 return "Path('" + p.String() + "')";
             })
        .def("__hash__",
             [](const Path& p) { return std::hash<Path>()(p); })
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def(nb::self < nb::self);

    nb::enum_<Dirty>(m, "Dirty", nb::is_flag(),
                     "What changed about a prim.")
        .value("NONE", Dirty::None)
        .value("ADDED", Dirty::Added)
        .value("REMOVED", Dirty::Removed)
        .value("TRANSFORM", Dirty::Transform)
        .value("TOPOLOGY", Dirty::Topology)
        .value("POINTS", Dirty::Points)
        .value("PRIMVARS", Dirty::Primvars)
        .value("MATERIAL", Dirty::Material)
        .value("PARAMS", Dirty::Params)
        .value("VISIBILITY", Dirty::Visibility);
    m.def(
        "dirty_of", &DirtyOf, nb::arg("attribute"),
        "The dirty bit that setting or writing `attribute` raises.");

    nb::class_<ChangeCursor>(m, "ChangeCursor",
                             "A consumer's position in a scene's "
                             "history; see Scene.changes.")
        .def(nb::init<>())
        .def_rw("version", &ChangeCursor::version);

    nb::class_<glm::dquat>(
        m, "Quat",
        "A rotation as a quaternion, Quat(w, x, y, z) with the real "
        "part first as USD writes one "
        "(xformOp:orient). Arrays of quaternions (PointInstancer "
        "orientations, xformOp:orient "
        "rows) are x, y, z, w instead, as in USD's memory; "
        "Quat.from_xyzw reads that order, as "
        "SciPy, ROS and Warp use.")
        .def(nb::init<double, double, double, double>(), nb::arg("w"),
             nb::arg("x"), nb::arg("y"), nb::arg("z"))
        .def_static(
            "from_xyzw",
            [](double x, double y, double z, double w) {
                return glm::dquat(w, x, y, z);
            },
            nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("w"))
        .def_static(
            "from_axis_angle",
            [](nb::handle axis, double degrees) {
                return glm::angleAxis(
                    glm::radians(degrees),
                    glm::normalize(Vec3FromPython(axis)));
            },
            nb::arg("axis"), nb::arg("degrees"),
            "The rotation by `degrees` about `axis`, right-handed.")
        .def_static(
            "from_euler",
            [](nb::handle degrees, std::string_view order) {
                return glm::normalize(glm::quat_cast(
                    EulerRotation(Vec3FromPython(degrees),
                                  RotationOrderFromPython(order))));
            },
            nb::arg("degrees"), nb::arg("order") = "XYZ",
            "The rotation of Euler angles (x, y, z) in degrees "
            "applied about fixed axes in "
            "`order`, as the xformOp:rotate ops (\"XYZ\": about X "
            "first, then Y, then Z).")
        .def_rw("w", &glm::dquat::w)
        .def_rw("x", &glm::dquat::x)
        .def_rw("y", &glm::dquat::y)
        .def_rw("z", &glm::dquat::z)
        .def_prop_ro("wxyz",
                     [](const glm::dquat& q) {
                         return nb::make_tuple(q.w, q.x, q.y, q.z);
                     })
        .def_prop_ro("xyzw",
                     [](const glm::dquat& q) {
                         return nb::make_tuple(q.x, q.y, q.z, q.w);
                     })
        .def(
            "euler",
            [](const glm::dquat& q, std::string_view order) {
                const glm::dvec3 degrees =
                    EulerAngles(glm::mat3_cast(glm::normalize(q)),
                                RotationOrderFromPython(order));
                return nb::make_tuple(degrees.x, degrees.y,
                                      degrees.z);
            },
            nb::arg("order") = "XYZ",
            "Euler angles (x, y, z) in degrees, as from_euler takes.")
        .def("__eq__", [](const glm::dquat& a,
                          const glm::dquat& b) { return a == b; })
        .def("__repr__", [](const glm::dquat& q) {
            return "Quat(w=" + std::to_string(q.w) +
                   ", x=" + std::to_string(q.x) +
                   ", y=" + std::to_string(q.y) +
                   ", z=" + std::to_string(q.z) + ")";
        });

    nb::class_<PrimSet>(
        m, "PrimSet",
        "A fixed list of prims to write rows of one array to at once "
        "(Scene.write), such as a "
        "simulation's bodies every step: a scene looks the prims up "
        "the first time and keeps "
        "them, looking them up again only after prims are removed.")
        .def(nb::init<std::vector<Path>>(), nb::arg("paths"))
        .def_prop_ro("paths", &PrimSet::Paths)
        .def("__len__", &PrimSet::Size)
        .def("__repr__", [](const PrimSet& prims) {
            return "PrimSet(" + std::to_string(prims.Size()) +
                   " prims)";
        });

    nb::class_<Prim>(m, "Prim", "Handle to a prim in a Scene.")
        .def_prop_ro("path", &Prim::GetPath)
        .def_prop_ro("type", &Prim::Type)
        .def_prop_ro("valid", &Prim::IsValid)
        .def(
            "attributes", &Prim::Attributes,
            "The attribute names set or written on the prim, in name "
            "order.")
        .def("__bool__", &Prim::IsValid)
        .def("define", &Prim::Define, nb::arg("relative"),
             nb::arg("type"), nb::keep_alive<0, 1>(),
             "Defines the prim at `relative` below this one "
             "(\"Rock\", or \"Rig/Arm\") with "
             "`type`, as Scene.define.")
        .def(
            "get_prim",
            [](const Prim& prim,
               std::string_view relative) -> nb::object {
                Prim found = prim.GetPrim(relative);
                return found ? nb::cast(found) : nb::none();
            },
            nb::arg("relative"), nb::keep_alive<0, 1>(),
            "The prim at `relative` below this one (\"Rock\", or "
            "\"Rig/Arm\"), or None, as "
            "Scene.get_prim.")
        .def(
            "set",
            [](Prim& prim, std::string_view attribute,
               nb::handle value) -> Prim& {
                return prim.Set(attribute, ValueFromPython(value));
            },
            nb::arg("attribute"), nb::arg("value").none(),
            nb::rv_policy::reference_internal,
            "Sets `attribute`; returns the prim for chaining.")
        .def(
            "set_xform",
            [](Prim& prim, nb::handle translate, nb::handle rotate,
               std::string_view order, nb::handle orient,
               nb::handle scale) -> Prim& {
                return prim.SetXform(XformFromPython(
                    translate, rotate, order, orient, scale));
            },
            nb::kw_only(), nb::arg("translate").none() = nb::none(),
            nb::arg("rotate").none() = nb::none(),
            nb::arg("rotation_order") = "XYZ",
            nb::arg("orient").none() = nb::none(),
            nb::arg("scale").none() = nb::none(),
            nb::rv_policy::reference_internal,
            "Sets the prim's transform as translate, rotate (Euler "
            "angles in degrees in "
            "`rotation_order`) or orient (a Quat), and scale (three "
            "numbers or one) ops, with "
            "xformOpOrder listing the given ones in that order; "
            "returns the prim.")
        .def(
            "get",
            [](const Prim& prim, std::string_view attribute,
               nb::object fallback) {
                const Value* value = prim.Get(attribute);
                return value != nullptr ? ValueToPython(*value)
                                        : fallback;
            },
            nb::arg("attribute"), nb::arg("default") = nb::none())
        .def(
            "write",
            [](Prim& prim, std::string_view attribute,
               const AnyArray& array) -> Prim& {
                return prim.Write(attribute, TensorFromArray(array));
            },
            nb::arg("attribute"), nb::arg("array"),
            nb::rv_policy::reference_internal,
            "Writes any DLPack array as `attribute` without copying; "
            "returns the prim.")
        .def(
            "read",
            [](const Prim& prim,
               std::string_view attribute) -> nb::object {
                const Tensor* tensor = prim.Read(attribute);
                return tensor != nullptr ? ArrayFromTensor(*tensor)
                                         : nb::none();
            },
            nb::arg("attribute"))
        .def("__repr__", [](const Prim& prim) {
            return "Prim('" + prim.GetPath().String() + "')";
        });

    nb::class_<AttributeMapping>(
        m, "AttributeMapping",
        "An array mapped for writing in place (Scene.map); a context "
        "manager returning the "
        "array and unmapping on exit.")
        .def_ro("array", &AttributeMapping::array,
                "The writable array.")
        .def_prop_ro("path",
                     [](const AttributeMapping& mapping) {
                         return mapping.path;
                     })
        .def_prop_ro("attribute",
                     [](const AttributeMapping& mapping) {
                         return mapping.attribute;
                     })
        .def("__enter__",
             [](AttributeMapping& mapping) { return mapping.array; })
        .def(
            "__exit__",
            [](AttributeMapping& mapping, nb::handle, nb::handle,
               nb::handle) { mapping.Unmap({}); },
            nb::arg("type").none(), nb::arg("value").none(),
            nb::arg("traceback").none())
        .def(
            "unmap",
            [](AttributeMapping& mapping, nb::handle stream) {
                mapping.Unmap(stream);
            },
            nb::arg("stream").none() = nb::none(),
            "Commits the writes, on the GPU after the work queued on "
            "`stream` (the mapping's "
            "when None).");

    nb::class_<Scene>(
        m, "Scene",
        "A USD-shaped scene: typed prims named by Path holding "
        "attributes by USD "
        "name, with bulk data as DLPack arrays and change tracking.")
        .def(nb::init<>())
        .def("define", &Scene::Define, nb::arg("path"),
             nb::arg("type"), nb::keep_alive<0, 1>(),
             "Defines the prim at `path` with `type`, and any "
             "missing ancestors untyped.")
        .def(
            "connect",
            [](Scene& scene, const Path& path, std::string_view input,
               const Path& source, std::string_view output) {
                Connect(scene, path, input, source, output);
            },
            nb::arg("path"), nb::arg("input"), nb::arg("source"),
            nb::arg("output"),
            "Connects shader input `input` (e.g. "
            "tokens.DIFFUSE_COLOR) of the prim at `path` to "
            "output `output` (e.g. 'outputs:rgb') of the Shader at "
            "`source`, as UsdShade does: "
            "the attribute '<input>.connect' holding "
            "'<source>.<output>'.")
        .def(
            "connection",
            [](const Scene& scene, const Path& path,
               std::string_view input)
                -> std::optional<
                    std::pair<std::string, std::string>> {
                const std::optional<Connection> connection =
                    GetConnection(scene, path, input);
                if (!connection) {
                    return std::nullopt;
                }
                return std::pair(connection->prim.String(),
                                 connection->output);
            },
            nb::arg("path"), nb::arg("input"),
            "The (source prim path, output) shader input `input` of "
            "the prim at `path` connects "
            "to, or None.")
        .def("clone", &Scene::Clone, nb::arg("source"),
             nb::arg("destination"), nb::keep_alive<0, 1>(),
             "Copies `source` and its descendants to `destination`, "
             "sharing tensors and pointing relationships and shader "
             "connections inside `source` at their copies; returns "
             "the copy.")
        .def("remove", &Scene::Remove, nb::arg("path"))
        .def("has", &Scene::Has, nb::arg("path"))
        .def("__contains__", &Scene::Has, nb::arg("path"))
        .def(
            "get_prim",
            [](Scene& scene, const Path& path) -> nb::object {
                Prim prim = scene.GetPrim(path);
                return prim ? nb::cast(prim) : nb::none();
            },
            nb::arg("path"), nb::keep_alive<0, 1>())
        .def("type_of", &Scene::TypeOf, nb::arg("path"))
        .def("children", &Scene::Children, nb::arg("path"))
        .def("prims", &Scene::Prims, nb::arg("type") = "")
        .def(
            "set",
            [](Scene& scene, const Path& path,
               std::string_view attribute, nb::handle value) {
                scene.Set(path, attribute, ValueFromPython(value));
            },
            nb::arg("path"), nb::arg("attribute"),
            nb::arg("value").none())
        .def(
            "get",
            [](const Scene& scene, const Path& path,
               std::string_view attribute, nb::object fallback) {
                const Value* value = scene.Get(path, attribute);
                return value != nullptr ? ValueToPython(*value)
                                        : fallback;
            },
            nb::arg("path"), nb::arg("attribute"),
            nb::arg("default") = nb::none())
        .def(
            "write",
            [](Scene& scene, const Path& path,
               std::string_view attribute, const AnyArray& array,
               nb::handle stream) {
                scene.Write(path, attribute, TensorFromArray(array),
                            StreamOf(stream));
            },
            nb::arg("path"), nb::arg("attribute"), nb::arg("array"),
            nb::arg("stream").none() = nb::none(),
            "Writes any DLPack array (NumPy, PyTorch, JAX, Warp) as "
            "`attribute` without "
            "copying. For a CUDA array, `stream` (an int, a "
            "torch.cuda.Stream, or None for the "
            "legacy default stream) is the one its contents are "
            "written on.")
        .def(
            "map",
            [](nb::handle self, const Path& path,
               std::string_view attribute, nb::handle stream) {
                Scene& scene = nb::cast<Scene&>(self);
                AttributeMapping mapping;
                mapping.scene = nb::borrow(self);
                mapping.path = path;
                mapping.attribute = std::string(attribute);
                mapping.stream = nb::borrow(stream);
                mapping.array =
                    ArrayFromTensor(scene.Map(path, attribute), true);
                return mapping;
            },
            nb::arg("path"), nb::arg("attribute"),
            nb::arg("stream").none() = nb::none(),
            "Maps the array written as `attribute` for writing in "
            "place: `with scene.map(...) "
            "as array:` writes into the array (on its device) and "
            "unmaps on exit, with `stream` "
            "the CUDA stream the writes are queued on.")
        .def(
            "unmap",
            [](Scene& scene, const Path& path,
               std::string_view attribute, nb::handle stream) {
                scene.Unmap(path, attribute, StreamOf(stream));
            },
            nb::arg("path"), nb::arg("attribute"),
            nb::arg("stream").none() = nb::none(),
            "Commits writes into a mapped array: renderers read it "
            "again, on the GPU after the "
            "work queued on `stream`.")
        .def(
            "unmap",
            [](Scene& scene, const PrimSet& prims,
               std::string_view attribute, nb::handle stream) {
                scene.Unmap(prims, attribute, StreamOf(stream));
            },
            nb::arg("prims"), nb::arg("attribute"),
            nb::arg("stream").none() = nb::none(),
            "As the unmap of one path, for each prim of a PrimSet: "
            "rows of one array mapped over "
            "them (`Renderer.map` of a PrimSet).")
        .def(
            "unmap",
            [](Scene& scene, const std::vector<Path>& paths,
               std::string_view attribute, nb::handle stream) {
                scene.Unmap(PrimSet(paths), attribute,
                            StreamOf(stream));
            },
            nb::arg("paths"), nb::arg("attribute"),
            nb::arg("stream").none() = nb::none(),
            "As the unmap of a PrimSet, for a list of paths.")
        .def(
            "write",
            [](Scene& scene, const std::vector<Path>& paths,
               std::string_view attribute, const AnyArray& rows,
               nb::handle stream) {
                scene.Write(paths, attribute, TensorFromArray(rows),
                            StreamOf(stream));
            },
            nb::arg("paths"), nb::arg("attribute"), nb::arg("rows"),
            nb::arg("stream").none() = nb::none(),
            "Writes row i of `rows` as `attribute` of `paths[i]`, "
            "sharing one array. Transform "
            "ops written so from CUDA arrays (float32 rows of "
            "xformOp:translate, xformOp:orient as "
            "x, y, z, w, and xformOp:scale) pose the prims on the "
            "GPU: the meshes under them "
            "render without the poses coming to the host (give each "
            "prim ernie:xform:extent, or "
            "the poses are copied to the host to bound them). "
            "`stream` is as for `write`.")
        .def(
            "write",
            [](Scene& scene, const PrimSet& prims,
               std::string_view attribute, const AnyArray& rows,
               nb::handle stream) {
                scene.Write(prims, attribute, TensorFromArray(rows),
                            StreamOf(stream));
            },
            nb::arg("prims"), nb::arg("attribute"), nb::arg("rows"),
            nb::arg("stream").none() = nb::none(),
            "As the write of many paths, to a PrimSet: its prims are "
            "looked up once and kept, the "
            "way to write a simulation's arrays every step.")
        .def(
            "read",
            [](const Scene& scene, const Path& path,
               std::string_view attribute) -> nb::object {
                const Tensor* tensor = scene.Read(path, attribute);
                return tensor != nullptr ? ArrayFromTensor(*tensor)
                                         : nb::none();
            },
            nb::arg("path"), nb::arg("attribute"),
            "A read-only view of the array written as `attribute`, "
            "or None.")
        .def("attributes", &Scene::Attributes, nb::arg("path"))
        .def("clear", &Scene::Clear, nb::arg("path"),
             nb::arg("attribute"))
        .def("attribute_version", &Scene::AttributeVersion,
             nb::arg("path"), nb::arg("attribute"))
        .def(
            "local_transform",
            [](const Scene& scene, const Path& path) {
                return MatrixToPython(scene.LocalTransform(path));
            },
            nb::arg("path"))
        .def(
            "world_transform",
            [](const Scene& scene, const Path& path) {
                return MatrixToPython(scene.WorldTransform(path));
            },
            nb::arg("path"))
        .def(
            "set_local_transform",
            [](Scene& scene, const Path& path, nb::handle matrix) {
                scene.SetLocalTransform(path,
                                        TransformFromPython(matrix));
            },
            nb::arg("path"), nb::arg("matrix"),
            "Gives the prim the local transform `matrix` (4 x 4), "
            "through its own transform ops "
            "(translate, a rotate op or orient, scale) where they "
            "can represent it, else as "
            "xformOp:transform.")
        .def("is_visible", &Scene::IsVisible, nb::arg("path"))
        .def("overlay", &Scene::GetOverlay, nb::arg("name"),
             nb::rv_policy::reference_internal,
             "The overlay (debug drawing layer) named `name`, "
             "created empty if missing.")
        .def("overlay_names", &Scene::OverlayNames)
        .def("remove_overlay", &Scene::RemoveOverlay, nb::arg("name"))
        .def_prop_ro("version", &Scene::Version)
        .def(
            "changes",
            [](const Scene& scene, ChangeCursor& cursor) {
                std::vector<std::pair<Path, Dirty>> result;
                for (const Change& change : scene.Changes(cursor)) {
                    result.emplace_back(change.path, change.bits);
                }
                return result;
            },
            nb::arg("cursor"),
            "(path, Dirty) for every prim changed since `cursor`, "
            "which advances.");

    const auto triple = [](const glm::dvec3& v) {
        return nb::make_tuple(v.x, v.y, v.z);
    };
    nb::class_<IsdLineScanner>(
        m, "IsdLineScanner",
        "A line scanner (pushbroom camera) from its image support "
        "data (ALE's ISD JSON, as "
        "usgscsm reads it): its timing, detector and trajectory in "
        "the body-fixed frame, metres. "
        "Image coordinates are CSM's (line, sample), the first "
        "pixel's centre at (0.5, 0.5).")
        .def_static(
            "from_file",
            [](const std::string& path) {
                return IsdLineScanner(ReadIsdFile(path));
            },
            nb::arg("path"),
            "The line scanner of the ISD file at `path`.")
        .def_static(
            "from_json",
            [](const std::string& json) {
                return IsdLineScanner(ReadIsd(json));
            },
            nb::arg("json"),
            "The line scanner of ISD JSON text `json`.")
        .def_prop_ro(
            "lines",
            [](const IsdLineScanner& s) { return s.Isd().lines; })
        .def_prop_ro(
            "samples",
            [](const IsdLineScanner& s) { return s.Isd().samples; })
        .def_prop_ro(
            "sensor",
            [](const IsdLineScanner& s) { return s.Isd().sensor; })
        .def_prop_ro(
            "platform",
            [](const IsdLineScanner& s) { return s.Isd().platform; })
        .def("image_time", &IsdLineScanner::ImageTime,
             nb::arg("line"),
             "Ephemeris seconds (TDB past J2000) at which `line` is "
             "exposed.")
        .def(
            "position",
            [triple](const IsdLineScanner& s, double time) {
                return triple(s.Position(time));
            },
            nb::arg("time"),
            "The sensor's body-fixed position at `time`.")
        .def(
            "sun_position",
            [triple](const IsdLineScanner& s, double time) {
                return triple(s.SunPosition(time));
            },
            nb::arg("time"),
            "The sun's body-fixed position at `time`.")
        .def(
            "image_to_ground",
            [triple](const IsdLineScanner& s, double line,
                     double sample, double height) {
                return triple(s.ImageToGround(line, sample, height));
            },
            nb::arg("line"), nb::arg("sample"),
            nb::arg("height") = 0.0,
            "Where pixel (`line`, `sample`)'s ray meets the body's "
            "ellipsoid raised by `height`.")
        .def(
            "ground_to_image",
            [](const IsdLineScanner& s, nb::handle ground) {
                const glm::dvec2 image =
                    s.GroundToImage(Vec3FromPython(ground));
                return nb::make_tuple(image.x, image.y);
            },
            nb::arg("ground"),
            "The (line, sample) that sees body-fixed point `ground`.")
        .def(
            "strip",
            [](const IsdLineScanner& s, std::uint32_t first,
               std::uint32_t lines) {
                const IsdLineScanner::Strip strip =
                    s.StripOf(first, lines);
                return nb::make_tuple(MatrixToPython(strip.from),
                                      MatrixToPython(strip.to),
                                      strip.fromTime, strip.toTime);
            },
            nb::arg("first_line"), nb::arg("lines"),
            "(from, to, from_time, to_time): the camera poses and "
            "times bounding the strip of "
            "`lines` lines from `first_line`.")
        .def(
            "author_strip",
            [](const IsdLineScanner& s, Scene& scene,
               const PrimPath& pathArg, std::uint32_t first,
               std::uint32_t lines, bool atEnd) {
                const Path path = pathArg;
                const IsdLineScanner::Strip strip =
                    s.StripOf(first, lines);
                CameraParams camera = strip.camera;
                // The camera's geometry and line time only: its
                // f-number, ISO, lens transmission and sensor stay as
                // authored (by a demo's instrument, say).
                camera.transform = atEnd ? strip.to : strip.from;
                AuthorIsdCamera(scene, path, camera);
                return nb::make_tuple(camera.width, camera.height);
            },
            nb::arg("scene"), nb::arg("path"), nb::arg("first_line"),
            nb::arg("lines"), nb::arg("at_end") = true,
            "Authors the Camera prim at `path` (a str, Path, Prim or "
            "typed prim) as the pushbroom "
            "of the strip of `lines` lines from `first_line` (with "
            "the lens's distortion along the "
            "detector fitted as "
            "`ernie:lens:distortion`), posed at its end (or its "
            "start), and returns the render "
            "resolution it needs, (samples, lines). Render an image "
            "strip by strip: author the "
            "first strip at its start and step once, then author "
            "each strip at its end and "
            "step.");

    nb::class_<IsdPushFrame>(
        m, "IsdPushFrame",
        "A push-frame camera (LROC's WAC, MRO's MARCI) from its "
        "image support data, as usgscsm's "
        "push-frame model reads it: the image is a stack of "
        "framelets, each a frame of "
        "`framelet_lines` lines exposed at once. Body-fixed frame, "
        "metres; CSM image "
        "coordinates.")
        .def_static(
            "from_file",
            [](const std::string& path) {
                return IsdPushFrame(ReadIsdFile(path));
            },
            nb::arg("path"),
            "The push-frame camera of the ISD file at `path`.")
        .def_static(
            "from_json",
            [](const std::string& json) {
                return IsdPushFrame(ReadIsd(json));
            },
            nb::arg("json"),
            "The push-frame camera of ISD JSON text `json`.")
        .def_prop_ro(
            "lines",
            [](const IsdPushFrame& s) { return s.Isd().lines; })
        .def_prop_ro(
            "samples",
            [](const IsdPushFrame& s) { return s.Isd().samples; })
        .def_prop_ro(
            "sensor",
            [](const IsdPushFrame& s) { return s.Isd().sensor; })
        .def_prop_ro(
            "platform",
            [](const IsdPushFrame& s) { return s.Isd().platform; })
        .def_prop_ro("framelet_lines", &IsdPushFrame::FrameletLines)
        .def_prop_ro("framelets", &IsdPushFrame::Framelets)
        .def("image_time", &IsdPushFrame::ImageTime, nb::arg("line"),
             "Ephemeris seconds (TDB past J2000) at the middle of "
             "`line`'s framelet's exposure.")
        .def(
            "position",
            [triple](const IsdPushFrame& s, double time) {
                return triple(s.Position(time));
            },
            nb::arg("time"),
            "The sensor's body-fixed position at `time`.")
        .def(
            "sun_position",
            [triple](const IsdPushFrame& s, double time) {
                return triple(s.SunPosition(time));
            },
            nb::arg("time"),
            "The sun's body-fixed position at `time`.")
        .def(
            "image_to_ground",
            [triple](const IsdPushFrame& s, double line,
                     double sample, double height) {
                return triple(s.ImageToGround(line, sample, height));
            },
            nb::arg("line"), nb::arg("sample"),
            nb::arg("height") = 0.0,
            "Where pixel (`line`, `sample`)'s ray meets the body's "
            "ellipsoid raised by `height`.")
        .def(
            "ground_to_image",
            [](const IsdPushFrame& s, nb::handle ground) {
                const glm::dvec2 image =
                    s.GroundToImage(Vec3FromPython(ground));
                return nb::make_tuple(image.x, image.y);
            },
            nb::arg("ground"),
            "The (line, sample) that sees body-fixed point `ground`.")
        .def(
            "author_framelet",
            [](const IsdPushFrame& s, Scene& scene,
               const PrimPath& pathArg, std::uint32_t index) {
                const Path path = pathArg;
                const IsdPushFrame::Framelet framelet =
                    s.FrameletOf(index);
                const CameraParams& camera = framelet.camera;
                // The camera's geometry and exposure only.
                AuthorIsdCamera(scene, path, camera);
                return nb::make_tuple(
                    camera.width, camera.height, framelet.firstLine,
                    framelet.rowsReversed, framelet.time);
            },
            nb::arg("scene"), nb::arg("path"), nb::arg("framelet"),
            "Authors the Camera prim at `path` (a str, Path, Prim or "
            "typed prim) as framelet "
            "`framelet` (in image order, from the top): a frame "
            "camera posed when it is exposed, "
            "with the lens's distortion fitted as "
            "`ernie:lens:distortion`. Returns (samples, lines, "
            "first_line, rows_reversed, time): "
            "render at that resolution, flip the rows if "
            "`rows_reversed`, and place them at "
            "`first_line` of the image.");

    nb::class_<CylindricalDem>(
        m, "CylindricalDem",
        "A planet's radius on a simple cylindrical grid, as LOLA's "
        "LDEM products hold it: heights in metres above "
        "reference_radius, rows from the northern, each from the "
        "western sample eastwards. Made empty, it is the bare "
        "reference sphere of `reference_radius`.")
        .def(
            "__init__",
            [](CylindricalDem* dem, double referenceRadius) {
                new (dem) CylindricalDem();
                dem->referenceRadius = referenceRadius;
            },
            nb::arg("reference_radius"))
        .def_ro("lines", &CylindricalDem::lines)
        .def_ro("samples", &CylindricalDem::samples)
        .def_ro("pixels_per_degree", &CylindricalDem::pixelsPerDegree)
        .def_ro("max_latitude", &CylindricalDem::maxLatitudeDeg)
        .def_ro("west_longitude", &CylindricalDem::westLongitudeDeg)
        .def_ro("standard_parallel",
                &CylindricalDem::standardParallelDeg)
        .def_ro("reference_radius", &CylindricalDem::referenceRadius)
        .def("height_at", &CylindricalDem::HeightAt,
             nb::arg("latitude"), nb::arg("longitude"),
             "The height (m) at planetocentric latitude and east "
             "longitude (degrees), bilinear between cell centres.");
    m.def("read_cylindrical_dem", &ReadCylindricalDem,
          nb::arg("label_path"),
          "Reads the DEM a PDS3 label describes (LOLA's LDEM "
          "products: 16- or 32-bit integers or 32-bit reals on a "
          "simple cylindrical grid).");
    nb::class_<PolarStereographicDem>(
        m, "PolarStereographicDem",
        "A planet's radius on a polar stereographic grid (spherical, "
        "true at the pole), as LOLA's polar LDEM products hold it: "
        "heights in metres above reference_radius, map_scale metres "
        "a "
        "pixel. Made empty, it is the bare reference sphere of "
        "`reference_radius`.")
        .def(
            "__init__",
            [](PolarStereographicDem* dem, double referenceRadius) {
                new (dem) PolarStereographicDem();
                dem->referenceRadius = referenceRadius;
            },
            nb::arg("reference_radius"))
        .def_ro("lines", &PolarStereographicDem::lines)
        .def_ro("samples", &PolarStereographicDem::samples)
        .def_ro("map_scale", &PolarStereographicDem::mapScale)
        .def_ro("center_latitude",
                &PolarStereographicDem::centerLatitudeDeg)
        .def_ro("center_longitude",
                &PolarStereographicDem::centerLongitudeDeg)
        .def_ro("reference_radius",
                &PolarStereographicDem::referenceRadius)
        .def(
            "to_grid",
            [](const PolarStereographicDem& dem, double latitude,
               double longitude) {
                const glm::dvec2 at = dem.ToGrid(latitude, longitude);
                return nb::make_tuple(at.x, at.y);
            },
            nb::arg("latitude"), nb::arg("longitude"),
            "The fractional (line, sample) of a latitude and east "
            "longitude (degrees), cell centres at whole numbers.")
        .def("height_at", &PolarStereographicDem::HeightAt,
             nb::arg("latitude"), nb::arg("longitude"),
             "The height (m) at planetocentric latitude and east "
             "longitude (degrees), bilinear between cell centres.");
    m.def("read_polar_stereographic_dem", &ReadPolarStereographicDem,
          nb::arg("label_path"),
          "Reads the DEM a PDS3 label describes on a polar "
          "stereographic grid (LOLA's polar LDEM products).");
    const auto localTerrain = [](const LocalTerrain& terrain) {
        nb::dict result;
        result["heights"] = ArrayFromTensor(terrain.heights);
        result["origin"] =
            nb::make_tuple(terrain.origin.x, terrain.origin.y);
        result["cell_size"] = terrain.cellSize;
        result["body_to_local"] = MatrixToPython(terrain.bodyToLocal);
        return result;
    };
    // One function for either grid, not overloads: the API docs
    // (mkdocstrings) cannot resolve a stub function that is only
    // overloads.
    m.def(
        "local_terrain_of",
        [localTerrain](std::variant<const PolarStereographicDem*,
                                    const CylindricalDem*>
                           dem,
                       double latitude, double longitude,
                       std::uint32_t cells, double cellSize,
                       const CylindricalDem* fallback) {
            if (fallback != nullptr) {
                const auto* const* grid =
                    std::get_if<const CylindricalDem*>(&dem);
                if (grid == nullptr) {
                    throw std::invalid_argument(
                        "A fallback DEM backs a cylindrical DEM "
                        "only");
                }
                return localTerrain(
                    LocalTerrainOf(**grid, *fallback, latitude,
                                   longitude, cells, cellSize));
            }
            return std::visit(
                [&](const auto* grid) {
                    return localTerrain(LocalTerrainOf(
                        *grid, latitude, longitude, cells, cellSize));
                },
                dem);
        },
        nb::arg("dem"), nb::arg("latitude"), nb::arg("longitude"),
        nb::arg("cells"), nb::arg("cell_size"),
        nb::arg("fallback").none() = nb::none(),
        "The cells x cells heightfield of `dem`, `cell_size` m "
        "apart, "
        "about a latitude and east longitude (degrees), in a local "
        "frame there (x east, y north, z up, its origin on the "
        "reference sphere), the planet's curvature in the heights: a "
        "dict of heights (float32, the first row northern), origin "
        "(x, y of the north-western cell), cell_size and "
        "body_to_local (body-fixed metres to the local frame). Where "
        "`dem` has no data, `fallback`'s heights (a cylindrical DEM "
        "about a high-resolution DTM), else the reference sphere.");

    nb::class_<CahvoreModel>(
        m, "CahvoreModel",
        "A camera in JPL's CAHV, CAHVOR or CAHVORE model, as Mars "
        "rovers' cameras are calibrated. "
        "Image coordinates are the models': (0, 0) the centre of the "
        "upper-left pixel, x right, "
        "y down.")
        .def_prop_ro("kind",
                     [](const CahvoreModel& model) {
                         return model.kind == CahvoreModel::Kind::Cahv
                                    ? "cahv"
                                : model.kind ==
                                        CahvoreModel::Kind::Cahvor
                                    ? "cahvor"
                                    : "cahvore";
                     })
        .def_prop_ro(
            "width",
            [](const CahvoreModel& model) { return model.width; })
        .def_prop_ro(
            "height",
            [](const CahvoreModel& model) { return model.height; })
        .def(
            "image_to_ray",
            [triple](const CahvoreModel& model, double x, double y) {
                const CahvoreModel::Ray ray =
                    model.ImageToRay({x, y});
                return nb::make_tuple(triple(ray.origin),
                                      triple(ray.direction));
            },
            nb::arg("x"), nb::arg("y"),
            "(origin, unit direction) of the ray through image (x, "
            "y).")
        .def(
            "point_to_image",
            [](const CahvoreModel& model, nb::handle point) {
                const glm::dvec2 image =
                    model.PointToImage(Vec3FromPython(point));
                return nb::make_tuple(image.x, image.y);
            },
            nb::arg("point"), "The image (x, y) of 3D point `point`.")
        .def(
            "fit",
            [](const CahvoreModel& model, double pixelPitchUm) {
                const CahvoreModel::ErnieCamera fit =
                    model.ToCamera(pixelPitchUm);
                nb::dict out;
                out["residual_pixels"] = fit.residualPixels;
                out["pupil_shift_max"] = fit.pupilShiftMax;
                out["field_angle_max"] = fit.fieldAngleMax;
                out["focal_length_mm"] = fit.camera.focalLengthMm;
                out["fisheye"] = fit.camera.lens.projection ==
                                 LensProjection::Fisheye;
                out["rows_reversed"] = fit.rowsReversed;
                return out;
            },
            nb::arg("pixel_pitch_um"),
            "How the model converts to Ernie's camera at pixel pitch "
            "`pixel_pitch_um`: its "
            "residual_pixels, pupil_shift_max, field_angle_max "
            "(radians), focal_length_mm and "
            "whether it is a fisheye.")
        .def(
            "author_camera",
            [](const CahvoreModel& model, Scene& scene,
               const PrimPath& pathArg, double pixelPitchUm,
               std::uint32_t raysStep) {
                const Path path = pathArg;
                const CahvoreModel::ErnieCamera fit =
                    model.ToCamera(pixelPitchUm);
                const CameraParams& camera = fit.camera;
                using namespace tokens;
                // The camera's geometry only, as
                // `IsdFrame.author_camera`; its pose in the model's
                // frame (parent the prim to that frame's Xform).
                Prim prim = scene.Has(Path(path))
                                ? scene.GetPrim(Path(path))
                                : scene.Define(Path(path), kCamera);
                if (raysStep > 0) {
                    // The model's own rays, which the sensor
                    // interpolates in place of the fit.
                    const LensRayTable table =
                        model.RayTable(raysStep);
                    std::vector<float> values;
                    values.reserve(table.rays.size() * 3);
                    for (const glm::vec3& ray : table.rays) {
                        values.insert(values.end(),
                                      {ray.x, ray.y, ray.z});
                    }
                    scene.Write(
                        Path(path), kLensRays,
                        Tensor::Copy(std::span<const float>(values),
                                     {std::int64_t(table.rows),
                                      std::int64_t(table.columns),
                                      3}));
                    scene.Set(Path(path), kLensRaysStep,
                              static_cast<int>(raysStep));
                }
                const bool fisheye =
                    camera.lens.projection == LensProjection::Fisheye;
                prim.Set(kTransform, camera.transform)
                    .Set(kCameraScan, std::string(kFrameScan))
                    .Set(kLensProjection,
                         std::string(fisheye ? kFisheye
                                             : kPerspective))
                    .Set(kFocalLength, camera.focalLengthMm)
                    .Set(kVerticalAperture, camera.verticalApertureMm)
                    .Set(kHorizontalAperture,
                         camera.verticalApertureMm * camera.width /
                             camera.height)
                    .Set(kHorizontalApertureOffset,
                         camera.horizontalApertureOffsetMm)
                    .Set(kVerticalApertureOffset,
                         camera.verticalApertureOffsetMm)
                    .Set(kLensDistortion,
                         std::vector<float>{
                             static_cast<float>(camera.lens.k1),
                             static_cast<float>(camera.lens.k2),
                             static_cast<float>(camera.lens.k3),
                             static_cast<float>(camera.lens.p1),
                             static_cast<float>(camera.lens.p2)})
                    .Set(kLensFisheye,
                         std::vector<float>{
                             static_cast<float>(
                                 camera.lens.fisheye[0]),
                             static_cast<float>(
                                 camera.lens.fisheye[1]),
                             static_cast<float>(
                                 camera.lens.fisheye[2]),
                             static_cast<float>(
                                 camera.lens.fisheye[3])})
                    .Set(kClippingRange, glm::dvec2(camera.nearPlane,
                                                    camera.farPlane));
                return nb::make_tuple(camera.width, camera.height,
                                      fit.rowsReversed,
                                      fit.residualPixels);
            },
            nb::arg("scene"), nb::arg("path"),
            nb::arg("pixel_pitch_um"), nb::arg("rays_step") = 8,
            "Authors the Camera prim at `path` (a str, Path, Prim or "
            "typed prim) as the model in "
            "Ernie's camera (perspective with "
            "Brown-Conrady terms, or a Kannala-Brandt fisheye for "
            "CAHVORE), posed in the model's "
            "frame, and with `rays_step` above 0 the model's own "
            "rays every that many pixels "
            "(`ernie:lens:rays`), which the sensor follows in place "
            "of the fit: exact but for "
            "CAHVORE's moving entrance pupil. Returns (width, "
            "height, rows_reversed, "
            "residual_pixels), the residual the fit's alone: render "
            "at that resolution and flip "
            "the rows if `rows_reversed`.");
    m.def(
        "read_cahvore_models",
        [](const std::string& path) {
            nb::list out;
            for (NamedCahvore& named : ReadCahvoreModelsFile(path)) {
                out.append(nb::make_tuple(named.name, named.frame,
                                          named.model));
            }
            return out;
        },
        nb::arg("path"),
        "(name, frame, CahvoreModel) for each model of the camera "
        "model JSON file at `path`, "
        "in NASA-JPL's m2020-urdf-models form.");

    nb::class_<IsdFrame>(
        m, "IsdFrame",
        "A frame camera from its image support data, as usgscsm's "
        "frame model reads it: the "
        "whole image exposed at once from one pose. Body-fixed "
        "frame, metres; CSM image "
        "coordinates.")
        .def_static(
            "from_file",
            [](const std::string& path) {
                return IsdFrame(ReadIsdFile(path));
            },
            nb::arg("path"),
            "The frame camera of the ISD file at `path`.")
        .def_static(
            "from_json",
            [](const std::string& json) {
                return IsdFrame(ReadIsd(json));
            },
            nb::arg("json"),
            "The frame camera of ISD JSON text `json`.")
        .def_prop_ro("lines",
                     [](const IsdFrame& s) { return s.Isd().lines; })
        .def_prop_ro(
            "samples",
            [](const IsdFrame& s) { return s.Isd().samples; })
        .def_prop_ro("sensor",
                     [](const IsdFrame& s) { return s.Isd().sensor; })
        .def_prop_ro(
            "platform",
            [](const IsdFrame& s) { return s.Isd().platform; })
        .def_prop_ro("image_time", &IsdFrame::ImageTime,
                     "Ephemeris seconds (TDB past J2000) at which "
                     "the image is exposed.")
        .def(
            "position",
            [triple](const IsdFrame& s, double time) {
                return triple(s.Position(time));
            },
            nb::arg("time"),
            "The sensor's body-fixed position at `time`.")
        .def(
            "sun_position",
            [triple](const IsdFrame& s, double time) {
                return triple(s.SunPosition(time));
            },
            nb::arg("time"),
            "The sun's body-fixed position at `time`.")
        .def(
            "image_to_ground",
            [triple](const IsdFrame& s, double line, double sample,
                     double height) {
                return triple(s.ImageToGround(line, sample, height));
            },
            nb::arg("line"), nb::arg("sample"),
            nb::arg("height") = 0.0,
            "Where pixel (`line`, `sample`)'s ray first meets the "
            "body's ellipsoid raised by "
            "`height`.")
        .def(
            "ground_to_image",
            [](const IsdFrame& s, nb::handle ground) {
                const glm::dvec2 image =
                    s.GroundToImage(Vec3FromPython(ground));
                return nb::make_tuple(image.x, image.y);
            },
            nb::arg("ground"),
            "The (line, sample) that sees body-fixed point `ground`.")
        .def(
            "jitter",
            [](const IsdFrame& s, double line) {
                const glm::dvec2 jitter = s.Jitter(line);
                return nb::make_tuple(jitter.x, jitter.y);
            },
            nb::arg("line"),
            "The rolling shutter's jitter (line, sample) in pixels "
            "at image line `line`, as "
            "usgscsm's `removeJitter` takes it; (0, 0) when the ISD "
            "has no `jitter`.")
        .def(
            "author_camera",
            [](const IsdFrame& s, Scene& scene,
               const PrimPath& pathArg) {
                const Path path = pathArg;
                const IsdFrame::FrameView view = s.View();
                const CameraParams& camera = view.camera;
                using namespace tokens;
                // The camera's geometry only, as
                // `IsdLineScanner.author_strip`.
                Prim prim = scene.Has(Path(path))
                                ? scene.GetPrim(Path(path))
                                : scene.Define(Path(path), kCamera);
                prim.Set(kTransform, camera.transform)
                    .Set(kCameraScan, std::string(kFrameScan))
                    .Set(kFocalLength, camera.focalLengthMm)
                    .Set(kVerticalAperture, camera.verticalApertureMm)
                    .Set(kHorizontalAperture,
                         camera.verticalApertureMm * camera.width /
                             camera.height)
                    .Set(kHorizontalApertureOffset,
                         camera.horizontalApertureOffsetMm)
                    .Set(kVerticalApertureOffset,
                         camera.verticalApertureOffsetMm)
                    .Set(kPushbroomLineOffset, 0.0)
                    .Set(kLensDistortion,
                         std::vector<float>{
                             static_cast<float>(camera.lens.k1),
                             static_cast<float>(camera.lens.k2),
                             static_cast<float>(camera.lens.k3),
                             static_cast<float>(camera.lens.p1),
                             static_cast<float>(camera.lens.p2)})
                    .Set(kClippingRange, glm::dvec2(camera.nearPlane,
                                                    camera.farPlane));
                return nb::make_tuple(camera.width, camera.height,
                                      view.rowsReversed);
            },
            nb::arg("scene"), nb::arg("path"),
            "Authors the Camera prim at `path` (a str, Path, Prim or "
            "typed prim) as the image's "
            "frame camera, posed when it is "
            "exposed, with the lens's distortion fitted as "
            "`ernie:lens:distortion`. Returns "
            "(samples, lines, rows_reversed): render at that "
            "resolution and flip the rows if "
            "`rows_reversed`.");

    m.def(
        "author_singlet_chromatic",
        [](Scene& scene, const PrimPath& camera, double nd,
           double abbe, double lateral) {
            using namespace tokens;
            const Path path = camera;
            if (!scene.Has(path) || scene.TypeOf(path) != kCamera) {
                throw nb::value_error(
                    ("'" + path.String() + "' is not a Camera")
                        .c_str());
            }
            LensModel lens;
            try {
                SetSingletChromatic(
                    lens, Camera(scene, path).GetFocalLength(), nd,
                    abbe, lateral);
            } catch (const std::invalid_argument& error) {
                throw nb::value_error(error.what());
            }
            const auto pairs = [](const Spectrum& spectrum) {
                std::vector<float> out;
                for (double nm = 300.0; nm <= 1100.0 + 1e-9;
                     nm += 10.0) {
                    out.push_back(static_cast<float>(nm));
                    out.push_back(static_cast<float>(spectrum(nm)));
                }
                return out;
            };
            scene.GetPrim(path)
                .Set(kLensChromaticScale, pairs(lens.chromaticScale))
                .Set(kLensChromaticFocus,
                     pairs(lens.chromaticFocusMm));
        },
        nb::arg("scene"), nb::arg("camera"), nb::arg("nd") = 1.5168,
        nb::arg("abbe") = 64.17, nb::arg("lateral") = 0.0,
        "Authors Camera `camera`'s chromatic aberration "
        "(`ernie:lens:chromatic:scale` and "
        "`:focus`) as a thin singlet of its focal length in a glass "
        "of index `nd` and Abbe number "
        "`abbe` (by default SCHOTT N-BK7) would have: axial colour "
        "from the glass's dispersion "
        "(Cauchy, f_C - f_F = f / V), lateral colour `lateral` times "
        "its relative focal change "
        "(0 with the stop at the lens, up to 1 for a distant stop). "
        "The `defocus` filter blurs "
        "by the axial colour; the lens resamples each band by the "
        "lateral. `camera` is a str, "
        "Path, Prim or typed prim.");
    m.def(
        "look_at_transform",
        [](nb::handle eye, nb::handle target, nb::handle up) {
            return MatrixToPython(LookAtTransform(
                Vec3FromPython(eye), Vec3FromPython(target),
                Vec3FromPython(up)));
        },
        nb::arg("eye"), nb::arg("target"),
        nb::arg("up") = nb::make_tuple(0.0, 0.0, 1.0),
        "Transform of a Camera at `eye` looking at `target`.");
    m.def(
        "distant_light_transform",
        [](nb::handle towards) {
            return MatrixToPython(
                DistantLightTransform(Vec3FromPython(towards)));
        },
        nb::arg("towards_light"),
        "Transform of a DistantLight shining from `towards_light`.");
    m.def(
        "load_image",
        [](const std::filesystem::path& path, bool srgb) {
            const Image image =
                LoadImageFile(path, srgb ? ImageEncoding::Srgb
                                         : ImageEncoding::Linear);
            const auto* values =
                reinterpret_cast<const float*>(image.pixels.data());
            return ArrayFromTensor(
                Tensor::Copy<float>(
                    std::span(values, image.pixels.size() * 4),
                    {static_cast<std::int64_t>(image.height),
                     static_cast<std::int64_t>(image.width), 4}),
                true);
        },
        nb::arg("path"), nb::arg("srgb") = true,
        "A PNG, JPEG or OpenEXR image as a linear float32 (rows, "
        "cols, 4) array: PNG and JPEG "
        "values decode as sRGB (srgb) or as linear data such as "
        "normal maps; EXR is linear.");
    m.def(
        "noise_permutation",
        [](std::uint32_t seed) {
            const auto table = GradientNoise(seed).Permutation();
            return ArrayFromTensor(Tensor::Copy<std::uint8_t>(
                std::span(table.data(), table.size()),
                {static_cast<std::int64_t>(table.size())}));
        },
        nb::sig("def noise_permutation(seed: int = 0) -> "
                "NDArray[numpy.uint8]"),
        nb::arg("seed") = 0,
        "The lattice permutation of `fbm_noise`'s noise for `seed`: "
        "256 entries twice over, uint8, for kernels on the GPU to "
        "evaluate the same noise.");
    m.def(
        "fbm_noise",
        [](nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig,
                       nb::device::cpu>
               points,
           int octaves, double lacunarity, double gain,
           std::uint32_t seed) {
            const GradientNoise noise(seed);
            const auto count =
                static_cast<std::int64_t>(points.shape(0));
            Tensor values = Tensor::Zeros(DType::Float64, {count});
            double* out = values.As<double>();
            for (std::int64_t i = 0; i < count; ++i) {
                const double* p = points.data() + 3 * i;
                out[i] = noise.Fbm(glm::dvec3(p[0], p[1], p[2]),
                                   octaves, lacunarity, gain);
            }
            return ArrayFromTensor(values);
        },
        nb::sig("def fbm_noise(points: NDArray[numpy.float64], "
                "octaves: int = 5, lacunarity: float = 2.0, "
                "gain: float = 0.5, seed: int = 0) -> "
                "NDArray[numpy.float64]"),
        nb::arg("points"), nb::arg("octaves") = 5,
        nb::arg("lacunarity") = 2.0, nb::arg("gain") = 0.5,
        nb::arg("seed") = 0,
        "Perlin's improved gradient noise summed over octaves "
        "(fractional Brownian motion) at each of the float64 (n, 3) "
        "points: as C++ GradientNoise::Fbm, within about [-1, 1].");
    m.def(
        "dense_nanovdb_grid",
        [](nb::ndarray<const float, nb::ndim<3>, nb::c_contig,
                       nb::device::cpu>
               values,
           double voxelSize, std::array<double, 3> origin,
           const std::string& name) {
            const std::vector<std::uint8_t> bytes = DenseNanoVdbGrid(
                std::span(values.data(), values.size()),
                glm::ivec3(static_cast<int>(values.shape(0)),
                           static_cast<int>(values.shape(1)),
                           static_cast<int>(values.shape(2))),
                voxelSize,
                glm::dvec3(origin[0], origin[1], origin[2]), name);
            return ArrayFromTensor(Tensor::Copy<std::uint8_t>(
                bytes, {static_cast<std::int64_t>(bytes.size())}));
        },
        nb::sig(
            "def dense_nanovdb_grid(values: NDArray[numpy.float32], "
            "voxel_size: float, origin: tuple[float, float, float] = "
            "(0.0, 0.0, 0.0), name: str = '') -> "
            "NDArray[numpy.uint8]"),
        nb::arg("values"), nb::arg("voxel_size"),
        nb::arg("origin") = std::array<double, 3>{0.0, 0.0, 0.0},
        nb::arg("name") = "",
        "A NanoVDB float grid of a dense float32 (x, y, z) array, "
        "active where not 0, as uint8 bytes for a NanoVDBField's "
        "buffer: index (i, j, k) at origin + voxel_size (i, j, k).");
    m.def("julian_date", &JulianDate, nb::arg("year"),
          nb::arg("month"), nb::arg("day"), nb::arg("hours") = 0.0,
          "The Julian date of `hours` UT past midnight of a "
          "Gregorian date (Meeus, chapter 7).");
    m.def("greenwich_sidereal_time", &GreenwichMeanSiderealTime,
          nb::arg("julian_date"),
          "Greenwich mean sidereal time in degrees at a Julian date "
          "(Meeus, equation 12.4).");
    m.def(
        "celestial_to_local",
        [](double latitude, double longitude, double julianDate) {
            return MatrixToPython(
                CelestialToLocal(latitude, longitude, julianDate));
        },
        nb::arg("latitude"), nb::arg("longitude"),
        nb::arg("julian_date"),
        "A StarField's transform for the sky over a site (degrees, "
        "east positive) at a Julian "
        "date: J2000 equatorial to x east, y north, z up.");
    m.def("star_illuminance", &StarIlluminance, nb::arg("magnitude"),
          nb::arg("zero_point") = kMagnitudeZeroLux,
          "Lux of a star of visual magnitude `magnitude` (2.54e-6 "
          "lux at magnitude 0).");
    m.def("color_index_to_temperature", &ColorIndexToTemperature,
          nb::arg("color_index"),
          "A star's temperature in kelvin from its B-V colour index "
          "(Ballesteros 2012).");
    m.def(
        "reflectance_library",
        [] {
            nb::list entries;
            for (const MeasuredReflectance& entry :
                 ReflectanceLibrary()) {
                nb::dict item;
                item["id"] = std::string(entry.id);
                item["name"] = std::string(entry.name);
                item["source"] = std::string(entry.source);
                item["wavelengths"] = entry.spectrum.Wavelengths();
                item["values"] = entry.spectrum.Values();
                entries.append(item);
            }
            return entries;
        },
        "Ernie's measured reflectance spectra (ECOSTRESS), for "
        "materials to name in reflectance_library: dicts of id, "
        "name, source (the ECOSTRESS file), and wavelengths (nm) "
        "and values (fractions) from 0.35 um to 2.5 um and for most "
        "into the thermal infrared.");
    m.def(
        "synthetic_stars",
        [](std::size_t count, std::uint64_t seed, double faintest) {
            StarCatalogue stars =
                SyntheticStars(count, seed, faintest);
            nb::dict result;
            result["right_ascension"] = stars.rightAscension;
            result["declination"] = stars.declination;
            result["magnitude"] = stars.magnitude;
            result["color_index"] = stars.colorIndex;
            return result;
        },
        nb::arg("count"), nb::arg("seed") = 1,
        nb::arg("faintest") = 6.5,
        "`count` made-up stars as lists of right_ascension and "
        "declination (degrees), "
        "magnitude and color_index, the same for the same seed.");

    nb::module_ tokens =
        m.def_submodule("tokens", "Prim type and attribute names.");
    const std::pair<const char*, std::string_view> names[] = {
        {"UP_AXIS", tokens::kUpAxis},
        {"METERS_PER_UNIT", tokens::kMetersPerUnit},
        {"KILOGRAMS_PER_UNIT", tokens::kKilogramsPerUnit},
        {"TIME_CODES_PER_SECOND", tokens::kTimeCodesPerSecond},
        {"FRAMES_PER_SECOND", tokens::kFramesPerSecond},
        {"START_TIME_CODE", tokens::kStartTimeCode},
        {"END_TIME_CODE", tokens::kEndTimeCode},
        {"DEFAULT_PRIM", tokens::kDefaultPrim},
        {"TIME_CODE", tokens::kTimeCode},
        {"XFORM", tokens::kXform},
        {"SCOPE", tokens::kScope},
        {"MESH", tokens::kMesh},
        {"POINT_INSTANCER", tokens::kPointInstancer},
        {"GEOM_POINTS", tokens::kGeomPoints},
        {"VOLUME", tokens::kVolume},
        {"OPENVDB_ASSET", tokens::kOpenVDBAsset},
        {"NANOVDB_FIELD", tokens::kNanoVDBField},
        {"NANOVDB_BUFFER", tokens::kNanoVdbBuffer},
        {"DENSE_FIELD", tokens::kDenseField},
        {"DENSE_VALUES", tokens::kDenseValues},
        {"VOXEL_SIZE", tokens::kVoxelSize},
        {"HENYEY_GREENSTEIN", tokens::kHenyeyGreenstein},
        {"DOUBLE_HENYEY_GREENSTEIN", tokens::kDoubleHenyeyGreenstein},
        {"CLOUD_DROPLETS", tokens::kCloudDroplets},
        {"PHASE_B", tokens::kPhaseB},
        {"PHASE_C", tokens::kPhaseC},
        {"FILE_PATH", tokens::kFilePath},
        {"FIELD_NAME", tokens::kFieldName},
        {"FIELD_INDEX", tokens::kFieldIndex},
        {"VOLUME_DENSITY", tokens::kVolumeDensity},
        {"VOLUME_ALBEDO", tokens::kVolumeAlbedo},
        {"VOLUME_EMISSION_STRENGTH", tokens::kVolumeEmissionStrength},
        {"VOLUME_TEMPERATURE", tokens::kVolumeTemperature},
        {"CAMERA", tokens::kCamera},
        {"DISTANT_LIGHT", tokens::kDistantLight},
        {"GROUND_PLANE", tokens::kGroundPlane},
        {"GROUND_HEIGHT", tokens::kGroundHeight},
        {"CHECKER", tokens::kChecker},
        {"CHECKER_SIZE", tokens::kCheckerSize},
        {"CHECKER_COLOR0", tokens::kCheckerColor0},
        {"CHECKER_COLOR1", tokens::kCheckerColor1},
        {"CUBE", tokens::kCube},
        {"SPHERE", tokens::kSphere},
        {"CYLINDER", tokens::kCylinder},
        {"CONE", tokens::kCone},
        {"CAPSULE", tokens::kCapsule},
        {"PLANE", tokens::kPlane},
        {"SIZE", tokens::kSize},
        {"GEOM_RADIUS", tokens::kGeomRadius},
        {"GEOM_HEIGHT", tokens::kGeomHeight},
        {"GEOM_WIDTH", tokens::kGeomWidth},
        {"GEOM_LENGTH", tokens::kGeomLength},
        {"AXIS", tokens::kAxis},
        {"TESSELLATION", tokens::kTessellation},
        {"TESSELLATION_LEVEL", tokens::kTessellationLevel},
        {"UV_SPHERE", tokens::kUvSphere},
        {"ICOSPHERE", tokens::kIcosphere},
        {"CUBE_SPHERE", tokens::kCubeSphere},
        {"HEALPIX", tokens::kHealpix},
        {"SPHERE_LIGHT", tokens::kSphereLight},
        {"RECT_LIGHT", tokens::kRectLight},
        {"DISK_LIGHT", tokens::kDiskLight},
        {"CYLINDER_LIGHT", tokens::kCylinderLight},
        {"LASER_LIGHT", tokens::kLaserLight},
        {"LASER_PATTERN_IMAGE_ARRAY",
         tokens::kLaserPatternImageArray},
        {"DOME_LIGHT", tokens::kDomeLight},
        {"MATERIAL", tokens::kMaterial},
        {"TERRAIN", tokens::kTerrain},
        {"RENDER_SETTINGS", tokens::kRenderSettings},
        {"RENDER_PRODUCT", tokens::kRenderProduct},
        {"RENDER_VAR", tokens::kRenderVar},
        {"TRANSFORM", tokens::kTransform},
        {"XFORM_OP_ORDER", tokens::kXformOpOrder},
        {"XFORM_EXTENT", tokens::kXformExtent},
        {"QUATERNION_ORDER", tokens::kQuaternionOrder},
        {"XYZW", tokens::kXyzw},
        {"WXYZ", tokens::kWxyz},
        {"RESET_XFORM_STACK", tokens::kResetXformStack},
        {"TRANSLATE", tokens::kTranslate},
        {"SCALE", tokens::kScale},
        {"ORIENT", tokens::kOrient},
        {"ROTATE_X", tokens::kRotateX},
        {"ROTATE_Y", tokens::kRotateY},
        {"ROTATE_Z", tokens::kRotateZ},
        {"ROTATE_XYZ", tokens::kRotateXYZ},
        {"ROTATE_XZY", tokens::kRotateXZY},
        {"ROTATE_YXZ", tokens::kRotateYXZ},
        {"ROTATE_YZX", tokens::kRotateYZX},
        {"ROTATE_ZXY", tokens::kRotateZXY},
        {"ROTATE_ZYX", tokens::kRotateZYX},
        {"VISIBILITY", tokens::kVisibility},
        {"INHERITED", tokens::kInherited},
        {"INVISIBLE", tokens::kInvisible},
        {"POINTS", tokens::kPoints},
        {"NORMALS", tokens::kNormals},
        {"THERMAL_TEMPERATURES", tokens::kThermalTemperatures},
        {"THERMAL_NODE_IDS", tokens::kThermalNodeIds},
        {"THERMAL_NODE_TEMPERATURES",
         tokens::kThermalNodeTemperatures},
        {"THERMAL_MODEL", tokens::kThermalModel},
        {"THERMAL_MODEL_PARAMS", tokens::kThermalModelParams},
        {"THERMAL_MODEL_DATA", tokens::kThermalModelData},
        {"THERMAL_OUTPUT", tokens::kThermalOutput},
        {"ERNIE_THERMAL", tokens::kErnieThermal},
        {"THERMAL_TEMPERATURE_INPUT",
         tokens::kThermalTemperatureInput},
        {"THERMAL_EMISSIVITY_INPUT", tokens::kThermalEmissivityInput},
        {"THERMAL_MODEL_INPUT", tokens::kThermalModelInput},
        {"THERMAL_MODEL_PARAMS_INPUT",
         tokens::kThermalModelParamsInput},
        {"THERMAL_MODEL_DATA_INPUT", tokens::kThermalModelDataInput},
        {"WIDTHS", tokens::kWidths},
        {"FACE_VERTEX_COUNTS", tokens::kFaceVertexCounts},
        {"FACE_VERTEX_INDICES", tokens::kFaceVertexIndices},
        {"ACTIVE_COUNTS", tokens::kActiveCounts},
        {"DISPLAY_COLOR", tokens::kDisplayColor},
        {"POSITIONS", tokens::kPositions},
        {"ORIENTATIONS", tokens::kOrientations},
        {"SCALES", tokens::kScales},
        {"VELOCITIES", tokens::kVelocities},
        {"EXTENT", tokens::kExtent},
        {"PROTO_INDICES", tokens::kProtoIndices},
        {"PROTOTYPES", tokens::kPrototypes},
        {"MATERIAL_BINDING", tokens::kMaterialBinding},
        {"FOCAL_LENGTH", tokens::kFocalLength},
        {"HORIZONTAL_APERTURE", tokens::kHorizontalAperture},
        {"VERTICAL_APERTURE", tokens::kVerticalAperture},
        {"CLIPPING_RANGE", tokens::kClippingRange},
        {"PROJECTION", tokens::kProjection},
        {"PERSPECTIVE", tokens::kPerspective},
        {"ORTHOGRAPHIC", tokens::kOrthographic},
        {"CAMERA_EXPOSURE", tokens::kCameraExposure},
        {"INTENSITY", tokens::kIntensity},
        {"EXPOSURE", tokens::kExposure},
        {"COLOR", tokens::kColor},
        {"ANGLE", tokens::kAngle},
        {"RADIUS", tokens::kRadius},
        {"TREAT_AS_POINT", tokens::kTreatAsPoint},
        {"SHAPING_CONE_ANGLE", tokens::kShapingConeAngle},
        {"SHAPING_CONE_SOFTNESS", tokens::kShapingConeSoftness},
        {"WIDTH", tokens::kWidth},
        {"HEIGHT", tokens::kHeight},
        {"TEXTURE_FILE", tokens::kTextureFile},
        {"SHADOW_ENABLE", tokens::kShadowEnable},
        {"COLOR_TEMPERATURE", tokens::kColorTemperature},
        {"ENABLE_COLOR_TEMPERATURE", tokens::kEnableColorTemperature},
        {"LIGHT_IRRADIANCE", tokens::kLightIrradiance},
        {"LIGHT_POWER", tokens::kLightPower},
        {"LIGHT_TEXTURE", tokens::kLightTexture},
        {"LIGHT_SPECTRUM", tokens::kLightSpectrum},
        {"FILTERS", tokens::kFilters},
        {"OUTPUTS_IN_PLACE", tokens::kOutputsInPlace},
        {"SENSOR_FILTER", tokens::kSensorFilter},
        {"FILTER_TYPE", tokens::kFilterType},
        {"HORIZONTAL_APERTURE_OFFSET",
         tokens::kHorizontalApertureOffset},
        {"VERTICAL_APERTURE_OFFSET", tokens::kVerticalApertureOffset},
        {"EXPOSURE_TIME", tokens::kExposureTime},
        {"EXPOSURE_ISO", tokens::kExposureIso},
        {"EXPOSURE_FSTOP", tokens::kExposureFStop},
        {"EXPOSURE_RESPONSIVITY", tokens::kExposureResponsivity},
        {"EXPOSURE_MODE", tokens::kExposureMode},
        {"EXPOSURE_ADAPTATION", tokens::kExposureAdaptation},
        {"MANUAL", tokens::kManual},
        {"AUTO", tokens::kAuto},
        {"F_STOP", tokens::kFStop},
        {"FOCUS_DISTANCE", tokens::kFocusDistance},
        {"SHUTTER_OPEN", tokens::kShutterOpen},
        {"SHUTTER_CLOSE", tokens::kShutterClose},
        {"SHUTTER_SAMPLES", tokens::kShutterSamples},
        {"SHUTTER_ROLLING", tokens::kShutterRolling},
        {"SPECTRAL_RANGE", tokens::kSpectralRange},
        {"SPECTRAL_BANDS", tokens::kSpectralBands},
        {"SPECTRAL_MODE", tokens::kSpectralMode},
        {"SPECTRAL", tokens::kSpectral},
        {"RGB", tokens::kRgb},
        {"SPECTRAL_PRECISION", tokens::kSpectralPrecision},
        {"FLOAT32", tokens::kFloat32},
        {"FLOAT16", tokens::kFloat16},
        {"CAMERA_SCAN", tokens::kCameraScan},
        {"FRAME_SCAN", tokens::kFrameScan},
        {"PUSHBROOM", tokens::kPushbroom},
        {"PUSHBROOM_LINE_OFFSET", tokens::kPushbroomLineOffset},
        {"LENS_PROJECTION", tokens::kLensProjection},
        {"FISHEYE", tokens::kFisheye},
        {"LENS_FISHEYE", tokens::kLensFisheye},
        {"LENS_RAYS", tokens::kLensRays},
        {"LENS_RAYS_STEP", tokens::kLensRaysStep},
        {"LENS_CHROMATIC_SCALE", tokens::kLensChromaticScale},
        {"LENS_CHROMATIC_FOCUS", tokens::kLensChromaticFocus},
        {"LENS_DISTORTION", tokens::kLensDistortion},
        {"LENS_VIGNETTING", tokens::kLensVignetting},
        {"LENS_TRANSMISSION", tokens::kLensTransmission},
        {"SENSOR_TYPE", tokens::kSensorType},
        {"SENSOR_QE_PREFIX", tokens::kSensorQePrefix},
        {"SENSOR_FULL_WELL", tokens::kSensorFullWell},
        {"SENSOR_READ_NOISE", tokens::kSensorReadNoise},
        {"SENSOR_DARK_CURRENT", tokens::kSensorDarkCurrent},
        {"SENSOR_PRNU", tokens::kSensorPrnu},
        {"SENSOR_DSNU", tokens::kSensorDsnu},
        {"SENSOR_GAIN", tokens::kSensorGain},
        {"SENSOR_ADC_BITS", tokens::kSensorAdcBits},
        {"SENSOR_BLACK_LEVEL", tokens::kSensorBlackLevel},
        {"SENSOR_PIXEL_PITCH", tokens::kSensorPixelPitch},
        {"SENSOR_NOISE", tokens::kSensorNoise},
        {"SENSOR_SEED", tokens::kSensorSeed},
        {"HEIGHTS", tokens::kHeights},
        {"ORIGIN", tokens::kOrigin},
        {"CELL_SIZE", tokens::kCellSize},
        {"PLANET_RADIUS", tokens::kPlanetRadius},
        {"PLANET_RADII", tokens::kPlanetRadii},
        {"FINEST_CELL", tokens::kFinestCell},
        {"NO_DATA", tokens::kNoData},
        {"EXAGGERATION", tokens::kExaggeration},
        {"HEIGHTS_DIRTY", tokens::kHeightsDirty},
        {"BRDF", tokens::kBrdf},
        {"ALBEDO", tokens::kAlbedo},
        {"REFLECTANCE", tokens::kReflectance},
        {"REFLECTANCE_LIBRARY", tokens::kReflectanceLibrary},
        {"ALBEDO_MAP", tokens::kAlbedoMap},
        {"ALBEDO_MAP_ORIGIN", tokens::kAlbedoMapOrigin},
        {"ALBEDO_MAP_CELL_SIZE", tokens::kAlbedoMapCellSize},
        {"ALBEDO_MAP_NO_DATA", tokens::kAlbedoMapNoData},
        {"ALBEDO_MAP_ENABLE", tokens::kAlbedoMapEnable},
        {"DETAIL_ALBEDO", tokens::kDetailAlbedo},
        {"DETAIL_NORMAL", tokens::kDetailNormal},
        {"DETAIL_TILE_SIZE", tokens::kDetailTileSize},
        {"DETAIL_ALBEDO_STRENGTH", tokens::kDetailAlbedoStrength},
        {"DETAIL_NORMAL_STRENGTH", tokens::kDetailNormalStrength},
        {"DETAIL_DIRECTX_NORMALS", tokens::kDetailDirectXNormals},
        {"DETAIL_HEX_TILING", tokens::kDetailHexTiling},
        {"DETAIL_HEX_ROTATION", tokens::kDetailHexRotation},
        {"DETAIL_HEX_CONTRAST", tokens::kDetailHexContrast},
        {"DETAIL_HEX_SHARPNESS", tokens::kDetailHexSharpness},
        {"SHADER", tokens::kShader},
        {"INFO_ID", tokens::kInfoId},
        {"USD_PREVIEW_SURFACE", tokens::kUsdPreviewSurface},
        {"USD_UV_TEXTURE", tokens::kUsdUVTexture},
        {"USD_PRIMVAR_READER_FLOAT2",
         tokens::kUsdPrimvarReaderFloat2},
        {"USD_TRANSFORM2D", tokens::kUsdTransform2d},
        {"SURFACE_OUTPUT", tokens::kSurfaceOutput},
        {"DIFFUSE_COLOR", tokens::kDiffuseColor},
        {"NORMAL_INPUT", tokens::kNormalInput},
        {"ROUGHNESS", tokens::kRoughness},
        {"METALLIC", tokens::kMetallic},
        {"IOR", tokens::kIor},
        {"USE_SPECULAR_WORKFLOW", tokens::kUseSpecularWorkflow},
        {"SPECULAR_COLOR", tokens::kSpecularColor},
        {"CLEARCOAT", tokens::kClearcoat},
        {"CLEARCOAT_ROUGHNESS", tokens::kClearcoatRoughness},
        {"EMISSIVE_COLOR", tokens::kEmissiveColor},
        {"OPACITY", tokens::kOpacity},
        {"OPACITY_THRESHOLD", tokens::kOpacityThreshold},
        {"FILE", tokens::kFile},
        {"SHADER_TEXTURE", tokens::kShaderTexture},
        {"ST", tokens::kSt},
        {"WRAP_S", tokens::kWrapS},
        {"WRAP_T", tokens::kWrapT},
        {"SHADER_SCALE", tokens::kShaderScale},
        {"SHADER_BIAS", tokens::kShaderBias},
        {"SOURCE_COLOR_SPACE", tokens::kSourceColorSpace},
        {"VARNAME", tokens::kVarname},
        {"IN", tokens::kIn},
        {"ROTATION", tokens::kRotation},
        {"TRANSLATION", tokens::kTranslation},
        {"CAMERA_REL", tokens::kCameraRel},
        {"RESOLUTION", tokens::kResolution},
        {"POSES", tokens::kPoses},
        {"POSES_DEVICE", tokens::kPosesDevice},
        {"HOST", tokens::kHost},
        {"WORLD", tokens::kWorld},
        {"WORLDS", tokens::kWorlds},
        {"PRODUCTS", tokens::kProducts},
        {"ORDERED_VARS", tokens::kOrderedVars},
        {"SOURCE_NAME", tokens::kSourceName},
        {"AMBIENT", tokens::kAmbient},
        {"AMBIENT_COLOR", tokens::kAmbientColor},
        {"COLOR_MODE", tokens::kColorMode},
        {"SHADED", tokens::kShaded},
        {"COLORMAP", tokens::kColormap},
        {"SHADED_COLORMAP", tokens::kShadedColormap},
        {"HILLSHADE", tokens::kHillshade},
        {"HILLSHADE_AZIMUTH", tokens::kHillshadeAzimuth},
        {"HILLSHADE_ALTITUDE", tokens::kHillshadeAltitude},
        {"WIREFRAME", tokens::kWireframe},
        {"OFF", tokens::kOff},
        {"OVERLAY", tokens::kOverlay},
        {"HIDDEN_LINE", tokens::kHiddenLine},
        {"CULLING", tokens::kCulling},
        {"CPU", tokens::kCpu},
        {"GPU", tokens::kGpu},
        {"OCCLUSION", tokens::kOcclusion},
        {"SHADOW_WORLD_RESOLUTION", tokens::kShadowWorldResolution},
        {"SHADOW_WORLDS", tokens::kShadowWorlds},
        {"SEPARATE", tokens::kSeparate},
        {"SHARED", tokens::kShared},
        {"MERGED", tokens::kMerged},
        {"SHADOW_WORLD_FIT", tokens::kShadowWorldFit},
        {"SCENE", tokens::kScene},
        {"WORLD_FIT", tokens::kWorldFit},
        {"SHADOW_WORLD_MARGIN", tokens::kShadowWorldMargin},
        {"EXPOSURE_COMPENSATION", tokens::kExposureCompensation},
        {"METER", tokens::kMeter},
        {"PHYSICAL_ILLUMINANT", tokens::kPhysicalIlluminant},
        {"PHYSICAL_CUSTOM_ILLUMINANT",
         tokens::kPhysicalCustomIlluminant},
        {"PHOTOMETRIC_ILLUMINANCE", tokens::kPhotometricIlluminance},
        {"PHOTOMETRIC_ILLUMINANCE_DISTANCE",
         tokens::kPhotometricIlluminanceDistance},
        {"PHOTOMETRIC_POWER", tokens::kPhotometricPower},
        {"VISIBLE_IN_PRIMARY_RAY", tokens::kVisibleInPrimaryRay},
        {"STAR_FIELD", tokens::kStarField},
        {"STARS_RIGHT_ASCENSION", tokens::kStarsRightAscension},
        {"STARS_DECLINATION", tokens::kStarsDeclination},
        {"STARS_DIRECTION", tokens::kStarsDirection},
        {"STARS_MAGNITUDE", tokens::kStarsMagnitude},
        {"STARS_COLOR_INDEX", tokens::kStarsColorIndex},
        {"STARS_TEMPERATURE", tokens::kStarsTemperature},
        {"STARS_MAGNITUDE_LIMIT", tokens::kStarsMagnitudeLimit},
        {"STARS_ZERO_POINT", tokens::kStarsZeroPoint},
        {"ATMOSPHERE", tokens::kAtmosphere},
        {"ATMOSPHERE_PRESET", tokens::kAtmospherePreset},
        {"EARTH", tokens::kEarth},
        {"MARS", tokens::kMars},
        {"ATMOSPHERE_PLANET_RADIUS", tokens::kAtmospherePlanetRadius},
        {"ATMOSPHERE_GROUND_ALBEDO", tokens::kAtmosphereGroundAlbedo},
        {"ATMOSPHERE_HAZE_OPTICAL_DEPTH",
         tokens::kAtmosphereHazeOpticalDepth},
        {"ATMOSPHERE_OZONE_COLUMN", tokens::kAtmosphereOzoneColumn},
        {"ATMOSPHERE_SODIUM_COLUMN", tokens::kAtmosphereSodiumColumn},
        {"ATMOSPHERE_DUST_OPTICAL_DEPTH",
         tokens::kAtmosphereDustOpticalDepth},
        {"ATMOSPHERE_CLOUD_TOP", tokens::kAtmosphereCloudTop},
        {"ATMOSPHERE_CLOUD_COVER", tokens::kAtmosphereCloudCover},
        {"ATMOSPHERE_ANCHOR", tokens::kAtmosphereAnchor},
        {"ANCHOR_GROUND", tokens::kAnchorGround},
        {"ANCHOR_CENTRE", tokens::kAnchorCentre},
        {"CELESTIAL_BODY", tokens::kCelestialBody},
        {"BODY_RADIUS", tokens::kBodyRadius},
        {"BODY_ALBEDO", tokens::kBodyAlbedo},
        {"WHITE", tokens::kWhite},
        {"BLACKBODY", tokens::kBlackbody},
        {"ILLUMINANT_D", tokens::kIlluminantD},
        {"CUSTOM", tokens::kCustom},
        {"SUN", tokens::kSun},
        {"OVERLAYS", tokens::kOverlays},
        {"MINIMAL_MODE", tokens::kMinimalMode},
        {"MINIMAL_CONSTANT_COLOR", tokens::kMinimalConstantColor},
        {"NONE", tokens::kNone},
        {"CONSTANT_DIFFUSE", tokens::kConstantDiffuse},
        {"TEXTURED_DIFFUSE", tokens::kTexturedDiffuse},
        {"DIFFUSE_GLOSSY_EMISSION", tokens::kDiffuseGlossyEmission},
        {"OCCLUDERS", tokens::kOccluders},
        {"TWO_PASS", tokens::kTwoPass},
        {"CLEAR_COLOR", tokens::kClearColor},
        {"SHADOW_CASCADES", tokens::kShadowCascades},
        {"SHADOW_RESOLUTION", tokens::kShadowResolution},
        {"SHADOW_SOFTNESS", tokens::kShadowSoftness},
        {"SHADOW_NORMAL_OFFSET", tokens::kShadowNormalOffset},
        {"SHADOW_SPLIT_EXPONENT", tokens::kShadowSplitExponent},
        {"SHADOW_SHOW_CASCADES", tokens::kShadowShowCascades},
        {"LOD_ENABLE", tokens::kLodEnable},
        {"LOD_DETAIL", tokens::kLodDetail},
        {"LOD_MORPH_START", tokens::kLodMorphStart},
        {"LOD_FREEZE", tokens::kLodFreeze},
        {"LOD_SHOW_LEVELS", tokens::kLodShowLevels},
        {"LOD_MESH_ERROR", tokens::kLodMeshError},
        {"LDR_COLOR", tokens::kLdrColor},
    };
    for (const auto& [name, token] : names) {
        tokens.attr(name) = nb::str(token.data(), token.size());
    }
}

}  // namespace ernie::python
