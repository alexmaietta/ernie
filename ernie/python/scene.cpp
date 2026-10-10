// SPDX-License-Identifier: BSD-3-Clause
#include "bindings.h"

#include "ernie/scene/cahvore.h"
#include "ernie/scene/camera_schemas.h"
#include "ernie/scene/instruments.h"
#include "ernie/scene/isd.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"
#include "ernie/util/astronomy.h"
#include "ernie/util/image.h"
#include "ernie/util/tensor.h"

#include <nanobind/operators.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ernie::python {

using namespace nb::literals;

namespace {

/// An array mapped for writing in place by `Scene.map`, unmapped once.
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
            StreamOf(override.is_valid() && !override.is_none() ? override : nb::handle(stream)));
    }
};

/// A 4 x 4 NumPy array of `matrix` in math convention: column vectors, translation in the last
/// column.
nb::object MatrixToPython(const glm::dmat4& matrix) {
    auto* values = new std::array<double, 16>();
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            (*values)[static_cast<std::size_t>(row * 4 + col)] = matrix[col][row];
        }
    }
    nb::capsule owner(values,
                      [](void* p) noexcept { delete static_cast<std::array<double, 16>*>(p); });
    const std::size_t shape[2] = {4, 4};
    return nb::cast(nb::ndarray<nb::numpy, double>(values->data(), 2, shape, owner),
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
    return nb::isinstance<nb::float_>(item) || nb::isinstance<nb::int_>(item);
}

/// `object` as a C-contiguous NumPy array of `dtype` and rows of `columns` (any leading shape).
nb::ndarray<nb::numpy, nb::c_contig> RowsOf(nb::handle object, const char* dtype,
                                             std::size_t columns) {
    nb::module_ numpy = nb::module_::import_("numpy");
    nb::object array = numpy.attr("ascontiguousarray")(object, "dtype"_a = dtype);
    if (nb::cast<std::size_t>(array.attr("size")) % columns != 0) {
        throw nb::value_error(("expected rows of " + std::to_string(columns) + " values").c_str());
    }
    array = array.attr("reshape")(-1, columns);
    return nb::cast<nb::ndarray<nb::numpy, nb::c_contig>>(array);
}

/// Points as rows of three numbers: an (N, 3) array or a sequence of 3-sequences.
std::vector<glm::dvec3> PointsFromPython(nb::handle object) {
    const auto rows = RowsOf(object, "float64", 3);
    const auto* values = static_cast<const double*>(rows.data());
    std::vector<glm::dvec3> points(rows.shape(0));
    for (std::size_t i = 0; i < points.size(); ++i) {
        points[i] = {values[3 * i], values[3 * i + 1], values[3 * i + 2]};
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
    nb::object array = numpy.attr("asarray")(object, "dtype"_a = "float32");
    const std::size_t width = nb::cast<std::size_t>(array.attr("shape")[nb::int_(-1)]);
    if (width != 3 && width != 4) {
        throw nb::value_error("expected colours as rows of RGB or RGBA");
    }
    const auto rows = RowsOf(array, "float32", width);
    const auto* values = static_cast<const float*>(rows.data());
    colors.resize(rows.shape(0));
    for (std::size_t i = 0; i < colors.size(); ++i) {
        const float* row = values + width * i;
        colors[i] = {row[0], row[1], row[2], width == 4 ? row[3] : 1.0f};
    }
    return colors;
}

std::vector<std::uint32_t> IndicesFromPython(nb::handle object) {
    const auto rows = RowsOf(object, "uint32", 1);
    const auto* values = static_cast<const std::uint32_t*>(rows.data());
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
    throw nb::value_error("depth must be \"test\", \"xray\" or \"always\"");
}

MarkerShape ShapeFromPython(std::string_view name) {
    constexpr std::pair<std::string_view, MarkerShape> kShapes[] = {
        {"circle", MarkerShape::Circle}, {"ring", MarkerShape::Ring},
        {"square", MarkerShape::Square}, {"diamond", MarkerShape::Diamond},
        {"cross", MarkerShape::Cross},   {"plus", MarkerShape::Plus}};
    for (const auto& [token, shape] : kShapes) {
        if (token == name) {
            return shape;
        }
    }
    throw nb::value_error(
        "shape must be \"circle\", \"ring\", \"square\", \"diamond\", \"cross\" or \"plus\"");
}

/// An `OverlayStyle` from keyword arguments: color, width, size, depth ("test", "xray" or
/// "always"), world and background.
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
            style.depth = DepthFromPython(nb::cast<std::string>(value));
        } else if (name == "world") {
            style.world = nb::cast<std::int32_t>(value);
        } else if (name == "background") {
            style.background = ColorFromPython(value);
        } else {
            throw nb::type_error(("unknown overlay style '" + name +
                                  "'; expected color, width, size, depth, world or background")
                                     .c_str());
        }
    }
    return style;
}

constexpr const char* kStyleDoc =
    " Style keywords: color (RGB or RGBA display colour), width (line pixels), size (marker or "
    "text pixels), depth (\"test\", \"xray\" or \"always\"), world (-1 for every view) and "
    "background (labels' backing colour). Returns the overlay, for chaining.";

void BindOverlay(nb::module_& m) {
    nb::class_<Overlay>(
        m, "Overlay",
        "A named layer of debug drawing in world coordinates (`Scene.overlay`): lines, shapes, "
        "markers and labels drawn over the LdrColor of render products whose `ernie:overlays` "
        "names it (\"*\" for all; the viewer shows all). Items stay until `clear`.")
        .def("clear", &Overlay::Clear, nb::rv_policy::reference_internal)
        .def_prop_rw("visible", &Overlay::IsVisible,
                     [](Overlay& overlay, bool visible) { overlay.SetVisible(visible); })
        .def_prop_ro("version", &Overlay::Version)
        .def_prop_ro("empty", &Overlay::Empty)
        .def(
            "line",
            [](Overlay& o, nb::handle a, nb::handle b, const nb::kwargs& style) -> Overlay& {
                return o.Line(Vec3FromPython(a), Vec3FromPython(b), StyleFromPython(style));
            },
            nb::arg("a"), nb::arg("b"), nb::arg("style"), nb::rv_policy::reference_internal,
            (std::string("A segment from `a` to `b`.") + kStyleDoc).c_str())
        .def(
            "lines",
            [](Overlay& o, nb::handle endpoints, nb::handle colors,
               const nb::kwargs& style) -> Overlay& {
                return o.Lines(PointsFromPython(endpoints), StyleFromPython(style),
                               ColorsFromPython(colors));
            },
            nb::arg("endpoints"), nb::arg("colors") = nb::none(), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string("Many segments: `endpoints` as (N, 2, 3) or (2N, 3), coloured by "
                         "`colors` (N rows of RGB or RGBA) when given.") +
             kStyleDoc)
                .c_str())
        .def(
            "polyline",
            [](Overlay& o, nb::handle points, bool closed, const nb::kwargs& style) -> Overlay& {
                return o.Polyline(PointsFromPython(points), StyleFromPython(style), closed);
            },
            nb::arg("points"), nb::arg("closed") = false, nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string("Segments through (N, 3) `points`, back to the first when `closed`.") +
             kStyleDoc)
                .c_str())
        .def(
            "arrow",
            [](Overlay& o, nb::handle start, nb::handle end, double head,
               const nb::kwargs& style) -> Overlay& {
                return o.Arrow(Vec3FromPython(start), Vec3FromPython(end),
                               StyleFromPython(style), head);
            },
            nb::arg("start"), nb::arg("end"), nb::arg("head") = 0.0, nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string("A line with a cone head `head` long (a fifth of it when 0).") +
             kStyleDoc)
                .c_str())
        .def(
            "box",
            [](Overlay& o, nb::handle low, nb::handle high, nb::handle transform,
               const nb::kwargs& style) -> Overlay& {
                return o.Box(Vec3FromPython(low), Vec3FromPython(high), StyleFromPython(style),
                             TransformFromPython(transform));
            },
            nb::arg("low"), nb::arg("high"), nb::arg("transform") = nb::none(), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string("The edges of a box, placed by a 4 x 4 `transform`.") + kStyleDoc)
                .c_str())
        .def(
            "circle",
            [](Overlay& o, nb::handle centre, nb::handle normal, double radius, int segments,
               const nb::kwargs& style) -> Overlay& {
                return o.Circle(Vec3FromPython(centre), Vec3FromPython(normal), radius,
                                StyleFromPython(style), segments);
            },
            nb::arg("centre"), nb::arg("normal"), nb::arg("radius"), nb::arg("segments") = 48,
            nb::arg("style"), nb::rv_policy::reference_internal,
            (std::string("A circle in the plane facing `normal`.") + kStyleDoc).c_str())
        .def(
            "sphere",
            [](Overlay& o, nb::handle centre, double radius, const nb::kwargs& style) -> Overlay& {
                return o.Sphere(Vec3FromPython(centre), radius, StyleFromPython(style));
            },
            nb::arg("centre"), nb::arg("radius"), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string("Three great circles of a sphere.") + kStyleDoc).c_str())
        .def(
            "axes",
            [](Overlay& o, nb::handle transform, double length,
               const nb::kwargs& style) -> Overlay& {
                return o.Axes(TransformFromPython(transform), length, StyleFromPython(style));
            },
            nb::arg("transform"), nb::arg("length") = 1.0, nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string("A transform's x, y and z axes in red, green and blue.") + kStyleDoc)
                .c_str())
        .def(
            "wire_mesh",
            [](Overlay& o, nb::handle points, nb::handle indices, nb::handle transform,
               const nb::kwargs& style) -> Overlay& {
                return o.WireMesh(PointsFromPython(points), IndicesFromPython(indices),
                                  StyleFromPython(style), TransformFromPython(transform));
            },
            nb::arg("points"), nb::arg("indices"), nb::arg("transform") = nb::none(),
            nb::arg("style"), nb::rv_policy::reference_internal,
            (std::string("The edges of triangles (`indices` in threes over `points`).") +
             kStyleDoc)
                .c_str())
        .def(
            "triangles",
            [](Overlay& o, nb::handle points, nb::handle indices, nb::handle transform,
               const nb::kwargs& style) -> Overlay& {
                return o.Triangles(PointsFromPython(points), IndicesFromPython(indices),
                                   StyleFromPython(style), TransformFromPython(transform));
            },
            nb::arg("points"), nb::arg("indices"), nb::arg("transform") = nb::none(),
            nb::arg("style"), nb::rv_policy::reference_internal,
            (std::string("Filled triangles (`indices` in threes over `points`).") + kStyleDoc)
                .c_str())
        .def(
            "marker",
            [](Overlay& o, nb::handle point, std::string_view shape,
               const nb::kwargs& style) -> Overlay& {
                return o.Marker(Vec3FromPython(point), StyleFromPython(style),
                                ShapeFromPython(shape));
            },
            nb::arg("point"), nb::arg("shape") = "circle", nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string("A marker of `size` pixels: circle, ring, square, diamond, cross or "
                         "plus.") +
             kStyleDoc)
                .c_str())
        .def(
            "markers",
            [](Overlay& o, nb::handle points, nb::handle colors, std::string_view shape,
               const nb::kwargs& style) -> Overlay& {
                return o.Markers(PointsFromPython(points), StyleFromPython(style),
                                 ShapeFromPython(shape), ColorsFromPython(colors));
            },
            nb::arg("points"), nb::arg("colors") = nb::none(), nb::arg("shape") = "circle",
            nb::arg("style"), nb::rv_policy::reference_internal,
            (std::string("Markers at (N, 3) `points`, coloured by `colors` when given.") +
             kStyleDoc)
                .c_str())
        .def(
            "label",
            [](Overlay& o, nb::handle point, std::string_view text,
               const nb::kwargs& style) -> Overlay& {
                return o.Label(Vec3FromPython(point), text, StyleFromPython(style));
            },
            nb::arg("point"), nb::arg("text"), nb::arg("style"),
            nb::rv_policy::reference_internal,
            (std::string("ASCII `text`, `size` pixels tall, beside `point`.") + kStyleDoc)
                .c_str())
        .def("__repr__", [](const Overlay& o) {
            return "Overlay(" + std::to_string(o.Segments().size()) + " segments, " +
                   std::to_string(o.Fills().size()) + " triangles, " +
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
    return nb::isinstance<nb::str>(object) || nb::isinstance<Path>(object) ||
           nb::isinstance<Prim>(object) || WrappedPrim(object).is_valid();
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
    throw nb::type_error(("A prim path is a str, Path, Prim or typed prim, not " +
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
            throw nb::value_error("Integer scene values must fit in 32 bits");
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
    if (!nb::isinstance<nb::list>(sequence) && !nb::isinstance<nb::tuple>(sequence)) {
        throw nb::type_error("Scene values are bool, int, float, str, Path, Quat, a sequence of "
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
                throw nb::type_error("A list of targets must hold only Path values");
            }
            paths.push_back(nb::cast<Path>(item));
        }
        return paths;
    }
    if (!items.empty() && nb::isinstance<nb::str>(items[0])) {
        std::vector<std::string> tokens;
        for (const nb::object& item : items) {
            if (!nb::isinstance<nb::str>(item)) {
                throw nb::type_error("A list of tokens must hold only str values");
            }
            tokens.push_back(nb::cast<std::string>(item));
        }
        return tokens;
    }
    if (items.size() == 4 && !IsNumber(items[0])) {
        glm::dmat4 matrix(1.0);
        for (int row = 0; row < 4; ++row) {
            const std::vector<double> values =
                nb::cast<std::vector<double>>(items[static_cast<std::size_t>(row)]);
            if (values.size() != 4) {
                throw nb::type_error("A matrix needs 4 rows of 4 numbers");
            }
            for (int col = 0; col < 4; ++col) {
                matrix[col][row] = values[static_cast<std::size_t>(col)];
            }
        }
        return matrix;
    }
    bool integers = true;
    std::vector<double> numbers;
    for (const nb::object& item : items) {
        if (!IsNumber(item) || nb::isinstance<nb::bool_>(item)) {
            throw nb::type_error("Sequences in scene values must hold numbers");
        }
        integers = integers && nb::isinstance<nb::int_>(item);
        numbers.push_back(nb::cast<double>(item));
    }
    if (numbers.size() == 2) {
        if (integers) {
            return glm::ivec2(static_cast<int>(numbers[0]), static_cast<int>(numbers[1]));
        }
        return glm::dvec2(numbers[0], numbers[1]);
    }
    if (numbers.size() == 3) {
        return glm::dvec3(numbers[0], numbers[1], numbers[2]);
    }
    if (numbers.size() == 4) {
        return glm::vec4(numbers[0], numbers[1], numbers[2], numbers[3]);
    }
    return std::vector<float>(numbers.begin(), numbers.end());
}

RotationOrder RotationOrderFromPython(std::string_view name) {
    constexpr std::pair<std::string_view, RotationOrder> kOrders[] = {
        {"XYZ", RotationOrder::XYZ}, {"XZY", RotationOrder::XZY}, {"YXZ", RotationOrder::YXZ},
        {"YZX", RotationOrder::YZX}, {"ZXY", RotationOrder::ZXY}, {"ZYX", RotationOrder::ZYX}};
    for (const auto& [token, order] : kOrders) {
        if (token == name) {
            return order;
        }
    }
    throw nb::value_error("rotation_order must be \"XYZ\", \"XZY\", \"YXZ\", \"YZX\", \"ZXY\" or "
                          "\"ZYX\"");
}

/// An `XformOps` from `set_xform`'s keyword arguments; a single number scales uniformly.
XformOps XformFromPython(nb::handle translate, nb::handle rotate, std::string_view rotationOrder,
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
            throw nb::type_error("orient must be a Quat (Quat(w, x, y, z) or Quat.from_xyzw)");
        }
        xform.orient = nb::cast<glm::dquat>(orient);
    }
    if (!scale.is_none()) {
        xform.scale = IsNumber(scale) ? glm::dvec3(nb::cast<double>(scale)) : Vec3FromPython(scale);
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
            } else if constexpr (std::is_same_v<T, glm::vec2> || std::is_same_v<T, glm::dvec2>) {
                return nb::make_tuple(double(v.x), double(v.y));
            } else if constexpr (std::is_same_v<T, glm::vec3> || std::is_same_v<T, glm::dvec3>) {
                return nb::make_tuple(double(v.x), double(v.y), double(v.z));
            } else if constexpr (std::is_same_v<T, glm::vec4> || std::is_same_v<T, glm::dvec4>) {
                return nb::make_tuple(double(v.x), double(v.y), double(v.z), double(v.w));
            } else if constexpr (std::is_same_v<T, glm::dmat4>) {
                return MatrixToPython(v);
            } else if constexpr (std::is_same_v<T, std::vector<Path>>) {
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
                     "Absolute path of a prim, as in USD: `/` is the root and `/World/Sun` the "
                     "child `Sun` of `/World`. Functions taking a path also accept a str.")
        .def(nb::init<>())
        .def(nb::init_implicit<std::string>(), nb::arg("text"))
        .def_static("root", &Path::Root)
        .def_static("is_valid_name", &Path::IsValidName, nb::arg("name"))
        .def_prop_ro("name", [](const Path& p) { return std::string(p.Name()); })
        .def_prop_ro("parent", &Path::Parent)
        .def_prop_ro("is_root", &Path::IsRoot)
        .def_prop_ro("is_empty", &Path::IsEmpty)
        .def("child", &Path::Child, nb::arg("name"))
        .def("has_prefix", &Path::HasPrefix, nb::arg("ancestor"))
        .def("__str__", &Path::String)
        .def("__repr__", [](const Path& p) { return "Path('" + p.String() + "')"; })
        .def("__hash__", [](const Path& p) { return std::hash<Path>()(p); })
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def(nb::self < nb::self);

    nb::enum_<Dirty>(m, "Dirty", nb::is_flag(), "What changed about a prim.")
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
    m.def("dirty_of", &DirtyOf, nb::arg("attribute"),
          "The dirty bit that setting or writing `attribute` raises.");

    nb::class_<ChangeCursor>(m, "ChangeCursor",
                             "A consumer's position in a scene's history; see Scene.changes.")
        .def(nb::init<>())
        .def_rw("version", &ChangeCursor::version);

    nb::class_<glm::dquat>(
        m, "Quat",
        "A rotation as a quaternion, Quat(w, x, y, z) with the real part first as USD writes one "
        "(xformOp:orient). Arrays of quaternions (PointInstancer orientations, xformOp:orient "
        "rows) are x, y, z, w instead, as in USD's memory; Quat.from_xyzw reads that order, as "
        "SciPy, ROS and Warp use.")
        .def(nb::init<double, double, double, double>(), nb::arg("w"), nb::arg("x"), nb::arg("y"),
             nb::arg("z"))
        .def_static(
            "from_xyzw",
            [](double x, double y, double z, double w) { return glm::dquat(w, x, y, z); },
            nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("w"))
        .def_static(
            "from_axis_angle",
            [](nb::handle axis, double degrees) {
                return glm::angleAxis(glm::radians(degrees), glm::normalize(Vec3FromPython(axis)));
            },
            nb::arg("axis"), nb::arg("degrees"),
            "The rotation by `degrees` about `axis`, right-handed.")
        .def_static(
            "from_euler",
            [](nb::handle degrees, std::string_view order) {
                return glm::normalize(glm::quat_cast(
                    EulerRotation(Vec3FromPython(degrees), RotationOrderFromPython(order))));
            },
            nb::arg("degrees"), nb::arg("order") = "XYZ",
            "The rotation of Euler angles (x, y, z) in degrees applied about fixed axes in "
            "`order`, as the xformOp:rotate ops (\"XYZ\": about X first, then Y, then Z).")
        .def_rw("w", &glm::dquat::w)
        .def_rw("x", &glm::dquat::x)
        .def_rw("y", &glm::dquat::y)
        .def_rw("z", &glm::dquat::z)
        .def_prop_ro("wxyz", [](const glm::dquat& q) { return nb::make_tuple(q.w, q.x, q.y, q.z); })
        .def_prop_ro("xyzw", [](const glm::dquat& q) { return nb::make_tuple(q.x, q.y, q.z, q.w); })
        .def(
            "euler",
            [](const glm::dquat& q, std::string_view order) {
                const glm::dvec3 degrees =
                    EulerAngles(glm::mat3_cast(glm::normalize(q)), RotationOrderFromPython(order));
                return nb::make_tuple(degrees.x, degrees.y, degrees.z);
            },
            nb::arg("order") = "XYZ", "Euler angles (x, y, z) in degrees, as from_euler takes.")
        .def("__eq__", [](const glm::dquat& a, const glm::dquat& b) { return a == b; })
        .def("__repr__", [](const glm::dquat& q) {
            return "Quat(w=" + std::to_string(q.w) + ", x=" + std::to_string(q.x) +
                   ", y=" + std::to_string(q.y) + ", z=" + std::to_string(q.z) + ")";
        });

    nb::class_<PrimSet>(
        m, "PrimSet",
        "A fixed list of prims to write rows of one array to at once (Scene.write), such as a "
        "simulation's bodies every step: a scene looks the prims up the first time and keeps "
        "them, looking them up again only after prims are removed.")
        .def(nb::init<std::vector<Path>>(), nb::arg("paths"))
        .def_prop_ro("paths", &PrimSet::Paths)
        .def("__len__", &PrimSet::Size)
        .def("__repr__", [](const PrimSet& prims) {
            return "PrimSet(" + std::to_string(prims.Size()) + " prims)";
        });

    nb::class_<Prim>(m, "Prim", "Handle to a prim in a Scene.")
        .def_prop_ro("path", &Prim::GetPath)
        .def_prop_ro("type", &Prim::Type)
        .def_prop_ro("valid", &Prim::IsValid)
        .def("__bool__", &Prim::IsValid)
        .def("define", &Prim::Define, nb::arg("relative"), nb::arg("type"), nb::keep_alive<0, 1>(),
             "Defines the prim at `relative` below this one (\"Rock\", or \"Rig/Arm\") with "
             "`type`, as Scene.define.")
        .def(
            "get_prim",
            [](const Prim& prim, std::string_view relative) -> nb::object {
                Prim found = prim.GetPrim(relative);
                return found ? nb::cast(found) : nb::none();
            },
            nb::arg("relative"), nb::keep_alive<0, 1>(),
            "The prim at `relative` below this one (\"Rock\", or \"Rig/Arm\"), or None, as "
            "Scene.get_prim.")
        .def(
            "set",
            [](Prim& prim, std::string_view attribute, nb::handle value) -> Prim& {
                return prim.Set(attribute, ValueFromPython(value));
            },
            nb::arg("attribute"), nb::arg("value").none(), nb::rv_policy::reference_internal,
            "Sets `attribute`; returns the prim for chaining.")
        .def(
            "set_xform",
            [](Prim& prim, nb::handle translate, nb::handle rotate, std::string_view order,
               nb::handle orient, nb::handle scale) -> Prim& {
                return prim.SetXform(XformFromPython(translate, rotate, order, orient, scale));
            },
            nb::kw_only(), nb::arg("translate").none() = nb::none(),
            nb::arg("rotate").none() = nb::none(), nb::arg("rotation_order") = "XYZ",
            nb::arg("orient").none() = nb::none(), nb::arg("scale").none() = nb::none(),
            nb::rv_policy::reference_internal,
            "Sets the prim's transform as translate, rotate (Euler angles in degrees in "
            "`rotation_order`) or orient (a Quat), and scale (three numbers or one) ops, with "
            "xformOpOrder listing the given ones in that order; returns the prim.")
        .def(
            "get",
            [](const Prim& prim, std::string_view attribute, nb::object fallback) {
                const Value* value = prim.Get(attribute);
                return value != nullptr ? ValueToPython(*value) : fallback;
            },
            nb::arg("attribute"), nb::arg("default") = nb::none())
        .def(
            "write",
            [](Prim& prim, std::string_view attribute, const AnyArray& array) -> Prim& {
                return prim.Write(attribute, TensorFromArray(array));
            },
            nb::arg("attribute"), nb::arg("array"), nb::rv_policy::reference_internal,
            "Writes any DLPack array as `attribute` without copying; returns the prim.")
        .def(
            "read",
            [](const Prim& prim, std::string_view attribute) -> nb::object {
                const Tensor* tensor = prim.Read(attribute);
                return tensor != nullptr ? ArrayFromTensor(*tensor) : nb::none();
            },
            nb::arg("attribute"))
        .def("__repr__",
             [](const Prim& prim) { return "Prim('" + prim.GetPath().String() + "')"; });

    nb::class_<AttributeMapping>(
        m, "AttributeMapping",
        "An array mapped for writing in place (Scene.map); a context manager returning the "
        "array and unmapping on exit.")
        .def_ro("array", &AttributeMapping::array, "The writable array.")
        .def_prop_ro("path", [](const AttributeMapping& mapping) { return mapping.path; })
        .def_prop_ro("attribute", [](const AttributeMapping& mapping) { return mapping.attribute; })
        .def("__enter__", [](AttributeMapping& mapping) { return mapping.array; })
        .def(
            "__exit__",
            [](AttributeMapping& mapping, nb::handle, nb::handle, nb::handle) {
                mapping.Unmap({});
            },
            nb::arg("type").none(), nb::arg("value").none(), nb::arg("traceback").none())
        .def(
            "unmap", [](AttributeMapping& mapping, nb::handle stream) { mapping.Unmap(stream); },
            nb::arg("stream").none() = nb::none(),
            "Commits the writes, on the GPU after the work queued on `stream` (the mapping's "
            "when None).");

    nb::class_<Scene>(m, "Scene",
                      "A USD-shaped scene: typed prims named by Path holding attributes by USD "
                      "name, with bulk data as DLPack arrays and change tracking.")
        .def(nb::init<>())
        .def("define", &Scene::Define, nb::arg("path"), nb::arg("type"), nb::keep_alive<0, 1>(),
             "Defines the prim at `path` with `type`, and any missing ancestors untyped.")
        .def(
            "connect",
            [](Scene& scene, const Path& path, std::string_view input, const Path& source,
               std::string_view output) { Connect(scene, path, input, source, output); },
            nb::arg("path"), nb::arg("input"), nb::arg("source"), nb::arg("output"),
            "Connects shader input `input` (e.g. tokens.DIFFUSE_COLOR) of the prim at `path` to "
            "output `output` (e.g. 'outputs:rgb') of the Shader at `source`, as UsdShade does: "
            "the attribute '<input>.connect' holding '<source>.<output>'.")
        .def(
            "connection",
            [](const Scene& scene, const Path& path,
               std::string_view input) -> std::optional<std::pair<std::string, std::string>> {
                const std::optional<Connection> connection = GetConnection(scene, path, input);
                if (!connection) {
                    return std::nullopt;
                }
                return std::pair(connection->prim.String(), connection->output);
            },
            nb::arg("path"), nb::arg("input"),
            "The (source prim path, output) shader input `input` of the prim at `path` connects "
            "to, or None.")
        .def("clone", &Scene::Clone, nb::arg("source"), nb::arg("destination"),
             nb::keep_alive<0, 1>(),
             "Copies `source` and its descendants to `destination`, sharing tensors and pointing "
             "relationships inside `source` at their copies; returns the copy.")
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
            [](Scene& scene, const Path& path, std::string_view attribute, nb::handle value) {
                scene.Set(path, attribute, ValueFromPython(value));
            },
            nb::arg("path"), nb::arg("attribute"), nb::arg("value").none())
        .def(
            "get",
            [](const Scene& scene, const Path& path, std::string_view attribute,
               nb::object fallback) {
                const Value* value = scene.Get(path, attribute);
                return value != nullptr ? ValueToPython(*value) : fallback;
            },
            nb::arg("path"), nb::arg("attribute"), nb::arg("default") = nb::none())
        .def(
            "write",
            [](Scene& scene, const Path& path, std::string_view attribute, const AnyArray& array,
               nb::handle stream) {
                scene.Write(path, attribute, TensorFromArray(array), StreamOf(stream));
            },
            nb::arg("path"), nb::arg("attribute"), nb::arg("array"),
            nb::arg("stream").none() = nb::none(),
            "Writes any DLPack array (NumPy, PyTorch, JAX, Warp) as `attribute` without "
            "copying. For a CUDA array, `stream` (an int, a torch.cuda.Stream, or None for the "
            "legacy default stream) is the one its contents are written on.")
        .def(
            "map",
            [](nb::handle self, const Path& path, std::string_view attribute, nb::handle stream) {
                Scene& scene = nb::cast<Scene&>(self);
                AttributeMapping mapping;
                mapping.scene = nb::borrow(self);
                mapping.path = path;
                mapping.attribute = std::string(attribute);
                mapping.stream = nb::borrow(stream);
                mapping.array = ArrayFromTensor(scene.Map(path, attribute), true);
                return mapping;
            },
            nb::arg("path"), nb::arg("attribute"), nb::arg("stream").none() = nb::none(),
            "Maps the array written as `attribute` for writing in place: `with scene.map(...) "
            "as array:` writes into the array (on its device) and unmaps on exit, with `stream` "
            "the CUDA stream the writes are queued on.")
        .def(
            "unmap",
            [](Scene& scene, const Path& path, std::string_view attribute, nb::handle stream) {
                scene.Unmap(path, attribute, StreamOf(stream));
            },
            nb::arg("path"), nb::arg("attribute"), nb::arg("stream").none() = nb::none(),
            "Commits writes into a mapped array: renderers read it again, on the GPU after the "
            "work queued on `stream`.")
        .def(
            "unmap",
            [](Scene& scene, const PrimSet& prims, std::string_view attribute, nb::handle stream) {
                scene.Unmap(prims, attribute, StreamOf(stream));
            },
            nb::arg("prims"), nb::arg("attribute"), nb::arg("stream").none() = nb::none(),
            "As the unmap of one path, for each prim of a PrimSet: rows of one array mapped over "
            "them (`Renderer.map` of a PrimSet).")
        .def(
            "unmap",
            [](Scene& scene, const std::vector<Path>& paths, std::string_view attribute,
               nb::handle stream) { scene.Unmap(PrimSet(paths), attribute, StreamOf(stream)); },
            nb::arg("paths"), nb::arg("attribute"), nb::arg("stream").none() = nb::none(),
            "As the unmap of a PrimSet, for a list of paths.")
        .def(
            "write",
            [](Scene& scene, const std::vector<Path>& paths, std::string_view attribute,
               const AnyArray& rows, nb::handle stream) {
                scene.Write(paths, attribute, TensorFromArray(rows), StreamOf(stream));
            },
            nb::arg("paths"), nb::arg("attribute"), nb::arg("rows"),
            nb::arg("stream").none() = nb::none(),
            "Writes row i of `rows` as `attribute` of `paths[i]`, sharing one array. Transform "
            "ops written so from CUDA arrays (float32 rows of xformOp:translate, xformOp:orient as "
            "x, y, z, w, and xformOp:scale) pose the prims on the GPU: the meshes under them "
            "render without the poses coming to the host (give each prim ernie:xform:extent, or "
            "the poses are copied to the host to bound them). `stream` is as for `write`.")
        .def(
            "write",
            [](Scene& scene, const PrimSet& prims, std::string_view attribute,
               const AnyArray& rows, nb::handle stream) {
                scene.Write(prims, attribute, TensorFromArray(rows), StreamOf(stream));
            },
            nb::arg("prims"), nb::arg("attribute"), nb::arg("rows"),
            nb::arg("stream").none() = nb::none(),
            "As the write of many paths, to a PrimSet: its prims are looked up once and kept, the "
            "way to write a simulation's arrays every step.")
        .def(
            "read",
            [](const Scene& scene, const Path& path, std::string_view attribute) -> nb::object {
                const Tensor* tensor = scene.Read(path, attribute);
                return tensor != nullptr ? ArrayFromTensor(*tensor) : nb::none();
            },
            nb::arg("path"), nb::arg("attribute"),
            "A read-only view of the array written as `attribute`, or None.")
        .def("attributes", &Scene::Attributes, nb::arg("path"))
        .def("clear", &Scene::Clear, nb::arg("path"), nb::arg("attribute"))
        .def("attribute_version", &Scene::AttributeVersion, nb::arg("path"), nb::arg("attribute"))
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
                scene.SetLocalTransform(path, TransformFromPython(matrix));
            },
            nb::arg("path"), nb::arg("matrix"),
            "Gives the prim the local transform `matrix` (4 x 4), through its own transform ops "
            "(translate, a rotate op or orient, scale) where they can represent it, else as "
            "xformOp:transform.")
        .def("is_visible", &Scene::IsVisible, nb::arg("path"))
        .def("overlay", &Scene::GetOverlay, nb::arg("name"), nb::rv_policy::reference_internal,
             "The overlay (debug drawing layer) named `name`, created empty if missing.")
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
            "(path, Dirty) for every prim changed since `cursor`, which advances.");

    const auto triple = [](const glm::dvec3& v) { return nb::make_tuple(v.x, v.y, v.z); };
    nb::class_<IsdLineScanner>(
        m, "IsdLineScanner",
        "A line scanner (pushbroom camera) from its image support data (ALE's ISD JSON, as "
        "usgscsm reads it): its timing, detector and trajectory in the body-fixed frame, metres. "
        "Image coordinates are CSM's (line, sample), the first pixel's centre at (0.5, 0.5).")
        .def_static(
            "from_file",
            [](const std::string& path) { return IsdLineScanner(ReadIsdFile(path)); },
            nb::arg("path"), "The line scanner of the ISD file at `path`.")
        .def_static(
            "from_json", [](const std::string& json) { return IsdLineScanner(ReadIsd(json)); },
            nb::arg("json"), "The line scanner of ISD JSON text `json`.")
        .def_prop_ro("lines", [](const IsdLineScanner& s) { return s.Isd().lines; })
        .def_prop_ro("samples", [](const IsdLineScanner& s) { return s.Isd().samples; })
        .def_prop_ro("sensor", [](const IsdLineScanner& s) { return s.Isd().sensor; })
        .def_prop_ro("platform", [](const IsdLineScanner& s) { return s.Isd().platform; })
        .def("image_time", &IsdLineScanner::ImageTime, nb::arg("line"),
             "Ephemeris seconds (TDB past J2000) at which `line` is exposed.")
        .def(
            "position",
            [triple](const IsdLineScanner& s, double time) { return triple(s.Position(time)); },
            nb::arg("time"), "The sensor's body-fixed position at `time`.")
        .def(
            "sun_position",
            [triple](const IsdLineScanner& s, double time) {
                return triple(s.SunPosition(time));
            },
            nb::arg("time"), "The sun's body-fixed position at `time`.")
        .def(
            "image_to_ground",
            [triple](const IsdLineScanner& s, double line, double sample, double height) {
                return triple(s.ImageToGround(line, sample, height));
            },
            nb::arg("line"), nb::arg("sample"), nb::arg("height") = 0.0,
            "Where pixel (`line`, `sample`)'s ray meets the body's ellipsoid raised by `height`.")
        .def(
            "ground_to_image",
            [](const IsdLineScanner& s, nb::handle ground) {
                const glm::dvec2 image = s.GroundToImage(Vec3FromPython(ground));
                return nb::make_tuple(image.x, image.y);
            },
            nb::arg("ground"), "The (line, sample) that sees body-fixed point `ground`.")
        .def(
            "strip",
            [](const IsdLineScanner& s, std::uint32_t first, std::uint32_t lines) {
                const IsdLineScanner::Strip strip = s.StripOf(first, lines);
                return nb::make_tuple(MatrixToPython(strip.from), MatrixToPython(strip.to),
                                      strip.fromTime, strip.toTime);
            },
            nb::arg("first_line"), nb::arg("lines"),
            "(from, to, from_time, to_time): the camera poses and times bounding the strip of "
            "`lines` lines from `first_line`.")
        .def(
            "author_strip",
            [](const IsdLineScanner& s, Scene& scene, const PrimPath& pathArg, std::uint32_t first,
               std::uint32_t lines, bool atEnd) {
                const Path path = pathArg;
                const IsdLineScanner::Strip strip = s.StripOf(first, lines);
                const CameraParams& camera =strip.camera;
                using namespace tokens;
                // The camera's geometry and line time only: its f-number, ISO, lens
                // transmission and sensor stay as authored (`author_lroc_nac`, say).
                Prim prim = scene.Has(Path(path)) ? scene.GetPrim(Path(path))
                                                  : scene.Define(Path(path), kCamera);
                prim.Set(kTransform, atEnd ? strip.to : strip.from)
                    .Set(kCameraScan, std::string(kPushbroom))
                    .Set(kFocalLength, camera.focalLengthMm)
                    .Set(kVerticalAperture, camera.verticalApertureMm)
                    .Set(kHorizontalAperture,
                         camera.verticalApertureMm * camera.width / camera.height)
                    .Set(kHorizontalApertureOffset, camera.horizontalApertureOffsetMm)
                    .Set(kVerticalApertureOffset, 0.0)
                    .Set(kPushbroomLineOffset, camera.lineOffsetMm)
                    .Set(kLensDistortion,
                         std::vector<float>{static_cast<float>(camera.lens.k1),
                                            static_cast<float>(camera.lens.k2),
                                            static_cast<float>(camera.lens.k3), 0.0f, 0.0f})
                    .Set(kClippingRange, glm::dvec2(camera.nearPlane, camera.farPlane))
                    .Set(kExposureTime, camera.exposureTime);
                return nb::make_tuple(camera.width, camera.height);
            },
            nb::arg("scene"), nb::arg("path"), nb::arg("first_line"), nb::arg("lines"),
            nb::arg("at_end") = true,
            "Authors the Camera prim at `path` (a str, Path, Prim or typed prim) as the pushbroom "
            "of the strip of `lines` lines from `first_line` (with the lens's distortion along the detector fitted as "
            "`ernie:lens:distortion`), posed at its end (or its start), and returns the render "
            "resolution it needs, (samples, lines). Render an image strip by strip: author the "
            "first strip at its start and step once, then author each strip at its end and "
            "step.");

    nb::class_<IsdPushFrame>(
        m, "IsdPushFrame",
        "A push-frame camera (LROC's WAC, MRO's MARCI) from its image support data, as usgscsm's "
        "push-frame model reads it: the image is a stack of framelets, each a frame of "
        "`framelet_lines` lines exposed at once. Body-fixed frame, metres; CSM image "
        "coordinates.")
        .def_static(
            "from_file", [](const std::string& path) { return IsdPushFrame(ReadIsdFile(path)); },
            nb::arg("path"), "The push-frame camera of the ISD file at `path`.")
        .def_static(
            "from_json", [](const std::string& json) { return IsdPushFrame(ReadIsd(json)); },
            nb::arg("json"), "The push-frame camera of ISD JSON text `json`.")
        .def_prop_ro("lines", [](const IsdPushFrame& s) { return s.Isd().lines; })
        .def_prop_ro("samples", [](const IsdPushFrame& s) { return s.Isd().samples; })
        .def_prop_ro("sensor", [](const IsdPushFrame& s) { return s.Isd().sensor; })
        .def_prop_ro("platform", [](const IsdPushFrame& s) { return s.Isd().platform; })
        .def_prop_ro("framelet_lines", &IsdPushFrame::FrameletLines)
        .def_prop_ro("framelets", &IsdPushFrame::Framelets)
        .def("image_time", &IsdPushFrame::ImageTime, nb::arg("line"),
             "Ephemeris seconds (TDB past J2000) at the middle of `line`'s framelet's exposure.")
        .def(
            "position",
            [triple](const IsdPushFrame& s, double time) { return triple(s.Position(time)); },
            nb::arg("time"), "The sensor's body-fixed position at `time`.")
        .def(
            "sun_position",
            [triple](const IsdPushFrame& s, double time) { return triple(s.SunPosition(time)); },
            nb::arg("time"), "The sun's body-fixed position at `time`.")
        .def(
            "image_to_ground",
            [triple](const IsdPushFrame& s, double line, double sample, double height) {
                return triple(s.ImageToGround(line, sample, height));
            },
            nb::arg("line"), nb::arg("sample"), nb::arg("height") = 0.0,
            "Where pixel (`line`, `sample`)'s ray meets the body's ellipsoid raised by `height`.")
        .def(
            "ground_to_image",
            [](const IsdPushFrame& s, nb::handle ground) {
                const glm::dvec2 image = s.GroundToImage(Vec3FromPython(ground));
                return nb::make_tuple(image.x, image.y);
            },
            nb::arg("ground"), "The (line, sample) that sees body-fixed point `ground`.")
        .def(
            "author_framelet",
            [](const IsdPushFrame& s, Scene& scene, const PrimPath& pathArg, std::uint32_t index) {
                const Path path = pathArg;
                const IsdPushFrame::Framelet framelet = s.FrameletOf(index);
                const CameraParams& camera =framelet.camera;
                using namespace tokens;
                // The camera's geometry and exposure only, as `IsdLineScanner.author_strip`.
                Prim prim = scene.Has(Path(path)) ? scene.GetPrim(Path(path))
                                                  : scene.Define(Path(path), kCamera);
                prim.Set(kTransform, camera.transform)
                    .Set(kCameraScan, std::string(kFrameScan))
                    .Set(kFocalLength, camera.focalLengthMm)
                    .Set(kVerticalAperture, camera.verticalApertureMm)
                    .Set(kHorizontalAperture,
                         camera.verticalApertureMm * camera.width / camera.height)
                    .Set(kHorizontalApertureOffset, camera.horizontalApertureOffsetMm)
                    .Set(kVerticalApertureOffset, camera.verticalApertureOffsetMm)
                    .Set(kPushbroomLineOffset, 0.0)
                    .Set(kLensDistortion,
                         std::vector<float>{static_cast<float>(camera.lens.k1),
                                            static_cast<float>(camera.lens.k2),
                                            static_cast<float>(camera.lens.k3),
                                            static_cast<float>(camera.lens.p1),
                                            static_cast<float>(camera.lens.p2)})
                    .Set(kClippingRange, glm::dvec2(camera.nearPlane, camera.farPlane))
                    .Set(kExposureTime, camera.exposureTime);
                return nb::make_tuple(camera.width, camera.height, framelet.firstLine,
                                      framelet.rowsReversed, framelet.time);
            },
            nb::arg("scene"), nb::arg("path"), nb::arg("framelet"),
            "Authors the Camera prim at `path` (a str, Path, Prim or typed prim) as framelet "
            "`framelet` (in image order, from the top): a frame camera posed when it is exposed, with the lens's distortion fitted as "
            "`ernie:lens:distortion`. Returns (samples, lines, first_line, rows_reversed, time): "
            "render at that resolution, flip the rows if `rows_reversed`, and place them at "
            "`first_line` of the image.");

    nb::class_<CahvoreModel>(
        m, "CahvoreModel",
        "A camera in JPL's CAHV, CAHVOR or CAHVORE model, as Mars rovers' cameras are calibrated. "
        "Image coordinates are the models': (0, 0) the centre of the upper-left pixel, x right, "
        "y down.")
        .def_prop_ro("kind",
                     [](const CahvoreModel& model) {
                         return model.kind == CahvoreModel::Kind::Cahv     ? "cahv"
                                : model.kind == CahvoreModel::Kind::Cahvor ? "cahvor"
                                                                           : "cahvore";
                     })
        .def_prop_ro("width", [](const CahvoreModel& model) { return model.width; })
        .def_prop_ro("height", [](const CahvoreModel& model) { return model.height; })
        .def(
            "image_to_ray",
            [triple](const CahvoreModel& model, double x, double y) {
                const CahvoreModel::Ray ray = model.ImageToRay({x, y});
                return nb::make_tuple(triple(ray.origin), triple(ray.direction));
            },
            nb::arg("x"), nb::arg("y"), "(origin, unit direction) of the ray through image (x, y).")
        .def(
            "point_to_image",
            [](const CahvoreModel& model, nb::handle point) {
                const glm::dvec2 image = model.PointToImage(Vec3FromPython(point));
                return nb::make_tuple(image.x, image.y);
            },
            nb::arg("point"), "The image (x, y) of 3D point `point`.")
        .def(
            "fit",
            [](const CahvoreModel& model, double pixelPitchUm) {
                const CahvoreModel::ErnieCamera fit = model.ToCamera(pixelPitchUm);
                nb::dict out;
                out["residual_pixels"] = fit.residualPixels;
                out["pupil_shift_max"] = fit.pupilShiftMax;
                out["field_angle_max"] = fit.fieldAngleMax;
                out["focal_length_mm"] = fit.camera.focalLengthMm;
                out["fisheye"] = fit.camera.lens.projection == LensProjection::Fisheye;
                out["rows_reversed"] = fit.rowsReversed;
                return out;
            },
            nb::arg("pixel_pitch_um"),
            "How the model converts to Ernie's camera at pixel pitch `pixel_pitch_um`: its "
            "residual_pixels, pupil_shift_max, field_angle_max (radians), focal_length_mm and "
            "whether it is a fisheye.")
        .def(
            "author_camera",
            [](const CahvoreModel& model, Scene& scene, const PrimPath& pathArg,
               double pixelPitchUm, std::uint32_t raysStep) {
                const Path path = pathArg;
                const CahvoreModel::ErnieCamera fit = model.ToCamera(pixelPitchUm);
                const CameraParams& camera =fit.camera;
                using namespace tokens;
                // The camera's geometry only, as `IsdFrame.author_camera`; its pose in the
                // model's frame (parent the prim to that frame's Xform).
                Prim prim = scene.Has(Path(path)) ? scene.GetPrim(Path(path))
                                                  : scene.Define(Path(path), kCamera);
                if (raysStep > 0) {
                    // The model's own rays, which the sensor interpolates in place of the fit.
                    const LensRayTable table = model.RayTable(raysStep);
                    std::vector<float> values;
                    values.reserve(table.rays.size() * 3);
                    for (const glm::vec3& ray : table.rays) {
                        values.insert(values.end(), {ray.x, ray.y, ray.z});
                    }
                    scene.Write(Path(path), kLensRays,
                                Tensor::Copy(std::span<const float>(values),
                                             {std::int64_t(table.rows), std::int64_t(table.columns),
                                              3}));
                    scene.Set(Path(path), kLensRaysStep, static_cast<int>(raysStep));
                }
                const bool fisheye = camera.lens.projection == LensProjection::Fisheye;
                prim.Set(kTransform, camera.transform)
                    .Set(kCameraScan, std::string(kFrameScan))
                    .Set(kLensProjection, std::string(fisheye ? kFisheye : kPerspective))
                    .Set(kFocalLength, camera.focalLengthMm)
                    .Set(kVerticalAperture, camera.verticalApertureMm)
                    .Set(kHorizontalAperture,
                         camera.verticalApertureMm * camera.width / camera.height)
                    .Set(kHorizontalApertureOffset, camera.horizontalApertureOffsetMm)
                    .Set(kVerticalApertureOffset, camera.verticalApertureOffsetMm)
                    .Set(kLensDistortion,
                         std::vector<float>{static_cast<float>(camera.lens.k1),
                                            static_cast<float>(camera.lens.k2),
                                            static_cast<float>(camera.lens.k3),
                                            static_cast<float>(camera.lens.p1),
                                            static_cast<float>(camera.lens.p2)})
                    .Set(kLensFisheye,
                         std::vector<float>{static_cast<float>(camera.lens.fisheye[0]),
                                            static_cast<float>(camera.lens.fisheye[1]),
                                            static_cast<float>(camera.lens.fisheye[2]),
                                            static_cast<float>(camera.lens.fisheye[3])})
                    .Set(kClippingRange, glm::dvec2(camera.nearPlane, camera.farPlane));
                return nb::make_tuple(camera.width, camera.height, fit.rowsReversed,
                                      fit.residualPixels);
            },
            nb::arg("scene"), nb::arg("path"), nb::arg("pixel_pitch_um"),
            nb::arg("rays_step") = 8,
            "Authors the Camera prim at `path` (a str, Path, Prim or typed prim) as the model in "
            "Ernie's camera (perspective with "
            "Brown-Conrady terms, or a Kannala-Brandt fisheye for CAHVORE), posed in the model's "
            "frame, and with `rays_step` above 0 the model's own rays every that many pixels "
            "(`ernie:lens:rays`), which the sensor follows in place of the fit: exact but for "
            "CAHVORE's moving entrance pupil. Returns (width, height, rows_reversed, "
            "residual_pixels), the residual the fit's alone: render at that resolution and flip "
            "the rows if `rows_reversed`.");
    m.def(
        "read_cahvore_models",
        [](const std::string& path) {
            nb::list out;
            for (NamedCahvore& named : ReadCahvoreModelsFile(path)) {
                out.append(nb::make_tuple(named.name, named.frame, named.model));
            }
            return out;
        },
        nb::arg("path"),
        "(name, frame, CahvoreModel) for each model of the camera model JSON file at `path`, "
        "in NASA-JPL's m2020-urdf-models form.");

    nb::class_<IsdFrame>(
        m, "IsdFrame",
        "A frame camera from its image support data, as usgscsm's frame model reads it: the "
        "whole image exposed at once from one pose. Body-fixed frame, metres; CSM image "
        "coordinates.")
        .def_static(
            "from_file", [](const std::string& path) { return IsdFrame(ReadIsdFile(path)); },
            nb::arg("path"), "The frame camera of the ISD file at `path`.")
        .def_static(
            "from_json", [](const std::string& json) { return IsdFrame(ReadIsd(json)); },
            nb::arg("json"), "The frame camera of ISD JSON text `json`.")
        .def_prop_ro("lines", [](const IsdFrame& s) { return s.Isd().lines; })
        .def_prop_ro("samples", [](const IsdFrame& s) { return s.Isd().samples; })
        .def_prop_ro("sensor", [](const IsdFrame& s) { return s.Isd().sensor; })
        .def_prop_ro("platform", [](const IsdFrame& s) { return s.Isd().platform; })
        .def_prop_ro("image_time", &IsdFrame::ImageTime,
                     "Ephemeris seconds (TDB past J2000) at which the image is exposed.")
        .def(
            "position",
            [triple](const IsdFrame& s, double time) { return triple(s.Position(time)); },
            nb::arg("time"), "The sensor's body-fixed position at `time`.")
        .def(
            "sun_position",
            [triple](const IsdFrame& s, double time) { return triple(s.SunPosition(time)); },
            nb::arg("time"), "The sun's body-fixed position at `time`.")
        .def(
            "image_to_ground",
            [triple](const IsdFrame& s, double line, double sample, double height) {
                return triple(s.ImageToGround(line, sample, height));
            },
            nb::arg("line"), nb::arg("sample"), nb::arg("height") = 0.0,
            "Where pixel (`line`, `sample`)'s ray first meets the body's ellipsoid raised by "
            "`height`.")
        .def(
            "ground_to_image",
            [](const IsdFrame& s, nb::handle ground) {
                const glm::dvec2 image = s.GroundToImage(Vec3FromPython(ground));
                return nb::make_tuple(image.x, image.y);
            },
            nb::arg("ground"), "The (line, sample) that sees body-fixed point `ground`.")
        .def(
            "author_camera",
            [](const IsdFrame& s, Scene& scene, const PrimPath& pathArg) {
                const Path path = pathArg;
                const IsdFrame::FrameView view = s.View();
                const CameraParams& camera =view.camera;
                using namespace tokens;
                // The camera's geometry only, as `IsdLineScanner.author_strip`.
                Prim prim = scene.Has(Path(path)) ? scene.GetPrim(Path(path))
                                                  : scene.Define(Path(path), kCamera);
                prim.Set(kTransform, camera.transform)
                    .Set(kCameraScan, std::string(kFrameScan))
                    .Set(kFocalLength, camera.focalLengthMm)
                    .Set(kVerticalAperture, camera.verticalApertureMm)
                    .Set(kHorizontalAperture,
                         camera.verticalApertureMm * camera.width / camera.height)
                    .Set(kHorizontalApertureOffset, camera.horizontalApertureOffsetMm)
                    .Set(kVerticalApertureOffset, camera.verticalApertureOffsetMm)
                    .Set(kPushbroomLineOffset, 0.0)
                    .Set(kLensDistortion,
                         std::vector<float>{static_cast<float>(camera.lens.k1),
                                            static_cast<float>(camera.lens.k2),
                                            static_cast<float>(camera.lens.k3),
                                            static_cast<float>(camera.lens.p1),
                                            static_cast<float>(camera.lens.p2)})
                    .Set(kClippingRange, glm::dvec2(camera.nearPlane, camera.farPlane));
                return nb::make_tuple(camera.width, camera.height, view.rowsReversed);
            },
            nb::arg("scene"), nb::arg("path"),
            "Authors the Camera prim at `path` (a str, Path, Prim or typed prim) as the image's "
            "frame camera, posed when it is "
            "exposed, with the lens's distortion fitted as `ernie:lens:distortion`. Returns "
            "(samples, lines, rows_reversed): render at that resolution and flip the rows if "
            "`rows_reversed`.");

    m.def(
        "author_lroc_nac",
        [](Scene& scene, const PrimPath& camera, const PrimPath& product,
           const std::string& side) {
            if (side != "left" && side != "right") {
                throw nb::value_error("side must be 'left' or 'right'");
            }
            AuthorLrocNac(scene, Path(camera), Path(product),
                          side == "left" ? LrocNac::Left : LrocNac::Right);
        },
        nb::arg("scene"), nb::arg("camera"), nb::arg("product"), nb::arg("side") = "left",
        "Authors one of LROC's Narrow Angle Cameras (NAC-L or NAC-R) as Robinson et al. 2010 "
        "publish it (Table 2, Appendix A): Camera `camera`'s sensor (400-760 nm, gain, detector "
        "noise, full well, 12-bit ADC, f-number) and RenderProduct `product`'s readout chain "
        "(qe, noise, adc, even/odd readoutTaps, compandTable code 0) whose RawDN is the 8-bit "
        "numbers the NAC sends. Its geometry comes from the image's ISD "
        "(`IsdLineScanner.author_strip`). `camera` and `product` are each a str, Path, Prim or "
        "typed prim.");

    m.def(
        "author_lroc_wac",
        [](Scene& scene, const PrimPath& camera, const PrimPath& product, int band) {
            try {
                AuthorLrocWac(scene, Path(camera), Path(product), LrocWacBandNamed(band));
            } catch (const std::invalid_argument& error) {
                throw nb::value_error(error.what());
            }
        },
        nb::arg("scene"), nb::arg("camera"), nb::arg("product"), nb::arg("band") = 605,
        "Authors one band of LROC's Wide Angle Camera (320, 360, 415, 565, 605, 645 or 690 nm, "
        "by name or effective wavelength) as Robinson et al. 2010 publish it (Tables 3 and 4): "
        "Camera `camera`'s sensor (the passband at the effective QE that gives Table 4's "
        "responsivity, gain, detector noise, full well, 11-bit ADC, f-number; its spectral "
        "range narrowed to the passband) and RenderProduct `product`'s readout chain (qe, "
        "noise, adc). Its geometry comes from the image's ISD "
        "(`IsdPushFrame.author_framelet`). `camera` and `product` are each a str, Path, Prim or "
        "typed prim.");

    const auto bayerPattern = [](const std::string& name) {
        const std::pair<const char*, BayerPattern> patterns[] = {{"rggb", BayerPattern::Rggb},
                                                                 {"bggr", BayerPattern::Bggr},
                                                                 {"grbg", BayerPattern::Grbg},
                                                                 {"gbrg", BayerPattern::Gbrg}};
        for (const auto& [text, pattern] : patterns) {
            if (name == text) {
                return pattern;
            }
        }
        throw nb::value_error("pattern must be 'rggb', 'bggr', 'grbg' or 'gbrg'");
    };
    m.def(
        "author_mars2020_navcam",
        [bayerPattern](Scene& scene, const PrimPath& camera, const PrimPath& product,
                       const std::string& pattern, double jpegQuality) {
            AuthorMars2020Navcam(scene, Path(camera), Path(product), bayerPattern(pattern),
                                 jpegQuality);
        },
        nb::arg("scene"), nb::arg("camera"), nb::arg("product"), nb::arg("pattern") = "rggb",
        nb::arg("jpeg_quality") = 85.0,
        "Authors one of the Mars 2020 rover's Navcams as Maki et al. 2020 publish it (Table 2): "
        "Camera `camera`'s f-theta fisheye (19.1 mm, f/12, 96 x 73 degrees over 5120 x 3840 "
        "pixels of 6.4 um) and RGB sensor (gain 0.24 DN/e-, 8 e- dark noise, 15,000 e- full "
        "well, 12-bit ADC), and RenderProduct `product`'s chain (qe, bayer, noise, adc, "
        "demosaic; cie, exposure, jpeg). The paper gives neither the Bayer layout (`pattern`) "
        "nor the JPEG quality (`jpeg_quality`). `camera` and `product` are each a str, Path, "
        "Prim or typed prim.");
    m.def(
        "author_mars2020_navcam_pair",
        [bayerPattern](Scene& scene, const PrimPath& rig, const PrimPath& left,
                       const PrimPath& right, const std::string& pattern, double jpegQuality) {
            AuthorMars2020NavcamPair(scene, Path(rig), Path(left), Path(right),
                                     bayerPattern(pattern), jpegQuality);
        },
        nb::arg("scene"), nb::arg("rig"), nb::arg("left_product"), nb::arg("right_product"),
        nb::arg("pattern") = "rggb", nb::arg("jpeg_quality") = 85.0,
        "The Navcam stereo pair on Xform `rig` (defined if missing; pose it as the cameras' "
        "frame, looking down -z with +y up): Cameras `rig`/Left and `rig`/Right 42.4 cm apart "
        "along x, each as `author_mars2020_navcam` with its product. `rig` and the products are "
        "each a str, Path, Prim or typed prim.");
    m.def(
        "author_mars2020_hazcam",
        [bayerPattern](Scene& scene, const PrimPath& camera, const PrimPath& product,
                       const std::string& pattern, double jpegQuality) {
            AuthorMars2020Hazcam(scene, Path(camera), Path(product), bayerPattern(pattern),
                                 jpegQuality);
        },
        nb::arg("scene"), nb::arg("camera"), nb::arg("product"), nb::arg("pattern") = "rggb",
        nb::arg("jpeg_quality") = 85.0,
        "Authors one of the Mars 2020 rover's Hazcams as Maki et al. 2020 publish it (Table 2), as "
        "`author_mars2020_navcam` does a Navcam but through the Hazcam's f-theta fisheye (14.0 mm, "
        "f/12, 136 x 102 degrees, focused at 0.9 m), which renders on cube faces. `camera` and "
        "`product` are each a str, Path, Prim or typed prim.");
    m.def(
        "author_mars2020_hazcam_pair",
        [bayerPattern](Scene& scene, const PrimPath& rig, const PrimPath& left,
                       const PrimPath& right, const std::string& which,
                       const std::string& pattern, double jpegQuality) {
            if (which != "front" && which != "rear") {
                throw nb::value_error("which must be 'front' or 'rear'");
            }
            AuthorMars2020HazcamPair(scene, Path(rig), Path(left), Path(right),
                                     which == "front" ? Mars2020Hazcams::Front
                                                      : Mars2020Hazcams::Rear,
                                     bayerPattern(pattern), jpegQuality);
        },
        nb::arg("scene"), nb::arg("rig"), nb::arg("left_product"), nb::arg("right_product"),
        nb::arg("which") = "front", nb::arg("pattern") = "rggb", nb::arg("jpeg_quality") = 85.0,
        "A Hazcam stereo pair on Xform `rig`: Cameras `rig`/Left and `rig`/Right 24.8 cm apart "
        "(front) or 93.4 cm (rear) along x, each as `author_mars2020_hazcam` with its product. "
        "`rig` and the products are each a str, Path, Prim or typed prim.");
    m.def(
        "author_singlet_chromatic",
        [](Scene& scene, const PrimPath& camera, double nd, double abbe, double lateral) {
            using namespace tokens;
            const Path path = camera;
            if (!scene.Has(path) || scene.TypeOf(path) != kCamera) {
                throw nb::value_error(("'" + path.String() + "' is not a Camera").c_str());
            }
            LensModel lens;
            try {
                SetSingletChromatic(lens, Camera(scene, path).GetFocalLength(), nd, abbe, lateral);
            } catch (const std::invalid_argument& error) {
                throw nb::value_error(error.what());
            }
            const auto pairs = [](const Spectrum& spectrum) {
                std::vector<float> out;
                for (double nm = 300.0; nm <= 1100.0 + 1e-9; nm += 10.0) {
                    out.push_back(static_cast<float>(nm));
                    out.push_back(static_cast<float>(spectrum(nm)));
                }
                return out;
            };
            scene.GetPrim(path)
                .Set(kLensChromaticScale, pairs(lens.chromaticScale))
                .Set(kLensChromaticFocus, pairs(lens.chromaticFocusMm));
        },
        nb::arg("scene"), nb::arg("camera"), nb::arg("nd") = 1.5168, nb::arg("abbe") = 64.17,
        nb::arg("lateral") = 0.0,
        "Authors Camera `camera`'s chromatic aberration (`ernie:lens:chromatic:scale` and "
        "`:focus`) as a thin singlet of its focal length in a glass of index `nd` and Abbe number "
        "`abbe` (by default SCHOTT N-BK7) would have: axial colour from the glass's dispersion "
        "(Cauchy, f_C - f_F = f / V), lateral colour `lateral` times its relative focal change "
        "(0 with the stop at the lens, up to 1 for a distant stop). The `defocus` filter blurs "
        "by the axial colour; the lens resamples each band by the lateral. `camera` is a str, "
        "Path, Prim or typed prim.");
    m.def(
        "author_mastcamz",
        [bayerPattern](Scene& scene, const PrimPath& camera, const PrimPath& product,
                       const std::string& eye, int filter, double focalLength,
                       const std::string& pattern) {
            if (eye != "left" && eye != "right") {
                throw nb::value_error("eye must be 'left' or 'right'");
            }
            try {
                AuthorMastcamZ(scene, Path(camera), Path(product),
                               eye == "left" ? MastcamZEye::Left : MastcamZEye::Right, filter,
                               focalLength, bayerPattern(pattern));
            } catch (const std::invalid_argument& error) {
                throw nb::value_error(error.what());
            }
        },
        nb::arg("scene"), nb::arg("camera"), nb::arg("product"), nb::arg("eye") = "left",
        nb::arg("filter") = 0, nb::arg("focal_length") = 26.0, nb::arg("pattern") = "rggb",
        "Authors one Mastcam-Z camera through filter `filter` as Bell et al. 2021 publish it "
        "(Tables 3 and 5): the KAI-2020CM (1600 x 1200 pixels of 7.4 um, 15.6 e-/DN, 11-bit) "
        "behind the zoom at `focal_length` (26 to 110 mm). Filter 0 is the Bayer RGB; 1 to 6 "
        "the narrow bands (the left eye's 442 to 800 nm, the right's 800 to 1022 nm); 7 the "
        "neutral-density solar filter. `camera` and `product` are each a str, Path, Prim or "
        "typed prim.");
    m.def(
        "author_mastcamz_pair",
        [bayerPattern](Scene& scene, const PrimPath& rig, const PrimPath& left,
                       const PrimPath& right, int leftFilter, int rightFilter,
                       double focalLength, const std::string& pattern) {
            try {
                AuthorMastcamZPair(scene, Path(rig), Path(left), Path(right), leftFilter,
                                   rightFilter, focalLength, bayerPattern(pattern));
            } catch (const std::invalid_argument& error) {
                throw nb::value_error(error.what());
            }
        },
        nb::arg("scene"), nb::arg("rig"), nb::arg("left_product"), nb::arg("right_product"),
        nb::arg("left_filter") = 0, nb::arg("right_filter") = 0, nb::arg("focal_length") = 26.0,
        nb::arg("pattern") = "rggb",
        "The Mastcam-Z pair on Xform `rig`: Cameras `rig`/Left and `rig`/Right 24.4 cm apart "
        "along x, each toed in 1.15 degrees, through their filters. `rig` and the products are "
        "each a str, Path, Prim or typed prim.");

    m.def(
        "look_at_transform",
        [](nb::handle eye, nb::handle target, nb::handle up) {
            return MatrixToPython(
                LookAtTransform(Vec3FromPython(eye), Vec3FromPython(target), Vec3FromPython(up)));
        },
        nb::arg("eye"), nb::arg("target"), nb::arg("up") = nb::make_tuple(0.0, 0.0, 1.0),
        "Transform of a Camera at `eye` looking at `target`.");
    m.def(
        "distant_light_transform",
        [](nb::handle towards) {
            return MatrixToPython(DistantLightTransform(Vec3FromPython(towards)));
        },
        nb::arg("towards_light"), "Transform of a DistantLight shining from `towards_light`.");
    m.def(
        "load_image",
        [](const std::filesystem::path& path, bool srgb) {
            const Image image =
                LoadImage(path, srgb ? ImageEncoding::Srgb : ImageEncoding::Linear);
            const auto* values = reinterpret_cast<const float*>(image.pixels.data());
            return ArrayFromTensor(Tensor::Copy<float>(
                std::span(values, image.pixels.size() * 4),
                {static_cast<std::int64_t>(image.height), static_cast<std::int64_t>(image.width),
                 4}), true);
        },
        nb::arg("path"), nb::arg("srgb") = true,
        "A PNG, JPEG or OpenEXR image as a linear float32 (rows, cols, 4) array: PNG and JPEG "
        "values decode as sRGB (srgb) or as linear data such as normal maps; EXR is linear.");
    m.def("julian_date", &JulianDate, nb::arg("year"), nb::arg("month"), nb::arg("day"),
          nb::arg("hours") = 0.0,
          "The Julian date of `hours` UT past midnight of a Gregorian date (Meeus, chapter 7).");
    m.def("greenwich_sidereal_time", &GreenwichMeanSiderealTime, nb::arg("julian_date"),
          "Greenwich mean sidereal time in degrees at a Julian date (Meeus, equation 12.4).");
    m.def(
        "celestial_to_local",
        [](double latitude, double longitude, double julianDate) {
            return MatrixToPython(CelestialToLocal(latitude, longitude, julianDate));
        },
        nb::arg("latitude"), nb::arg("longitude"), nb::arg("julian_date"),
        "A StarField's transform for the sky over a site (degrees, east positive) at a Julian "
        "date: J2000 equatorial to x east, y north, z up.");
    m.def("star_illuminance", &StarIlluminance, nb::arg("magnitude"),
          nb::arg("zero_point") = kMagnitudeZeroLux,
          "Lux of a star of visual magnitude `magnitude` (2.54e-6 lux at magnitude 0).");
    m.def("color_index_to_temperature", &ColorIndexToTemperature, nb::arg("color_index"),
          "A star's temperature in kelvin from its B-V colour index (Ballesteros 2012).");
    m.def(
        "synthetic_stars",
        [](std::size_t count, std::uint64_t seed, double faintest) {
            StarCatalogue stars = SyntheticStars(count, seed, faintest);
            nb::dict result;
            result["right_ascension"] = stars.rightAscension;
            result["declination"] = stars.declination;
            result["magnitude"] = stars.magnitude;
            result["color_index"] = stars.colorIndex;
            return result;
        },
        nb::arg("count"), nb::arg("seed") = 1, nb::arg("faintest") = 6.5,
        "`count` made-up stars as lists of right_ascension and declination (degrees), "
        "magnitude and color_index, the same for the same seed.");

    nb::module_ tokens = m.def_submodule("tokens", "Prim type and attribute names.");
    const std::pair<const char*, std::string_view> names[] = {
        {"XFORM", tokens::kXform},
        {"SCOPE", tokens::kScope},
        {"MESH", tokens::kMesh},
        {"POINT_INSTANCER", tokens::kPointInstancer},
        {"GEOM_POINTS", tokens::kGeomPoints},
        {"VOLUME", tokens::kVolume},
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
        {"SPHERE_LIGHT", tokens::kSphereLight},
        {"RECT_LIGHT", tokens::kRectLight},
        {"DISK_LIGHT", tokens::kDiskLight},
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
        {"WIDTHS", tokens::kWidths},
        {"FACE_VERTEX_COUNTS", tokens::kFaceVertexCounts},
        {"FACE_VERTEX_INDICES", tokens::kFaceVertexIndices},
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
        {"SENSOR_FILTER", tokens::kSensorFilter},
        {"FILTER_TYPE", tokens::kFilterType},
        {"HORIZONTAL_APERTURE_OFFSET", tokens::kHorizontalApertureOffset},
        {"VERTICAL_APERTURE_OFFSET", tokens::kVerticalApertureOffset},
        {"EXPOSURE_TIME", tokens::kExposureTime},
        {"EXPOSURE_ISO", tokens::kExposureIso},
        {"EXPOSURE_FSTOP", tokens::kExposureFStop},
        {"EXPOSURE_RESPONSIVITY", tokens::kExposureResponsivity},
        {"F_STOP", tokens::kFStop},
        {"FOCUS_DISTANCE", tokens::kFocusDistance},
        {"SHUTTER_OPEN", tokens::kShutterOpen},
        {"SHUTTER_CLOSE", tokens::kShutterClose},
        {"SHUTTER_SAMPLES", tokens::kShutterSamples},
        {"SHUTTER_ROLLING", tokens::kShutterRolling},
        {"SPECTRAL_RANGE", tokens::kSpectralRange},
        {"SPECTRAL_BANDS", tokens::kSpectralBands},
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
        {"NO_DATA", tokens::kNoData},
        {"EXAGGERATION", tokens::kExaggeration},
        {"HEIGHTS_DIRTY", tokens::kHeightsDirty},
        {"BRDF", tokens::kBrdf},
        {"ALBEDO", tokens::kAlbedo},
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
        {"USD_PRIMVAR_READER_FLOAT2", tokens::kUsdPrimvarReaderFloat2},
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
        {"COLOR_MODE", tokens::kColorMode},
        {"HILLSHADE", tokens::kHillshade},
        {"COLORMAP", tokens::kColormap},
        {"SHADED_COLORMAP", tokens::kShadedColormap},
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
        {"SHADOW_FIT", tokens::kShadowFit},
        {"EXPOSURE_COMPENSATION", tokens::kExposureCompensation},
        {"METER", tokens::kMeter},
        {"PHYSICAL_ILLUMINANT", tokens::kPhysicalIlluminant},
        {"PHYSICAL_CUSTOM_ILLUMINANT", tokens::kPhysicalCustomIlluminant},
        {"PHOTOMETRIC_ILLUMINANCE", tokens::kPhotometricIlluminance},
        {"PHOTOMETRIC_ILLUMINANCE_DISTANCE", tokens::kPhotometricIlluminanceDistance},
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
        {"WHITE", tokens::kWhite},
        {"BLACKBODY", tokens::kBlackbody},
        {"ILLUMINANT_D", tokens::kIlluminantD},
        {"CUSTOM", tokens::kCustom},
        {"VISIBLE", tokens::kVisible},
        {"OVERLAYS", tokens::kOverlays},
        {"MINIMAL_MODE", tokens::kMinimalMode},
        {"MINIMAL_CONSTANT_COLOR", tokens::kMinimalConstantColor},
        {"NONE", tokens::kNone},
        {"CONSTANT_DIFFUSE", tokens::kConstantDiffuse},
        {"TEXTURED_DIFFUSE", tokens::kTexturedDiffuse},
        {"OCCLUDERS", tokens::kOccluders},
        {"TWO_PASS", tokens::kTwoPass},
        {"CLEAR_COLOR", tokens::kClearColor},
        {"SHADOW_CASCADES", tokens::kShadowCascades},
        {"SHADOW_RESOLUTION", tokens::kShadowResolution},
        {"SHADOW_SOFTNESS", tokens::kShadowSoftness},
        {"SHADOW_NORMAL_OFFSET", tokens::kShadowNormalOffset},
        {"SHADOW_SPLIT_LAMBDA", tokens::kShadowSplitLambda},
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
