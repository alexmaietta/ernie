#include "bindings.h"

#include "ernie/scene/scene.h"
#include "ernie/scene/schema.h"
#include "ernie/scene/tokens.h"

#include <nanobind/operators.h>
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

}  // namespace

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
    nb::object sequence = nb::borrow(object);
    if (nb::hasattr(object, "tolist")) {
        sequence = object.attr("tolist")();
        if (IsNumber(sequence)) {
            return ValueFromPython(sequence);
        }
    }
    if (!nb::isinstance<nb::list>(sequence) && !nb::isinstance<nb::tuple>(sequence)) {
        throw nb::type_error("Scene values are bool, int, float, str, Path, a sequence of numbers "
                             "or a 4 x 4 matrix");
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
            } else if constexpr (std::is_same_v<T, glm::vec4>) {
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

    nb::class_<Prim>(m, "Prim", "Handle to a prim in a Scene.")
        .def_prop_ro("path", &Prim::GetPath)
        .def_prop_ro("type", &Prim::Type)
        .def_prop_ro("valid", &Prim::IsValid)
        .def("__bool__", &Prim::IsValid)
        .def(
            "set",
            [](Prim& prim, std::string_view attribute, nb::handle value) -> Prim& {
                return prim.Set(attribute, ValueFromPython(value));
            },
            nb::arg("attribute"), nb::arg("value").none(), nb::rv_policy::reference_internal,
            "Sets `attribute`; returns the prim for chaining.")
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
            "write",
            [](Scene& scene, const std::vector<Path>& paths, std::string_view attribute,
               const AnyArray& rows) { scene.Write(paths, attribute, TensorFromArray(rows)); },
            nb::arg("paths"), nb::arg("attribute"), nb::arg("rows"),
            "Writes row i of `rows` as `attribute` of `paths[i]`, sharing one array.")
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

    nb::module_ tokens = m.def_submodule("tokens", "Prim type and attribute names.");
    const std::pair<const char*, std::string_view> names[] = {
        {"XFORM", tokens::kXform},
        {"SCOPE", tokens::kScope},
        {"MESH", tokens::kMesh},
        {"POINT_INSTANCER", tokens::kPointInstancer},
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
        {"VISIBILITY", tokens::kVisibility},
        {"INHERITED", tokens::kInherited},
        {"INVISIBLE", tokens::kInvisible},
        {"POINTS", tokens::kPoints},
        {"NORMALS", tokens::kNormals},
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
        {"SPECTRAL_RANGE", tokens::kSpectralRange},
        {"SPECTRAL_BANDS", tokens::kSpectralBands},
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
        {"SENSOR_NOISE", tokens::kSensorNoise},
        {"SENSOR_SEED", tokens::kSensorSeed},
        {"HEIGHTS", tokens::kHeights},
        {"ORIGIN", tokens::kOrigin},
        {"CELL_SIZE", tokens::kCellSize},
        {"NO_DATA", tokens::kNoData},
        {"EXAGGERATION", tokens::kExaggeration},
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
        {"WHITE", tokens::kWhite},
        {"BLACKBODY", tokens::kBlackbody},
        {"ILLUMINANT_D", tokens::kIlluminantD},
        {"CUSTOM", tokens::kCustom},
        {"VISIBLE", tokens::kVisible},
        {"OVERLAYS", tokens::kOverlays},
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
