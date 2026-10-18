// SPDX-License-Identifier: BSD-3-Clause
/// @file scene.cpp
/// @brief Python bindings of the scene: prims, attributes as Python values, arrays shared through
/// DLPack, JSON scene files and transforms. Values pass as scene files write them, so Python reads
/// and authors attributes by the same rules as JSON.
#include "ernie/python/scene.h"

#include "ernie/python/dlpack.h"
#include "ernie/scene/json.h"
#include "ernie/scene/json_value.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/xform.h"
#include "ernie/util/algorithm.h"

#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace nb = nanobind;

namespace ernie::python {
namespace {

/// A prim of a scene, as long as it is not removed.
struct Prim {
    PrimId id;
};

/// A prim, or the path of one.
using Target = std::variant<Prim, std::string>;

PrimId Resolve(const Scene& scene, const Target& target) {
    if (const Prim* prim = std::get_if<Prim>(&target)) {
        if (!scene.Store().Alive(prim->id)) {
            throw nb::value_error("The prim has been removed");
        }
        return prim->id;
    }
    return scene.Require(Path(std::get<std::string>(target)));
}

/// @return A Python value as JSON: numbers, strings and None, and sequences of them nested.
Json ToJson(nb::handle value) {
    if (value.is_none()) {
        return nullptr;
    }
    if (nb::isinstance<nb::bool_>(value)) {
        return nb::cast<bool>(value);
    }
    if (nb::isinstance<nb::int_>(value)) {
        return nb::cast<std::int64_t>(value);
    }
    if (nb::isinstance<nb::float_>(value)) {
        return nb::cast<double>(value);
    }
    if (nb::isinstance<nb::str>(value)) {
        return nb::cast<std::string>(value);
    }
    if (nb::isinstance<nb::list>(value) || nb::isinstance<nb::tuple>(value)) {
        Json items = Json::array();
        for (nb::handle item : value) {
            items.push_back(ToJson(item));
        }
        return items;
    }
    // NumPy's scalars, which are no Python numbers.
    if (nb::hasattr(value, "__index__")) {
        return nb::cast<std::int64_t>(value.attr("__index__")());
    }
    if (nb::hasattr(value, "__float__")) {
        return nb::cast<double>(value.attr("__float__")());
    }
    throw nb::type_error(("A scene value cannot be a " +
                          nb::cast<std::string>(nb::str(value.type().attr("__name__"))))
                             .c_str());
}

/// @return JSON as Python values: arrays as lists.
nb::object ToPython(const Json& json) {
    switch (json.type()) {
    case Json::value_t::boolean:
        return nb::bool_(json.get<bool>());
    case Json::value_t::number_integer:
        return nb::int_(json.get<std::int64_t>());
    case Json::value_t::number_unsigned:
        return nb::int_(json.get<std::uint64_t>());
    case Json::value_t::number_float:
        return nb::float_(json.get<double>());
    case Json::value_t::string:
        return nb::str(json.get_ref<const std::string&>().c_str());
    case Json::value_t::array: {
        nb::list items;
        for (const Json& item : json) {
            items.append(ToPython(item));
        }
        return items;
    }
    default:
        return nb::none();
    }
}

/// Runs `edit`, naming the prim and attribute in what it throws.
void Attempt(const Scene& scene, PrimId prim, std::string_view attribute,
             const std::function<void()>& edit) {
    try {
        edit();
    } catch (const std::exception& error) {
        const std::string message =
            "'" + scene.PrimPath(prim).String() + "' " + std::string(attribute) + ": ";
        throw nb::value_error((message + error.what()).c_str());
    }
}

/// @return Whether a value is an array to share: a NumPy array or another DLPack producer.
bool IsArray(nb::handle value) {
    return nb::hasattr(value, "__dlpack__");
}

/// @return Whether DLPack places a tensor on the host or a GPU, with its device index.
std::tuple<int, int> DlpackDevice(const Tensor& tensor) {
    const Device device = tensor.Location();
    switch (device.kind) {
    case DeviceKind::Cuda:
        return {static_cast<int>(kDLCUDA), device.id};
    case DeviceKind::Rocm:
        return {static_cast<int>(kDLROCM), device.id};
    case DeviceKind::Cpu:
        break;
    }
    return {static_cast<int>(kDLCPU), 0};
}

/// An authored array, which NumPy, PyTorch and any DLPack library view in place.
struct SceneArray {
    Tensor tensor;
};

nb::object TransformRows(const glm::dmat4& matrix) {
    return ToPython(ernie::ToJson(Value(matrix)));
}

XformOps ReadXformArguments(const nb::object& translate, const nb::object& rotate,
                            const std::string& rotationOrder, const nb::object& orient,
                            const nb::object& scale) {
    Json xform = Json::object();
    const auto add = [&](const char* key, const nb::object& value) {
        if (!value.is_none()) {
            xform[key] = ToJson(value);
        }
    };
    add("translate", translate);
    add("rotate", rotate);
    add("orient", orient);
    add("scale", scale);
    xform["rotationOrder"] = rotationOrder;
    return ReadXform(xform);
}

}  // namespace

void BindScene(nb::module_& module) {
    nb::class_<Prim>(module, "Prim", "A prim of a scene, valid until it is removed.")
        .def("__eq__", [](const Prim& self, const Prim& other) { return self.id == other.id; })
        .def("__hash__",
             [](const Prim& self) {
                 return (static_cast<std::uint64_t>(self.id.generation) << 32) | self.id.slot;
             })
        .def("__repr__", [](const Prim& self) {
            return "Prim(" + std::to_string(self.id.slot) + ", " +
                   std::to_string(self.id.generation) + ")";
        });

    nb::class_<SceneArray>(module, "SceneArray",
                           "An array attribute's values, shared in place through DLPack.")
        .def_prop_ro("shape", [](const SceneArray& self) { return self.tensor.Shape(); })
        .def(
            "__dlpack__",
            [](const SceneArray& self, const nb::object&, const nb::object& maxVersion,
               const nb::object&, const nb::object& copy) {
                if (!copy.is_none() && nb::cast<bool>(copy)) {
                    throw nb::buffer_error("Scene arrays are shared in place, never copied");
                }
                const bool versioned =
                    !maxVersion.is_none() && nb::cast<std::tuple<int, int>>(maxVersion) >=
                                                 std::tuple<int, int>(DLPACK_MAJOR_VERSION, 0);
                return Capsule(self.tensor.ToDlpack(), versioned);
            },
            nb::kw_only(), nb::arg("stream") = nb::none(), nb::arg("max_version") = nb::none(),
            nb::arg("dl_device") = nb::none(), nb::arg("copy") = nb::none())
        .def("__dlpack_device__", [](const SceneArray& self) { return DlpackDevice(self.tensor); });

    nb::class_<Scene>(module, "Scene", "Typed prims at paths, their attributes in columns by type.")
        .def(nb::init<>())
        .def(
            "define",
            [](Scene& self, const std::string& path, const std::string& type) {
                return Prim{self.Define(Path(path), type)};
            },
            nb::arg("path"), nb::arg("type") = "",
            "Defines a prim of a concrete type, or none, with its missing ancestors.")
        .def(
            "remove", [](Scene& self, const std::string& path) { return self.Remove(Path(path)); },
            nb::arg("path"), "Removes a prim and its descendants; whether there was one.")
        .def(
            "find",
            [](const Scene& self, const std::string& path) -> std::optional<Prim> {
                const std::optional<PrimId> prim = self.Find(Path(path));
                return prim ? std::optional<Prim>(Prim{*prim}) : std::nullopt;
            },
            nb::arg("path"), "The prim at a path, or None.")
        .def("__contains__",
             [](const Scene& self, const std::string& path) { return self.Has(Path(path)); })
        .def(
            "path",
            [](const Scene& self, const Target& prim) {
                return self.PrimPath(Resolve(self, prim)).String();
            },
            nb::arg("prim"), "A prim's path.")
        .def(
            "type_name",
            [](const Scene& self, const Target& prim) {
                return std::string(self.TypeName(Resolve(self, prim)));
            },
            nb::arg("prim"), "A prim's type; empty for none.")
        .def(
            "children",
            [](const Scene& self, const std::string& path) {
                return Map(self.Children(Path(path)), &Path::String);
            },
            nb::arg("path") = "/", "The paths of a prim's children.")
        .def(
            "prims",
            [](const Scene& self, const std::string& type) {
                return Map(self.Prims(type), &Path::String);
            },
            nb::arg("type") = "", "The paths of every prim of a type or one deriving from it.")
        .def(
            "apply_api",
            [](Scene& self, const Target& prim, const std::string& api) {
                self.ApplyApi(Resolve(self, prim), api);
            },
            nb::arg("prim"), nb::arg("api"),
            "Applies an API schema, or an instance of one, as 'CollectionAPI:lights'.")
        .def(
            "attributes",
            [](const Scene& self, const Target& prim) {
                return self.Attributes(self.PrimPath(Resolve(self, prim)));
            },
            nb::arg("prim"), "The authored attributes' names.")
        .def(
            "set",
            [](Scene& self, const Target& target, const std::string& attribute,
               const nb::object& value) {
                const PrimId prim = Resolve(self, target);
                if (IsArray(value)) {
                    Tensor array = TakeTensor(value, nb::none());
                    Attempt(self, prim, attribute,
                            [&] { self.Write(prim, attribute, std::move(array)); });
                    return;
                }
                const Json json = ToJson(value);
                Attempt(self, prim, attribute, [&] { LoadAttribute(self, prim, attribute, json); });
            },
            nb::arg("prim"), nb::arg("attribute"), nb::arg("value"),
            "Authors an attribute: a number, string, sequence for a vector, quaternion "
            "(w, x, y, z), matrix (4 rows) or array, list of strings for tokens or paths, or a "
            "NumPy or DLPack array, shared in place.")
        .def(
            "get",
            [](const Scene& self, const Target& target,
               const std::string& attribute) -> nb::object {
                const PrimId prim = Resolve(self, target);
                const std::optional<Value> value = self.Get(prim, attribute);
                if (!value) {
                    return nb::none();
                }
                nb::object result;
                Attempt(self, prim, attribute, [&] { result = ToPython(ernie::ToJson(*value)); });
                return result;
            },
            nb::arg("prim"), nb::arg("attribute"),
            "An authored value as scene files write it, or None; arrays are read with `read`.")
        .def(
            "write",
            [](Scene& self, const Target& target, const std::string& attribute,
               const nb::object& array, const nb::object& stream) {
                const PrimId prim = Resolve(self, target);
                Tensor tensor = TakeTensor(array, stream);
                Attempt(self, prim, attribute,
                        [&] { self.Write(prim, attribute, std::move(tensor)); });
            },
            nb::arg("prim"), nb::arg("attribute"), nb::arg("array"), nb::arg("stream") = nb::none(),
            "Authors an array attribute with a NumPy or DLPack array, shared, not copied.")
        .def(
            "read",
            [](const Scene& self, const Target& target,
               const std::string& attribute) -> std::optional<SceneArray> {
                const Tensor* array = self.Read(Resolve(self, target), attribute);
                return array != nullptr ? std::optional<SceneArray>(SceneArray{*array})
                                        : std::nullopt;
            },
            nb::arg("prim"), nb::arg("attribute"),
            "An authored array, shared in place through DLPack (numpy.from_dlpack), or None.")
        .def(
            "clear",
            [](Scene& self, const Target& prim, const std::string& attribute) {
                return self.Clear(Resolve(self, prim), attribute);
            },
            nb::arg("prim"), nb::arg("attribute"), "Unauthors an attribute; whether it was.")
        .def(
            "set_xform",
            [](Scene& self, const Target& target, const nb::object& translate,
               const nb::object& rotate, const std::string& rotationOrder, const nb::object& orient,
               const nb::object& scale) {
                const Path path = self.PrimPath(Resolve(self, target));
                SetXform(self, path,
                         ReadXformArguments(translate, rotate, rotationOrder, orient, scale));
            },
            nb::arg("prim"), nb::kw_only(), nb::arg("translate") = nb::none(),
            nb::arg("rotate") = nb::none(), nb::arg("rotation_order") = "XYZ",
            nb::arg("orient") = nb::none(), nb::arg("scale") = nb::none(),
            "Authors a prim's transform ops: translate, rotate (degrees, in `rotation_order`) or "
            "orient (w, x, y, z), and scale.")
        .def(
            "local_transform",
            [](const Scene& self, const Target& prim) {
                return TransformRows(self.LocalTransform(Resolve(self, prim)));
            },
            nb::arg("prim"), "A prim's own transform, as 4 rows.")
        .def(
            "world_transform",
            [](const Scene& self, const Target& prim) {
                return TransformRows(self.WorldTransform(Resolve(self, prim)));
            },
            nb::arg("prim"),
            "A prim's transform to world space, as 4 rows: column vectors, translation in the "
            "last column.")
        .def("update", &Scene::Update,
             "Brings derived data, such as inherited attributes and extents, up to date.")
        .def(
            "load_json",
            [](Scene& self, std::string_view text, const std::vector<nb::object>& arrays) {
                const std::vector<Tensor> tensors = Map(
                    arrays, [](const nb::object& array) { return TakeTensor(array, nb::none()); });
                LoadScene(self, text, tensors);
            },
            nb::arg("text"), nb::arg("arrays") = std::vector<nb::object>(),
            "Builds the prims a scene file's text describes; {\"$array\": i} names arrays[i], "
            "shared in place.")
        .def("save_json", &SaveScene, "The scene as a scene file's text.")
        .def(
            "load",
            [](Scene& self, const std::filesystem::path& path) { LoadSceneFile(self, path); },
            nb::arg("path"), "Builds the prims a scene file describes.")
        .def(
            "save",
            [](const Scene& self, const std::filesystem::path& path) { SaveSceneFile(self, path); },
            nb::arg("path"), "Writes the scene to a scene file.");
}

}  // namespace ernie::python
