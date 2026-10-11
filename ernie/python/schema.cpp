// SPDX-License-Identifier: BSD-3-Clause
#include "bindings.h"

#include "ernie/scene/prim_type.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/stage_schemas.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

namespace ernie::python {

namespace {

nb::list AttributeList(const std::vector<const AttributeDef*>& defs) {
    nb::list list;
    for (const AttributeDef* def : defs) {
        list.append(nb::cast(def, nb::rv_policy::reference));
    }
    return list;
}

}  // namespace

void BindSchema(nb::module_& m) {
    nb::enum_<AttributeKind>(m, "AttributeKind",
                             "What a registered attribute holds.")
        .value("BOOL", AttributeKind::Bool)
        .value("INT", AttributeKind::Int)
        .value("DOUBLE", AttributeKind::Double)
        .value("VEC2", AttributeKind::Vec2)
        .value("VEC3", AttributeKind::Vec3)
        .value("VEC4", AttributeKind::Vec4)
        .value("TOKEN", AttributeKind::Token)
        .value("RELATIONSHIP", AttributeKind::Relationship)
        .value("RELATIONSHIPS", AttributeKind::Relationships)
        .value("FLOAT_ARRAY", AttributeKind::FloatArray)
        .value("TENSOR", AttributeKind::Tensor)
        .value("QUAT", AttributeKind::Quat)
        .value("MATRIX", AttributeKind::Matrix);

    nb::class_<AttributeDef>(m, "AttributeDef",
                             "An attribute a registered prim type "
                             "declares: its USD and Python "
                             "names, kind, default and description.")
        .def_prop_ro(
            "name",
            [](const AttributeDef& def) {
                return std::string(def.name);
            },
            "Its USD name, such as 'inputs:radius'.")
        .def_ro("py_name", &AttributeDef::pyName,
                "Its Python name, such as 'radius'.")
        .def_ro("label", &AttributeDef::label)
        .def_ro("kind", &AttributeDef::kind)
        .def_prop_ro(
            "default",
            [](const AttributeDef& def) {
                return ValueToPython(def.fallback);
            },
            "What Ernie uses while it is unset; None for single "
            "relationships and arrays.")
        .def_prop_ro("doc",
                     [](const AttributeDef& def) {
                         return std::string(def.meta.doc);
                     })
        .def_prop_ro("group",
                     [](const AttributeDef& def) {
                         return std::string(def.meta.group);
                     })
        .def_prop_ro(
            "min",
            [](const AttributeDef& def) { return def.meta.min; })
        .def_prop_ro(
            "max",
            [](const AttributeDef& def) { return def.meta.max; })
        .def_prop_ro("options",
                     [](const AttributeDef& def) {
                         return std::vector<std::string>(
                             def.meta.options.begin(),
                             def.meta.options.end());
                     })
        .def_prop_ro(
            "is_color",
            [](const AttributeDef& def) { return def.meta.color; })
        .def_prop_ro(
            "is_asset",
            [](const AttributeDef& def) { return def.meta.asset; })
        .def_prop_ro(
            "plural",
            [](const AttributeDef& def) {
                return std::string(def.meta.plural);
            },
            "For a list of targets usually holding one: the Python "
            "name of the whole list, its "
            "own name reading the first target; else empty.")
        .def_prop_ro(
            "is_integer",
            [](const AttributeDef& def) { return def.meta.integer; },
            "Whether it is an array of whole numbers (indices, "
            "counts), loaded as int32.")
        .def_prop_ro(
            "takes_array",
            [](const AttributeDef& def) { return def.meta.array; },
            "Whether it also takes an array in place of its value: "
            "colours per vertex for "
            "display_color, a width per point, a pose per prim for a "
            "transform op.")
        .def(
            "read",
            [](const AttributeDef& def,
               const Prim& prim) -> nb::object {
                if (def.kind == AttributeKind::Tensor ||
                    (def.meta.array &&
                     prim.Get(def.name) == nullptr &&
                     prim.Read(def.name))) {
                    const Tensor* tensor = prim.Read(def.name);
                    return tensor != nullptr
                               ? ArrayFromTensor(*tensor)
                               : nb::none();
                }
                const Value* value = prim.Get(def.name);
                if (def.kind == AttributeKind::Relationships &&
                    value != nullptr) {
                    // A list, also of the one target stored alone;
                    // `Coerce` throws for others.
                    def.Coerce(*value);
                    return ValueToPython(Value(
                        *AttributeTraits<std::vector<Path>>::From(
                            *value)));
                }
                return ValueToPython(value != nullptr
                                         ? def.Coerce(*value)
                                         : def.fallback);
            },
            nb::arg("prim"),
            "The attribute of `prim` as its declared type, or its "
            "default while unset; a list of "
            "paths for a relationship to several prims; an array (or "
            "None) for tensors, and the "
            "array written in place of the value of one that takes "
            "one (takes_array).")
        .def(
            "write",
            [](const AttributeDef& def, Prim& prim,
               nb::handle value) {
                AnyArray array;
                if (def.kind == AttributeKind::Tensor) {
                    prim.Write(
                        def.name,
                        TensorFromArray(nb::cast<AnyArray>(value)));
                } else if (def.meta.array &&
                           nb::try_cast(value, array)) {
                    prim.Write(def.name, TensorFromArray(array));
                } else {
                    prim.Set(def.name,
                             def.Coerce(ValueFromPython(value)));
                }
            },
            nb::arg("prim"), nb::arg("value"),
            "Sets the attribute of `prim`, converted to its declared "
            "type (an int to a float, a "
            "str to a Path; a relationship to several prims takes "
            "one target or a list); writes "
            "an array for tensors, and for one that takes an array "
            "in place of its value "
            "(takes_array: colours per vertex for display_color).")
        .def("__repr__", [](const AttributeDef& def) {
            return "AttributeDef('" + def.pyName + "', '" +
                   std::string(def.name) + "')";
        });

    nb::class_<PrimTypeDef>(m, "PrimType",
                            "A registered prim type: its name, base "
                            "and declared attributes.")
        .def_prop_ro("name",
                     [](const PrimTypeDef& type) {
                         return std::string(type.name);
                     })
        .def_prop_ro(
            "base", [](const PrimTypeDef& type) { return type.base; },
            nb::rv_policy::reference)
        .def_ro("abstract", &PrimTypeDef::abstract)
        .def_prop_ro("doc",
                     [](const PrimTypeDef& type) {
                         return std::string(type.doc);
                     })
        .def_prop_ro(
            "attributes",
            [](const PrimTypeDef& type) {
                std::vector<const AttributeDef*> own;
                for (const AttributeDef& def : type.attributes) {
                    own.push_back(&def);
                }
                return AttributeList(own);
            },
            "The attributes it declares itself.")
        .def(
            "all_attributes",
            [](const PrimTypeDef& type) {
                return AttributeList(type.AllAttributes());
            },
            "Every attribute, the root base's first.")
        .def(
            "find",
            [](const PrimTypeDef& type,
               std::string_view name) -> const AttributeDef* {
                const AttributeDef* def = type.FindByPyName(name);
                return def != nullptr ? def : type.Find(name);
            },
            nb::arg("name"), nb::rv_policy::reference,
            "The attribute with Python or USD name `name`, here or "
            "on a base, or None.")
        .def(
            "is_a",
            [](const PrimTypeDef& type, std::string_view other) {
                const PrimTypeDef* found = FindPrimType(other);
                return found != nullptr && type.IsA(*found);
            },
            nb::arg("type"),
            "Whether this is the type named `type` or derives from "
            "it.")
        .def("__repr__", [](const PrimTypeDef& type) {
            return "PrimType('" + std::string(type.name) + "')";
        });

    m.def(
        "prim_types",
        [] {
            nb::list list;
            for (const PrimTypeDef* type : PrimTypes()) {
                list.append(nb::cast(type, nb::rv_policy::reference));
            }
            return list;
        },
        "Every registered prim type, bases before the types derived "
        "from them.");
    m.def("find_prim_type", &FindPrimType, nb::arg("name"),
          nb::rv_policy::reference,
          "The registered prim type named `name`, or None.");
    m.def(
        "stage_type", [] { return &Stage::Definition(); },
        nb::rv_policy::reference,
        "The schema of a scene's stage metadata, on its root prim: "
        "not a prim type, as no prim is defined as one.");
}

}  // namespace ernie::python
