// SPDX-License-Identifier: BSD-3-Clause
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "ernie/device/queue.h"
#include "ernie/render/shaders.h"
#include "ernie/stage/views/poses.h"
#include "values.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace ernie::python {

SchemaId FindSchema(std::string_view name) {
    for (uint32_t index = 0; index < SchemaCount(); ++index) {
        if (Info(SchemaId(index)).name == name) {
            return SchemaId(index);
        }
    }
    throw nb::value_error(("No schema is called " + std::string(name)).c_str());
}

namespace {

struct SceneSelection {
    Ref<Scene> scene;
    View view;

    std::vector<Prim> Prims() const {
        std::vector<Prim> prims;
        for (const uint32_t index : view.Prims()) {
            prims.push_back(scene->ByIndex(index));
        }
        return prims;
    }
};

struct SceneMapping {
    Ref<Scene> scene;
    Mapping mapping;
    std::optional<uintptr_t> stream;
};

SceneSelection Single(const ScenePrim& prim) {
    return {prim.scene, prim.scene->Select(prim.prim)};
}

std::vector<ScenePrim> ScenePrims(const Ref<Scene>& scene, std::span<const Prim> prims) {
    std::vector<ScenePrim> wrapped;
    for (const Prim& prim : prims) {
        wrapped.push_back({scene, prim});
    }
    return wrapped;
}

bool IsText(ColumnType type) {
    return type == ColumnType::Enum || type == ColumnType::Token || type == ColumnType::Asset;
}

std::string NumpyType(DLDataType dtype) {
    const char* kind = dtype.code == kDLFloat ? "float" : dtype.code == kDLInt ? "int" : "uint";
    return kind + std::to_string(dtype.bits);
}

std::vector<std::byte> ConvertCells(Scene& scene, const ColumnInfo& column, nb::handle value) {
    switch (column.type) {
    case ColumnType::Enum: {
        std::vector<uint32_t> words;
        if (nb::isinstance<nb::str>(value) || nb::isinstance<nb::int_>(value)) {
            words.push_back(EnumIndex(value, column.tokens));
        } else {
            for (nb::handle item : value) {
                words.push_back(EnumIndex(item, column.tokens));
            }
        }
        return Bytes(words);
    }
    case ColumnType::Token:
    case ColumnType::Asset: {
        std::vector<uint32_t> words;
        for (const std::string& text : Strings(value)) {
            words.push_back(scene.Token(text).id);
        }
        return Bytes(words);
    }
    case ColumnType::Bool:
        return NumpyBytes(nb::module_::import_("numpy").attr("asarray")(value, "dtype"_a = "bool"),
                          "uint32");
    default:
        return NumpyBytes(value, NumpyType(Layout(column.type).dtype));
    }
}

bool OnDevice(nb::handle value) {
    if (!nb::hasattr(value, "__dlpack_device__")) {
        return false;
    }
    const nb::tuple device = nb::cast<nb::tuple>(value.attr("__dlpack_device__")());
    return nb::cast<int>(device[0]) != kDLCPU;
}

nb::object MaskWords(nb::handle mask) {
    if (mask.is_none() || nb::hasattr(mask, "__dlpack_device__")) {
        const bool host =
            mask.is_none() ||
            nb::cast<int>(nb::cast<nb::tuple>(mask.attr("__dlpack_device__")())[0]) == kDLCPU;
        if (!host || mask.is_none()) {
            return nb::borrow(mask);
        }
    }
    const nb::module_ numpy = nb::module_::import_("numpy");
    const nb::object array = numpy.attr("asarray")(mask);
    if (nb::cast<std::string>(array.attr("dtype").attr("kind")) != "b") {
        return array;
    }
    nb::object bytes = numpy.attr("packbits")(array, "bitorder"_a = "little");
    const size_t padding = (8 - nb::len(bytes) % 8) % 8;
    bytes = numpy.attr("pad")(bytes, nb::make_tuple(0, padding));
    return bytes.attr("view")(numpy.attr("uint64"));
}

void WriteTensor(Scene& scene, UntypedAttribute& target, nb::handle value,
                 std::optional<uintptr_t> stream, nb::handle mask = nb::none()) {
    HeldTensors held;
    if (nb::isinstance<device::Tensor>(value)) {
        const nb::object words = MaskWords(mask);
        const device::Tensor bits =
            words.is_none() ? device::Tensor()
            : nb::isinstance<device::Tensor>(words)
                ? nb::cast<device::Tensor>(words)
                : held.Import(scene.GetDevice(), words, stream, "attribute mask");
        const device::Tensor values = nb::cast<device::Tensor>(value);
        const Stream producer =
            held.Producer().IsNone() && stream ? ToStream(stream) : held.Producer();
        nb::gil_scoped_release release;
        held.Retire(scene, target.Write(values, producer, bits));
        return;
    }
    const DLTensor* values = held.Take(value, stream);
    const DLTensor* bits = held.Take(MaskWords(mask), stream);
    nb::gil_scoped_release release;
    held.Retire(scene, target.Write(*values, held.Producer(), bits));
}

void AssignCells(Scene& scene, UntypedAttribute& target, nb::handle value,
                 std::optional<uintptr_t> stream) {
    const ColumnInfo& column = *target.Reference().column;
    if (OnDevice(value)) {
        return WriteTensor(scene, target, value, stream);
    }

    const std::vector<std::byte> cells = ConvertCells(scene, column, value);
    if (cells.size() == ElementBytes(column.type)) {
        target.Fill(cells);
    } else {
        target.Write(cells);
    }
}

bool IsPrimvars(const ColumnRef& column) {
    return column.column->field == "primvars";
}

void SetArray(UntypedAttribute& target, uint32_t index, nb::handle value) {
    const ColumnRef column = target.Reference();
    switch (column.column->type) {
    case ColumnType::PrimArray:
        return target.SetTargets(index, Strings(value));
    case ColumnType::TokenArray:
        return target.SetTokens(index, Strings(value));
    case ColumnType::FloatArray:
    case ColumnType::Float2Array:
    case ColumnType::Float3Array:
    case ColumnType::Float4Array:
        return target.SetArray(index, NumpyBytes(value, "float32"));
    case ColumnType::Int64Array:
        return target.SetArray(index, NumpyBytes(value, "int64"));
    default:
        if (IsPrimvars(column)) {
            return target.SetPrimvars(index, nb::cast<std::vector<Primvar>>(value));
        }
        return target.SetArray(index, NumpyBytes(value, "int32"));
    }
}

size_t ArrayLanes(ColumnType type) {
    switch (type) {
    case ColumnType::Float2Array:
        return 2;
    case ColumnType::Float3Array:
        return 3;
    case ColumnType::Float4Array:
        return 4;
    default:
        return 1;
    }
}

bool IsNumericArray(const ColumnRef& column) {
    return column.column->type != ColumnType::PrimArray &&
           column.column->type != ColumnType::TokenArray && !IsPrimvars(column);
}

std::string ShapeText(const nb::object& array) {
    return nb::cast<std::string>(array.attr("shape").attr("__str__")());
}

void CheckRows(const UntypedAttribute& target, std::string_view field, nb::handle value) {
    const ColumnRef column = target.Reference();
    const size_t lanes = ArrayLanes(column.column->type);
    if (!IsNumericArray(column) || lanes == 1) {
        return;
    }
    const nb::object array = nb::module_::import_("numpy").attr("asarray")(value);
    const size_t rank = nb::cast<size_t>(array.attr("ndim"));
    const size_t size = nb::cast<size_t>(array.attr("size"));
    const bool rows = rank == 2 && nb::cast<size_t>(array.attr("shape")[nb::int_(1)]) == lanes;
    const bool flat = rank == 1 && size % lanes == 0;
    if (!rows && !flat) {
        throw nb::value_error((std::string(field) + " takes rows of " + std::to_string(lanes) +
                               " values, shaped (rows, " + std::to_string(lanes) + "), got " +
                               ShapeText(array))
                                  .c_str());
    }
}

void AssignArrays(SceneSelection& source, UntypedAttribute& target, std::string_view field,
                  nb::handle value) {
    const uint32_t count = source.view.PrimCount();
    const ColumnRef column = target.Reference();
    if (!IsNumericArray(column)) {
        for (uint32_t index = 0; index < count; ++index) {
            SetArray(target, index, value);
        }
        return;
    }
    const nb::object array = nb::module_::import_("numpy").attr("asarray")(value);
    const size_t lanes = ArrayLanes(column.column->type);
    const size_t rank = nb::cast<size_t>(array.attr("ndim"));
    const size_t batched = lanes > 1 ? 3 : 2;
    if (rank == batched) {
        const nb::object shape = array.attr("shape");
        const bool leading = nb::cast<size_t>(shape[nb::int_(0)]) == count;
        const bool rows = lanes == 1 || nb::cast<size_t>(shape[nb::int_(2)]) == lanes;
        if (!leading || !rows) {
            const std::string row = lanes > 1 ? ", rows, " + std::to_string(lanes) : ", rows";
            throw nb::value_error((std::string(field) + " takes one array for each of the " +
                                   std::to_string(count) + " prims, shaped (" +
                                   std::to_string(count) + row + "), got " + ShapeText(array) +
                                   ". Use set_each for a different number of rows per prim")
                                      .c_str());
        }
        for (uint32_t index = 0; index < count; ++index) {
            const nb::object rows = array[nb::int_(index)];
            SetArray(target, index, rows);
        }
        return;
    }
    if (rank > batched || (lanes == 1 && rank > 1)) {
        throw nb::value_error((std::string(field) +
                               " takes one array for every prim or one per prim, got " +
                               ShapeText(array))
                                  .c_str());
    }
    CheckRows(target, field, array);
    for (uint32_t index = 0; index < count; ++index) {
        SetArray(target, index, array);
    }
}

void Assign(SceneSelection& source, std::string_view field, nb::handle value,
            std::optional<uintptr_t> stream = std::nullopt, nb::handle mask = nb::none()) {
    UntypedAttribute target = source.view.Get(field);
    if (!mask.is_none()) {
        if (IsArray(target.Reference().column->type)) {
            throw nb::value_error("Masked writes take cell attributes, not arrays");
        }
        const nb::object tensor =
            nb::hasattr(value, "__dlpack__")
                ? nb::borrow(value)
                : nb::module_::import_("numpy").attr("ascontiguousarray")(value);
        return WriteTensor(*source.scene, target, tensor, stream, mask);
    }
    if (!IsArray(target.Reference().column->type)) {
        return AssignCells(*source.scene, target, value, stream);
    }
    AssignArrays(source, target, field, value);
}

void AssignEach(SceneSelection& source, std::string_view field, nb::handle values) {
    UntypedAttribute target = source.view.Get(field);
    const uint32_t count = source.view.PrimCount();
    if (nb::len(values) != count) {
        throw nb::value_error(("set_each(\"" + std::string(field) +
                               "\") takes one value for each of the " + std::to_string(count) +
                               " prims, got " + std::to_string(nb::len(values)))
                                  .c_str());
    }
    if (!IsArray(target.Reference().column->type)) {
        return AssignCells(*source.scene, target, values, std::nullopt);
    }
    uint32_t index = 0;
    for (nb::handle value : values) {
        CheckRows(target, field, value);
        SetArray(target, index++, value);
    }
}

nb::object ReadArray(const UntypedAttribute& origin, uint32_t index) {
    const ColumnRef column = origin.Reference();
    switch (column.column->type) {
    case ColumnType::PrimArray:
        return nb::cast(origin.GetTargets(index));
    case ColumnType::TokenArray:
        return nb::cast(origin.GetTokens(index));
    default:
        if (IsPrimvars(column)) {
            return nb::cast(origin.GetPrimvars(index));
        }
    }

    std::vector<std::byte> bytes;
    {
        nb::gil_scoped_release release;
        bytes = origin.GetArray(index);
    }
    if (column.column->type == ColumnType::FloatArray) {
        const size_t count = bytes.size() / 4;
        return Numpy(std::move(bytes), {count}, {kDLFloat, 32, 1});
    }
    if (column.column->type == ColumnType::Float2Array ||
        column.column->type == ColumnType::Float3Array ||
        column.column->type == ColumnType::Float4Array) {
        const size_t lanes = column.column->type == ColumnType::Float2Array   ? 2
                             : column.column->type == ColumnType::Float3Array ? 3
                                                                              : 4;
        const size_t rows = bytes.size() / (lanes * 4);
        return Numpy(std::move(bytes), {rows, lanes}, {kDLFloat, 32, 1});
    }
    if (column.column->type == ColumnType::Int64Array) {
        const size_t count = bytes.size() / 8;
        return Numpy(std::move(bytes), {count}, {kDLInt, 64, 1});
    }
    const size_t count = bytes.size() / 4;
    return Numpy(std::move(bytes), {count}, {kDLInt, 32, 1});
}

nb::object ReadCells(Scene& scene, const UntypedAttribute& origin) {
    std::vector<std::byte> bytes;
    {
        nb::gil_scoped_release release;
        bytes = origin.ReadHost();
    }

    const ColumnInfo& column = *origin.Reference().column;
    const ColumnType type = column.type;
    if (IsText(type)) {
        nb::list texts;
        for (const uint32_t word : Items<uint32_t>(bytes)) {
            const std::string_view text =
                type == ColumnType::Enum ? column.tokens[word] : scene.Text(Token{word});
            texts.append(nb::str(text.data(), text.size()));
        }
        return texts;
    }
    if (type == ColumnType::Bool) {
        return Numpy(std::move(bytes), {origin.Size()}, {kDLUInt, 32, 1}).attr("astype")("bool");
    }

    const ColumnLayout layout = Layout(type);
    std::vector<size_t> shape = {origin.Size()};
    if (layout.lanes > 1) {
        shape.push_back(layout.lanes);
    }
    return Numpy(std::move(bytes), shape, layout.dtype);
}

nb::object Read(SceneSelection& source, std::string_view field) {
    const UntypedAttribute origin = source.view.Get(field);
    if (!IsArray(origin.Reference().column->type)) {
        return ReadCells(*source.scene, origin);
    }
    nb::list values;
    for (uint32_t index = 0; index < origin.Size(); ++index) {
        values.append(ReadArray(origin, index));
    }
    return values;
}

void AssignOne(ScenePrim& prim, std::string_view field, nb::handle value,
               std::optional<uintptr_t> stream) {
    SceneSelection single = Single(prim);
    UntypedAttribute target = single.view.Get(field);
    if (IsArray(target.Reference().column->type)) {
        CheckRows(target, field, value);
        return SetArray(target, 0, value);
    }
    Assign(single, field, value, stream);
}

nb::object ReadFirst(SceneSelection& single, std::string_view field) {
    const nb::object values = Read(single, field);
    const ColumnType type = single.view.Column(field).column->type;
    if (IsArray(type) || IsText(type)) {
        return values[0];
    }

    const nb::object first = values[0];
    return Layout(type).lanes > 1 ? nb::tuple(first.attr("tolist")()) : first.attr("item")();
}

AttachFrom FindAttachFrom(std::string_view from) {
    if (from == "tensor") {
        return AttachFrom::Tensor;
    }
    if (from == "scene") {
        return AttachFrom::Scene;
    }
    throw nb::value_error("Values are attached from the tensor or the scene");
}

void Attach(const SceneSelection& source, std::string_view field, nb::handle tensor,
            std::string_view initial, std::optional<uintptr_t> stream) {
    UntypedAttribute target = source.view.Get(field);
    const AttachFrom values = FindAttachFrom(initial);
    const TakenTensor taken = TakeTensor(tensor, stream);
    try {
        nb::gil_scoped_release release;
        target.Attach(taken.managed, taken.producer, values);
    } catch (...) {
        if (taken.managed->deleter) {
            taken.managed->deleter(taken.managed);
        }
        throw;
    }
}

void MarkChanged(const SceneSelection& source, std::string_view field,
                 std::optional<uintptr_t> stream, nb::handle mask) {
    UntypedAttribute target = source.view.Get(field);
    HeldTensors held;
    const DLTensor* bits = held.Take(MaskWords(mask), stream);
    nb::gil_scoped_release release;
    held.Retire(*source.scene, target.MarkChanged(ToStream(stream), bits));
}

void WaitForReads(const SceneSelection& source, std::string_view field,
                  std::optional<uintptr_t> stream) {
    const UntypedAttribute target = source.view.Get(field);
    nb::gil_scoped_release release;
    target.WaitForReads(stream ? ToStream(stream) : Stream::Host());
}

void Detach(const SceneSelection& source, std::string_view field, std::optional<uintptr_t> stream) {
    UntypedAttribute target = source.view.Get(field);
    nb::gil_scoped_release release;
    target.Detach(ToStream(stream));
}

QuaternionOrder FindOrder(std::string_view order) {
    if (order == "xyzw") {
        return QuaternionOrder::Xyzw;
    }
    if (order == "wxyz") {
        return QuaternionOrder::Wxyz;
    }
    throw nb::value_error("Quaternions are ordered xyzw or wxyz");
}

MapAccess Access(std::string_view access) {
    if (access == "read") {
        return MapAccess::Read;
    }
    if (access == "write") {
        return MapAccess::Write;
    }
    if (access == "read_write") {
        return MapAccess::ReadWrite;
    }
    throw nb::value_error("Access is read, write or read_write");
}

SchemaId DefinedSchema(Scene& scene, std::string_view name, const std::optional<nb::dict>& values) {
    const SchemaId schema = FindSchema(name);
    if (schema != SchemaId::Shader) {
        return schema;
    }
    const auto text = [&](const char* key) {
        return values && values->contains(key) ? nb::cast<std::string>((*values)[key])
                                               : std::string();
    };
    if (text("implementationSource") == "sourceAsset") {
        const std::string node = text("slangSourceAssetSubIdentifier");
        const std::string file = text("slangSourceAsset");
        if (node.empty() || file.empty()) {
            throw nb::value_error("A Shader loaded from a source asset needs slangSourceAsset and "
                                  "slangSourceAssetSubIdentifier");
        }
        if (ShaderSchema(node) == SchemaId::None) {
            LoadShaders(scene, file);
        }
        const SchemaId loaded = ShaderSchema(node);
        if (loaded == SchemaId::None) {
            throw nb::value_error((file + " has no material node called " + node).c_str());
        }
        return loaded;
    }
    if (!values || !values->contains("id")) {
        throw nb::value_error("A Shader needs an id naming its node, such as UsdPreviewSurface");
    }
    const std::string id = nb::cast<std::string>((*values)["id"]);
    const SchemaId shader = ShaderSchema(id);
    if (shader == SchemaId::None) {
        throw nb::value_error(("No shader has info:id " + id).c_str());
    }
    return shader;
}

std::vector<Prim> Handles(std::span<const ScenePrim> prims) {
    std::vector<Prim> handles;
    handles.reserve(prims.size());
    for (const ScenePrim& prim : prims) {
        handles.push_back(prim.prim);
    }
    return handles;
}

SceneSelection DefineBatch(Ref<Scene> scene, const std::vector<std::string>& paths,
                           std::string_view name, std::optional<nb::dict> values,
                           std::optional<nb::dict> apis, bool attachable) {
    const SchemaId schema = DefinedSchema(*scene, name, values);
    const PrimRange range = scene->Define(paths, schema, {.attachable = attachable});
    SceneSelection defined{scene, scene->Select(range, schema)};
    if (values) {
        for (auto [key, value] : *values) {
            Assign(defined, nb::cast<std::string>(key), value);
        }
    }
    if (apis) {
        const std::vector<Prim> prims = scene->Handles(range);
        for (auto [key, value] : *apis) {
            ApplySchema(*scene, prims, FindSchema(nb::cast<std::string>(key)), "",
                        nb::cast<nb::dict>(value));
        }
    }
    return defined;
}

ScenePrim Define(Ref<Scene> scene, std::string_view path, std::string_view name,
                 std::optional<nb::dict> values, std::optional<nb::dict> apis) {
    const SchemaId schema = DefinedSchema(*scene, name, values);
    if (IsRuntime(schema)) {
        DefineBatch(scene, {std::string(path)}, name, values, apis, false);
        const std::optional<Prim> prim = scene->Find(path);
        return {std::move(scene), *prim};
    }
    const Prim prim = DefineSchema(*scene, path, schema, values ? *values : nb::dict(),
                                   apis ? *apis : nb::dict());
    return {std::move(scene), prim};
}

Interpolation FindInterpolation(std::string_view name) {
    for (uint32_t mode = 0; mode < std::size(kInterpolationTokens); ++mode) {
        if (kInterpolationTokens[mode] == name) {
            return Interpolation(mode);
        }
    }
    throw nb::value_error((std::string(name) + " isn't an interpolation").c_str());
}

Primvar CreatePrimvar(std::string name, nb::handle values, std::string_view interpolation,
                      std::optional<nb::handle> indices, uint32_t elementSize) {
    const nb::object numpy = nb::module_::import_("numpy");
    nb::object array = numpy.attr("asarray")(values);
    const bool integral =
        nb::cast<bool>(numpy.attr("issubdtype")(array.attr("dtype"), numpy.attr("integer")));
    const size_t ndim = nb::cast<size_t>(array.attr("ndim"));
    const uint32_t components =
        ndim > 1 ? nb::cast<uint32_t>(nb::cast<nb::tuple>(array.attr("shape"))[ndim - 1]) : 1;
    if (ndim > 2 || components < 1 || components > 4) {
        throw nb::value_error("A primvar's values are n values or n rows of 2, 3 or 4");
    }

    Primvar primvar;
    primvar.name = std::move(name);
    primvar.type =
        PrimvarType(uint32_t(integral ? PrimvarType::Int : PrimvarType::Float) + components - 1);
    primvar.interpolation = FindInterpolation(interpolation);
    primvar.elementSize = elementSize;
    primvar.words = Items<uint32_t>(NumpyBytes(array, integral ? "int32" : "float32"));
    if (indices) {
        primvar.indices = Items<int32_t>(NumpyBytes(*indices, "int32"));
    }
    return primvar;
}

void RegisterPrimvar(nb::module_& module) {
    nb::class_<Primvar>(module, "Primvar")
        .def(nb::new_(&CreatePrimvar), nb::arg("name"), nb::arg("values"),
             nb::arg("interpolation") = "vertex", nb::arg("indices") = nb::none(),
             nb::arg("element_size") = 1)
        .def_ro("name", &Primvar::name)
        .def_prop_ro("interpolation",
                     [](const Primvar& primvar) {
                         return std::string(kInterpolationTokens[uint32_t(primvar.interpolation)]);
                     })
        .def_ro("element_size", &Primvar::elementSize)
        .def_prop_ro("values",
                     [](const Primvar& primvar) {
                         const uint32_t components = PrimvarComponents(primvar.type);
                         std::vector<size_t> shape = {primvar.words.size() / components};
                         if (components > 1) {
                             shape.push_back(components);
                         }
                         const bool integral = PrimvarIsInt(primvar.type);
                         return Numpy(Bytes(primvar.words), shape,
                                      {uint8_t(integral ? kDLInt : kDLFloat), 32, 1});
                     })
        .def_prop_ro("indices",
                     [](const Primvar& primvar) {
                         return Numpy(Bytes(primvar.indices), {primvar.indices.size()},
                                      {kDLInt, 32, 1});
                     })
        .def("__repr__", [](const Primvar& primvar) {
            return "Primvar(" + primvar.name + ", " +
                   std::string(kInterpolationTokens[uint32_t(primvar.interpolation)]) + ")";
        });
}

void RegisterScene(nb::module_& module) {
    nb::class_<Scene>(module, "Scene")
        .def(nb::new_([](std::optional<Ref<device::Device>> device, device::Traversal traversal,
                         uint32_t maxRefits, uint32_t pointsPerBox, bool waitForKernels) {
                 nb::gil_scoped_release release;
                 const SceneDesc desc{.traversal = traversal,
                                      .maxRefits = maxRefits,
                                      .pointsPerBox = pointsPerBox,
                                      .waitForKernels = waitForKernels};
                 return device ? Scene::Create(**device, desc) : Scene::Create(desc);
             }),
             nb::arg("device") = nb::none(), nb::kw_only(),
             nb::arg("traversal") = device::Traversal::Auto,
             nb::arg("max_refits") = SceneDesc{}.maxRefits,
             nb::arg("points_per_box") = SceneDesc{}.pointsPerBox,
             nb::arg("wait_for_kernels") = true)
        .def(
            "load_shaders",
            [](Ref<Scene> scene, const std::string& path) { return LoadShaders(*scene, path); },
            nb::arg("path"))
        .def("reload_shaders", [](Ref<Scene> scene) { return ReloadShaders(*scene); })
        .def("define", &Define, nb::arg("path"), nb::arg("schema"), nb::arg("values") = nb::none(),
             nb::kw_only(), nb::arg("apis") = nb::none())
        .def("define", &DefineBatch, nb::arg("paths"), nb::arg("schema"),
             nb::arg("values") = nb::none(), nb::kw_only(), nb::arg("apis") = nb::none(),
             nb::arg("attachable") = false)
        .def(
            "apply",
            [](Scene& scene, const std::vector<ScenePrim>& prims, std::string_view api,
               std::optional<nb::dict> values, std::optional<std::string_view> instance) {
                const SchemaId schema = FindSchema(api);
                if (IsMultipleApi(schema) && !instance) {
                    throw nb::value_error((std::string(api) + " needs an instance name").c_str());
                }
                if (!IsMultipleApi(schema) && instance) {
                    throw nb::value_error((std::string(api) + " takes no instance name").c_str());
                }
                ApplySchema(scene, Handles(prims), schema, instance.value_or(""),
                            values ? *values : nb::dict());
            },
            nb::arg("prims"), nb::arg("api"), nb::arg("values") = nb::none(), nb::kw_only(),
            nb::arg("instance") = nb::none())
        .def(
            "unapply",
            [](Scene& scene, const std::vector<ScenePrim>& prims, std::string_view api,
               std::optional<std::string_view> instance) {
                if (instance) {
                    scene.Unapply(Handles(prims), FindSchema(api), *instance);
                } else {
                    scene.Unapply(Handles(prims), FindSchema(api));
                }
            },
            nb::arg("prims"), nb::arg("api"), nb::kw_only(), nb::arg("instance") = nb::none())
        .def(
            "remove", [](Scene& scene, const ScenePrim& prim) { scene.Remove(prim.prim); },
            nb::arg("prim"))
        .def(
            "move",
            [](Ref<Scene> scene, const ScenePrim& prim, std::string_view path) {
                return ScenePrim{scene, scene->Move(prim.prim, path)};
            },
            nb::arg("prim"), nb::arg("path"))
        .def(
            "rename",
            [](Ref<Scene> scene, const ScenePrim& prim, std::string_view name) {
                return ScenePrim{scene, scene->Rename(prim.prim, name)};
            },
            nb::arg("prim"), nb::arg("name"))
        .def(
            "copy", [](Scene& scene, const ScenePrim& prim) { return scene.Copy(prim.prim); },
            nb::arg("prim"))
        .def(
            "paste",
            [](Ref<Scene> scene, const PrimCopy& copy, std::string_view path) {
                return ScenePrim{scene, scene->Paste(copy, path)};
            },
            nb::arg("copy"), nb::arg("path"))
        .def(
            "find",
            [](Ref<Scene> scene, std::string_view path) -> std::optional<ScenePrim> {
                if (const std::optional<Prim> prim = scene->Find(path)) {
                    return ScenePrim{scene, *prim};
                }
                return std::nullopt;
            },
            nb::arg("path"))
        .def(
            "select",
            [](Ref<Scene> scene, std::string_view pattern, std::optional<std::string_view> schema,
               std::optional<std::string_view> instance) {
                View view = instance
                                ? scene->Select(pattern, FindSchema(schema.value_or("")), *instance)
                            : schema ? scene->Select(pattern, FindSchema(*schema))
                                     : scene->Select(pattern);
                return SceneSelection{std::move(scene), std::move(view)};
            },
            nb::arg("pattern"), nb::arg("schema") = nb::none(), nb::kw_only(),
            nb::arg("instance") = nb::none())
        .def(
            "select",
            [](Ref<Scene> scene, const std::vector<std::string>& paths) {
                View view = scene->Select(paths);
                return SceneSelection{std::move(scene), std::move(view)};
            },
            nb::arg("paths"))
        .def(
            "clone",
            [](Ref<Scene> scene, const ScenePrim& source, const std::vector<std::string>& targets,
               bool moveSource, bool attachable) {
                scene->Clone(source.prim, targets,
                             {.moveSource = moveSource, .attachable = attachable});
                std::vector<ScenePrim> roots;
                for (const std::string& target : targets) {
                    roots.push_back({scene, *scene->Find(target)});
                }
                return roots;
            },
            nb::arg("source"), nb::arg("targets"), nb::kw_only(), nb::arg("move_source") = false,
            nb::arg("attachable") = false)
        .def(
            "clone",
            [](Ref<Scene> scene, const ScenePrim& source, uint32_t count, bool moveSource,
               bool attachable) {
                const PrimRange copies = scene->Clone(
                    source.prim, count, {.moveSource = moveSource, .attachable = attachable});
                std::vector<ScenePrim> roots;
                for (uint32_t copy = 0; copy < copies.copies; ++copy) {
                    roots.push_back({scene, scene->ByIndex(copies.Index(copy, 0))});
                }
                return roots;
            },
            nb::arg("source"), nb::arg("count"), nb::kw_only(), nb::arg("move_source") = false,
            nb::arg("attachable") = false)
        .def(
            "publish",
            [](Scene& scene, std::optional<Ordinal> ordinal, std::optional<uintptr_t> stream,
               bool wait, std::optional<double> dt) {
                scene.Publish({.ordinal = ordinal, .dt = dt, .wait = wait}, ToStream(stream));
            },
            nb::arg("ordinal") = nb::none(), nb::arg("stream") = nb::none(), nb::kw_only(),
            nb::arg("wait") = false, nb::arg("dt") = nb::none(),
            nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("published", &Scene::Published)
        .def_prop_rw("time", &Scene::Time, &Scene::SetTime)
        .def_prop_ro("structure_version", &Scene::StructureVersion)
        .def(
            "acquire",
            [](Scene& scene, std::optional<uintptr_t> stream) {
                return scene.Acquire(stream ? ToStream(stream) : Stream::Host());
            },
            nb::arg("stream") = nb::none(), nb::call_guard<nb::gil_scoped_release>())
        .def(
            "release",
            [](Scene& scene, const Snapshot& snapshot, std::optional<uintptr_t> stream) {
                scene.Release(snapshot, ToStream(stream));
            },
            nb::arg("snapshot"), nb::arg("stream") = nb::none(),
            nb::call_guard<nb::gil_scoped_release>())
        .def(
            "bounds",
            [](const Scene& scene, std::optional<uint32_t> world) -> nb::object {
                const std::optional<Box> box = scene.Bounds(world.value_or(kGlobalWorld));
                if (!box) {
                    return nb::none();
                }
                return nb::make_tuple(nb::make_tuple(box->lower.x, box->lower.y, box->lower.z),
                                      nb::make_tuple(box->upper.x, box->upper.y, box->upper.z));
            },
            nb::arg("world") = nb::none())
        .def(
            "world_transforms",
            [](const Scene& scene, const std::vector<std::string>& paths) {
                std::vector<WorldTransform> transforms;
                {
                    const nb::gil_scoped_release release;
                    transforms = scene.WorldTransforms(Span<const std::string>(paths));
                }
                nb::list rows;
                for (const WorldTransform& t : transforms) {
                    rows.append(nb::make_tuple(
                        nb::make_tuple(t.x.x, t.x.y, t.x.z), nb::make_tuple(t.y.x, t.y.y, t.y.z),
                        nb::make_tuple(t.z.x, t.z.y, t.z.z),
                        nb::make_tuple(double(t.high.x) + t.low.x, double(t.high.y) + t.low.y,
                                       double(t.high.z) + t.low.z)));
                }
                return nb::module_::import_("numpy").attr("asarray")(rows, "dtype"_a = "float64");
            },
            nb::arg("paths"),
            "World transforms of the prims at paths as an (N, 4, 3) array of the x, y and z axes "
            "and the translation, as of the latest publish. Prims without a transform read as "
            "zeros.")
        .def(
            "prim_bounds",
            [](const Scene& scene, const std::vector<std::string>& paths) {
                std::vector<Box> boxes;
                {
                    const nb::gil_scoped_release release;
                    boxes = scene.PrimBounds(Span<const std::string>(paths));
                }
                nb::list rows;
                for (const Box& box : boxes) {
                    rows.append(
                        nb::make_tuple(nb::make_tuple(box.lower.x, box.lower.y, box.lower.z),
                                       nb::make_tuple(box.upper.x, box.upper.y, box.upper.z)));
                }
                return nb::module_::import_("numpy").attr("asarray")(rows, "dtype"_a = "float32");
            },
            nb::arg("paths"),
            "World-space boxes of the prims at paths, relative to each prim's world origin, as an "
            "(N, 2, 3) array of lower and upper corners as of the latest publish. Prims without "
            "bounds read as zeros.")
        .def_prop_ro("pending", &Scene::Pending)
        .def_prop_ro("world_count", &Scene::WorldCount)
        .def_prop_ro("prim_count", &Scene::PrimCount)
        .def_prop_ro("device",
                     [](Scene& scene) { return Ref<device::Device>(&scene.GetDevice()); });
}

void RegisterPrimCopy(nb::module_& module) {
    nb::class_<PrimCopy>(module, "PrimCopy")
        .def_prop_ro("empty", &PrimCopy::Empty)
        .def_prop_ro("name", [](const PrimCopy& copy) { return std::string(copy.Name()); })
        .def_prop_ro("prim_count", &PrimCopy::PrimCount);
}

void RegisterPrim(nb::module_& module) {
    nb::class_<ScenePrim>(module, "Prim")
        .def_prop_ro("path", [](const ScenePrim& prim) { return prim.prim.Path(); })
        .def_prop_ro("name", [](const ScenePrim& prim) { return std::string(prim.prim.Name()); })
        .def_prop_ro("index", [](const ScenePrim& prim) { return prim.prim.Index(); })
        .def_prop_ro("world",
                     [](const ScenePrim& prim) -> std::optional<uint32_t> {
                         const uint32_t world = prim.prim.World();
                         if (world == kGlobalWorld) {
                             return std::nullopt;
                         }
                         return world;
                     })
        .def_prop_ro("generation", [](const ScenePrim& prim) { return prim.prim.Generation(); })
        .def_prop_ro("valid", [](const ScenePrim& prim) { return prim.prim.Valid(); })
        .def_prop_ro(
            "schema",
            [](const ScenePrim& prim) { return std::string(Info(prim.prim.Schema()).name); })
        .def_prop_ro("scene", [](const ScenePrim& prim) { return prim.scene; })
        .def_prop_ro("parent",
                     [](const ScenePrim& prim) -> std::optional<ScenePrim> {
                         if (const std::optional<Prim> parent = prim.prim.Parent()) {
                             return ScenePrim{prim.scene, *parent};
                         }
                         return std::nullopt;
                     })
        .def_prop_ro(
            "children",
            [](const ScenePrim& prim) { return ScenePrims(prim.scene, prim.prim.Children()); })
        .def_prop_ro("connections",
                     [](const ScenePrim& prim) {
                         std::vector<std::tuple<std::string, std::string, std::string>> found;
                         for (const Connection& connection : prim.prim.Connections()) {
                             found.emplace_back(connection.input, connection.source,
                                                connection.output);
                         }
                         return found;
                     })
        .def(
            "connect",
            [](const ScenePrim& prim, std::string_view input, const ScenePrim& source,
               std::string_view output) { prim.prim.Connect(input, source.prim, output); },
            nb::arg("input"), nb::arg("source"), nb::arg("output"))
        .def(
            "disconnect",
            [](const ScenePrim& prim, std::string_view input) { prim.prim.Disconnect(input); },
            nb::arg("input"))
        .def(
            "is_a",
            [](const ScenePrim& prim, std::string_view schema) {
                return prim.prim.IsA(FindSchema(schema));
            },
            nb::arg("schema"))
        .def(
            "has_api",
            [](const ScenePrim& prim, std::string_view api) {
                return prim.prim.HasApi(FindSchema(api));
            },
            nb::arg("api"))
        .def(
            "attach",
            [](const ScenePrim& prim, std::string_view field, nb::handle tensor,
               std::string_view initial, std::optional<uintptr_t> stream) {
                Attach(Single(prim), field, tensor, initial, stream);
            },
            nb::arg("field"), nb::arg("tensor"), nb::kw_only(), nb::arg("initial") = "tensor",
            nb::arg("stream") = nb::none())
        .def(
            "mark_changed",
            [](const ScenePrim& prim, std::string_view field, std::optional<uintptr_t> stream) {
                MarkChanged(Single(prim), field, stream, nb::none());
            },
            nb::arg("field"), nb::arg("stream") = nb::none())
        .def(
            "detach",
            [](const ScenePrim& prim, std::string_view field, std::optional<uintptr_t> stream) {
                Detach(Single(prim), field, stream);
            },
            nb::arg("field"), nb::arg("stream") = nb::none())
        .def(
            "wait_for_reads",
            [](const ScenePrim& prim, std::string_view field, std::optional<uintptr_t> stream) {
                WaitForReads(Single(prim), field, stream);
            },
            nb::arg("field"), nb::arg("stream") = nb::none())
        .def(
            "attached",
            [](const ScenePrim& prim, std::string_view field) {
                return Single(prim).view.Get(field).Attached();
            },
            nb::arg("field"))
        .def("__getitem__",
             [](ScenePrim& prim, std::string_view field) {
                 SceneSelection single = Single(prim);
                 return ReadFirst(single, field);
             })
        .def("__setitem__", [](ScenePrim& prim, std::string_view field,
                               nb::handle value) { AssignOne(prim, field, value, std::nullopt); })
        .def(
            "instances",
            [](const ScenePrim& prim, std::string_view api) {
                return prim.prim.Instances(FindSchema(api));
            },
            nb::arg("api"))
        .def(
            "get_instance",
            [](ScenePrim& prim, std::string_view api, std::string_view instance,
               std::string_view field) {
                SceneSelection single{prim.scene,
                                      prim.scene->Select(prim.prim, FindSchema(api), instance)};
                return ReadFirst(single, field);
            },
            nb::arg("api"), nb::arg("instance"), nb::arg("field"))
        .def(
            "set_instance",
            [](ScenePrim& prim, std::string_view api, std::string_view instance,
               std::string_view field, nb::handle value) {
                SceneSelection single{prim.scene,
                                      prim.scene->Select(prim.prim, FindSchema(api), instance)};
                Assign(single, field, value);
            },
            nb::arg("api"), nb::arg("instance"), nb::arg("field"), nb::arg("value"))
        .def(
            "instance_values",
            [](ScenePrim& prim, std::string_view api, std::string_view field) {
                const SchemaId schema = FindSchema(api);
                nb::dict values;
                for (const std::string& name : prim.prim.Instances(schema)) {
                    SceneSelection single{prim.scene, prim.scene->Select(prim.prim, schema, name)};
                    values[nb::str(name.c_str())] = ReadFirst(single, field);
                }
                return values;
            },
            nb::arg("api"), nb::arg("field"))
        .def(
            "set_instances",
            [](ScenePrim& prim, std::string_view api, std::string_view field,
               const nb::dict& values) {
                const SchemaId schema = FindSchema(api);
                for (const std::string& name : prim.prim.Instances(schema)) {
                    if (!values.contains(name.c_str())) {
                        prim.scene->Unapply(Span<const Prim>(&prim.prim, 1), schema, name);
                    }
                }
                for (auto [key, value] : values) {
                    const std::string name = nb::cast<std::string>(key);
                    prim.scene->Apply(Span<const Prim>(&prim.prim, 1), schema, name);
                    SceneSelection single{prim.scene, prim.scene->Select(prim.prim, schema, name)};
                    Assign(single, field, value);
                }
            },
            nb::arg("api"), nb::arg("field"), nb::arg("values"))
        .def("__eq__",
             [](const ScenePrim& prim, const ScenePrim& other) {
                 return prim.scene == other.scene && prim.prim == other.prim;
             })
        .def("__hash__",
             [](const ScenePrim& prim) {
                 return (size_t(prim.prim.Index()) << 32) ^ prim.prim.Generation() ^
                        size_t(prim.scene.Get());
             })
        .def("__repr__", [](const ScenePrim& prim) {
            return "Prim(" + prim.prim.Path() + ", " + std::string(Info(prim.prim.Schema()).name) +
                   ")";
        });
}

void RegisterMapping(nb::module_& module) {
    nb::class_<SceneMapping>(module, "Mapping")
        .def_prop_ro("open", [](const SceneMapping& held) { return held.mapping.Open(); })
        .def_prop_ro("aliased", [](const SceneMapping& held) { return held.mapping.Aliased(); })
        .def_prop_ro("values", [](const SceneMapping& held) { return held.mapping.Values(); })
        .def_prop_ro("rows",
                     [](const SceneMapping& held) -> std::optional<device::Tensor> {
                         if (held.mapping.Rows().Empty()) {
                             return std::nullopt;
                         }
                         return held.mapping.Rows();
                     })
        .def_prop_ro("offsets",
                     [](const SceneMapping& held) -> std::optional<device::Tensor> {
                         if (held.mapping.Offsets().Empty()) {
                             return std::nullopt;
                         }
                         return held.mapping.Offsets();
                     })
        .def(
            "close",
            [](SceneMapping& held, std::optional<uintptr_t> stream, nb::handle mask) {
                if (!held.mapping.Open()) {
                    return;
                }
                HeldTensors kept;
                const DLTensor* bits = kept.Take(MaskWords(mask), stream);
                nb::gil_scoped_release release;
                kept.Retire(*held.scene,
                            held.mapping.Close(ToStream(stream ? stream : held.stream), bits));
            },
            nb::kw_only(), nb::arg("stream") = nb::none(), nb::arg("mask") = nb::none())
        .def(
            "__enter__", [](SceneMapping& held) -> SceneMapping& { return held; },
            nb::rv_policy::reference)
        .def(
            "__exit__",
            [](SceneMapping& held, nb::handle, nb::handle, nb::handle) {
                if (held.mapping.Open()) {
                    nb::gil_scoped_release release;
                    held.mapping.Close(ToStream(held.stream));
                }
            },
            nb::arg("type").none(), nb::arg("value").none(), nb::arg("traceback").none());
}

void RegisterSnapshot(nb::module_& module) {
    nb::class_<Snapshot>(module, "Snapshot")
        .def_ro("ordinal", &Snapshot::ordinal)
        .def_ro("tables", &Snapshot::tables)
        .def_prop_ro("ready", [](const Snapshot& snapshot) { return snapshot.ready.Ready(); })
        .def_prop_ro("version", [](const Snapshot& snapshot) { return snapshot.Version(); })
        .def(
            "table_version",
            [](const Snapshot& snapshot, std::string_view table) {
                return snapshot.Version(table);
            },
            nb::arg("table"));
}

void RegisterSelection(nb::module_& module) {
    nb::class_<SceneSelection>(module, "Selection")
        .def("__len__", [](const SceneSelection& source) { return source.view.PrimCount(); })
        .def_prop_ro(
            "prims",
            [](const SceneSelection& source) { return ScenePrims(source.scene, source.Prims()); })
        .def_prop_ro("paths",
                     [](const SceneSelection& source) {
                         std::vector<std::string> paths;
                         for (const Prim& prim : source.Prims()) {
                             paths.push_back(prim.Path());
                         }
                         return paths;
                     })
        .def_prop_ro("dense",
                     [](const SceneSelection& source) {
                         return source.view.Schema() != SchemaId::None &&
                                source.view.Dense(source.view.Schema());
                     })
        .def(
            "world",
            [](const SceneSelection& source, uint32_t world) {
                return SceneSelection{source.scene, source.view.World(world)};
            },
            nb::arg("world"))
        .def("attach", &Attach, nb::arg("field"), nb::arg("tensor"), nb::kw_only(),
             nb::arg("initial") = "tensor", nb::arg("stream") = nb::none())
        .def("mark_changed", &MarkChanged, nb::arg("field"), nb::arg("stream") = nb::none(),
             nb::kw_only(), nb::arg("mask") = nb::none())
        .def("detach", &Detach, nb::arg("field"), nb::arg("stream") = nb::none())
        .def("wait_for_reads", &WaitForReads, nb::arg("field"), nb::arg("stream") = nb::none())
        .def(
            "attached",
            [](const SceneSelection& source, std::string_view field) {
                return source.view.Get(field).Attached();
            },
            nb::arg("field"))
        .def(
            "read",
            [](const SceneSelection& source, std::string_view field,
               std::optional<uintptr_t> stream) {
                const UntypedAttribute origin = source.view.Get(field);
                nb::gil_scoped_release release;
                return origin.Read(stream ? ToStream(stream) : Stream::Host());
            },
            nb::arg("field"), nb::arg("stream") = nb::none())
        .def(
            "write",
            [](SceneSelection& source, std::string_view field, nb::handle value,
               std::optional<uintptr_t> stream,
               nb::handle mask) { Assign(source, field, value, stream, mask); },
            nb::arg("field"), nb::arg("value"), nb::kw_only(), nb::arg("stream") = nb::none(),
            nb::arg("mask") = nb::none())
        .def(
            "map",
            [](const SceneSelection& source, std::string_view field, std::string_view access,
               std::optional<uintptr_t> stream) {
                UntypedAttribute target = source.view.Get(field);
                nb::gil_scoped_release release;
                return new SceneMapping{source.scene, target.Map(Access(access), ToStream(stream)),
                                        stream};
            },
            nb::arg("field"), nb::arg("access") = "write", nb::arg("stream") = nb::none(),
            nb::keep_alive<0, 1>())
        .def(
            "map_primvar",
            [](const SceneSelection& source, std::string_view name, std::string_view access,
               std::optional<uintptr_t> stream) {
                UntypedAttribute target = source.view.Get("primvars");
                nb::gil_scoped_release release;
                return new SceneMapping{source.scene,
                                        target.MapPrimvar(name, Access(access), ToStream(stream)),
                                        stream};
            },
            nb::arg("name"), nb::arg("access") = "write", nb::arg("stream") = nb::none(),
            nb::keep_alive<0, 1>())
        .def(
            "read_poses",
            [](const SceneSelection& source, device::DType dtype, std::string_view order,
               std::optional<uintptr_t> stream) {
                const QuaternionOrder chosen = FindOrder(order);
                nb::gil_scoped_release release;
                return source.view.ReadPoses(dtype, stream ? ToStream(stream) : Stream::Host(),
                                             chosen);
            },
            nb::arg("dtype") = device::DType::Float32, nb::kw_only(), nb::arg("order") = "xyzw",
            nb::arg("stream") = nb::none())
        .def(
            "write_poses",
            [](const SceneSelection& target, nb::handle tensor, nb::handle translation,
               nb::handle rotation, std::string_view order, nb::handle rows, nb::handle mask,
               std::optional<uintptr_t> stream) {
                const bool split = !translation.is_none() || !rotation.is_none();
                if (split == !tensor.is_none() ||
                    (split && (translation.is_none() || rotation.is_none()))) {
                    throw nb::value_error("Give poses as one [..., 7] tensor, or as a translation "
                                          "and a rotation");
                }
                device::Device& device = target.scene->GetDevice();
                HeldTensors held;
                const auto input = [&](nb::handle value, const char* label) {
                    if (value.is_none()) {
                        return device::Tensor();
                    }
                    if (nb::isinstance<device::Tensor>(value)) {
                        return nb::cast<device::Tensor>(value);
                    }
                    return held.Import(device, value, stream, label);
                };
                const device::Tensor packed = input(tensor, "pose staging");
                const device::Tensor translations = input(translation, "translation staging");
                const device::Tensor rotations = input(rotation, "rotation staging");
                const QuaternionOrder chosen = FindOrder(order);
                const device::Tensor sources = input(rows, "pose rows");
                const device::Tensor bits = input(MaskWords(mask), "pose mask");
                const Stream producer =
                    held.Producer().IsNone() && stream ? ToStream(stream) : held.Producer();
                nb::gil_scoped_release release;
                held.Retire(*target.scene,
                            split
                                ? target.view.WritePoses(translations, rotations, producer, chosen,
                                                         sources, bits)
                                : target.view.WritePoses(packed, producer, chosen, sources, bits));
            },
            nb::arg("tensor") = nb::none(), nb::kw_only(), nb::arg("translation") = nb::none(),
            nb::arg("rotation") = nb::none(), nb::arg("order") = "xyzw",
            nb::arg("rows") = nb::none(), nb::arg("mask") = nb::none(),
            nb::arg("stream") = nb::none())
        .def("__getitem__", &Read)
        .def("__setitem__", [](SceneSelection& source, std::string_view field,
                               nb::handle value) { Assign(source, field, value); })
        .def("set_each", &AssignEach, nb::arg("field"), nb::arg("values"))
        .def("__repr__", [](const SceneSelection& source) {
            return "Selection(" + std::to_string(source.view.PrimCount()) + " prims)";
        });
}

}

void BindScene(nb::module_& module) {
    RegisterPrimvar(module);
    RegisterPrimCopy(module);
    RegisterScene(module);
    RegisterPrim(module);
    RegisterMapping(module);
    RegisterSnapshot(module);
    RegisterSelection(module);
}

}
