// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/scene_json.h"

#include "ernie/scene/prim_type.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/value.h"
#include "ernie/scene/xform.h"
#include "ernie/util/tensor.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <locale>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace ernie {
namespace {

// Objects keep their keys in document order, so prims and attributes are set in the order they
// are written.
using Json = nlohmann::ordered_json;

/// An integer as an int.
/// @throws std::invalid_argument if it does not fit in 32 bits.
int IntOf(const Json& json) {
    if (json.is_number_unsigned()) {
        const auto value = json.get<std::uint64_t>();
        if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("integer values must fit in 32 bits");
        }
        return static_cast<int>(value);
    }
    const auto value = json.get<std::int64_t>();
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("integer values must fit in 32 bits");
    }
    return static_cast<int>(value);
}

/// A 4 x 4 matrix from 4 rows of 4 numbers (math convention).
/// @throws std::invalid_argument if `json` is not one.
glm::dmat4 MatrixOf(const Json& json) {
    if (!json.is_array() || json.size() != 4) {
        throw std::invalid_argument("a matrix is 4 rows of 4 numbers");
    }
    glm::dmat4 matrix(1.0);
    for (int row = 0; row < 4; ++row) {
        const Json& values = json[static_cast<std::size_t>(row)];
        if (!values.is_array() || values.size() != 4) {
            throw std::invalid_argument("a matrix is 4 rows of 4 numbers");
        }
        for (int col = 0; col < 4; ++col) {
            const Json& value = values[static_cast<std::size_t>(col)];
            if (!value.is_number()) {
                throw std::invalid_argument("a matrix is 4 rows of 4 numbers");
            }
            matrix[col][row] = value.get<double>();
        }
    }
    return matrix;
}

/// `json` as a scene value, as Python's `ValueFromPython` converts the same value.
Value ValueOf(const Json& json) {
    if (json.is_null()) {
        return {};
    }
    if (json.is_boolean()) {
        return json.get<bool>();
    }
    if (json.is_number_integer()) {
        return IntOf(json);
    }
    if (json.is_number()) {
        return json.get<double>();
    }
    if (json.is_string()) {
        return json.get<std::string>();
    }
    if (!json.is_array()) {
        throw std::invalid_argument("values are booleans, numbers, strings, arrays of numbers or "
                                    "strings, or 4 x 4 matrices, not objects");
    }
    if (!json.empty() && json.front().is_string()) {
        std::vector<std::string> strings;
        for (const Json& item : json) {
            if (!item.is_string()) {
                throw std::invalid_argument("an array of strings must hold only strings");
            }
            strings.push_back(item.get<std::string>());
        }
        return strings;
    }
    if (json.size() == 4 && json.front().is_array()) {
        return MatrixOf(json);
    }
    bool integers = true;
    std::vector<double> numbers;
    for (const Json& item : json) {
        if (!item.is_number()) {
            throw std::invalid_argument("arrays in values must hold numbers (or strings)");
        }
        integers = integers && item.is_number_integer();
        numbers.push_back(item.get<double>());
    }
    if (numbers.size() == 2) {
        if (integers) {
            return glm::ivec2(IntOf(json[0]), IntOf(json[1]));
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

/// Appends the shape of the nested arrays `json` to `shape` at `depth`, checking that every array
/// at a depth has the same length, and its numbers to `numbers`.
void Flatten(const Json& json, std::size_t depth, std::vector<std::int64_t>& shape,
             std::vector<const Json*>& numbers) {
    if (json.is_array()) {
        if (depth == shape.size()) {
            if (!numbers.empty()) {
                throw std::invalid_argument("nested arrays must be rectangular");
            }
            shape.push_back(static_cast<std::int64_t>(json.size()));
        } else if (shape[depth] != static_cast<std::int64_t>(json.size())) {
            throw std::invalid_argument("nested arrays must be rectangular");
        }
        for (const Json& item : json) {
            Flatten(item, depth + 1, shape, numbers);
        }
        return;
    }
    if (!json.is_number() || depth != shape.size()) {
        throw std::invalid_argument(json.is_number() ? "nested arrays must be rectangular"
                                                     : "arrays must hold only numbers");
    }
    numbers.push_back(&json);
}

/// A (possibly nested) array as a CPU tensor shaped as it nests: int32 for an attribute of whole
/// numbers (`AttributeMeta::integer`), float32 otherwise, whatever the numbers look like.
Tensor TensorOf(const Json& json, bool integer) {
    if (!json.is_array()) {
        throw std::invalid_argument("an array attribute takes an array of numbers");
    }
    std::vector<std::int64_t> shape;
    std::vector<const Json*> numbers;
    Flatten(json, 0, shape, numbers);
    if (integer) {
        for (const Json* number : numbers) {
            if (!number->is_number_integer()) {
                throw std::invalid_argument("an array of whole numbers takes no fractions");
            }
        }
        std::vector<std::int32_t> values;
        for (const Json* number : numbers) {
            values.push_back(IntOf(*number));
        }
        return Tensor::Copy(std::span<const std::int32_t>(values), std::move(shape));
    }
    std::vector<float> values;
    for (const Json* number : numbers) {
        values.push_back(number->get<float>());
    }
    return Tensor::Copy(std::span<const float>(values), std::move(shape));
}

/// Whether `json` is an array of `count` numbers.
bool IsNumbers(const Json& json, std::size_t count) {
    return json.is_array() && json.size() == count &&
           std::all_of(json.begin(), json.end(), [](const Json& n) { return n.is_number(); });
}

/// Whether `json` is nested arrays other than a 4 x 4 matrix: a tensor on an undeclared
/// attribute.
bool IsTensorArray(const Json& json) {
    if (!json.is_array() || json.empty() || !json.front().is_array()) {
        return false;
    }
    const bool matrix =
        json.size() == 4 &&
        std::all_of(json.begin(), json.end(), [](const Json& r) { return IsNumbers(r, 4); });
    return !matrix;
}

/// Three numbers as a vector.
glm::dvec3 Vec3Of(const Json& json, std::string_view what) {
    if (!json.is_array() || json.size() != 3 || !json[0].is_number() || !json[1].is_number() ||
        !json[2].is_number()) {
        throw std::invalid_argument(std::string(what) + " takes 3 numbers");
    }
    return {json[0].get<double>(), json[1].get<double>(), json[2].get<double>()};
}

RotationOrder RotationOrderOf(const Json& json) {
    constexpr std::pair<std::string_view, RotationOrder> kOrders[] = {
        {"XYZ", RotationOrder::XYZ}, {"XZY", RotationOrder::XZY}, {"YXZ", RotationOrder::YXZ},
        {"YZX", RotationOrder::YZX}, {"ZXY", RotationOrder::ZXY}, {"ZYX", RotationOrder::ZYX}};
    if (json.is_string()) {
        for (const auto& [token, order] : kOrders) {
            if (token == json.get_ref<const std::string&>()) {
                return order;
            }
        }
    }
    throw std::invalid_argument(R"(rotation_order is "XYZ", "XZY", "YXZ", "YZX", "ZXY" or "ZYX")");
}

/// Sets one key of a prim: a transform part into `xform`, or an attribute.
void SetKey(Scene& scene, const Path& path, const PrimTypeDef* type, const std::string& key,
            const Json& json, XformOps& xform, bool& hasXform) {
    if (key == "translate") {
        xform.translate = Vec3Of(json, key);
    } else if (key == "rotate") {
        xform.rotate = Vec3Of(json, key);
    } else if (key == "rotation_order") {
        xform.rotationOrder = RotationOrderOf(json);
    } else if (key == "orient") {
        if (!IsNumbers(json, 4)) {
            throw std::invalid_argument("orient takes a quaternion as 4 numbers, [w, x, y, z]");
        }
        xform.orient = glm::dquat(json[0].get<double>(), json[1].get<double>(),
                                  json[2].get<double>(), json[3].get<double>());
    } else if (key == "scale") {
        xform.scale = json.is_number() ? glm::dvec3(json.get<double>()) : Vec3Of(json, key);
    } else if (key == "transform") {
        scene.Set(path, tokens::kTransform, Value(MatrixOf(json)));
        return;
    } else {
        const AttributeDef* def = nullptr;
        if (type != nullptr) {
            def = type->FindByPyName(key);
            def = def != nullptr ? def : type->Find(key);
        }
        const std::string_view name = def != nullptr ? def->name : std::string_view(key);
        if (def != nullptr && def->kind == AttributeKind::Tensor) {
            scene.Write(path, name, TensorOf(json, def->meta.integer));
        } else if (IsTensorArray(json)) {
            // Per-element data for a value attribute too, such as per-vertex displayColor.
            scene.Write(path, name, TensorOf(json, false));
        } else if (def != nullptr) {
            // A declared 4-vector keeps double precision (ValueOf makes 4 numbers a float vec4).
            Value value = def->kind == AttributeKind::Vec4 && IsNumbers(json, 4)
                              ? Value(glm::dvec4(json[0].get<double>(), json[1].get<double>(),
                                                 json[2].get<double>(), json[3].get<double>()))
                              : ValueOf(json);
            try {
                value = def->Coerce(value);
            } catch (const std::invalid_argument&) {
                // A lenient attribute's readers take other types too: keep what it cannot convert.
                if (!def->meta.lenient) {
                    throw;
                }
            }
            scene.Set(path, name, std::move(value));
        } else if ((name == tokens::kOrient ||
                    name.starts_with(std::string(tokens::kOrient) + ":")) &&
                   IsNumbers(json, 4)) {
            scene.Set(path, name,
                      glm::dquat(json[0].get<double>(), json[1].get<double>(),
                                 json[2].get<double>(), json[3].get<double>()));
        } else {
            scene.Set(path, name, ValueOf(json));
        }
        return;
    }
    hasXform = true;
}

/// Throws the error `message` about `key` (none: the prim) of the prim at `path`.
[[noreturn]] void Fail(const Path& path, std::string_view key, std::string_view message) {
    throw std::invalid_argument("Scene JSON, '" + path.String() + "'" +
                                (key.empty() ? "" : ", \"" + std::string(key) + "\"") + ": " +
                                std::string(message));
}

/// Defines (or finds) the prim at `path` as `body` describes it, then its children.
void LoadPrim(Scene& scene, const Path& path, const Json& body) {
    const auto fail = [&](std::string_view key, std::string_view message) {
        Fail(path, key, message);
    };
    if (!body.is_object()) {
        fail({}, "a prim is an object of attributes and children");
    }
    const auto found = body.find("type");
    const bool typed = found != body.end() && !found->is_null();
    if (typed && !found->is_string()) {
        fail("type", "a prim's type is a string");
    }
    try {
        if (typed || !scene.Has(path)) {
            scene.Define(path, typed ? found->get_ref<const std::string&>() : std::string());
        }
    } catch (const std::exception& error) {
        fail("type", error.what());
    }
    const PrimTypeDef* type = FindPrimType(scene.TypeOf(path));
    XformOps xform;
    bool hasXform = false;
    for (const auto& [key, value] : body.items()) {
        if (key == "type" || value.is_object()) {
            continue;
        }
        try {
            SetKey(scene, path, type, key, value, xform, hasXform);
        } catch (const std::exception& error) {
            fail(key, error.what());
        }
    }
    if (hasXform) {
        try {
            SetXform(scene, path, xform);
        } catch (const std::exception& error) {
            fail("translate/rotate/orient/scale", error.what());
        }
    }
    for (const auto& [key, value] : body.items()) {
        if (value.is_object()) {
            Path child;
            try {
                child = Path(path.String() + "/" + key);
            } catch (const std::exception& error) {
                fail(key, error.what());
            }
            LoadPrim(scene, child, value);
        }
    }
}

}  // namespace

void LoadScene(Scene& scene, std::string_view json) {
    Json description;
    try {
        description = Json::parse(json);
    } catch (const Json::parse_error& error) {
        // Its message gives the line and column (nlohmann::json's parse_error::what).
        throw std::invalid_argument(std::string("Scene JSON does not parse: ") + error.what());
    }
    if (!description.is_object()) {
        throw std::invalid_argument("Scene JSON is an object of prim paths, not " +
                                    std::string(description.type_name()));
    }
    for (const auto& [key, body] : description.items()) {
        Path path;
        try {
            path = Path(key.starts_with('/') ? key : "/" + key);
        } catch (const std::exception& error) {
            throw std::invalid_argument("Scene JSON, \"" + key + "\": " + error.what());
        }
        LoadPrim(scene, path, body);
    }
}

void LoadSceneFile(Scene& scene, const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::invalid_argument("Cannot read the scene file " + path.string());
    }
    std::ostringstream text;
    text << file.rdbuf();
    try {
        LoadScene(scene, text.str());
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(path.string() + ": " + error.what());
    }
}

namespace {

/// The keys `LoadScene` reads as parts of a prim's transform rather than as attributes.
constexpr std::string_view kXformKeys[] = {"translate", "rotate", "rotation_order",
                                           "orient",    "scale",  "transform"};

/// A double as a JSON number.
/// @throws std::invalid_argument for NaN or infinity, which JSON cannot hold.
Json NumberJson(double value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("NaN and infinite numbers cannot be written to JSON");
    }
    return value;
}

/// A float as the double of its shortest decimal form, which JSON writes as briefly ("0.1", not
/// "0.10000000149011612") and which reads back as the same float.
Json NumberJson(float value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("NaN and infinite numbers cannot be written to JSON");
    }
    char text[32];
    const auto written = std::to_chars(std::begin(text), std::end(text), value);
    // Read back through a stream in the classic locale: Apple's libc++ has no floating-point
    // from_chars, and strtod follows the C locale's decimal separator.
    std::istringstream in(std::string(std::begin(text), written.ptr));
    in.imbue(std::locale::classic());
    double shortest = 0.0;
    in >> shortest;
    return shortest;
}

template <typename T, glm::length_t N> Json VectorJson(const glm::vec<N, T>& vector) {
    Json json = Json::array();
    for (glm::length_t i = 0; i < N; ++i) {
        json.push_back(NumberJson(vector[i]));
    }
    return json;
}

/// A value as `LoadScene` reads it back.
Json ValueJson(const Value& value) {
    return std::visit(
        [](const auto& data) -> Json {
            using T = std::decay_t<decltype(data)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                return nullptr;
            } else if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, int> ||
                                 std::is_same_v<T, std::string>) {
                return data;
            } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                return NumberJson(data);
            } else if constexpr (std::is_same_v<T, glm::ivec2>) {
                return Json::array({data.x, data.y});
            } else if constexpr (std::is_same_v<T, glm::dmat4>) {
                Json rows = Json::array();
                for (int row = 0; row < 4; ++row) {
                    rows.push_back(
                        Json::array({NumberJson(data[0][row]), NumberJson(data[1][row]),
                                     NumberJson(data[2][row]), NumberJson(data[3][row])}));
                }
                return rows;
            } else if constexpr (std::is_same_v<T, glm::dquat>) {
                return Json::array({NumberJson(data.w), NumberJson(data.x), NumberJson(data.y),
                                    NumberJson(data.z)});
            } else if constexpr (std::is_same_v<T, Path>) {
                return data.String();
            } else if constexpr (std::is_same_v<T, std::vector<float>>) {
                Json json = Json::array();
                for (const float number : data) {
                    json.push_back(NumberJson(number));
                }
                return json;
            } else if constexpr (std::is_same_v<T, std::vector<Path>>) {
                Json json = Json::array();
                for (const Path& target : data) {
                    json.push_back(target.String());
                }
                return json;
            } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
                return data;
            } else {
                return VectorJson(data);
            }
        },
        value.Raw());
}

/// IEEE 754 half-precision bits as a float.
float HalfToFloat(std::uint16_t half) {
    const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000u) << 16;
    const std::uint32_t exponent = (half >> 10) & 0x1fu;
    const std::uint32_t mantissa = half & 0x3ffu;
    float magnitude = 0.0f;
    if (exponent == 0) {
        magnitude = std::ldexp(static_cast<float>(mantissa), -24);  // zero or subnormal
    } else if (exponent == 31) {
        magnitude = mantissa == 0 ? std::numeric_limits<float>::infinity()
                                  : std::numeric_limits<float>::quiet_NaN();
    } else {
        magnitude =
            std::ldexp(static_cast<float>(mantissa | 0x400u), static_cast<int>(exponent) - 25);
    }
    return sign != 0 ? -magnitude : magnitude;
}

/// Element `index` of the contiguous CPU tensor `tensor` as a JSON number.
Json ElementJson(const Tensor& tensor, std::size_t index) {
    switch (tensor.Type()) {
    case DType::UInt8:
        return tensor.As<std::uint8_t>()[index];
    case DType::UInt16:
        return tensor.As<std::uint16_t>()[index];
    case DType::UInt32:
        return tensor.As<std::uint32_t>()[index];
    case DType::Int32:
        return tensor.As<std::int32_t>()[index];
    case DType::Int64:
        return tensor.As<std::int64_t>()[index];
    case DType::Float16:
        return NumberJson(HalfToFloat(static_cast<const std::uint16_t*>(tensor.Data())[index]));
    case DType::Float32:
        return NumberJson(tensor.As<float>()[index]);
    case DType::Float64:
        return NumberJson(tensor.As<double>()[index]);
    }
    throw std::invalid_argument("unknown tensor element type");
}

/// The elements of dimensions `depth` on of `tensor` from flat index `index`, nested as its shape.
Json NestedJson(const Tensor& tensor, std::size_t depth, std::size_t& index) {
    Json json = Json::array();
    const auto length = static_cast<std::size_t>(tensor.Shape()[depth]);
    for (std::size_t i = 0; i < length; ++i) {
        json.push_back(depth + 1 == tensor.Rank() ? ElementJson(tensor, index++)
                                                  : NestedJson(tensor, depth + 1, index));
    }
    return json;
}

/// A CPU tensor as arrays nested by its shape; a rank-0 one as an array of its one element.
/// @throws std::invalid_argument for a GPU tensor.
Json TensorJson(const Tensor& tensor) {
    if (tensor.Location().kind != DeviceKind::Cpu) {
        throw std::invalid_argument(
            "GPU tensors cannot be written; write a CPU copy of it to the scene first");
    }
    const Tensor contiguous = tensor.Contiguous();
    if (contiguous.Rank() == 0) {
        return Json::array({ElementJson(contiguous, 0)});
    }
    std::size_t index = 0;
    return NestedJson(contiguous, 0, index);
}

/// Throws the error `message` about attribute `key` (none: the prim) of the prim at `path`.
[[noreturn]] void SaveFail(const Path& path, std::string_view key, std::string_view message) {
    throw std::invalid_argument("SaveScene, '" + path.String() + "'" +
                                (key.empty() ? "" : ", \"" + std::string(key) + "\"") + ": " +
                                std::string(message));
}

/// The prim at `path` and its descendants as `LoadScene` reads them.
Json PrimJson(const Scene& scene, const Path& path) {
    Json body = Json::object();
    const std::string& typeName = scene.TypeOf(path);
    if (!typeName.empty()) {
        body["type"] = typeName;
    }
    const PrimTypeDef* type = FindPrimType(typeName);
    for (const std::string& name : scene.Attributes(path)) {
        // LoadScene takes these keys for something else, so the attribute would not come back.
        if (name == "type" ||
            std::find(std::begin(kXformKeys), std::end(kXformKeys), name) != std::end(kXformKeys)) {
            SaveFail(path, name, "LoadScene reads this key as the prim's type or transform");
        }
        if (const AttributeDef* other = type != nullptr ? type->FindByPyName(name) : nullptr;
            other != nullptr && other->name != name) {
            SaveFail(path, name,
                     "LoadScene reads this key as '" + std::string(other->name) + "' of a " +
                         typeName);
        }
        try {
            if (const Value* value = scene.Get(path, name)) {
                if (!value->Empty()) {
                    body[name] = ValueJson(*value);
                }
            } else if (const Tensor* tensor = scene.Read(path, name)) {
                body[name] = TensorJson(*tensor);
            }
        } catch (const std::invalid_argument& error) {
            SaveFail(path, name, error.what());
        }
    }
    for (const Path& child : scene.Children(path)) {
        const std::string name(child.Name());
        if (name == "type" || body.contains(name)) {
            SaveFail(child, {},
                     "a child named as its parent's type key or attribute cannot be "
                     "written");
        }
        body[name] = PrimJson(scene, child);
    }
    return body;
}

/// Whether every item of the array `json` is a number, boolean, string or null.
bool IsFlat(const Json& json) {
    return std::none_of(json.begin(), json.end(),
                        [](const Json& item) { return item.is_structured(); });
}

/// Appends `json` to `out` pretty-printed at `indent`: objects a key per line, flat arrays on one
/// line, other arrays an item per line.
void Print(const Json& json, int indent, std::string& out) {
    const auto pad = [&](int width) { out.append(static_cast<std::size_t>(width), ' '); };
    if (json.is_object() && !json.empty()) {
        out += "{\n";
        bool first = true;
        for (const auto& [key, value] : json.items()) {
            out += first ? "" : ",\n";
            first = false;
            pad(indent + 2);
            out += Json(key).dump();
            out += ": ";
            Print(value, indent + 2, out);
        }
        out += "\n";
        pad(indent);
        out += "}";
    } else if (json.is_array() && !json.empty() && !IsFlat(json)) {
        out += "[\n";
        for (std::size_t i = 0; i < json.size(); ++i) {
            out += i == 0 ? "" : ",\n";
            pad(indent + 2);
            Print(json[i], indent + 2, out);
        }
        out += "\n";
        pad(indent);
        out += "]";
    } else if (json.is_array()) {
        out += "[";
        for (std::size_t i = 0; i < json.size(); ++i) {
            out += i == 0 ? "" : ", ";
            out += json[i].dump();
        }
        out += "]";
    } else {
        out += json.dump();
    }
}

}  // namespace

std::string SaveScene(const Scene& scene) {
    Json description = Json::object();
    const Path session(tokens::kSessionRoot);
    for (const Path& root : scene.Children(Path::Root())) {
        if (root != session) {
            description[root.String()] = PrimJson(scene, root);
        }
    }
    std::string text;
    Print(description, 0, text);
    text += "\n";
    return text;
}

void SaveSceneFile(const Scene& scene, const std::filesystem::path& path) {
    const std::string text = SaveScene(scene);
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file || !file.write(text.data(), static_cast<std::streamsize>(text.size())) ||
        !file.flush()) {
        throw std::invalid_argument("Cannot write the scene file " + path.string());
    }
}

}  // namespace ernie
