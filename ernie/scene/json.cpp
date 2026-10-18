// SPDX-License-Identifier: BSD-3-Clause
/// @file json.cpp
/// @brief Reading scene files into the typed API and writing them back.
#include "ernie/scene/json.h"

#include "ernie/scene/json_value.h"
#include "ernie/scene/scene.h"
#include "ernie/scene/tokens.h"
#include "ernie/scene/xform.h"
#include "ernie/util/algorithm.h"
#include "ernie/util/file.h"

#include <glm/gtc/matrix_access.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace ernie {
namespace {

// Reading.

template <glm::length_t N> glm::vec<N, double> Vector(const Json& json) {
    if (!json.is_array() || json.size() != N) {
        throw std::invalid_argument("takes " + std::to_string(N) + " numbers");
    }
    glm::vec<N, double> vector;
    for (glm::length_t i = 0; i < N; ++i) {
        vector[i] = json[static_cast<std::size_t>(i)].get<double>();
    }
    return vector;
}

glm::dquat Quaternion(const Json& json) {
    const glm::dvec4 wxyz = Vector<4>(json);
    return {wxyz[0], wxyz[1], wxyz[2], wxyz[3]};
}

glm::dmat4 Matrix(const Json& json) {
    if (!json.is_array() || json.size() != 4) {
        throw std::invalid_argument("takes 4 rows of 4 numbers");
    }
    glm::dmat4 matrix;
    for (int row = 0; row < 4; ++row) {
        const glm::dvec4 numbers = Vector<4>(json[static_cast<std::size_t>(row)]);
        for (int column = 0; column < 4; ++column) {
            matrix[column][row] = numbers[column];
        }
    }
    return matrix;
}

bool IsMatrix(const Json& json) {
    return json.size() == 4 && std::ranges::all_of(json, [](const Json& row) {
               return row.is_array() && row.size() == 4;
           });
}

/// @return The column type an undeclared attribute's JSON suggests.
ColumnType Infer(std::string_view name, const Json& json) {
    if (name.starts_with(tokens::kOrient)) {
        return ColumnType::Quat;
    }
    if (json.is_boolean()) {
        return ColumnType::Bool;
    }
    if (json.is_number_integer()) {
        return ColumnType::Int;
    }
    if (json.is_number()) {
        return ColumnType::Double;
    }
    if (json.is_string()) {
        return ColumnType::TokenValue;
    }
    if (!json.is_array()) {
        throw std::invalid_argument("is not a value");
    }
    if (!json.empty() && std::ranges::all_of(json, [](const Json& j) { return j.is_string(); })) {
        return ColumnType::TokenArray;
    }
    if (!json.empty() && json.front().is_array()) {
        return IsMatrix(json) ? ColumnType::Matrix : ColumnType::Tensor;
    }
    static constexpr std::array kVectors{ColumnType::Vec2, ColumnType::Vec3, ColumnType::Vec4};
    return json.size() >= 2 && json.size() <= 4 ? kVectors[json.size() - 2]
                                                : ColumnType::FloatArray;
}

Value ReadValue(const Json& json, ColumnType type) {
    switch (type) {
    case ColumnType::Bool:
        return json.get<bool>();
    case ColumnType::Int:
        return json.get<int>();
    case ColumnType::Double:
        return json.get<double>();
    case ColumnType::Vec2:
        return Vector<2>(json);
    case ColumnType::Vec3:
        return Vector<3>(json);
    case ColumnType::Vec4:
        return Vector<4>(json);
    case ColumnType::Quat:
        return Quaternion(json);
    case ColumnType::Matrix:
        return Matrix(json);
    case ColumnType::TokenValue:
        return json.get<std::string>();
    case ColumnType::Relationship:
        return Path(json.get<std::string>());
    case ColumnType::Relationships: {
        std::vector<Path> targets;
        for (const Json& target : json.is_string() ? Json::array({json}) : json) {
            targets.emplace_back(target.get<std::string>());
        }
        return targets;
    }
    case ColumnType::FloatArray:
        return json.get<std::vector<float>>();
    case ColumnType::TokenArray:
        return json.get<std::vector<std::string>>();
    case ColumnType::Tensor:
        break;
    }
    throw std::invalid_argument("is an array, not a value");
}

/// @return Nested arrays as a CPU tensor: int32 for integer attributes, bytes for raw ones and
/// float32 for the rest.
Tensor ReadTensor(const Json& json, const AttributeMeta* meta) {
    std::vector<std::int64_t> shape;
    for (const Json* level = &json; level->is_array(); level = &level->front()) {
        shape.push_back(static_cast<std::int64_t>(level->size()));
        if (level->empty()) {
            break;
        }
    }
    std::vector<double> numbers;
    const auto flatten = [&](const auto& self, const Json& level, std::size_t depth) -> void {
        if (depth == shape.size()) {
            numbers.push_back(level.get<double>());
            return;
        }
        if (!level.is_array() || static_cast<std::int64_t>(level.size()) != shape[depth]) {
            throw std::invalid_argument("rows of an array are all one length");
        }
        for (const Json& item : level) {
            self(self, item, depth + 1);
        }
    };
    flatten(flatten, json, 0);
    const auto copy = [&]<typename T>(std::type_identity<T>) {
        std::vector<T> values(numbers.size());
        std::ranges::transform(numbers, values.begin(),
                               [](double number) { return static_cast<T>(number); });
        return Tensor::Copy<T>(values, shape);
    };
    if (meta != nullptr && meta->bytes) {
        return copy(std::type_identity<std::uint8_t>{});
    }
    if (meta != nullptr && meta->integer) {
        return copy(std::type_identity<std::int32_t>{});
    }
    return copy(std::type_identity<float>{});
}

RotationOrder ReadRotationOrder(const Json& json) {
    const std::string name =
        std::string(tokens::kXformOpPrefix) + "rotate" + json.get<std::string>();
    for (const RotationOrder order : {RotationOrder::XYZ, RotationOrder::XZY, RotationOrder::YXZ,
                                      RotationOrder::YZX, RotationOrder::ZXY, RotationOrder::ZYX}) {
        if (RotateOp(order) == name) {
            return order;
        }
    }
    throw std::invalid_argument("rotationOrder is one of XYZ, XZY, YXZ, YZX, ZXY or ZYX");
}

XformOps ReadXform(const Json& json) {
    XformOps xform;
    for (const auto& [key, value] : json.items()) {
        if (key == "translate") {
            xform.translate = Vector<3>(value);
        } else if (key == "rotate") {
            xform.rotate = Vector<3>(value);
        } else if (key == "rotationOrder") {
            xform.rotationOrder = ReadRotationOrder(value);
        } else if (key == "orient") {
            xform.orient = Quaternion(value);
        } else if (key == "scale") {
            xform.scale = value.is_number() ? glm::dvec3(value.get<double>()) : Vector<3>(value);
        } else {
            throw std::invalid_argument("has no op '" + key + "'");
        }
    }
    return xform;
}

/// Authors an attribute's value, or with `time` its sample there.
void LoadAttribute(Scene& scene, PrimId prim, const std::string& name, const Json& json,
                   std::span<const Tensor> arrays, std::optional<double> time = std::nullopt) {
    const auto write = [&](Tensor array) {
        time ? scene.WriteAt(prim, name, *time, std::move(array))
             : scene.Write(prim, name, std::move(array));
    };
    const AttributeDef* def = scene.Declaration(prim, name);
    if (json.is_object() && json.contains(kArray)) {
        const auto index = json[kArray].get<std::size_t>();
        if (index >= arrays.size()) {
            throw std::invalid_argument("names array " + std::to_string(index) + " of " +
                                        std::to_string(arrays.size()));
        }
        write(arrays[index]);
        return;
    }
    const ColumnType type = def != nullptr ? def->type : Infer(name, json);
    if (type == ColumnType::Tensor) {
        write(ReadTensor(json, def != nullptr ? &def->meta : nullptr));
    } else {
        const Value value = ReadValue(json, type);
        time ? scene.SetAt(prim, name, *time, value) : scene.Set(prim, name, value);
    }
}

/// Authors an attribute's samples: `"radius.timeSamples": {"0": 1, "24": 2}`, as USD writes them.
void LoadSamples(Scene& scene, PrimId prim, const std::string& name, const Json& json,
                 std::span<const Tensor> arrays) {
    if (!json.is_object()) {
        throw std::invalid_argument("maps time codes to values");
    }
    for (const auto& [time, value] : json.items()) {
        std::size_t end = 0;
        const double code = std::stod(time, &end);
        if (end != time.size()) {
            throw std::invalid_argument("'" + time + "' is no time code");
        }
        LoadAttribute(scene, prim, name, value, arrays, code);
    }
}

/// @return The scene's message without the prim it opens with ("'/Ball' (Sphere) has ..."), which
/// the problem names already.
std::string WithoutPrim(std::string what, const Path& path) {
    const std::string prim = "'" + path.String() + "'";
    if (what.starts_with(prim)) {
        what.erase(0, prim.size());
        if (what.starts_with(" (")) {
            what.erase(0, what.find(')') + 1);
        }
        what.erase(0, what.find_first_not_of(": "));
    }
    return what;
}

/// Runs `load`, a problem it throws recorded in `problems` when given, else thrown on.
/// @return Whether it loaded.
template <typename Load>
bool Attempt(std::vector<SceneProblem>* problems, const Path& path, std::string_view key,
             const Load& load) {
    try {
        load();
        return true;
    } catch (const std::exception& error) {
        const SceneProblem problem{path.String(), std::string(key),
                                   WithoutPrim(error.what(), path)};
        if (problems == nullptr) {
            throw std::invalid_argument(problem.Message());
        }
        problems->push_back(problem);
        return false;
    }
}

void LoadPrim(Scene& scene, const Path& path, const Json& json, std::span<const Tensor> arrays,
              std::vector<SceneProblem>* problems) {
    PrimId prim;
    const bool defined = Attempt(problems, path, "", [&] {
        if (!json.is_object()) {
            throw std::invalid_argument("a prim is an object");
        }
        prim = json.contains(kTypeName) || !scene.Has(path)
                   ? scene.Define(path, json.value(kTypeName, std::string()))
                   : scene.Require(path);
    });
    if (!defined) {
        return;
    }
    for (const Json& api : json.value(kApis, Json::array())) {
        Attempt(problems, path, kApis, [&] {
            if (!scene.HasApi(prim, api.get<std::string>())) {
                scene.ApplyApi(prim, api.get<std::string>());
            }
        });
    }
    for (const auto& [key, value] : json.items()) {
        if (key == kTypeName || key == kApis) {
            continue;
        }
        Attempt(problems, path, key, [&] {
            if (key == kXform) {
                SetXform(scene, path, ReadXform(value));
            } else if (key.ends_with(kTimeSamples)) {
                LoadSamples(scene, prim, key.substr(0, key.size() - kTimeSamples.size()), value,
                            arrays);
            } else {
                LoadAttribute(scene, prim, key, value, arrays);
            }
        });
    }
}

// Writing.

Json Finite(double number) {
    if (!std::isfinite(number)) {
        throw std::invalid_argument("JSON has no NaN or infinite numbers");
    }
    return number;
}

/// @return A float as the double of its shortest decimal form, so it prints as written.
Json Shortest(float number) {
    if (!std::isfinite(number)) {
        return Finite(number);
    }
    std::array<char, 32> text{};
    const auto end = std::to_chars(text.data(), text.data() + text.size(), number).ptr;
    *end = '\0';
    return std::strtod(text.data(), nullptr);
}

template <typename Vector> Json Components(const Vector& vector) {
    Json json = Json::array();
    for (glm::length_t i = 0; i < vector.length(); ++i) {
        if constexpr (std::is_same_v<typename Vector::value_type, float>) {
            json.push_back(Shortest(vector[i]));
        } else if constexpr (std::is_same_v<typename Vector::value_type, double>) {
            json.push_back(Finite(vector[i]));
        } else {
            json.push_back(vector[i]);
        }
    }
    return json;
}

Json WriteValue(const Value& value) {
    return std::visit(
        [](const auto& v) -> Json {
            using V = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<V, std::monostate>) {
                return nullptr;
            } else if constexpr (std::is_same_v<V, float>) {
                return Shortest(v);
            } else if constexpr (std::is_same_v<V, double>) {
                return Finite(v);
            } else if constexpr (std::is_same_v<V, glm::dquat>) {
                return Components(glm::dvec4(v.w, v.x, v.y, v.z));
            } else if constexpr (std::is_same_v<V, glm::dmat4>) {
                Json rows = Json::array();
                for (int row = 0; row < 4; ++row) {
                    rows.push_back(Components(glm::row(v, row)));
                }
                return rows;
            } else if constexpr (std::is_same_v<V, Path>) {
                return v.String();
            } else if constexpr (std::is_same_v<V, std::vector<Path>>) {
                return Map(v, [](const Path& path) { return path.String(); });
            } else if constexpr (std::is_same_v<V, std::vector<float>>) {
                Json numbers = Json::array();
                for (const float number : v) {
                    numbers.push_back(Shortest(number));
                }
                return numbers;
            } else if constexpr (requires { V::length(); }) {
                return Components(v);
            } else {
                return v;
            }
        },
        value.Raw());
}

/// @return A CPU tensor as arrays nested by its shape.
Json WriteTensor(const Tensor& tensor) {
    if (IsGpu(tensor.Location().kind)) {
        throw std::invalid_argument("a GPU tensor cannot be written");
    }
    std::vector<Json> numbers;
    const DType type = tensor.Type();
    if (type == DType::Float16 || type == DType::Float32) {
        for (const float number : CopyFloats(tensor, "a tensor")) {
            numbers.push_back(Shortest(number));
        }
    } else if (type == DType::Float64) {
        const Tensor packed = tensor.Contiguous();
        for (std::int64_t i = 0; i < packed.Elements(); ++i) {
            numbers.push_back(Finite(packed.As<double>()[i]));
        }
    } else {
        for (const std::int64_t number : CopyInts(tensor, "a tensor")) {
            numbers.push_back(number);
        }
    }
    const std::vector<std::int64_t>& shape = tensor.Shape();
    std::size_t next = 0;
    const auto nest = [&](const auto& self, std::size_t depth) -> Json {
        Json level = Json::array();
        const auto count = depth < shape.size() ? static_cast<std::size_t>(shape[depth]) : 1;
        for (std::size_t i = 0; i < count; ++i) {
            level.push_back(depth + 1 < shape.size() ? self(self, depth + 1) : numbers[next++]);
        }
        return level;
    };
    return nest(nest, 0);
}

Json WriteSamples(std::span<const TimeSample> samples) {
    Json json = Json::object();
    for (const TimeSample& sample : samples) {
        json[Finite(sample.time).dump()] =
            sample.tensor.Empty() ? WriteValue(sample.value) : WriteTensor(sample.tensor);
    }
    return json;
}

/// Prints JSON with 2-space indents, keeping arrays of scalars on one line.
void Print(const Json& json, std::size_t indent, std::string& out) {
    const std::string pad(indent + 2, ' ');
    if (json.is_object() && !json.empty()) {
        out += "{\n";
        for (auto item = json.begin(); item != json.end(); ++item) {
            out += pad + Json(item.key()).dump() + ": ";
            Print(item.value(), indent + 2, out);
            out += std::next(item) == json.end() ? "\n" : ",\n";
        }
        out += std::string(indent, ' ') + "}";
    } else if (json.is_array() &&
               std::ranges::any_of(json, [](const Json& j) { return j.is_structured(); })) {
        out += "[\n";
        for (std::size_t i = 0; i < json.size(); ++i) {
            out += pad;
            Print(json[i], indent + 2, out);
            out += i + 1 == json.size() ? "\n" : ",\n";
        }
        out += std::string(indent, ' ') + "]";
    } else if (json.is_array()) {
        out += "[";
        for (std::size_t i = 0; i < json.size(); ++i) {
            out += (i == 0 ? "" : ", ") + json[i].dump();
        }
        out += "]";
    } else {
        out += json.dump();
    }
}

/// @return The scene file's prims.
const Json& ParsePrims(std::string_view text, Json& root) {
    try {
        root = Json::parse(text);
    } catch (const Json::parse_error& error) {
        throw std::invalid_argument(std::string("Not JSON: ") + error.what());
    }
    if (!root.is_object() || !root.contains(kPrims) || !root[kPrims].is_object()) {
        throw std::invalid_argument("A scene file is an object with \"prims\"");
    }
    return root[kPrims];
}

}  // namespace

Json ToJson(const Value& value) {
    return WriteValue(value);
}

std::string PrintJson(const Json& json) {
    std::string out;
    Print(json, 0, out);
    return out + "\n";
}

void LoadScene(Scene& scene, std::string_view text, std::span<const Tensor> arrays) {
    Json root;
    for (const auto& [path, prim] : ParsePrims(text, root).items()) {
        LoadPrim(scene, Path(path), prim, arrays, nullptr);
    }
}

std::vector<SceneProblem> LoadSceneChecked(Scene& scene, std::string_view text,
                                           std::span<const Tensor> arrays) {
    std::vector<SceneProblem> problems;
    Json root;
    try {
        for (const auto& [path, prim] : ParsePrims(text, root).items()) {
            LoadPrim(scene, Path(path), prim, arrays, &problems);
        }
    } catch (const std::invalid_argument& error) {
        problems.push_back({"", "", error.what()});
    }
    return problems;
}

std::string SceneProblem::Message() const {
    std::string message = path.empty() ? "" : "'" + path + "'";
    message += key.empty() ? "" : (message.empty() ? "" : " ") + key;
    return message.empty() ? what : message + ": " + what;
}

void LoadSceneFile(Scene& scene, const std::filesystem::path& path) {
    try {
        LoadScene(scene, ReadTextFile(path));
    } catch (const std::runtime_error& error) {
        throw std::invalid_argument(error.what());
    }
}

std::string SaveScene(const Scene& scene) {
    Json prims = Json::object();
    for (const Path& path : scene.Prims()) {
        const PrimId prim = *scene.Find(path);
        Json json = Json::object();
        if (const std::string_view type = scene.TypeName(prim); !type.empty()) {
            json[kTypeName] = type;
        }
        if (const std::vector<std::string> apis = scene.AppliedApis(prim); !apis.empty()) {
            json[kApis] = apis;
        }
        std::vector<std::string> names = scene.Attributes(path);
        std::ranges::sort(names);
        for (const std::string& name : names) {
            try {
                const Tensor* tensor = scene.Read(prim, name);
                json[name] =
                    tensor != nullptr ? WriteTensor(*tensor) : WriteValue(*scene.Get(prim, name));
                if (const std::span<const TimeSample> samples = scene.Samples(prim, name);
                    !samples.empty()) {
                    json[name + std::string(kTimeSamples)] = WriteSamples(samples);
                }
            } catch (const std::exception& error) {
                throw std::invalid_argument(
                    SceneProblem{path.String(), name, error.what()}.Message());
            }
        }
        prims[path.String()] = std::move(json);
    }
    Json root = Json::object();
    root["$schema"] = std::string(kSceneSchemaUrl);
    root[kPrims] = std::move(prims);
    return PrintJson(root);
}

void SaveSceneFile(const Scene& scene, const std::filesystem::path& path) {
    const std::string text = SaveScene(scene);
    try {
        WriteTextFile(path, text);
    } catch (const std::runtime_error& error) {
        throw std::invalid_argument(error.what());
    }
}

}  // namespace ernie
