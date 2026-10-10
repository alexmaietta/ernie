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
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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
        if (!json.is_array() || json.size() != 4 ||
            !std::all_of(json.begin(), json.end(), [](const Json& n) { return n.is_number(); })) {
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
        } else if (def != nullptr) {
            scene.Set(path, name, def->Coerce(ValueOf(json)));
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

}  // namespace ernie
