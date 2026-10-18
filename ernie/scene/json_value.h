// SPDX-License-Identifier: BSD-3-Clause
/// @file json_value.h
/// @brief Scene values as JSON, shared by scene files, the schema reference and the Python
/// bindings. Internal.
#pragma once

#include "ernie/scene/storage.h"
#include "ernie/scene/value.h"
#include "ernie/util/tensor.h"

#include <nlohmann/json.hpp>

#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace ernie {

using Json = nlohmann::ordered_json;

/// The keys of scene files other than attributes: names no USD attribute has.
inline constexpr std::string_view kPrims = "prims";
inline constexpr std::string_view kTypeName = "typeName";
inline constexpr std::string_view kApis = "apiSchemas";
inline constexpr std::string_view kXform = "xform";
inline constexpr std::string_view kArray = "$array";
/// The suffix of an attribute's samples by time code, as USD's text format writes them.
inline constexpr std::string_view kTimeSamples = ".timeSamples";

/// @return A value as scene files write it: vectors as arrays, quaternions as [w, x, y, z],
/// matrices as 4 rows, paths as strings, floats in their shortest form.
/// @throws std::invalid_argument for a NaN or infinite number.
Json ToJson(const Value& value);

/// @return JSON with 2-space indents, arrays of scalars kept on one line, and a final newline.
std::string PrintJson(const Json& json);

/// @return A scene file of prims, naming its JSON Schema.
Json SceneFile(Json prims);

class Scene;
struct XformOps;

/// @brief Authors an attribute from its value as scene files write it, read as the attribute's
/// declared type, else the type the JSON suggests: how Python's values reach the scene too.
/// @param arrays The tensors `{"$array": i}` names.
/// @param time   The time code of a sample; none for the value itself.
/// @throws std::invalid_argument or nlohmann::json::exception for a value of the wrong kind.
void LoadAttribute(Scene& scene, PrimId prim, const std::string& name, const Json& json,
                   std::span<const Tensor> arrays = {}, std::optional<double> time = std::nullopt);

/// @return The transform ops of a scene file's `"xform"`.
/// @throws std::invalid_argument for an unknown op or a rotation order that is none.
XformOps ReadXform(const Json& json);

}  // namespace ernie
