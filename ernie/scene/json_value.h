// SPDX-License-Identifier: BSD-3-Clause
/// @file json_value.h
/// @brief Scene values as JSON, shared by scene files and the schema reference. Internal.
#pragma once

#include "ernie/scene/value.h"

#include <nlohmann/json.hpp>

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

}  // namespace ernie
