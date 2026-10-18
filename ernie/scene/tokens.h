// SPDX-License-Identifier: BSD-3-Clause
/// @file tokens.h
/// @brief Names the scene's code shares: namespaces, transform ops and attribute values. Prim types
/// name themselves (`Sphere::kName`).
#pragma once

#include <string_view>

namespace ernie::tokens {

inline constexpr std::string_view kFieldPrefix = "field:";

inline constexpr std::string_view kXformOpPrefix = "xformOp:";
inline constexpr std::string_view kPrimvarsPrefix = "primvars:";
inline constexpr std::string_view kXformOpOrder = "xformOpOrder";
inline constexpr std::string_view kTransform = "xformOp:transform";
inline constexpr std::string_view kTranslate = "xformOp:translate";
inline constexpr std::string_view kOrient = "xformOp:orient";
inline constexpr std::string_view kScale = "xformOp:scale";
inline constexpr std::string_view kResetXformStack = "!resetXformStack!";
inline constexpr std::string_view kInvertPrefix = "!invert!";

inline constexpr std::string_view kInherited = "inherited";
inline constexpr std::string_view kInvisible = "invisible";
inline constexpr std::string_view kDefault = "default";
inline constexpr std::string_view kRender = "render";
inline constexpr std::string_view kProxy = "proxy";
inline constexpr std::string_view kGuide = "guide";

}  // namespace ernie::tokens
