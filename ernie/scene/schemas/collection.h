// SPDX-License-Identifier: BSD-3-Clause
/// @file collection.h
/// @brief Usd's CollectionAPI: named sets of prims, applied once per collection.
///
/// References:
/// - Pixar, UsdCollectionAPI, https://openusd.org/release/api/class_usd_collection_a_p_i.html
#pragma once

#include "ernie/scene/view.h"

#include <array>
#include <string_view>
#include <vector>

// clang-format off
#define ERNIE_COLLECTION_API_ATTRIBUTES(X)                                                         \
    X(Choice<ExpansionRule>, ExpansionRule, "collection:__INSTANCE_NAME__:expansionRule",          \
      ExpansionRule::ExpandPrims,                                                                  \
      "explicitOnly: the targets alone; expandPrims: their descendants too; "                      \
      "expandPrimsAndProperties: and their properties.",                                           \
      .group = "Collection")                                                                       \
    X(bool, IncludeRoot, "collection:__INSTANCE_NAME__:includeRoot", false,                        \
      "Whether the root prim is a member.", .group = "Collection")                                 \
    X(std::vector<Path>, Includes, "collection:__INSTANCE_NAME__:includes", std::vector<Path>{},  \
      "The prims, with their descendants per the rule, in the collection.", .group = "Collection") \
    X(std::vector<Path>, Excludes, "collection:__INSTANCE_NAME__:excludes", std::vector<Path>{},  \
      "The prims, with their descendants, left out.", .group = "Collection")
// clang-format on

namespace ernie {

enum class ExpansionRule { ExplicitOnly, ExpandPrims, ExpandPrimsAndProperties };

template <> struct ChoiceTokens<ExpansionRule> {
    static constexpr std::array<std::string_view, 3> kTokens{"explicitOnly", "expandPrims",
                                                             "expandPrimsAndProperties"};
};

ERNIE_API_SCHEMA(CollectionAPI, ERNIE_COLLECTION_API_ATTRIBUTES,
                 "A named set of prims, as light linking and material collections use them.");

}  // namespace ernie
