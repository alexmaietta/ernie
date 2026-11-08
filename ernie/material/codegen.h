// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <functional>
#include <map>
#include <string>

#include "ernie/material/definition.h"

namespace ernie {

inline constexpr const char* kDispatchFile = "ernie/render/kernels/material/dispatch.slang";

std::map<std::string, std::string, std::less<>> DispatchSources(const DefinitionSet& set);

}
