// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "ernie/stage/primvar.h"
#include "ernie/stage/structure/tokens.h"

namespace ernie {

std::vector<int32_t> PackPrimvars(Tokens& tokens, std::span<const Primvar> primvars);
std::vector<Primvar> UnpackPrimvars(const Tokens& tokens, std::span<const int32_t> table);
std::vector<uint32_t> InternTokens(Tokens& tokens, std::span<const std::string> texts);

}
