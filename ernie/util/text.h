// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ernie {

std::string Lowercase(std::string_view text);

std::string Join(const std::vector<std::string>& parts, std::string_view separator = ", ");

std::vector<std::string_view> Split(std::string_view text, char separator);

std::pair<std::string_view, std::string_view> SplitLast(std::string_view text, char separator);

bool IsIdentifier(std::string_view text);

bool Glob(std::string_view pattern, std::string_view text);

std::string Digest(std::string_view bytes);

}
