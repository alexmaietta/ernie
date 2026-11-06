// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace ernie::app {

std::string FormatBytes(uint64_t bytes);

std::string Lowercase(std::string_view text);

void Row(const char* label, const std::string& value);

bool RowButton(const char* icon, const char* tip);

}
