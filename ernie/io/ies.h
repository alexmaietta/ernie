// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <filesystem>
#include <string_view>

#include "ernie/io/image.h"
#include "ernie/io/kernels/ies.h"

namespace ernie {

bool IsIes(const std::filesystem::path& path);

Image ReadIes(std::string_view text);

}
