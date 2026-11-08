// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "ernie/export.h"
#include "ernie/scene/scene.h"

namespace ernie {

ERNIE_API std::vector<std::string> LoadShaders(Scene& scene, const std::filesystem::path& file);

}
