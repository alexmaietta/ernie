// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "ernie/app/viewer/widgets/file_dialog.h"
#include "ernie/scene/scene.h"

namespace ernie::app {

enum class AssetKind { Image, Model, Any };

std::vector<FileFilter> AssetFilters(AssetKind kind);

std::string AddAsset(Scene& scene, const std::filesystem::path& file);

bool DrawAssetPath(const char* id, std::string& text, FileDialog& dialog, AssetKind kind,
                   std::function<void(const std::filesystem::path&)> chosen);

}
