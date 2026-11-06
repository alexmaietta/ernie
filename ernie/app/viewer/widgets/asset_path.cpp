// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/viewer/widgets/asset_path.h"

#include <fstream>
#include <iterator>
#include <memory>
#include <utility>

#include <imgui.h>
#include <imgui_stdlib.h>

#include "ernie/app/viewer/theme/theme.h"
#include "ernie/import/importer.h"

namespace ernie::app {

std::vector<FileFilter> AssetFilters(AssetKind kind) {
    std::vector<FileFilter> filters;
    if (kind == AssetKind::Image) {
        filters.push_back({"Images (.png .jpg .jpeg .exr)", {".png", ".jpg", ".jpeg", ".exr"}});
    } else if (kind == AssetKind::Model) {
        const std::vector<std::string> extensions = ImporterExtensions();
        std::string label = "Models (";
        for (size_t i = 0; i < extensions.size(); ++i) {
            label += (i > 0 ? " " : "") + extensions[i];
        }
        filters.push_back({label + ")", extensions});
    }
    filters.push_back({"All files", {}});
    return filters;
}

std::string AddAsset(Scene& scene, const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        throw Error("Can't read {}", file.string());
    }
    auto bytes = std::make_shared<std::vector<std::byte>>();
    for (auto it = std::istreambuf_iterator<char>(in); it != std::istreambuf_iterator<char>();
         ++it) {
        bytes->push_back(std::byte(*it));
    }
    std::string key = std::filesystem::absolute(file).lexically_normal().generic_string();
    scene.AddFile(key, std::move(bytes));
    return key;
}

bool DrawAssetPath(const char* id, std::string& text, FileDialog& dialog, AssetKind kind,
                   std::function<void(const std::filesystem::path&)> chosen) {
    ImGui::PushID(id);
    const float browse = ImGui::GetFrameHeight();
    ImGui::SetNextItemWidth(-(browse + ImGui::GetStyle().ItemInnerSpacing.x));
    ImGui::InputTextWithHint("##text", "No file", &text);
    const bool committed = ImGui::IsItemDeactivatedAfterEdit();
    ImGui::SameLine(0, ImGui::GetStyle().ItemInnerSpacing.x);
    if (ImGui::Button(icons::kFolderOpen, {browse, 0})) {
        dialog.Open({.title = "Choose a File", .filters = AssetFilters(kind), .start = text},
                    std::move(chosen));
    }
    ImGui::SetItemTooltip("Browse for a file");
    ImGui::PopID();
    return committed;
}

}
