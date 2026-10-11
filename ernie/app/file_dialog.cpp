// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/file_dialog.h"

#include "ernie/app/icons.h"
#include "ernie/app/panel.h"

#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iterator>
#include <system_error>
#include <utility>

namespace ernie::app {

namespace fs = std::filesystem;

namespace {

/// Folders kept in `FileDialog::Recent`.
constexpr std::size_t kRecentFolders = 8;

std::string Lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return text;
}

bool IsBlank(const std::string& text) {
    return std::all_of(text.begin(), text.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
}

bool IsDirectory(const fs::path& path) {
    std::error_code error;
    return fs::is_directory(path, error);
}

fs::path HomeDirectory() {
#ifdef _WIN32
    const char* home = std::getenv("USERPROFILE");
#else
    const char* home = std::getenv("HOME");
#endif
    return home != nullptr ? fs::path(home) : fs::path();
}

std::string SizeText(std::uintmax_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    double value = static_cast<double>(bytes);
    std::size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < std::size(units)) {
        value /= 1024.0;
        ++unit;
    }
    char text[32];
    std::snprintf(text, sizeof(text),
                  unit == 0 ? "%.0f %s" : "%.1f %s", value,
                  units[unit]);
    return text;
}

std::string TimeText(fs::file_time_type time) {
    const auto system = std::chrono::time_point_cast<
        std::chrono::system_clock::duration>(
        time - fs::file_time_type::clock::now() +
        std::chrono::system_clock::now());
    const std::time_t seconds =
        std::chrono::system_clock::to_time_t(system);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &seconds);
#else
    localtime_r(&seconds, &local);
#endif
    char text[32];
    std::strftime(text, sizeof(text), "%Y-%m-%d %H:%M", &local);
    return text;
}

}  // namespace

bool FileFilter::Matches(const fs::path& path) const {
    if (extensions.empty()) {
        return true;
    }
    const std::string extension = Lower(path.extension().string());
    return std::any_of(
        extensions.begin(), extensions.end(),
        [&](const std::string& e) { return Lower(e) == extension; });
}

std::vector<DirectoryEntry> ListDirectory(const fs::path& directory,
                                          const FileFilter* filter,
                                          bool showHidden) {
    std::vector<DirectoryEntry> entries;
    for (const fs::directory_entry& item :
         fs::directory_iterator(directory)) {
        std::error_code error;
        DirectoryEntry entry;
        entry.name = item.path().filename().string();
        if (!showHidden && !entry.name.empty() &&
            entry.name.front() == '.') {
            continue;
        }
        entry.directory = item.is_directory(error);
        if (error) {
            continue;
        }
        if (!entry.directory) {
            if (filter != nullptr && !filter->Matches(item.path())) {
                continue;
            }
            entry.size = item.file_size(error);
            if (error) {
                entry.size = 0;
            }
        }
        entry.modified = item.last_write_time(error);
        entries.push_back(std::move(entry));
    }
    std::sort(entries.begin(), entries.end(),
              [](const DirectoryEntry& a, const DirectoryEntry& b) {
                  if (a.directory != b.directory) {
                      return a.directory;
                  }
                  const std::string la = Lower(a.name);
                  const std::string lb = Lower(b.name);
                  return la != lb ? la < lb : a.name < b.name;
              });
    return entries;
}

std::optional<fs::path> ChosenPath(const fs::path& directory,
                                   const std::string& name,
                                   const FileFilter* filter,
                                   bool save) {
    if (IsBlank(name)) {
        return std::nullopt;
    }
    fs::path path = fs::path(name).is_absolute() ? fs::path(name)
                                                 : directory / name;
    path = path.lexically_normal();
    if (save && filter != nullptr && !filter->extensions.empty() &&
        !path.has_extension() && !IsDirectory(path)) {
        path += filter->extensions.front();
    }
    return path;
}

void FileDialog::Open(FileDialogOptions options, Done done) {
    options_ = std::move(options);
    done_ = std::move(done);
    open_ = true;
    popupRequested_ = true;
    filter_ = 0;
    name_.clear();
    replace_.reset();
    error_.clear();
    fs::path start = options_.start;
    if (!start.empty() && !IsDirectory(start)) {
        name_ = start.filename().string();
        start = start.parent_path();
    }
    if (start.empty() || !IsDirectory(start)) {
        start = !lastDirectory_.empty() && IsDirectory(lastDirectory_)
                    ? lastDirectory_
                    : fs::current_path();
    }
    Navigate(start);
}

bool FileDialog::Navigate(const fs::path& directory) {
    std::error_code error;
    const fs::path absolute =
        fs::absolute(directory, error).lexically_normal();
    if (error || !IsDirectory(absolute)) {
        error_ = "Not a folder: " + directory.string();
        return false;
    }
    try {
        entries_ =
            ListDirectory(absolute, ChosenFilter(), showHidden_);
    } catch (const fs::filesystem_error& failure) {
        error_ = "Cannot read " + absolute.string() + ": " +
                 failure.code().message();
        return false;
    }
    directory_ = absolute;
    pathField_ = directory_.string();
    replace_.reset();
    error_.clear();
    return true;
}

bool FileDialog::Up() {
    const fs::path parent = directory_.parent_path();
    return parent != directory_ && !parent.empty() &&
           Navigate(parent);
}

void FileDialog::SetFilter(std::size_t index) {
    if (index < options_.filters.size()) {
        filter_ = index;
        Refresh();
    }
}

void FileDialog::Refresh() {
    if (!directory_.empty()) {
        Navigate(directory_);
    }
}

const FileFilter* FileDialog::ChosenFilter() const {
    return filter_ < options_.filters.size()
               ? &options_.filters[filter_]
               : nullptr;
}

bool FileDialog::Accept(const std::string& name) {
    if (!open_) {
        return false;
    }
    const std::optional<fs::path> path =
        ChosenPath(directory_, name.empty() ? name_ : name,
                   ChosenFilter(), options_.save);
    if (!path) {
        error_ = options_.save ? "Type a file name" : "Choose a file";
        return false;
    }
    if (IsDirectory(*path)) {
        if (Navigate(*path)) {
            name_.clear();
        }
        return false;
    }
    std::error_code error;
    const bool exists = fs::exists(*path, error);
    if (!options_.save) {
        if (!exists) {
            error_ = "No such file: " + path->string();
            return false;
        }
        Choose(*path);
        return true;
    }
    if (!IsDirectory(path->parent_path())) {
        error_ = "No such folder: " + path->parent_path().string();
        return false;
    }
    if (exists && replace_ != path) {
        replace_ = *path;
        error_.clear();
        return false;
    }
    Choose(*path);
    return true;
}

bool FileDialog::ConfirmReplace() {
    if (!open_ || !replace_) {
        return false;
    }
    Choose(*replace_);
    return true;
}

void FileDialog::Cancel() {
    open_ = false;
    done_ = nullptr;
    replace_.reset();
}

void FileDialog::Choose(fs::path path) {
    lastDirectory_ = path.parent_path();
    recent_.erase(
        std::remove(recent_.begin(), recent_.end(), lastDirectory_),
        recent_.end());
    recent_.insert(recent_.begin(), lastDirectory_);
    if (recent_.size() > kRecentFolders) {
        recent_.resize(kRecentFolders);
    }
    open_ = false;
    replace_.reset();
    Done done = std::move(done_);
    done_ = nullptr;
    if (done) {
        done(path);
    }
}

void FileDialog::Draw() {
    if (!open_) {
        return;
    }
    const std::string id = options_.title + "###ernie_file_dialog";
    if (popupRequested_) {
        ImGui::OpenPopup(id.c_str());
        popupRequested_ = false;
    }
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->GetCenter(),
                            ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(
        ImVec2(std::min(860.0f, viewport->Size.x * 0.9f),
               std::min(540.0f, viewport->Size.y * 0.9f)),
        ImGuiCond_Appearing);
    bool keepOpen = true;
    if (!ImGui::BeginPopupModal(id.c_str(), &keepOpen)) {
        if (!keepOpen) {
            Cancel();
        }
        return;
    }

    // Path bar: up, the folder (editable), refresh.
    if (ImGui::Button(icons::kArrowUp)) {
        Up();
    }
    ImGui::SetItemTooltip("Up a folder");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(-1.0f);
    if (InputString("##path", pathField_,
                    ImGuiInputTextFlags_EnterReturnsTrue)) {
        if (!Navigate(pathField_)) {
            pathField_ = directory_.string();
        }
    }

    // Places on the left, the folder's contents on the right.
    const float footer = ImGui::GetFrameHeightWithSpacing() * 2.0f +
                         ImGui::GetStyle().ItemSpacing.y;
    if (ImGui::BeginChild("##places", ImVec2(170.0f, -footer),
                          ImGuiChildFlags_Borders)) {
        const auto place = [&](const char* icon,
                               const std::string& label,
                               const fs::path& path) {
            if (path.empty()) {
                return;
            }
            const std::string text = std::string(icon) + "  " +
                                     label + "##" + path.string();
            if (ImGui::Selectable(text.c_str(), path == directory_)) {
                Navigate(path);
            }
            ImGui::SetItemTooltip("%s", path.string().c_str());
        };
        place(icons::kHome, "Home", HomeDirectory());
        place(icons::kFolderOpen, "Working folder",
              fs::current_path());
        place(icons::kHardDrive, "Root", directory_.root_path());
        if (!recent_.empty()) {
            ImGui::SeparatorText("Recent");
            for (const fs::path& folder : recent_) {
                const std::string label =
                    folder.filename().empty()
                        ? folder.string()
                        : folder.filename().string();
                place(icons::kHistory, label, folder);
            }
        }
    }
    ImGui::EndChild();
    ImGui::SameLine();
    const ImGuiTableFlags tableFlags =
        ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
        ImGuiTableFlags_BordersOuter | ImGuiTableFlags_Resizable;
    std::optional<fs::path> enter;
    bool accepted = false;
    if (ImGui::BeginTable("##entries", 3, tableFlags,
                          ImVec2(0.0f, -footer))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Name",
                                ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn(
            "Size", ImGuiTableColumnFlags_WidthFixed, 90.0f);
        ImGui::TableSetupColumn(
            "Modified", ImGuiTableColumnFlags_WidthFixed, 140.0f);
        ImGui::TableHeadersRow();
        for (const DirectoryEntry& entry : entries_) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            const std::string label =
                std::string(entry.directory ? icons::kFolder
                                            : icons::kFile) +
                "  " + entry.name;
            const bool selected =
                !entry.directory && entry.name == name_;
            if (ImGui::Selectable(
                    label.c_str(), selected,
                    ImGuiSelectableFlags_SpanAllColumns |
                        ImGuiSelectableFlags_AllowDoubleClick)) {
                if (entry.directory) {
                    if (ImGui::IsMouseDoubleClicked(
                            ImGuiMouseButton_Left)) {
                        enter = directory_ / entry.name;
                    }
                } else {
                    name_ = entry.name;
                    replace_.reset();
                    accepted = ImGui::IsMouseDoubleClicked(
                        ImGuiMouseButton_Left);
                }
            }
            ImGui::TableNextColumn();
            if (!entry.directory) {
                ImGui::TextDisabled("%s",
                                    SizeText(entry.size).c_str());
            }
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%s",
                                TimeText(entry.modified).c_str());
        }
        ImGui::EndTable();
    }
    if (enter) {
        Navigate(*enter);
    }

    // Name, file type, hidden files, and the buttons.
    const float buttons = 180.0f;
    const float types = options_.filters.empty() ? 0.0f : 200.0f;
    ImGui::SetNextItemWidth(
        -(buttons + types + ImGui::GetStyle().ItemSpacing.x * 2.0f));
    if (InputString("##name", name_,
                    ImGuiInputTextFlags_EnterReturnsTrue)) {
        accepted = true;
    }
    if (!options_.filters.empty()) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(types);
        if (ImGui::BeginCombo(
                "##type", options_.filters[filter_].label.c_str())) {
            for (std::size_t i = 0; i < options_.filters.size();
                 ++i) {
                if (ImGui::Selectable(
                        options_.filters[i].label.c_str(),
                        i == filter_)) {
                    SetFilter(i);
                }
            }
            ImGui::EndCombo();
        }
    }
    ImGui::SameLine();
    if (ImGui::Button(options_.save ? "Save" : "Open",
                      ImVec2(buttons * 0.5f - 4.0f, 0.0f))) {
        accepted = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(-1.0f, 0.0f)) ||
        ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        Cancel();
    }
    bool hidden = showHidden_;
    if (ImGui::Checkbox("Hidden files", &hidden)) {
        SetShowHidden(hidden);
        Refresh();
    }
    ImGui::SameLine();
    if (replace_) {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                           "%s exists. Replace it?",
                           replace_->filename().string().c_str());
        ImGui::SameLine();
        if (ImGui::SmallButton("Replace")) {
            ConfirmReplace();
        }
    } else if (!error_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s",
                           error_.c_str());
    }
    if (accepted && open_) {
        Accept();
    }
    if (!open_) {
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

}  // namespace ernie::app
