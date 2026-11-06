// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace ernie::app {

struct FileFilter {
    std::string label;
    std::vector<std::string> extensions;

    bool Matches(const std::filesystem::path& path) const;
};

struct FileDialogOptions {
    std::string title = "Open";
    bool save = false;
    std::vector<FileFilter> filters;
    std::filesystem::path start;
};

struct DirectoryEntry {
    std::string name;
    bool directory = false;
    uintmax_t size = 0;
    std::filesystem::file_time_type modified{};
};

std::vector<DirectoryEntry> ListDirectory(const std::filesystem::path& directory,
                                          const FileFilter* filter, bool showHidden);

class FileDialog {
  public:
    using Done = std::function<void(const std::filesystem::path&)>;

    void Open(FileDialogOptions options, Done done);
    bool Navigate(const std::filesystem::path& directory);
    bool Accept(const std::string& name = {});
    void Cancel();
    void Draw();

    bool IsOpen() const {
        return m_open;
    }

    const std::filesystem::path& Directory() const {
        return m_directory;
    }

    const std::vector<DirectoryEntry>& Entries() const {
        return m_entries;
    }

    const std::string& Error() const {
        return m_error;
    }

    const std::optional<std::filesystem::path>& ReplacePending() const {
        return m_replace;
    }

  private:
    const FileFilter* Filter() const;
    void Choose(std::filesystem::path path);
    void DrawPlaces(float footer);
    std::optional<std::filesystem::path> DrawEntries(float footer, bool& accepted);
    void DrawFooter(bool& accepted);

    FileDialogOptions m_options;
    Done m_done;
    bool m_open = false;
    bool m_popupRequested = false;
    std::filesystem::path m_directory;
    std::string m_pathField;
    std::string m_name;
    size_t m_filter = 0;
    bool m_showHidden = false;
    std::vector<DirectoryEntry> m_entries;
    std::optional<std::filesystem::path> m_replace;
    std::string m_error;
    std::vector<std::filesystem::path> m_recent;
};

}
