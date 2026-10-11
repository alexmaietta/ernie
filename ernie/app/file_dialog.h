// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace ernie::app {

/// A choice of the file dialog's file type menu: a label and the
/// extensions it shows, such as
/// {"Terrain configurations", {".json"}}; no extensions shows every
/// file.
struct FileFilter {
    std::string label;
    std::vector<std::string> extensions;  ///< With the dot, matched
                                          ///< without regard to case.

    /// Whether `path`'s extension is one of `extensions`, or there
    /// are none.
    bool Matches(const std::filesystem::path& path) const;
};

/// What a file dialog asks for (`Viewer::OpenFileDialog`).
struct FileDialogOptions {
    std::string title = "Open";  ///< The dialog's title.
    /// Choose a file to write: a new name may be typed, the first
    /// extension of the chosen filter is added to a name without one,
    /// and replacing a file asks first. Otherwise choose an existing
    /// file to read.
    bool save = false;
    /// The file type menu, first entry chosen; empty shows every
    /// file.
    std::vector<FileFilter> filters;
    /// The folder to start in, or a file whose folder to start in and
    /// whose name to suggest; empty starts in the folder last used,
    /// else the working directory.
    std::filesystem::path start;
};

/// One row of a folder listing.
struct DirectoryEntry {
    std::string name;
    bool directory = false;
    std::uintmax_t size = 0;  ///< Bytes, for files.
    std::filesystem::file_time_type modified{};
};

/// The folders and the files `filter` matches (every file without
/// one) in `directory`, folders first, each sorted by name without
/// regard to case; names starting with a dot only with `showHidden`.
/// Entries that cannot be inspected are left out.
/// @throws std::filesystem::filesystem_error if `directory` cannot be
/// read.
std::vector<DirectoryEntry>
ListDirectory(const std::filesystem::path& directory,
              const FileFilter* filter, bool showHidden);

/// The path a dialog in `directory` returns for `name` as typed: an
/// absolute name as it is, else `directory / name`, made lexically
/// normal. Saving with `filter`, a name without an extension gets the
/// filter's first one. None for an empty or blank name.
std::optional<std::filesystem::path>
ChosenPath(const std::filesystem::path& directory,
           const std::string& name, const FileFilter* filter,
           bool save);

/// A modal file browser drawn with Dear ImGui, which `Viewer` shows
/// for `OpenFileDialog`: a folder's contents with an Up button, an
/// editable path and shortcuts to home, the working directory, the
/// root and the folders used recently, a name field, a file type menu
/// and Open (or Save) and Cancel. Double-clicking a folder enters it
/// and a file chooses it.
///
/// Its state and choices (`Navigate`, `Accept`) work without ImGui,
/// so applications and tests can drive it; `Draw` needs an ImGui
/// frame.
class FileDialog {
  public:
    using Done = std::function<void(const std::filesystem::path&)>;

    /// Opens the dialog, replacing one already open without calling
    /// its callback. `done` is called once with the chosen path when
    /// the user confirms; never if they cancel.
    void Open(FileDialogOptions options, Done done);

    bool IsOpen() const {
        return open_;
    }

    const FileDialogOptions& Options() const {
        return options_;
    }

    /// The folder shown.
    const std::filesystem::path& Directory() const {
        return directory_;
    }

    /// Shows `directory` when it is a readable folder; otherwise
    /// keeps the current one and sets `Error`. Returns whether it
    /// moved.
    bool Navigate(const std::filesystem::path& directory);

    /// The folder above `Directory()`, if any.
    bool Up();

    /// The name field.
    const std::string& Name() const {
        return name_;
    }

    void SetName(std::string name) {
        name_ = std::move(name);
    }

    /// The chosen entry of `Options().filters`.
    std::size_t Filter() const {
        return filter_;
    }

    void SetFilter(std::size_t index);

    bool ShowHidden() const {
        return showHidden_;
    }

    void SetShowHidden(bool show) {
        showHidden_ = show;
    }

    /// `Directory()`'s listing as the dialog shows it, refreshed by
    /// `Refresh` and on navigation.
    const std::vector<DirectoryEntry>& Entries() const {
        return entries_;
    }

    void Refresh();

    /// Chooses `name` (the name field when empty) as the user's
    /// confirm does: a folder is entered; opening needs an existing
    /// file; saving over an existing file first asks, and a second
    /// `Accept` of the same path, or `ConfirmReplace`, replaces it.
    /// On a choice the dialog closes and calls back. Returns whether
    /// it chose; otherwise `Error` or `ReplacePending` says why.
    bool Accept(const std::string& name = {});

    /// The existing file a save is waiting to replace, if it is.
    const std::optional<std::filesystem::path>&
    ReplacePending() const {
        return replace_;
    }

    /// Replaces the file `ReplacePending()` names, closing and
    /// calling back.
    bool ConfirmReplace();

    /// Closes without calling back.
    void Cancel();

    /// Why the last navigation or choice failed; empty when it did
    /// not.
    const std::string& Error() const {
        return error_;
    }

    /// Folders of past choices, the latest first.
    const std::vector<std::filesystem::path>& Recent() const {
        return recent_;
    }

    /// Draws the dialog while it is open, inside an ImGui frame.
    void Draw();

  private:
    const FileFilter* ChosenFilter() const;
    void Choose(std::filesystem::path path);

    FileDialogOptions options_;
    Done done_;
    bool open_ = false;
    bool popupRequested_ = false;
    std::filesystem::path directory_;
    std::string pathField_;
    std::string name_;
    std::size_t filter_ = 0;
    bool showHidden_ = false;
    std::vector<DirectoryEntry> entries_;
    std::optional<std::filesystem::path> replace_;
    std::string error_;
    std::vector<std::filesystem::path> recent_;
    std::filesystem::path lastDirectory_;
};

}  // namespace ernie::app
