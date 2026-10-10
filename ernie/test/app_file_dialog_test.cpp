// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/app/file_dialog.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using ernie::app::ChosenPath;
using ernie::app::DirectoryEntry;
using ernie::app::FileDialog;
using ernie::app::FileFilter;
using ernie::app::ListDirectory;

namespace {

/// A fresh folder with a.json, B.JSON, c.txt, .hidden and folders sub and Zed.
fs::path Folder(const std::string& name) {
    const fs::path dir = fs::temp_directory_path() / ("ernie_file_dialog_" + name);
    fs::remove_all(dir);
    fs::create_directories(dir / "sub");
    fs::create_directories(dir / "Zed");
    for (const char* file : {"a.json", "B.JSON", "c.txt", ".hidden"}) {
        std::ofstream(dir / file) << "12345";
    }
    return dir;
}

std::vector<std::string> Names(const std::vector<DirectoryEntry>& entries) {
    std::vector<std::string> names;
    for (const DirectoryEntry& entry : entries) {
        names.push_back(entry.name);
    }
    return names;
}

const FileFilter kJson{"JSON", {".json"}};

}  // namespace

TEST(FileFilter, MatchesExtensionsWithoutRegardToCase) {
    EXPECT_TRUE(kJson.Matches("x/a.json"));
    EXPECT_TRUE(kJson.Matches("B.JSON"));
    EXPECT_FALSE(kJson.Matches("c.txt"));
    EXPECT_FALSE(kJson.Matches("json"));
    const FileFilter all{"All", {}};
    EXPECT_TRUE(all.Matches("anything"));
}

TEST(FileDialogListing, PutsFoldersFirstAndFiltersFiles) {
    const fs::path dir = Folder("listing");
    EXPECT_EQ(Names(ListDirectory(dir, nullptr, false)),
              (std::vector<std::string>{"sub", "Zed", "a.json", "B.JSON", "c.txt"}));
    EXPECT_EQ(Names(ListDirectory(dir, &kJson, false)),
              (std::vector<std::string>{"sub", "Zed", "a.json", "B.JSON"}));
    const std::vector<DirectoryEntry> all = ListDirectory(dir, nullptr, true);
    EXPECT_EQ(all.size(), 6u);
    EXPECT_EQ(all[2].name, ".hidden");
    EXPECT_TRUE(all[0].directory);
    EXPECT_EQ(all[3].size, 5u);
    EXPECT_THROW(ListDirectory(dir / "missing", nullptr, false), fs::filesystem_error);
}

TEST(FileDialogListing, ResolvesTypedNames) {
    const fs::path dir = Folder("names");
    EXPECT_EQ(ChosenPath(dir, "a.json", &kJson, false), dir / "a.json");
    EXPECT_EQ(ChosenPath(dir, "sub/../c.txt", nullptr, false), dir / "c.txt");
    EXPECT_EQ(ChosenPath(dir, "/abs/x.json", nullptr, false), fs::path("/abs/x.json"));
    EXPECT_EQ(ChosenPath(dir, "new", &kJson, true), dir / "new.json") << "saving adds the type";
    EXPECT_EQ(ChosenPath(dir, "new", &kJson, false), dir / "new");
    EXPECT_EQ(ChosenPath(dir, "new.tif", &kJson, true), dir / "new.tif");
    EXPECT_EQ(ChosenPath(dir, "sub", &kJson, true), dir / "sub") << "folders stay folders";
    EXPECT_EQ(ChosenPath(dir, "  ", &kJson, true), std::nullopt);
}

TEST(FileDialog, OpensAnExistingFileAndEntersFolders) {
    const fs::path dir = Folder("open");
    FileDialog dialog;
    std::vector<fs::path> chosen;
    dialog.Open({"Open", false, {kJson}, dir / "a.json"},
                [&](const fs::path& path) { chosen.push_back(path); });
    ASSERT_TRUE(dialog.IsOpen());
    EXPECT_EQ(dialog.Directory(), dir);
    EXPECT_EQ(dialog.Name(), "a.json") << "a start file is suggested";
    EXPECT_EQ(Names(dialog.Entries()),
              (std::vector<std::string>{"sub", "Zed", "a.json", "B.JSON"}));

    EXPECT_FALSE(dialog.Accept("missing.json"));
    EXPECT_NE(dialog.Error().find("No such file"), std::string::npos);
    EXPECT_FALSE(dialog.Accept("   "));
    EXPECT_FALSE(dialog.Accept("sub"));
    EXPECT_EQ(dialog.Directory(), dir / "sub") << "accepting a folder enters it";
    EXPECT_TRUE(dialog.Up());
    EXPECT_EQ(dialog.Directory(), dir);
    EXPECT_FALSE(dialog.Navigate(dir / "missing"));
    EXPECT_EQ(dialog.Directory(), dir);
    EXPECT_FALSE(dialog.Error().empty());

    EXPECT_TRUE(dialog.Name().empty()) << "entering a folder clears the name";
    dialog.SetName("a.json");
    EXPECT_TRUE(dialog.Accept());
    EXPECT_FALSE(dialog.IsOpen());
    EXPECT_EQ(chosen, (std::vector<fs::path>{dir / "a.json"}));
    EXPECT_FALSE(dialog.Accept()) << "a closed dialog chooses nothing";
    ASSERT_FALSE(dialog.Recent().empty());
    EXPECT_EQ(dialog.Recent().front(), dir);
}

TEST(FileDialog, SavingAddsTheTypeAndAsksBeforeReplacing) {
    const fs::path dir = Folder("save");
    FileDialog dialog;
    fs::path chosen;
    const auto keep = [&](const fs::path& path) { chosen = path; };
    dialog.Open({"Save", true, {kJson, {"All files", {}}}, dir}, keep);
    EXPECT_TRUE(dialog.Accept("fresh"));
    EXPECT_EQ(chosen, dir / "fresh.json");

    dialog.Open({"Save", true, {kJson}, {}}, keep);
    EXPECT_EQ(dialog.Directory(), dir) << "starts where the last choice was";
    EXPECT_FALSE(dialog.Accept("a"));
    ASSERT_TRUE(dialog.ReplacePending().has_value());
    EXPECT_EQ(*dialog.ReplacePending(), dir / "a.json");
    EXPECT_TRUE(dialog.IsOpen());
    EXPECT_TRUE(dialog.Accept("a")) << "accepting the same file again replaces it";
    EXPECT_EQ(chosen, dir / "a.json");

    dialog.Open({"Save", true, {kJson}, dir}, keep);
    EXPECT_FALSE(dialog.Accept("a.json"));
    EXPECT_TRUE(dialog.ConfirmReplace());
    EXPECT_FALSE(dialog.ConfirmReplace());

    dialog.Open({"Save", true, {kJson}, dir}, keep);
    EXPECT_FALSE(dialog.Accept("nowhere/x.json"));
    EXPECT_NE(dialog.Error().find("No such folder"), std::string::npos);
}

TEST(FileDialog, CancelsWithoutChoosingAndFiltersByType) {
    const fs::path dir = Folder("cancel");
    FileDialog dialog;
    int calls = 0;
    dialog.Open({"Open", false, {kJson, {"Text", {".txt"}}}, dir},
                [&](const fs::path&) { ++calls; });
    dialog.SetFilter(1);
    EXPECT_EQ(dialog.Filter(), 1u);
    EXPECT_EQ(Names(dialog.Entries()), (std::vector<std::string>{"sub", "Zed", "c.txt"}));
    dialog.SetFilter(7);
    EXPECT_EQ(dialog.Filter(), 1u) << "an index past the filters is ignored";
    dialog.SetShowHidden(true);
    dialog.Refresh();
    EXPECT_EQ(dialog.Entries().size(), 3u) << ".hidden is no text file";
    dialog.Cancel();
    EXPECT_FALSE(dialog.IsOpen());
    EXPECT_EQ(calls, 0);
    EXPECT_TRUE(dialog.Recent().empty());
}
