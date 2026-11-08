// SPDX-License-Identifier: BSD-3-Clause
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/app/viewer/shell/capture.h"
#include "ernie/app/viewer/shell/selection.h"
#include "ernie/app/viewer/widgets/file_dialog.h"
#include "ernie/app/viewer/widgets/widgets.h"

namespace ernie::app {
namespace {

namespace fs = std::filesystem;

class TemporaryFolder {
  public:
    TemporaryFolder()
        : m_path(fs::temp_directory_path() /
                 ("ernie_viewer_" +
                  std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()))) {
        fs::remove_all(m_path);
        fs::create_directories(m_path);
    }

    ~TemporaryFolder() {
        std::error_code error;
        fs::remove_all(m_path, error);
    }

    const fs::path& Path() const {
        return m_path;
    }

    void Touch(const std::string& name) const {
        std::ofstream(m_path / name) << "x";
    }

  private:
    fs::path m_path;
};

std::vector<std::string> Names(const std::vector<DirectoryEntry>& entries) {
    std::vector<std::string> names;
    for (const DirectoryEntry& entry : entries) {
        names.push_back(entry.name);
    }
    return names;
}

TEST(SelectionTest, SetsTogglesAndClears) {
    Selection selection;
    selection.Set("/A");
    selection.Toggle("/B");
    EXPECT_EQ(selection.paths, (std::vector<std::string>{"/A", "/B"}));
    EXPECT_TRUE(selection.Contains("/B"));
    selection.Toggle("/A");
    EXPECT_EQ(selection.paths, std::vector<std::string>{"/B"});
    selection.Toggle("/B");
    EXPECT_TRUE(selection.paths.empty());
    selection.Set("/C");
    selection.Clear();
    EXPECT_FALSE(selection.Contains("/C"));
}

TEST(WidgetsTest, FormatsBytesInBinaryUnits) {
    EXPECT_EQ(FormatBytes(0), "0 B");
    EXPECT_EQ(FormatBytes(1023), "1023 B");
    EXPECT_EQ(FormatBytes(1536), "1.5 KiB");
    EXPECT_EQ(FormatBytes(uint64_t(3) << 30), "3.0 GiB");
    EXPECT_EQ(FormatBytes(uint64_t(5) << 50), "5120.0 TiB");
    EXPECT_EQ(Lowercase("MiXeD.PNG"), "mixed.png");
}

TEST(FileDialogTest, MatchesExtensionsWhateverTheirCase) {
    const FileFilter images{"Images", {".png", ".JPG"}};
    EXPECT_TRUE(images.Matches("a/b/photo.PNG"));
    EXPECT_TRUE(images.Matches("photo.jpg"));
    EXPECT_FALSE(images.Matches("photo.exr"));
    EXPECT_FALSE(images.Matches("png"));
    const FileFilter all{"All", {}};
    EXPECT_TRUE(all.Matches("anything"));
}

TEST(FileDialogTest, ListsFoldersFirstThenFilesByName) {
    const TemporaryFolder folder;
    fs::create_directory(folder.Path() / "zeta");
    fs::create_directory(folder.Path() / "Alpha");
    folder.Touch("b.png");
    folder.Touch("A.png");
    folder.Touch("notes.txt");
    folder.Touch(".hidden.png");

    const FileFilter images{"Images", {".png"}};
    EXPECT_EQ(Names(ListDirectory(folder.Path(), &images, false)),
              (std::vector<std::string>{"Alpha", "zeta", "A.png", "b.png"}));
    EXPECT_EQ(
        Names(ListDirectory(folder.Path(), nullptr, true)),
        (std::vector<std::string>{"Alpha", "zeta", ".hidden.png", "A.png", "b.png", "notes.txt"}));
}

TEST(CaptureTest, ExpandsPatternsIntoSafeFileNames) {
    EXPECT_EQ(ExpandPattern("{scene}-{camera}-{frame}", "demo", "/World/Camera", 7),
              "demo-_World_Camera-0007");
    EXPECT_EQ(ExpandPattern("{unknown}_{scene}", "s", "c", 0), "{unknown}_s");
    EXPECT_EQ(ExpandPattern("", "s", "c", 0), "capture");
    EXPECT_EQ(ExpandPattern("a:b*c?", "s", "c", 0), "a_b_c_");
    EXPECT_EQ(ExpandPattern("{date}", "s", "c", 0).size(), 10u);
}

TEST(CaptureTest, FindsAFreePathBesideTakenOnes) {
    const TemporaryFolder folder;
    EXPECT_EQ(FreePath(folder.Path() / "shot.png"), folder.Path() / "shot.png");
    folder.Touch("shot.png");
    EXPECT_EQ(FreePath(folder.Path() / "shot.png"), folder.Path() / "shot_1.png");
    folder.Touch("shot_1.png");
    EXPECT_EQ(FreePath(folder.Path() / "shot.png"), folder.Path() / "shot_2.png");
}

}
}
