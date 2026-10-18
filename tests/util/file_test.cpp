// SPDX-License-Identifier: BSD-3-Clause
/// @file file_test.cpp
/// @brief Tests of whole-file reads and file lookups.
#include "ernie/util/file.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace ernie {
namespace {

namespace fs = std::filesystem;

TEST(File, ReadTextFileReadsEveryByteAndReportsMissingFiles) {
    const fs::path path = fs::path(testing::TempDir()) / "ernie_file_test.txt";
    const std::string text("a\r\nb\0c", 6);
    std::ofstream(path, std::ios::binary) << text;
    EXPECT_EQ(ReadTextFile(path), text);
    fs::remove(path);
    EXPECT_THROW(ReadTextFile(path), std::runtime_error);
}

TEST(File, FindFileAnyCaseMatchesNamesInOtherCases) {
    const fs::path directory = fs::path(testing::TempDir()) / "ernie_file_test_case";
    fs::create_directories(directory);
    std::ofstream(directory / "ldem_4.img") << "x";
    // The same file, though case-insensitive file systems (Windows, macOS) find it by the name as
    // asked.
    EXPECT_TRUE(fs::equivalent(FindFileAnyCase(directory, "LDEM_4.IMG"), directory / "ldem_4.img"));
    EXPECT_EQ(FindFileAnyCase(directory, "ldem_4.img"), directory / "ldem_4.img");
    EXPECT_EQ(FindFileAnyCase(directory, "missing.lbl"), directory / "missing.lbl");
    EXPECT_EQ(FindFileAnyCase(directory / "absent", "a.lbl"), directory / "absent" / "a.lbl");
    fs::remove_all(directory);
}

}  // namespace
}  // namespace ernie
