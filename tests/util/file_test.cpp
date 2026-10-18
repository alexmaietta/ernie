// SPDX-License-Identifier: BSD-3-Clause
/// @file file_test.cpp
/// @brief Tests of whole-file reads, file lookups and mapped files.
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

fs::path WriteFile(const std::string& name, const std::string& contents) {
    const fs::path path = fs::path(testing::TempDir()) / name;
    std::ofstream(path, std::ios::binary) << contents;
    return path;
}

TEST(MappedFile, ExposesTheFilesBytes) {
    const MappedFile file(WriteFile("ernie_mapped.bin", "spectral"));
    const auto bytes = file.Bytes();
    ASSERT_EQ(bytes.size(), 8u);
    EXPECT_EQ(static_cast<char>(bytes[0]), 's');
    EXPECT_EQ(static_cast<char>(bytes[7]), 'l');
}

TEST(MappedFile, MapsAnEmptyFile) {
    const MappedFile file(WriteFile("ernie_mapped_empty.bin", ""));
    EXPECT_TRUE(file.Bytes().empty());
}

TEST(MappedFile, AMappedFileCanBeDeletedAndItsViewStays) {
    // As caches evict entries that live textures still map, on every platform.
    const fs::path path = WriteFile("ernie_mapped_deleted.bin", "spectral");
    const MappedFile file(path);
    EXPECT_TRUE(fs::remove(path));
    EXPECT_FALSE(fs::exists(path));
    EXPECT_EQ(static_cast<char>(file.Bytes()[7]), 'l');
}

TEST(MappedFile, ThrowsForAMissingFile) {
    EXPECT_THROW(MappedFile(fs::path(testing::TempDir()) / "ernie_missing.bin"),
                 std::runtime_error);
}

}  // namespace
}  // namespace ernie
