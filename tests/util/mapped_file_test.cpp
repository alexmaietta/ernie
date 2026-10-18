// SPDX-License-Identifier: BSD-3-Clause
/// @file mapped_file_test.cpp
/// @brief Tests of read-only memory-mapped files.
#include "ernie/util/mapped_file.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace ernie {
namespace {

std::filesystem::path WriteFile(const std::string& name, const std::string& contents) {
    const std::filesystem::path path = std::filesystem::path(testing::TempDir()) / name;
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
    const std::filesystem::path path = WriteFile("ernie_mapped_deleted.bin", "spectral");
    const MappedFile file(path);
    EXPECT_TRUE(std::filesystem::remove(path));
    EXPECT_FALSE(std::filesystem::exists(path));
    EXPECT_EQ(static_cast<char>(file.Bytes()[7]), 'l');
}

TEST(MappedFile, ThrowsForAMissingFile) {
    EXPECT_THROW(MappedFile(std::filesystem::path(testing::TempDir()) / "ernie_missing.bin"),
                 std::runtime_error);
}

}  // namespace
}  // namespace ernie
