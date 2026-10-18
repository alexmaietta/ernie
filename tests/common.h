// SPDX-License-Identifier: BSD-3-Clause
/// @file common.h
/// @brief Helpers shared by the tests: temporary and fixture paths, and small files.
#pragma once

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace ernie::test {

/// @return `name` in the test run's temporary directory.
inline std::filesystem::path TempPath(const std::string& name) {
    return std::filesystem::path(testing::TempDir()) / name;
}

/// @return `relative` under tests/fixtures.
inline std::filesystem::path FixturePath(const std::filesystem::path& relative) {
    return std::filesystem::path(ERNIE_TEST_FIXTURES) / relative;
}

/// @brief Writes bytes to a file, replacing it.
/// @return The path.
inline std::filesystem::path WriteFile(const std::filesystem::path& path, std::string_view bytes) {
    std::ofstream(path, std::ios::binary)
        .write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return path;
}

}  // namespace ernie::test
