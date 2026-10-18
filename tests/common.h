// SPDX-License-Identifier: BSD-3-Clause
/// @file common.h
/// @brief Helpers shared by the tests: temporary and fixture paths, small files, and timing.
#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
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

/// @return A time budget in milliseconds for this build: Debug and coverage builds run slower.
constexpr double Budget(double ms) {
    return ms * ERNIE_TEST_BUDGET_SCALE;
}

/// @brief Times `step` three times, after `setup` before each, to see past a noisy machine.
/// @return The fastest run, in milliseconds.
template <typename Setup, typename Step> double BestOf3(const Setup& setup, const Step& step) {
    double best = 1e30;
    for (int run = 0; run < 3; ++run) {
        setup();
        const auto start = std::chrono::steady_clock::now();
        step();
        const auto elapsed = std::chrono::steady_clock::now() - start;
        best = std::min(best, std::chrono::duration<double, std::milli>(elapsed).count());
    }
    return best;
}

}  // namespace ernie::test
