// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/disk_cache.h"

#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <string>

#include <gtest/gtest.h>

#include "ernie/util/text.h"
#include "ernie/util/version.h"
#include "support/support.h"

namespace ernie {
namespace {

size_t FileCount(const std::filesystem::path& directory) {
    size_t count = 0;
    for ([[maybe_unused]] const auto& entry : std::filesystem::directory_iterator(directory)) {
        ++count;
    }
    return count;
}

TEST(DiskCacheTest, StoredBytesComeBackUnderTheirKey) {
    const test::TemporaryDirectory directory;
    const DiskCache cache(directory.Path() / "nested");
    EXPECT_EQ(cache.Load("a"), std::nullopt);

    const std::string bytes("binary\0bytes\n", 13);
    cache.Store("a", bytes);
    EXPECT_EQ(cache.Load("a"), bytes);
    EXPECT_EQ(cache.Load("b"), std::nullopt);

    cache.Store("a", "replaced");
    EXPECT_EQ(cache.Load("a"), "replaced");
    EXPECT_EQ(FileCount(directory.Path() / "nested"), 1u);
}

TEST(DiskCacheTest, AnEntryForAnotherKeyIsAMiss) {
    const test::TemporaryDirectory directory;
    const DiskCache cache(directory.Path());
    const std::filesystem::path entry = directory.Path() / (Digest("a") + ".bin");
    std::ofstream(entry) << "ernie cache 1\nb\nbytes";
    EXPECT_EQ(cache.Load("a"), std::nullopt);
    std::ofstream(entry) << "garbage";
    EXPECT_EQ(cache.Load("a"), std::nullopt);
}

TEST(DiskCacheTest, AnUnwritableDirectoryOnlyLogsAndMisses) {
    const test::TemporaryDirectory directory;
    std::ofstream(directory.Path() / "file") << "in the way";
    const DiskCache cache(directory.Path() / "file" / "cache");
    const test::CapturedLog log;
    cache.Store("a", "bytes");
    EXPECT_EQ(cache.Load("a"), std::nullopt);
    ASSERT_EQ(log.Messages().size(), 1u);
    EXPECT_TRUE(log.Messages()[0].second.starts_with("can't write "));
}

TEST(DiskCacheTest, AFailedRenameLeavesNoTemporaryBehind) {
    const test::TemporaryDirectory directory;
    const std::filesystem::path blocked = directory.Path() / (Digest("a") + ".bin");
    std::filesystem::create_directory(blocked);
    std::ofstream(blocked / "file") << "keeps the directory non-empty";
    const DiskCache cache(directory.Path());
    const test::CapturedLog log;
    cache.Store("a", "bytes");
    ASSERT_EQ(log.Messages().size(), 1u);
    EXPECT_TRUE(log.Messages()[0].second.starts_with("can't add "));
    EXPECT_EQ(FileCount(directory.Path()), 1u);
}

void WriteFile(const std::filesystem::path& path, size_t bytes, std::chrono::hours age) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path) << std::string(bytes, 'x');
    std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now() - age);
}

TEST(DiskCacheTest, LoadingAnEntryMarksItRecentlyUsed) {
    const test::TemporaryDirectory directory;
    const DiskCache cache(directory.Path());
    cache.Store("a", "bytes");
    const std::filesystem::path entry = directory.Path() / (Digest("a") + ".bin");
    const auto old = std::filesystem::file_time_type::clock::now() - std::chrono::hours(48);
    std::filesystem::last_write_time(entry, old);
    EXPECT_EQ(cache.Load("a"), "bytes");
    EXPECT_GT(std::filesystem::last_write_time(entry), old + std::chrono::hours(47));
}

TEST(DiskCacheTest, KernelsLiveUnderTheirErnieVersion) {
    const Version version = GetBuildInfo().version;
    EXPECT_EQ(KernelCacheDirectory("/cache"),
              std::filesystem::path("/cache") / "kernels" /
                  std::format("{}.{}.{}", version.major, version.minor, version.patch));
}

TEST(DiskCacheTest, TrimmingDropsOnlyTheLeastRecentlyUsedKernels) {
    const test::TemporaryDirectory directory;
    const std::filesystem::path root = directory.Path();
    const std::filesystem::path current = KernelCacheDirectory(root);
    WriteFile(root / "0123456789abcdef.bin", 10, std::chrono::hours(1));
    WriteFile(root / "kernels" / "0.0.1" / "slang" / "stale.bin", 100, std::chrono::hours(4));
    WriteFile(root / "kernels" / "0.0.1" / "slang" / "recent.bin", 10, std::chrono::hours(1));
    WriteFile(root / "assets" / "stars.bin", 10, std::chrono::hours(1));
    WriteFile(current / "slang" / "oldest.bin", 100, std::chrono::hours(3));
    WriteFile(current / "cuda" / "older.bin", 100, std::chrono::hours(2));
    WriteFile(current / "vulkan" / "newest.bin", 100, std::chrono::hours(1));

    TrimKernelCache(root, 260);
    EXPECT_TRUE(std::filesystem::exists(root / "0123456789abcdef.bin"));
    EXPECT_FALSE(std::filesystem::exists(root / "kernels" / "0.0.1" / "slang" / "stale.bin"));
    EXPECT_TRUE(std::filesystem::exists(root / "kernels" / "0.0.1" / "slang" / "recent.bin"));
    EXPECT_TRUE(std::filesystem::exists(root / "assets" / "stars.bin"));
    EXPECT_FALSE(std::filesystem::exists(current / "slang" / "oldest.bin"));
    EXPECT_TRUE(std::filesystem::exists(current / "cuda" / "older.bin"));
    EXPECT_TRUE(std::filesystem::exists(current / "vulkan" / "newest.bin"));

    TrimKernelCache(directory.Path() / "missing", 0);
}

TEST(DiskCacheTest, ErnieCacheDirChoosesTheRoot) {
    const test::ScopedEnvironment chosen("ERNIE_CACHE_DIR", "/chosen");
    EXPECT_EQ(UserCacheDirectory(), std::filesystem::path("/chosen"));
}

#if !defined(_WIN32) && !defined(__APPLE__)
TEST(DiskCacheTest, TheUserDirectoryFollowsXdgThenHomeThenTemp) {
    const test::ScopedEnvironment unchosen("ERNIE_CACHE_DIR", std::nullopt);
    const test::ScopedEnvironment xdg("XDG_CACHE_HOME", "/xdg");
    const test::ScopedEnvironment home("HOME", "/home/user");
    EXPECT_EQ(UserCacheDirectory(), std::filesystem::path("/xdg/ernie"));
    {
        const test::ScopedEnvironment noXdg("XDG_CACHE_HOME", std::nullopt);
        EXPECT_EQ(UserCacheDirectory(), std::filesystem::path("/home/user/.cache/ernie"));
        const test::ScopedEnvironment noHome("HOME", std::nullopt);
        EXPECT_EQ(UserCacheDirectory(), std::filesystem::temp_directory_path() / "ernie-cache");
    }
}
#else
TEST(DiskCacheTest, TheUserDirectoryIsNamedForErnie) {
    const test::ScopedEnvironment unchosen("ERNIE_CACHE_DIR", std::nullopt);
    EXPECT_NE(UserCacheDirectory().string().find("ernie"), std::string::npos);
}
#endif

}
}
