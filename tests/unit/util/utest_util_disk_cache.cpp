// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/disk_cache.h"

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

#include <gtest/gtest.h>

#include "ernie/util/text.h"
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

#if !defined(_WIN32) && !defined(__APPLE__)
TEST(DiskCacheTest, TheUserDirectoryFollowsXdgThenHomeThenTemp) {
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
    EXPECT_NE(UserCacheDirectory().string().find("ernie"), std::string::npos);
}
#endif

}
}
