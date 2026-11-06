// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/log.h"

#include <cstdlib>
#include <optional>
#include <string>

#include <gtest/gtest.h>

#include "support/support.h"

namespace ernie {
namespace {

TEST(LogTest, FormatsMessagesForTheSink) {
    const test::CapturedLog log(LogLevel::Debug);
    Log(LogLevel::Info, "{} devices, {:.1f} GiB", 2, 7.5);
    ASSERT_EQ(log.Messages().size(), 1u);
    EXPECT_EQ(log.Messages()[0].first, LogLevel::Info);
    EXPECT_EQ(log.Messages()[0].second, "2 devices, 7.5 GiB");
}

TEST(LogTest, DropsMessagesBelowTheLevel) {
    const test::CapturedLog log(LogLevel::Warning);
    EXPECT_EQ(GetLogLevel(), LogLevel::Warning);
    Log(LogLevel::Debug, "dropped");
    Log(LogLevel::Info, "dropped");
    LogMessage(LogLevel::Info, "dropped");
    Log(LogLevel::Warning, "kept");
    Log(LogLevel::Error, "kept");
    EXPECT_EQ(log.Messages().size(), 2u);
}

TEST(LogTest, OffIsALevelNotAMessage) {
    const test::CapturedLog everything(LogLevel::Debug);
    LogMessage(LogLevel::Off, "dropped");
    SetLogLevel(LogLevel::Off);
    Log(LogLevel::Error, "dropped");
    EXPECT_TRUE(everything.Messages().empty());
}

TEST(LogTest, WritesPrefixedLinesToStderrWithoutASink) {
    const LogLevel level = GetLogLevel();
    SetLogLevel(LogLevel::Debug);
    testing::internal::CaptureStderr();
    LogMessage(LogLevel::Debug, "a");
    LogMessage(LogLevel::Info, "b");
    LogMessage(LogLevel::Warning, "c");
    LogMessage(LogLevel::Error, "d");
    const std::string written = testing::internal::GetCapturedStderr();
    SetLogLevel(level);
    EXPECT_EQ(written, "ernie: debug: a\nernie: b\nernie: warning: c\nernie: error: d\n");
}

class LogEnvironmentTest : public testing::Test {
  protected:
    void SetUp() override {
        test::UseFreshProcessForDeathTests();
    }
};

TEST_F(LogEnvironmentTest, DefaultsToInfo) {
    const test::ScopedEnvironment unset("ERNIE_LOG", std::nullopt);
    EXPECT_EXIT(std::exit(int(GetLogLevel())), testing::ExitedWithCode(int(LogLevel::Info)), "");
}

TEST_F(LogEnvironmentTest, ReadsTheLevelIgnoringCase) {
    const test::ScopedEnvironment error("ERNIE_LOG", "Error");
    EXPECT_EXIT(std::exit(int(GetLogLevel())), testing::ExitedWithCode(int(LogLevel::Error)), "");
}

TEST_F(LogEnvironmentTest, WarnsAboutAnUnknownLevelAndUsesInfo) {
    const test::ScopedEnvironment loud("ERNIE_LOG", "loud");
    EXPECT_EXIT(
        std::exit(int(GetLogLevel())), testing::ExitedWithCode(int(LogLevel::Info)),
        "ernie: warning: ERNIE_LOG is \"loud\"; it may be debug, info, warning, error or off");
}

}
}
