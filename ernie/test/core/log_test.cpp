// SPDX-License-Identifier: BSD-3-Clause
/// @file log_test.cpp
/// @brief Tests of Ernie's console messages.
#include "ernie/core/log.h"

#include <gtest/gtest.h>

namespace ernie::log {
namespace {

TEST(Log, VerbosityGatesMessages) {
    const Verbosity previous = GetVerbosity();

    SetVerbosity(Verbosity::Quiet);
    testing::internal::CaptureStdout();
    Info("hidden");
    Detail("hidden");
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "");

    SetVerbosity(Verbosity::Normal);
    testing::internal::CaptureStdout();
    Info("shown");
    Detail("hidden");
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "shown\n");

    SetVerbosity(Verbosity::Verbose);
    testing::internal::CaptureStdout();
    Detail("detail");
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "detail\n");

    SetVerbosity(previous);
}

}  // namespace
}  // namespace ernie::log
