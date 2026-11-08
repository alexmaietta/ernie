// SPDX-License-Identifier: BSD-3-Clause
#include <string>

#include <gtest/gtest.h>

#include "ernie/util/library.h"
#include "ernie/util/profiler.h"

namespace ernie {
namespace {

TEST(RoctxTest, ZonesPushAndPopRangesWhenAProfilerHasLoadedRoctx) {
    const Library roctx = Library::Find({"libroctx64.so"});
    if (!roctx) {
        GTEST_SKIP() << "runs with the stand-in libroctx64.so preloaded";
    }
    long (*pushes)() = nullptr;
    long (*pops)() = nullptr;
    const char* (*last)() = nullptr;
    ASSERT_TRUE(roctx.Symbol("FakeRoctxPushes", pushes));
    ASSERT_TRUE(roctx.Symbol("FakeRoctxPops", pops));
    ASSERT_TRUE(roctx.Symbol("FakeRoctxLastRange", last));
    const long pushed = pushes();
    const long popped = pops();
    {
        ERNIE_ZONE("Roctx/Test");
        EXPECT_EQ(pushes(), pushed + 1);
        EXPECT_EQ(std::string(last()), "ernie:Roctx/Test");
        EXPECT_EQ(pops(), popped);
    }
    EXPECT_EQ(pops(), popped + 1);
}

}
}
