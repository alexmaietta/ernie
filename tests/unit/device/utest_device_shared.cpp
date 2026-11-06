// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/kernels/shared.h"

#include <gtest/gtest.h>

namespace ernie {
namespace {

static_assert(RoundUp(0, 4) == 0);
static_assert(RoundUp(5, 4) == 8);

TEST(SharedTest, RoundsUpToAMultiple) {
    volatile uint64_t value = 257;
    EXPECT_EQ(RoundUp(value, 256), 512u);
    EXPECT_EQ(RoundUp(256, 256), 256u);
    EXPECT_EQ(RoundUp(1, 1), 1u);
}

}
}
