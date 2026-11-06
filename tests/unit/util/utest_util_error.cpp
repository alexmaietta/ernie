// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/error.h"

#include <string>

#include <gtest/gtest.h>

namespace ernie {
namespace {

TEST(ErrorTest, CarriesPlainAndFormattedMessages) {
    EXPECT_STREQ(Error(std::string("plain")).what(), "plain");
    EXPECT_STREQ(Error("{} of {}", 3, "four").what(), "3 of four");
}

TEST(ErrorTest, SpecializedErrorsAreErrors) {
    EXPECT_THROW(throw UnsupportedError("{} is missing", "a feature"), Error);
    EXPECT_THROW(throw DeviceLostError(std::string("lost")), Error);
    EXPECT_STREQ(UnsupportedError("{} is missing", "a feature").what(), "a feature is missing");
}

TEST(ErrorTest, ApiErrorsNameTheCallAndCode) {
    const ApiError described("cuMemAlloc", "CUDA_ERROR_OUT_OF_MEMORY", 2, "out of memory");
    EXPECT_STREQ(described.what(),
                 "cuMemAlloc failed: CUDA_ERROR_OUT_OF_MEMORY (2): out of memory");
    EXPECT_EQ(described.Call(), "cuMemAlloc");
    EXPECT_EQ(described.ErrorName(), "CUDA_ERROR_OUT_OF_MEMORY");
    EXPECT_EQ(described.Code(), 2);

    const ApiError bare("vkCreateDevice", "VK_ERROR_INITIALIZATION_FAILED", -3);
    EXPECT_STREQ(bare.what(), "vkCreateDevice failed: VK_ERROR_INITIALIZATION_FAILED (-3)");
    EXPECT_THROW(throw ApiError("hipMalloc", "hipErrorOutOfMemory", 2), Error);
}

}
}
