// SPDX-License-Identifier: BSD-3-Clause
/// @file common.h
/// @brief Helpers shared by the device tests: validating contexts that skip without Vulkan.
#pragma once

#include "ernie/device/context.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>

namespace ernie::test {

/// @return Whether CI requires a Vulkan driver (ERNIE_TEST_REQUIRE_VULKAN=1), so tests fail
/// rather than skip without one.
inline bool VulkanRequired() {
    const char* value = std::getenv("ERNIE_TEST_REQUIRE_VULKAN");
    return value != nullptr && std::string_view(value) == "1";
}

/// @return A validating context on `device`, or none when there is no Vulkan and it is not
/// required.
inline std::unique_ptr<DeviceContext> MakeContext(std::string_view device = {}) {
    try {
        return std::make_unique<DeviceContext>(
            DeviceContextOptions{.validation = true, .device = std::string(device)});
    } catch (const DeviceError&) {
        if (VulkanRequired()) {
            throw;
        }
        return nullptr;
    }
}

/// Fails the test on any validation message.
inline void ExpectClean(DeviceContext& context) {
    for (const std::string& message : context.TakeValidationMessages()) {
        ADD_FAILURE() << message;
    }
}

}  // namespace ernie::test

/// Declares a validating context `name`, skipping the test without Vulkan.
#define ERNIE_CONTEXT(name, ...)                                                                   \
    const std::unique_ptr<::ernie::DeviceContext> name = ::ernie::test::MakeContext(__VA_ARGS__);  \
    if (!name) {                                                                                   \
        GTEST_SKIP() << "No Vulkan device";                                                        \
    }
