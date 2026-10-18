// SPDX-License-Identifier: BSD-3-Clause
/// @file common.h
/// @brief Helpers shared by the device tests: validating contexts that skip without Vulkan.
#pragma once

#include "ernie/device/context.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <initializer_list>
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

/// @return A validating context made with `options`, or none when there is no Vulkan and it is
/// not required.
inline std::unique_ptr<DeviceContext> MakeContext(DeviceContextOptions options) {
    options.validation = true;
    try {
        return std::make_unique<DeviceContext>(options);
    } catch (const DeviceError&) {
        if (VulkanRequired()) {
            throw;
        }
        return nullptr;
    }
}

/// @return A validating context on `device`, or none when there is no Vulkan and it is not
/// required.
inline std::unique_ptr<DeviceContext> MakeContext(std::string_view device = {}) {
    return MakeContext(DeviceContextOptions{.device = std::string(device)});
}

/// Fails the test on any validation message but those naming one of `ignored`.
inline void ExpectClean(DeviceContext& context,
                        std::initializer_list<std::string_view> ignored = {}) {
    for (const std::string& message : context.TakeValidationMessages()) {
        if (std::ranges::none_of(ignored, [&](std::string_view id) {
                return message.find(id) != std::string::npos;
            })) {
            ADD_FAILURE() << message;
        }
    }
}

/// The layer checks imports of memory another API allocated against what Vulkan exported under
/// the same file descriptor number, which the OS reuses once a handle is closed.
inline constexpr std::string_view kForeignImport = "VUID-VkMemoryAllocateInfo-allocationSize-01742";

}  // namespace ernie::test

/// Declares a validating context `name`, skipping the test without Vulkan.
#define ERNIE_CONTEXT(name, ...)                                                                   \
    const std::unique_ptr<::ernie::DeviceContext> name = ::ernie::test::MakeContext(__VA_ARGS__);  \
    if (!name) {                                                                                   \
        GTEST_SKIP() << "No Vulkan device";                                                        \
    }
