// SPDX-License-Identifier: BSD-3-Clause
/// @file vulkan_devices_test.cpp
/// @brief Tests that a Vulkan 1.3 driver is reachable.
#include "ernie/device/vulkan_devices.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

/// CI sets ERNIE_TEST_REQUIRE_VULKAN=1 so a missing driver fails instead of skipping.
bool VulkanRequired() {
    const char* value = std::getenv("ERNIE_TEST_REQUIRE_VULKAN");
    return value != nullptr && std::string_view(value) == "1";
}

}  // namespace

TEST(VulkanDevices, AtLeastOneSupportsVulkan13) {
    std::vector<ernie::VulkanDeviceInfo> devices;
    try {
        devices = ernie::ListVulkanDevices();
    } catch (const std::runtime_error& error) {
        if (!VulkanRequired()) {
            GTEST_SKIP() << error.what();
        }
        FAIL() << error.what();
    }
    if (devices.empty() && !VulkanRequired()) {
        GTEST_SKIP() << "No Vulkan device";
    }
    ASSERT_FALSE(devices.empty());
    bool any13 = false;
    for (const ernie::VulkanDeviceInfo& device : devices) {
        any13 = any13 || device.apiMajor > 1 || (device.apiMajor == 1 && device.apiMinor >= 3);
    }
    EXPECT_TRUE(any13);
}
