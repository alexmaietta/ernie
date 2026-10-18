// SPDX-License-Identifier: BSD-3-Clause
/// @file vulkan_devices_test.cpp
/// @brief Tests that a Vulkan 1.3 driver is reachable.
#include "ernie/device/device.h"

#include "tests/device/common.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

using ernie::test::VulkanRequired;

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
