// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/device/vulkan_context.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ernie::testing {

/// Whether the environment requires a Vulkan device with the validation layer
/// (`ERNIE_TEST_REQUIRE_VALIDATION=1`, as CI sets): without one a test fails rather than skipping
/// or running unvalidated, so a missing driver or layer cannot pass as green.
inline bool RequireValidation() {
    const char* value = std::getenv("ERNIE_TEST_REQUIRE_VALIDATION");
    return value != nullptr && std::string(value) == "1";
}

/// A Vulkan context with validation enabled where the layer is installed; skips the test when no
/// Vulkan device is available, and fails it on any validation message.
class VulkanTest : public ::testing::Test {
  protected:
    void SetUp() override {
        try {
            VulkanContextOptions options;
            options.validation = true;
            context_ = VulkanContext::Create(options);
        } catch (const std::runtime_error& error) {
            if (RequireValidation()) {
                FAIL() << "No Vulkan device: " << error.what();
            }
            GTEST_SKIP() << "No Vulkan device: " << error.what();
        }
        if (RequireValidation() && !context_->ValidationEnabled()) {
            FAIL() << "The Vulkan validation layer is not installed";
        }
    }

    void TearDown() override {
        if (context_ != nullptr) {
            for (const std::string& message : context_->TakeValidationMessages()) {
                ADD_FAILURE() << "Vulkan validation: " << message;
            }
        }
    }

    std::unique_ptr<VulkanContext> context_;
};

}  // namespace ernie::testing
