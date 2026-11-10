// SPDX-License-Identifier: BSD-3-Clause
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/compute/runtime_compiler.h"
#include "ernie/device/device.h"
#include "ernie/device/hip/api.h"
#include "ernie/device/hip/device.h"
#include "ernie/util/error.h"

namespace ernie::device::hip {
namespace {

const char* const kNothingSource = R"(extern "C" __global__ void Nothing() {})";

class HipTest : public testing::Test {
  protected:
    void SetUp() override {
        if (!Available()) {
            GTEST_SKIP() << "no HIP runtime";
        }
        try {
            m_device = Device::Create({.backend = Backend::Hip});
        } catch (const Error& error) {
            GTEST_SKIP() << error.what();
        }
        const std::vector<std::string> options =
            Traits::CompileOptions(m_device->Adapter().architecture, Size3{1, 1, 1}, 0, false);
        const std::string binary = GetHiprtc().Compile(kNothingSource, {}, options, "Nothing");
        m_scope.emplace(GetApi(), hipDevice_t(m_device->Adapter().ordinal));
        m_module = Traits::LoadModule(GetApi(), binary);
        GetApi().hipGetLastError();
    }

    void TearDown() override {
        if (m_module != nullptr) {
            Traits::UnloadModule(GetApi(), m_module);
        }
    }

    Ref<Device> m_device;
    std::optional<DeviceScope> m_scope;
    hipModule_t m_module = nullptr;
};

TEST_F(HipTest, MissingSymbolsLeaveNoLastError) {
    const Api& api = GetApi();
    EXPECT_EQ(Traits::Symbol(api, m_module, "SLANG_globalParams").first, 0u);
    EXPECT_EQ(api.hipPeekAtLastError(), hipSuccess);
}

TEST_F(HipTest, MissingSymbolsKeepAnEarlierError) {
    const Api& api = GetApi();
    const hipError_t earlier = api.hipSetDevice(-1);
    ASSERT_NE(earlier, hipSuccess);
    Traits::Symbol(api, m_module, "SLANG_globalParams");
    EXPECT_EQ(api.hipGetLastError(), earlier);
}

}
}
