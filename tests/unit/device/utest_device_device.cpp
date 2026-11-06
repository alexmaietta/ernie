// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/device/device.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/accel.h"
#include "ernie/device/adapter.h"
#include "ernie/device/queue.h"
#include "ernie/device/texture.h"
#include "ernie/util/error.h"
#include "support/buffers.h"
#include "support/devices.h"

namespace ernie::device {
namespace {

using test::ForEachDevice;

TEST(DeviceTest, DescribesItsAdapter) {
    ForEachDevice([](Device& device) {
        const AdapterInfo& adapter = device.Adapter();
        EXPECT_FALSE(adapter.name.empty());
        EXPECT_FALSE(BackendName(adapter.backend).empty());
        bool listed = false;
        for (const AdapterInfo& other : EnumerateAdapters()) {
            listed = listed || other.alias == adapter.alias;
        }
        EXPECT_TRUE(listed);
        EXPECT_NE(device.GetTraversal(), Traversal::Auto);
        EXPECT_NE(device.GetSampling(), Sampling::Auto);
        EXPECT_FALSE(device.CacheDirectory().empty());
    });
    EXPECT_THROW(Device::Create({.adapter = "no-such-adapter:9"}), Error);
}

TEST(DeviceTest, UploadsTexturesLevelByLevel) {
    ForEachDevice([](Device& device) {
        const Ref<Texture> texture =
            device.CreateTexture({.width = 8, .height = 4, .levels = 2, .label = "checker"});
        EXPECT_EQ(texture->Desc().width, 8u);
        EXPECT_EQ(texture->LevelBytes(0), 8u * 4 * 4);
        EXPECT_EQ(texture->LevelBytes(1), 4u * 2 * 4);
        EXPECT_NE(texture->KernelHandle(TextureWrap::Clamp, TextureWrap::Repeat), 0u);

        const std::vector<uint32_t> base(32, 0xFF00FF00u);
        const Ref<Buffer> small =
            device.CreateBuffer({.label = "small level"}, std::vector<uint32_t>(8, 0xFFu));
        device.GetQueue().Write(*texture, 0, base.data());
        CommandEncoder encoder = device.GetQueue().CreateEncoder();
        encoder.Copy(*small, *texture, 1);
        device.GetQueue().Submit(encoder.Finish());
        device.GetQueue().WaitIdle();
        EXPECT_GT(device.MemoryUsage().textureBytes, 0u);
    });
}

TEST(DeviceTest, SizesAndBuildsHardwareAccels) {
    ForEachDevice([](Device& device) {
        if (device.GetTraversal() != Traversal::Hardware) {
            return;
        }
        const float corners[9] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
        const Ref<Buffer> vertices =
            device.CreateBuffer({.usage = BufferUsage::AccelBuildInput, .label = "corners"},
                                std::vector<float>(std::begin(corners), std::end(corners)));
        const AccelInput input =
            AccelTriangles{.vertices = *vertices, .vertexCount = 3, .count = 1};
        const AccelSizes sizes = device.AccelBuildSizes(input);
        EXPECT_GT(sizes.storage, 0u);
        const Ref<Accel> accel = device.CreateAccel(input, "triangle");
        EXPECT_NE(accel->Address(), 0u);
        const Ref<Buffer> scratch = device.CreateBuffer(
            {.bytes = sizes.scratch, .usage = BufferUsage::AccelStorage, .label = "scratch"});
        CommandEncoder encoder = device.GetQueue().CreateEncoder();
        encoder.BuildAccel(*accel, input, *scratch);
        device.GetQueue().Submit(encoder.Finish());
    });
}

}
}
