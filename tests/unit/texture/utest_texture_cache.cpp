// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/texture/cache.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ernie/device/queue.h"
#include "ernie/io/image.h"
#include "support/devices.h"
#include "support/support.h"

namespace ernie {
namespace {

using device::Device;
using test::ForEachDevice;

FileBytes ReadFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    std::vector<char> chars((std::istreambuf_iterator<char>(stream)), {});
    auto bytes = std::make_shared<std::vector<std::byte>>(chars.size());
    std::memcpy(bytes->data(), chars.data(), chars.size());
    return bytes;
}

Image Gradient(uint32_t width, uint32_t height) {
    Image image(width, height);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            image.Pixel(x, y) = {float(x) / float(width), float(y) / float(height), 0.5f, 1};
        }
    }
    return image;
}

class Files {
  public:
    Files() {
        WritePng(m_directory.Path() / "small.png", Gradient(5, 3));
        WritePng(m_directory.Path() / "large.png", Gradient(16, 8));
        WriteExr(m_directory.Path() / "linear.exr", Gradient(4, 4));
        for (const char* name : {"small.png", "large.png", "linear.exr"}) {
            m_files[name] = ReadFile(m_directory.Path() / name);
        }
        m_files["corrupt.png"] = std::make_shared<std::vector<std::byte>>(64, std::byte{0x42});
    }

    FileLookup Lookup() const {
        return [this](const std::string& path) -> FileBytes {
            const auto found = m_files.find(path);
            return found == m_files.end() ? nullptr : found->second;
        };
    }

  private:
    test::TemporaryDirectory m_directory;
    std::map<std::string, FileBytes> m_files;
};

void ExpectLoadsAndResolves(Device& device, const Files& files) {
    TextureCache cache(device, files.Lookup(), 2);
    std::vector<TextureBinding> bindings;
    for (const char* name : {"small.png", "large.png", "linear.exr", kCheckerboardTexture.data()}) {
        bindings.push_back(cache.Bind({name, true}, TextureWrap::Clamp, TextureWrap::Clamp));
        EXPECT_EQ(bindings.back().handle, 0u);
    }
    cache.Finish();
    EXPECT_EQ(cache.Collect().size(), 4u);
    EXPECT_FALSE(cache.Pending());
    EXPECT_EQ(cache.Loading(), 0u);
    EXPECT_EQ(cache.Failed(), 0u);
    EXPECT_EQ(cache.Size({"small.png", true}), (std::pair<uint32_t, uint32_t>{5, 3}));
    EXPECT_EQ(cache.Size({"large.png", true}), (std::pair<uint32_t, uint32_t>{16, 8}));
    for (TextureBinding& binding : bindings) {
        cache.Resolve(binding);
        EXPECT_NE(binding.handle, 0u) << binding.key.first;
    }
    for (const TextureBinding& binding : bindings) {
        cache.Unbind(binding);
    }
    EXPECT_EQ(cache.Images(), 0u);
    device.GetQueue().WaitIdle();
}

TEST(TextureCacheTest, LoadsImagesAndResolvesBindings) {
    const Files files;
    ForEachDevice([&](Device& device) { ExpectLoadsAndResolves(device, files); });
}

TEST(TextureCacheTest, ResolvesSoftwareSampledDescriptors) {
    const Files files;
    ForEachDevice([&](Device& device) {
        const Ref<Device> software = Device::Create(
            {.adapter = device.Adapter().alias, .sampling = device::Sampling::Software});
        ExpectLoadsAndResolves(*software, files);

        TextureCache cache(*software, files.Lookup(), 1);
        TextureBinding first =
            cache.Bind({"small.png", true}, TextureWrap::Repeat, TextureWrap::Repeat);
        cache.Finish();
        cache.Collect();
        cache.Resolve(first);
        ASSERT_NE(first.desc, ~0u);
        cache.Unbind(first);
        TextureBinding second =
            cache.Bind({"small.png", false}, TextureWrap::Mirror, TextureWrap::Mirror);
        cache.Finish();
        cache.Collect();
        cache.Resolve(second);
        EXPECT_EQ(second.desc, first.desc);
        cache.Unbind(second);
        software->GetQueue().WaitIdle();
    });
}

TEST(TextureCacheTest, CountsImagesThatCannotBeRead) {
    const Files files;
    ForEachDevice([&](Device& device) {
        TextureCache cache(device, files.Lookup(), 2);
        TextureBinding missing =
            cache.Bind({"missing.png", true}, TextureWrap::Repeat, TextureWrap::Repeat);
        TextureBinding corrupt =
            cache.Bind({"corrupt.png", true}, TextureWrap::Repeat, TextureWrap::Repeat);
        const TextureBinding none =
            cache.Bind({"", true}, TextureWrap::Repeat, TextureWrap::Repeat);
        EXPECT_EQ(none.handle, 0u);
        cache.Finish();
        cache.Collect();
        EXPECT_EQ(cache.Failed(), 2u);
        EXPECT_EQ(cache.Loading(), 0u);
        EXPECT_FALSE(cache.Size(missing.key));
        cache.Resolve(missing);
        cache.Resolve(corrupt);
        EXPECT_EQ(missing.handle, 0u);
        EXPECT_EQ(corrupt.handle, 0u);
        cache.Unbind(missing);
        cache.Unbind(corrupt);
        cache.Unbind(none);
    });
}

TEST(TextureCacheTest, SharesImagesUntilTheirLastBindingGoes) {
    const Files files;
    ForEachDevice([&](Device& device) {
        TextureCache cache(device, files.Lookup(), 1);
        const TextureBinding a =
            cache.Bind({"small.png", true}, TextureWrap::Repeat, TextureWrap::Repeat);
        const TextureBinding b =
            cache.Bind({"small.png", true}, TextureWrap::Clamp, TextureWrap::Clamp);
        const TextureBinding linear =
            cache.Bind({"small.png", false}, TextureWrap::Repeat, TextureWrap::Repeat);
        EXPECT_EQ(cache.Images(), 2u);
        cache.Finish();
        cache.Collect();
        cache.Unbind(a);
        EXPECT_EQ(cache.Images(), 2u);
        cache.Unbind(b);
        cache.Unbind(linear);
        EXPECT_EQ(cache.Images(), 0u);
        device.GetQueue().WaitIdle();
    });
}

TEST(TextureCacheTest, ReusesAndUploadsBindingSlots) {
    const Files files;
    ForEachDevice([&](Device& device) {
        TextureCache cache(device, files.Lookup(), 1);
        EXPECT_EQ(cache.Binding(0), 0u);
        EXPECT_EQ(cache.AllocateBindings(4), 1u);
        EXPECT_EQ(cache.AllocateBindings(2), 5u);
        cache.FreeBindings(1, 4);
        EXPECT_EQ(cache.AllocateBindings(4), 1u);
        EXPECT_EQ(cache.AllocateBindings(4), 7u);
        cache.SetBinding(5, 0x1234);
        EXPECT_EQ(cache.Binding(5), 0x1234u);
        const uint64_t table = cache.UploadBindings();
        EXPECT_NE(table, 0u);
        EXPECT_EQ(cache.UploadBindings(), table);
    });
}

}
}
