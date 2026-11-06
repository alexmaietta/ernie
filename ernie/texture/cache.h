// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ernie/device/buffer.h"
#include "ernie/device/device.h"
#include "ernie/device/kernels/texture.h"
#include "ernie/device/program.h"
#include "ernie/device/ref.h"
#include "ernie/device/texture.h"
#include "ernie/texture/kernels/mips.h"
#include "ernie/util/thread_pool.h"

namespace ernie {

using FileBytes = std::shared_ptr<const std::vector<std::byte>>;
using FileLookup = std::function<FileBytes(const std::string& path)>;
using TextureKey = std::pair<std::string, bool>;

inline constexpr std::string_view kCheckerboardTexture = "<checkerboard>";

struct TextureBinding {
    TextureKey key;
    TextureWrap wrap = TextureWrap::Repeat;
    uint32_t desc = ~0u;
    uint64_t handle = 0;
};

class TextureCache {
  public:
    TextureCache(device::Device& device, FileLookup files, unsigned threads);
    ~TextureCache();
    TextureCache(const TextureCache&) = delete;
    TextureCache& operator=(const TextureCache&) = delete;

    TextureBinding Bind(TextureKey key, TextureWrap wrap);
    void Resolve(TextureBinding& binding);
    void Unbind(const TextureBinding& binding);

    std::set<TextureKey> Collect();
    bool Pending() const;
    void Finish();

    std::optional<std::pair<uint32_t, uint32_t>> Size(const TextureKey& key) const;

    uint32_t Images() const {
        return uint32_t(m_images.size());
    }

    uint32_t Loading() const;

    uint32_t Loaded() const {
        return m_loaded;
    }

    uint32_t Failed() const {
        return m_failed;
    }

    uint32_t AllocateBindings(uint32_t count);
    void FreeBindings(uint32_t start, uint32_t count);
    void SetBinding(uint32_t index, uint64_t handle);

    uint64_t Binding(uint32_t index) const {
        return m_bindings[index];
    }

    uint64_t UploadBindings();

  private:
    struct Decoded {
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t levels = 0;
        TexelFormat format = TexelFormat::Rgba8;
        TexelColorSpace colorSpace = TexelColorSpace::Linear;
        std::vector<uint32_t> words;
        bool failed = false;
    };

    struct Entry {
        Ref<device::Buffer> texels;
        Ref<device::Texture> texture;
        TexelFormat format = TexelFormat::Rgba8;
        TexelColorSpace colorSpace = TexelColorSpace::Linear;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t levels = 0;
        std::array<uint32_t, kTextureMaxLevels> offsets{};
        uint32_t users = 0;
        bool ready = false;
        bool failed = false;
    };

    struct Mips {
        std::vector<std::vector<MipJob>> levels;
        std::vector<std::pair<Ref<device::Buffer>, const Entry*>> copies;
    };

    void Acquire(const TextureKey& key);
    void Upload(const TextureKey& key, Entry& entry, Decoded decoded, Mips& mips);
    void Downsample(const Mips& mips);
    device::Program& Kernels();
    uint32_t AllocateDesc();
    uint64_t DescAddress(uint32_t desc) const;

    device::Device& m_device;
    FileLookup m_files;
    std::map<TextureKey, Entry> m_images;
    std::vector<Ref<device::Buffer>> m_descChunks;
    std::vector<uint32_t> m_freeDescs;
    uint32_t m_descCount = 0;
    std::vector<uint64_t> m_bindings = {0};
    std::map<uint32_t, std::vector<uint32_t>> m_freeBindings;
    Ref<device::Buffer> m_bindingBuffer;
    bool m_bindingsChanged = true;
    uint32_t m_loaded = 0;
    uint32_t m_failed = 0;
    mutable std::mutex m_mutex;
    std::condition_variable m_settled;
    std::vector<std::pair<TextureKey, Decoded>> m_done;
    uint32_t m_inFlight = 0;
    Ref<device::Program> m_program;
    ThreadPool m_pool;
};

}
