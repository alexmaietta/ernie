// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/texture/cache.h"

#include <algorithm>
#include <bit>
#include <exception>
#include <fstream>
#include <span>
#include <sstream>

#include "ernie/device/kernels/shared.h"
#include "ernie/device/queue.h"
#include "ernie/io/ies.h"
#include "ernie/io/image.h"
#include "ernie/io/kernels/encoding.h"
#include "ernie/texture/kernels.h"
#include "ernie/util/error.h"
#include "ernie/util/log.h"
#include "ernie/util/profiler.h"

namespace ernie {
namespace {

constexpr uint32_t kDescsPerChunk = 256;
constexpr uint32_t kCheckerSide = 512;

Image Checkerboard() {
    Image image(kCheckerSide, kCheckerSide);
    const uint32_t half = kCheckerSide / 2;
    for (uint32_t y = 0; y < kCheckerSide; ++y) {
        for (uint32_t x = 0; x < kCheckerSide; ++x) {
            const float value = (x / half + y / half + 1) % 2 == 0 ? 0.0f : 1.0f;
            image.At(x, y) = {value, value, value, 1};
        }
    }
    return image;
}

void Encode(const Image& level, TexelFormat format, TexelColorSpace space,
            std::vector<uint32_t>& words) {
    for (const Rgba& texel : level.pixels) {
        if (format == TexelFormat::Rgba16f) {
            words.push_back(FloatToHalf(texel.r) | FloatToHalf(texel.g) << 16);
            words.push_back(FloatToHalf(texel.b) | FloatToHalf(texel.a) << 16);
            continue;
        }
        const auto color = [&](float value) {
            return EncodeUnorm8(space == TexelColorSpace::Srgb ? EncodeSrgb(value) : value);
        };
        words.push_back(color(texel.r) | color(texel.g) << 8 | color(texel.b) << 16 |
                        EncodeUnorm8(texel.a) << 24);
    }
}

std::optional<Image> ReadPixels(const std::string& path, bool srgb, const FileBytes& bytes) {
    const ImageEncoding encoding = srgb ? ImageEncoding::Srgb : ImageEncoding::Linear;
    try {
        return path == kCheckerboardTexture ? Checkerboard()
               : bytes                      ? LoadImageBytes(*bytes, path, encoding)
                                            : LoadImageFile(path, encoding);
    } catch (const std::exception& error) {
        Log(LogLevel::Warning, "texture {} can't be read: {}", path, error.what());
        return std::nullopt;
    }
}

std::optional<Image> ReadProfile(const std::string& path, const FileBytes& bytes) {
    try {
        std::string text;
        if (bytes) {
            text.assign(reinterpret_cast<const char*>(bytes->data()), bytes->size());
        } else {
            std::ifstream file(path, std::ios::binary);
            if (!file) {
                throw Error("it can't be opened");
            }
            std::ostringstream contents;
            contents << file.rdbuf();
            text = contents.str();
        }
        return ReadIes(text);
    } catch (const std::exception& error) {
        Log(LogLevel::Warning, "IES profile {} can't be read: {}", path, error.what());
        return std::nullopt;
    }
}

uint32_t LevelCount(uint32_t width, uint32_t height) {
    uint32_t levels = 1;
    while (levels < kTextureMaxLevels &&
           (TextureLevelSide(width, levels - 1) > 1 || TextureLevelSide(height, levels - 1) > 1)) {
        ++levels;
    }
    return levels;
}

}

TextureCache::TextureCache(device::Device& device, FileLookup files, unsigned threads)
    : m_device(device), m_files(std::move(files)), m_pool(threads) {}

TextureCache::~TextureCache() = default;

bool TextureCache::Pending() const {
    const std::lock_guard lock(m_mutex);
    return !m_done.empty() || m_inFlight > 0;
}

void TextureCache::Finish() {
    std::unique_lock lock(m_mutex);
    m_settled.wait(lock, [&] { return m_inFlight == 0; });
}

uint32_t TextureCache::Loading() const {
    uint32_t loading = 0;
    for (const auto& [key, entry] : m_images) {
        loading += entry.ready || entry.failed ? 0 : 1;
    }
    return loading;
}

std::optional<std::pair<uint32_t, uint32_t>> TextureCache::Size(const TextureKey& key) const {
    const auto found = m_images.find(key);
    if (found == m_images.end() || !found->second.ready) {
        return std::nullopt;
    }
    return std::pair{found->second.width, found->second.height};
}

uint32_t TextureCache::AllocateBindings(uint32_t count) {
    std::vector<uint32_t>& free = m_freeBindings[count];
    if (!free.empty()) {
        const uint32_t start = free.back();
        free.pop_back();
        return start;
    }
    const uint32_t start = uint32_t(m_bindings.size());
    m_bindings.resize(m_bindings.size() + count, 0);
    return start;
}

void TextureCache::FreeBindings(uint32_t start, uint32_t count) {
    m_freeBindings[count].push_back(start);
}

void TextureCache::SetBinding(uint32_t index, uint64_t handle) {
    m_bindings[index] = handle;
    m_bindingsChanged = true;
}

uint64_t TextureCache::UploadBindings() {
    if (m_bindingsChanged) {
        m_bindingsChanged = false;
        const uint64_t bytes = m_bindings.size() * sizeof(uint64_t);
        if (!m_bindingBuffer || m_bindingBuffer->Bytes() < bytes) {
            m_bindingBuffer = m_device.CreateBuffer(
                {.bytes = std::bit_ceil(std::max<uint64_t>(bytes, 64 * sizeof(uint64_t))),
                 .label = "texture bindings"});
        }
        m_device.GetQueue().Write(m_bindingBuffer->Slice(0, bytes), m_bindings.data());
    }
    return m_bindingBuffer->Address();
}

uint32_t TextureCache::AllocateDesc() {
    if (!m_freeDescs.empty()) {
        const uint32_t desc = m_freeDescs.back();
        m_freeDescs.pop_back();
        return desc;
    }
    if (m_descCount == m_descChunks.size() * kDescsPerChunk) {
        m_descChunks.push_back(m_device.CreateBuffer(
            {.bytes = kDescsPerChunk * sizeof(TextureDesc), .label = "texture descs"}));
    }
    return m_descCount++;
}

uint64_t TextureCache::DescAddress(uint32_t desc) const {
    return m_descChunks[desc / kDescsPerChunk]->Address() +
           uint64_t(desc % kDescsPerChunk) * sizeof(TextureDesc);
}

void TextureCache::Acquire(const TextureKey& key) {
    auto [found, added] = m_images.try_emplace(key);
    ++found->second.users;
    if (!added) {
        return;
    }
    const FileBytes bytes = key.first == kCheckerboardTexture ? nullptr : m_files(key.first);
    {
        const std::lock_guard lock(m_mutex);
        ++m_inFlight;
    }
    m_pool.Add(this, [this, key, bytes] {
        Decoded decoded;
        decoded.format = IsPngOrJpeg(key.first) || key.first == kCheckerboardTexture
                             ? TexelFormat::Rgba8
                             : TexelFormat::Rgba16f;
        decoded.colorSpace = key.second && decoded.format == TexelFormat::Rgba8
                                 ? TexelColorSpace::Srgb
                                 : TexelColorSpace::Linear;
        if (IsPngOrJpeg(key.first)) {
            try {
                Rgba8Image pixels =
                    bytes ? LoadRgba8Bytes(*bytes, key.first) : LoadRgba8File(key.first);
                decoded.width = pixels.width;
                decoded.height = pixels.height;
                decoded.words = std::move(pixels.texels);
            } catch (const std::exception& error) {
                Log(LogLevel::Warning, "texture {} can't be read: {}", key.first, error.what());
                decoded.failed = true;
            }
        } else if (std::optional<Image> pixels = IsIes(key.first)
                                                     ? ReadProfile(key.first, bytes)
                                                     : ReadPixels(key.first, key.second, bytes)) {
            decoded.width = pixels->width;
            decoded.height = pixels->height;
            Encode(*pixels, decoded.format, decoded.colorSpace, decoded.words);
        } else {
            decoded.failed = true;
        }
        decoded.levels = decoded.failed ? 0 : LevelCount(decoded.width, decoded.height);
        const std::lock_guard lock(m_mutex);
        m_done.emplace_back(key, std::move(decoded));
        --m_inFlight;
        m_settled.notify_all();
    });
}

void TextureCache::Upload(const TextureKey& key, Entry& entry, Decoded decoded, Mips& mips) {
    if (decoded.failed) {
        entry.failed = true;
        ++m_failed;
        return;
    }
    entry.format = decoded.format;
    entry.colorSpace = decoded.colorSpace;
    entry.width = decoded.width;
    entry.height = decoded.height;
    entry.levels = decoded.levels;
    uint32_t bytes = 0;
    for (uint32_t level = 0; level < entry.levels; ++level) {
        entry.offsets[level] = bytes;
        bytes += uint32_t(
            RoundUp(TextureLevelBytes(entry.format, entry.width, entry.height, level), 16));
    }
    const Ref<device::Buffer> texels = m_device.CreateBuffer({.bytes = bytes, .label = key.first});
    m_device.GetQueue().Write(texels->Slice(0, decoded.words.size() * sizeof(uint32_t)),
                              decoded.words.data());
    if (mips.levels.size() < entry.levels) {
        mips.levels.resize(entry.levels);
    }
    for (uint32_t level = 1; level < entry.levels; ++level) {
        mips.levels[level].push_back({.texels = {texels->Address()},
                                      .source = entry.offsets[level - 1] / 4,
                                      .target = entry.offsets[level] / 4,
                                      .sourceWidth = TextureLevelSide(entry.width, level - 1),
                                      .sourceHeight = TextureLevelSide(entry.height, level - 1),
                                      .width = TextureLevelSide(entry.width, level),
                                      .height = TextureLevelSide(entry.height, level),
                                      .format = entry.format,
                                      .colorSpace = entry.colorSpace});
    }
    if (m_device.GetSampling() == device::Sampling::Hardware) {
        entry.texture = m_device.CreateTexture({.width = entry.width,
                                                .height = entry.height,
                                                .levels = entry.levels,
                                                .format = entry.format,
                                                .colorSpace = entry.colorSpace,
                                                .label = key.first});
        mips.copies.emplace_back(texels, &entry);
    } else {
        entry.texels = texels;
    }
    entry.ready = true;
    ++m_loaded;
}

void TextureCache::Downsample(const Mips& mips) {
    if (mips.levels.size() < 2 && mips.copies.empty()) {
        return;
    }
    static const profiler::ZoneId zone = profiler::ZoneId::Named("Textures/Mips");
    device::Queue& queue = m_device.GetQueue();
    device::CommandEncoder encoder = queue.CreateEncoder();
    encoder.BeginZone(zone);
    for (uint32_t level = 1; level < mips.levels.size(); ++level) {
        std::vector<MipJob> jobs = mips.levels[level];
        uint32_t total = 0;
        for (MipJob& job : jobs) {
            job.first = total;
            total += job.width * job.height;
        }
        const Ref<device::Buffer> table =
            m_device.CreateBuffer({.label = "texture mips"}, std::span<const MipJob>(jobs));
        encoder.Barrier();
        encoder.Dispatch(Kernels().Entry("DownsampleLevels"),
                         MipParams{.jobs = {table->Address()},
                                   .jobCount = uint32_t(jobs.size()),
                                   .total = total},
                         {.x = device::GroupCount(total, kMipGroup)});
    }
    encoder.Barrier();
    for (const auto& [texels, entry] : mips.copies) {
        for (uint32_t level = 0; level < entry->levels; ++level) {
            encoder.Copy(texels->Slice(entry->offsets[level], entry->texture->LevelBytes(level)),
                         *entry->texture, level);
        }
    }
    encoder.EndZone();
    queue.Submit(encoder.Finish());
}

device::Program& TextureCache::Kernels() {
    if (!m_program) {
        m_program = m_device.CreateProgram({.modules = {"ernie.texture.kernels.mips"},
                                            .library = &kTextureKernels,
                                            .entries = {"DownsampleLevels"},
                                            .label = "textures"});
    }
    return *m_program;
}

std::set<TextureKey> TextureCache::Collect() {
    std::vector<std::pair<TextureKey, Decoded>> done;
    {
        const std::lock_guard lock(m_mutex);
        done.swap(m_done);
    }
    m_loaded = 0;
    m_failed = 0;
    std::set<TextureKey> finished;
    Mips mips;
    for (auto& [key, decoded] : done) {
        const auto entry = m_images.find(key);
        if (entry == m_images.end()) {
            continue;
        }
        Upload(key, entry->second, std::move(decoded), mips);
        finished.insert(key);
    }
    Downsample(mips);
    return finished;
}

TextureBinding TextureCache::Bind(TextureKey key, TextureWrap wrap) {
    TextureBinding binding{.key = std::move(key), .wrap = wrap};
    if (binding.key.first.empty()) {
        return binding;
    }
    Acquire(binding.key);
    Resolve(binding);
    return binding;
}

void TextureCache::Resolve(TextureBinding& binding) {
    if (binding.handle != 0 || binding.key.first.empty()) {
        return;
    }
    const Entry& entry = m_images.at(binding.key);
    if (!entry.ready) {
        return;
    }
    const TextureWrap wrap = binding.wrap;
    if (entry.texture) {
        binding.handle = entry.texture->KernelHandle(wrap, wrap);
    } else if (entry.texels) {
        TextureDesc desc{};
        desc.texels.address = entry.texels->Address();
        desc.width = entry.width;
        desc.height = entry.height;
        desc.numLevels = entry.levels;
        desc.format = entry.format;
        desc.colorSpace = entry.colorSpace;
        desc.wrapS = wrap;
        desc.wrapT = wrap;
        std::copy(entry.offsets.begin(), entry.offsets.end(), desc.levelOffsets);
        binding.desc = AllocateDesc();
        m_device.GetQueue().Write(
            m_descChunks[binding.desc / kDescsPerChunk]->Slice(
                uint64_t(binding.desc % kDescsPerChunk) * sizeof(TextureDesc), sizeof(TextureDesc)),
            &desc);
        binding.handle = DescAddress(binding.desc);
    }
}

void TextureCache::Unbind(const TextureBinding& binding) {
    if (binding.desc != ~0u) {
        m_freeDescs.push_back(binding.desc);
    }
    if (!binding.key.first.empty()) {
        const auto entry = m_images.find(binding.key);
        if (--entry->second.users == 0) {
            m_images.erase(entry);
        }
    }
}

}
