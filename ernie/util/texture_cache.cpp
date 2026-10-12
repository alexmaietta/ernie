// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/texture_cache.h"

#include "ernie/util/disk_cache.h"
#include "ernie/util/profile.h"
#include "ernie/util/texture_bake.h"

#include <algorithm>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace ernie {

namespace {

namespace fs = std::filesystem;

/// Bumped when what the cache stores, or how it keys it, changes.
constexpr std::string_view kCacheFormat = "ernie-texture-1";

}  // namespace

std::optional<fs::path> TextureCache::DefaultDirectory() {
    return CacheDirectory("ERNIE_TEXTURE_CACHE", "textures");
}

TextureCache::TextureCache(std::optional<fs::path> directory,
                           std::uintmax_t capacity)
    : directory_(std::move(directory)), capacity_(capacity) {}

fs::path TextureCache::PathOf(std::string_view key) const {
    CacheKey hash;
    hash.Add(kCacheFormat);
    hash.Add(key);
    return *directory_ / (hash.Hex() + ".dds");
}

std::optional<TextureData>
TextureCache::Read(std::string_view key) const {
    if (!directory_) {
        return std::nullopt;
    }
    ERNIE_ZONE("Texture cache read");
    const fs::path path = PathOf(key);
    std::error_code error;
    if (!fs::exists(path, error)) {
        return std::nullopt;
    }
    try {
        TextureData texture = LoadTextureFile(path);
        // Its modification time is its last use, for `Prune`.
        fs::last_write_time(path, fs::file_time_type::clock::now(),
                            error);
        return texture;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

void TextureCache::Write(std::string_view key,
                         const TextureData& texture) const {
    if (!directory_) {
        return;
    }
    ERNIE_ZONE("Texture cache write");
    const fs::path path = PathOf(key);
    // Written beside it, then renamed over it, so that no reader (in
    // this process or another) sees a part of it.
    fs::path partial = path;
    partial += ".part" + std::to_string(std::random_device{}());
    std::error_code error;
    fs::create_directories(*directory_, error);
    try {
        WriteDdsFile(partial, texture);
        fs::rename(partial, path, error);
    } catch (const std::exception&) {
        error = std::make_error_code(std::errc::io_error);
    }
    if (error) {
        fs::remove(partial, error);
        return;
    }
    Prune();
}

void TextureCache::Prune() const {
    ERNIE_ZONE("Texture cache prune");
    struct Entry {
        fs::path path;
        fs::file_time_type used;
        std::uintmax_t bytes;
    };
    std::vector<Entry> entries;
    std::uintmax_t total = 0;
    std::error_code error;
    for (const fs::directory_entry& file :
         fs::directory_iterator(*directory_, error)) {
        if (file.path().extension() != ".dds") {
            continue;
        }
        Entry entry{file.path(), file.last_write_time(error),
                    file.file_size(error)};
        if (error) {
            continue;
        }
        total += entry.bytes;
        entries.push_back(std::move(entry));
    }
    if (total <= capacity_) {
        return;
    }
    std::sort(entries.begin(), entries.end(),
              [](const Entry& a, const Entry& b) {
                  return a.used < b.used;
              });
    for (const Entry& entry : entries) {
        if (total <= capacity_) {
            break;
        }
        if (fs::remove(entry.path, error)) {
            total -= entry.bytes;
        }
    }
}

}  // namespace ernie
