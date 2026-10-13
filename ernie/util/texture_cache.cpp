// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/texture_cache.h"

#include "ernie/util/disk_cache.h"
#include "ernie/util/profile.h"
#include "ernie/util/texture_bake.h"

#include <algorithm>
#include <chrono>
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
    const std::optional<fs::path> partial = PartialPath(key);
    if (!partial) {
        return;
    }
    ERNIE_ZONE("Texture cache write");
    try {
        WriteDdsFile(*partial, texture);
    } catch (const std::exception&) {
        std::error_code error;
        fs::remove(*partial, error);
        return;
    }
    Publish(*partial, key);
}

std::optional<fs::path>
TextureCache::PartialPath(std::string_view key) const {
    if (!directory_) {
        return std::nullopt;
    }
    std::error_code error;
    fs::create_directories(*directory_, error);
    // Written beside its entry, then renamed over it, so that no
    // reader (in this process or another) sees a part of it.
    fs::path partial = PathOf(key);
    partial += ".part" + std::to_string(std::random_device{}());
    return partial;
}

bool TextureCache::Publish(const fs::path& partial,
                           std::string_view key) const {
    std::error_code error;
    if (!directory_) {
        fs::remove(partial, error);
        return false;
    }
    const fs::path path = PathOf(key);
    fs::rename(partial, path, error);
    if (error) {
        fs::remove(partial, error);
        return false;
    }
    Prune(path);
    return true;
}

void TextureCache::Remove(std::string_view key) const {
    if (directory_) {
        std::error_code error;
        fs::remove(PathOf(key), error);
    }
}

void TextureCache::Prune(const fs::path& kept) const {
    ERNIE_ZONE("Texture cache prune");
    struct Entry {
        fs::path path;
        fs::file_time_type used;
        std::uintmax_t bytes;
    };
    std::vector<Entry> entries;
    std::uintmax_t total = 0;
    std::error_code error;
    const fs::file_time_type now = fs::file_time_type::clock::now();
    for (const fs::directory_entry& file :
         fs::directory_iterator(*directory_, error)) {
        const fs::file_time_type used = file.last_write_time(error);
        if (error) {
            continue;
        }
        // Parts a day old are of writes that never finished.
        if (file.path().extension().string().starts_with(".part")) {
            if (now - used > std::chrono::hours(24)) {
                fs::remove(file.path(), error);
            }
            continue;
        }
        if (file.path().extension() != ".dds") {
            continue;
        }
        Entry entry{file.path(), used, file.file_size(error)};
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
        if (entry.path == kept) {
            continue;
        }
        if (fs::remove(entry.path, error)) {
            total -= entry.bytes;
        }
    }
}

}  // namespace ernie
