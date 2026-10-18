// SPDX-License-Identifier: BSD-3-Clause
/// @file disk_cache.cpp
/// @brief Cache folder lookup and cache key hashing.
#include "ernie/core/disk_cache.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <system_error>

namespace ernie {

std::optional<std::filesystem::path> CacheDirectory(const char* variable, std::string_view folder) {
    namespace fs = std::filesystem;
    if (const char* own = std::getenv(variable)) {
        const std::string_view value(own);
        if (value.empty() || value == "0") {
            return std::nullopt;
        }
        return fs::path(value);
    }
    if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg != nullptr && *xdg != '\0') {
        return fs::path(xdg) / "ernie" / folder;
    }
#if defined(_WIN32)
    if (const char* local = std::getenv("LOCALAPPDATA"); local != nullptr && *local != '\0') {
        return fs::path(local) / "ernie" / folder;
    }
#else
    if (const char* home = std::getenv("HOME"); home != nullptr && *home != '\0') {
        return fs::path(home) / ".cache" / "ernie" / folder;
    }
#endif
    return std::nullopt;
}

void CacheKey::Add(std::string_view text) {
    const std::uint64_t size = text.size();
    Bytes(&size, sizeof(size));
    Bytes(text.data(), text.size());
}

std::string CacheKey::Hex() const {
    char text[33];
    std::snprintf(text, sizeof(text), "%016llx%016llx", static_cast<unsigned long long>(hashes_[0]),
                  static_cast<unsigned long long>(hashes_[1]));
    return text;
}

void CacheKey::Bytes(const void* data, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::uint64_t& hash : hashes_) {
        for (std::size_t i = 0; i < size; ++i) {
            hash = (hash ^ bytes[i]) * 0x100000001b3ull;
        }
    }
}

std::optional<std::vector<float>> ReadCachedFloats(const std::filesystem::path& path,
                                                   std::size_t count) {
    std::ifstream in(path, std::ios::binary);
    std::vector<float> values(count);
    const auto bytes = static_cast<std::streamsize>(count * sizeof(float));
    if (in.read(reinterpret_cast<char*>(values.data()), bytes) &&
        in.peek() == std::ifstream::traits_type::eof()) {
        return values;
    }
    return std::nullopt;
}

void WriteCachedFloats(const std::filesystem::path& path, std::span<const float> values) {
    namespace fs = std::filesystem;
    fs::path partial = path;
    partial += ".part" + std::to_string(std::random_device{}());
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    {
        std::ofstream out(partial, std::ios::binary);
        out.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size_bytes()));
        if (!out) {
            error = std::make_error_code(std::errc::io_error);
        }
    }
    if (!error) {
        fs::rename(partial, path, error);
    }
    if (error) {
        fs::remove(partial, error);
    }
}

}  // namespace ernie
