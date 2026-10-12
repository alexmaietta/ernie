// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/disk_cache.h"

#include <cstdio>
#include <cstdlib>

namespace ernie {

std::optional<std::filesystem::path>
CacheDirectory(const char* variable, std::string_view folder) {
    namespace fs = std::filesystem;
    if (const char* own = std::getenv(variable)) {
        const std::string_view value(own);
        if (value.empty() || value == "0") {
            return std::nullopt;
        }
        return fs::path(value);
    }
    if (const char* xdg = std::getenv("XDG_CACHE_HOME");
        xdg != nullptr && *xdg != '\0') {
        return fs::path(xdg) / "ernie" / folder;
    }
#if defined(_WIN32)
    if (const char* local = std::getenv("LOCALAPPDATA");
        local != nullptr && *local != '\0') {
        return fs::path(local) / "ernie" / folder;
    }
#else
    if (const char* home = std::getenv("HOME");
        home != nullptr && *home != '\0') {
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
    std::snprintf(text, sizeof(text), "%016llx%016llx",
                  static_cast<unsigned long long>(hashes_[0]),
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

}  // namespace ernie
