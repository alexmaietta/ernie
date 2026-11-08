// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/disk_cache.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iterator>
#include <system_error>
#include <thread>
#include <vector>

#include <fmt/format.h>

#include "ernie/util/environment.h"
#include "ernie/util/log.h"
#include "ernie/util/text.h"
#include "ernie/util/version.h"

namespace ernie {
namespace {

std::string Preamble(std::string_view key) {
    return fmt::format("ernie cache 1\n{}\n", key);
}

std::string UniqueSuffix() {
    static std::atomic<uint64_t> counter{0};
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return fmt::format(".{:x}.{:x}.{:x}.tmp",
                       std::hash<std::thread::id>{}(std::this_thread::get_id()), uint64_t(now),
                       counter++);
}

bool WriteTemporary(const std::filesystem::path& temporary, std::string_view key,
                    std::string_view bytes) {
    std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
    file << Preamble(key) << bytes;
    return bool(file.flush());
}

}

std::filesystem::path DiskCache::Path(std::string_view key) const {
    return m_directory / (Digest(key) + ".bin");
}

std::optional<std::string> DiskCache::Load(std::string_view key) const {
    std::ifstream file(Path(key), std::ios::binary);
    if (!file) {
        return std::nullopt;
    }

    std::string bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    const std::string preamble = Preamble(key);
    if (!bytes.starts_with(preamble)) {
        return std::nullopt;
    }
    std::error_code error;
    std::filesystem::last_write_time(Path(key), std::filesystem::file_time_type::clock::now(),
                                     error);
    return bytes.substr(preamble.size());
}

void DiskCache::Store(std::string_view key, std::string_view bytes) const {
    const std::filesystem::path path = Path(key);
    const std::filesystem::path temporary = path.string() + UniqueSuffix();
    std::error_code error;
    std::filesystem::create_directories(m_directory, error);
    if (!WriteTemporary(temporary, key, bytes)) {
        Log(LogLevel::Debug, "can't write {} to the disk cache", temporary.string());
        std::filesystem::remove(temporary, error);
        return;
    }

    std::filesystem::rename(temporary, path, error);
    if (error) {
        Log(LogLevel::Debug, "can't add {} to the disk cache: {}", path.string(), error.message());
        std::filesystem::remove(temporary, error);
    }
}

std::filesystem::path UserCacheDirectory() {
    if (const std::string chosen = Environment("ERNIE_CACHE_DIR"); !chosen.empty()) {
        return chosen;
    }
#if defined(_WIN32)
    if (const std::string local = Environment("LOCALAPPDATA"); !local.empty()) {
        return std::filesystem::path(local) / "ernie" / "cache";
    }
#elif defined(__APPLE__)
    if (const std::string home = Environment("HOME"); !home.empty()) {
        return std::filesystem::path(home) / "Library" / "Caches" / "ernie";
    }
#else
    if (const std::string cache = Environment("XDG_CACHE_HOME"); !cache.empty()) {
        return std::filesystem::path(cache) / "ernie";
    }
    if (const std::string home = Environment("HOME"); !home.empty()) {
        return std::filesystem::path(home) / ".cache" / "ernie";
    }
#endif
    std::error_code error;
    return std::filesystem::temp_directory_path(error) / "ernie-cache";
}

std::filesystem::path KernelCacheDirectory(const std::filesystem::path& root) {
    const Version version = GetBuildInfo().version;
    return root / "kernels" / fmt::format("{}.{}.{}", version.major, version.minor, version.patch);
}

void TrimKernelCache(const std::filesystem::path& root, uint64_t limit) {
    namespace fs = std::filesystem;
    std::error_code error;

    struct Entry {
        fs::path path;
        fs::file_time_type used;
        uint64_t bytes;
    };

    std::vector<Entry> entries;
    uint64_t total = 0;
    for (const fs::directory_entry& entry :
         fs::recursive_directory_iterator(root / "kernels", error)) {
        if (!entry.is_regular_file(error)) {
            continue;
        }
        const uint64_t bytes = entry.file_size(error);
        entries.push_back({entry.path(), entry.last_write_time(error), bytes});
        total += bytes;
    }
    if (total <= limit) {
        return;
    }
    std::ranges::sort(entries, {}, &Entry::used);
    for (const Entry& entry : entries) {
        if (total <= limit) {
            break;
        }
        if (fs::remove(entry.path, error)) {
            total -= entry.bytes;
        }
    }
}

}
