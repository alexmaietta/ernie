// SPDX-License-Identifier: BSD-3-Clause
/// @file disk_cache.h
/// @brief Cache folders and keys for caches kept between runs.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace ernie {

/// @brief Finds where one of Ernie's caches keeps its files.
/// @param variable An environment variable naming the folder; empty
///                 or "0" disables the cache.
/// @param folder   The subfolder of the user's cache folder used
///                 when `variable` is unset (`$XDG_CACHE_HOME/ernie`,
///                 `~/.cache/ernie`, `%LOCALAPPDATA%\ernie`).
/// @return The folder, or nothing when the cache is disabled.
std::optional<std::filesystem::path> CacheDirectory(const char* variable, std::string_view folder);

/// @brief A 128-bit key over everything a cache entry depends on.
/// Two 64-bit FNV-1a hashes from different seeds; each piece is
/// length-prefixed.
class CacheKey {
  public:
    /// @brief Mixes a piece of text into the key.
    /// @param text The piece to add.
    void Add(std::string_view text);

    /// @return The key as 32 hex digits, for a file name.
    std::string Hex() const;

  private:
    void Bytes(const void* data, std::size_t size);

    std::array<std::uint64_t, 2> hashes_{0xcbf29ce484222325ull, 0x84222325cbf29ce4ull};
};

}  // namespace ernie
