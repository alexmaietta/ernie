// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace ernie {

/// Where one of Ernie's caches keeps its files between runs: the
/// environment variable `variable`'s folder (empty or "0" for none),
/// else `folder` in the user's cache folder
/// (`$XDG_CACHE_HOME/ernie`, `~/.cache/ernie`,
/// `%LOCALAPPDATA%\ernie`).
std::optional<std::filesystem::path>
CacheDirectory(const char* variable, std::string_view folder);

/// A cache's 128-bit key for everything an entry depends on: two
/// 64-bit FNV-1a hashes from different starts, each piece
/// length-prefixed.
class CacheKey {
  public:
    void Add(std::string_view text);

    /// The key as 32 hex digits, for a file name.
    std::string Hex() const;

  private:
    void Bytes(const void* data, std::size_t size);

    std::array<std::uint64_t, 2> hashes_{0xcbf29ce484222325ull,
                                         0x84222325cbf29ce4ull};
};

}  // namespace ernie
