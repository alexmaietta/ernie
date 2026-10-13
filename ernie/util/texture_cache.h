// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "ernie/util/texture_file.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string_view>

namespace ernie {

/// Textures prepared at load kept between runs as DDS files (as
/// `RasterDelegate` keeps image files' textures encoded to BC7 on the
/// GPU), so that a texture decodes, mips and encodes once until its
/// file, or how it is read, changes. Entries are keyed by a hash of
/// everything they depend on; past `capacity` bytes the least
/// recently used go. Any thread may read; failures to read or write
/// are a miss, never an error.
class TextureCache {
  public:
    /// The default folder: `ERNIE_TEXTURE_CACHE` (empty or "0" for
    /// none), else the user's cache folder
    /// (`$XDG_CACHE_HOME/ernie/textures`, `~/.cache/ernie/textures`).
    static std::optional<std::filesystem::path> DefaultDirectory();

    /// A cache in `directory`, none without one.
    explicit TextureCache(std::optional<std::filesystem::path>
                              directory = DefaultDirectory(),
                          std::uintmax_t capacity = std::uintmax_t{4}
                                                    << 30);

    bool Enabled() const {
        return directory_.has_value();
    }

    /// The texture cached under `key` (mapped, as `LoadTextureFile`),
    /// marked as just used; none if absent or unreadable.
    std::optional<TextureData> Read(std::string_view key) const;

    /// Caches `texture` under `key`, written whole or not at all,
    /// then removes the least recently used past the capacity.
    void Write(std::string_view key,
               const TextureData& texture) const;

    /// Forgets the texture cached under `key`, if any.
    void Remove(std::string_view key) const;

  private:
    std::filesystem::path PathOf(std::string_view key) const;
    void Prune() const;

    std::optional<std::filesystem::path> directory_;
    std::uintmax_t capacity_;
};

}  // namespace ernie
