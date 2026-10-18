// SPDX-License-Identifier: BSD-3-Clause
/// @file file.h
/// @brief Files on disk: whole-file reads, lookups by name, and read-only memory mapping.
#pragma once

#include "ernie/export.h"

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>

namespace ernie {

/// @brief Reads a whole file as text.
/// @param path The file.
/// @return Its bytes.
/// @throws std::runtime_error If it cannot be read.
ERNIE_API std::string ReadTextFile(const std::filesystem::path& path);

/// @brief Finds a file by name in any case, as archives named in capitals often unpack in small
/// letters.
/// @param directory The directory to look in.
/// @param name      The file's name.
/// @return The file with that name, exactly if there is one, else in any case; `directory / name`
///         if there is neither.
ERNIE_API std::filesystem::path FindFileAnyCase(const std::filesystem::path& directory,
                                                const std::string& name);

/// @brief A file mapped read-only into memory.
/// Pages load on first touch, so files larger than memory can be read in parts. The file may be
/// deleted or replaced while mapped, on every platform; the view stays valid until destroyed.
class ERNIE_API MappedFile {
  public:
    /// @brief Maps a file.
    /// @param path The file to map.
    /// @throws std::runtime_error If it cannot be opened or mapped.
    explicit MappedFile(const std::filesystem::path& path);
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    /// @return The file's bytes; empty for an empty file.
    std::span<const std::byte> Bytes() const {
        return {static_cast<const std::byte*>(data_), size_};
    }

  private:
    void* data_ = nullptr;
    std::size_t size_ = 0;
#ifdef _WIN32
    void* file_ = nullptr;
    void* mapping_ = nullptr;
#endif
};

}  // namespace ernie
