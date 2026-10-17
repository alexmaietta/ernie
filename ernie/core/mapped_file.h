// SPDX-License-Identifier: BSD-3-Clause
/// @file mapped_file.h
/// @brief Read-only memory-mapped files.
#pragma once

#include <cstddef>
#include <filesystem>
#include <span>

namespace ernie {

/// @brief A file mapped read-only into memory.
/// Pages load on first touch, so files larger than memory can be read
/// in parts.
class MappedFile {
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
