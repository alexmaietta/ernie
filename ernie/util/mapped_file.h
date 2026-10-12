// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <filesystem>
#include <span>

namespace ernie {

/// A file mapped read-only into memory: its bytes are read from disk
/// as they are first touched, so a file far larger than memory can be
/// read in parts, the operating system keeping what is used.
class MappedFile {
  public:
    /// @throws std::runtime_error if the file cannot be opened or
    /// mapped.
    explicit MappedFile(const std::filesystem::path& path);
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    /// The file's bytes; empty for an empty file.
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
