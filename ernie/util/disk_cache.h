// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace ernie {

class DiskCache {
  public:
    explicit DiskCache(std::filesystem::path directory) : m_directory(std::move(directory)) {}

    std::optional<std::string> Load(std::string_view key) const;

    void Store(std::string_view key, std::string_view bytes) const;

  private:
    std::filesystem::path Path(std::string_view key) const;

    std::filesystem::path m_directory;
};

std::filesystem::path UserCacheDirectory();

std::filesystem::path KernelCacheDirectory(const std::filesystem::path& root);

void TrimKernelCache(const std::filesystem::path& root, uint64_t limit);

}
