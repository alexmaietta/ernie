// SPDX-License-Identifier: BSD-3-Clause
/// @file file.cpp
/// @brief Whole-file reads, file lookups, and read-only mapping on POSIX and Windows.
#include "ernie/util/file.h"

#include "ernie/util/text.h"

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ernie {

namespace fs = std::filesystem;

std::string ReadTextFile(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot read " + path.string());
    }
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void WriteTextFile(const fs::path& path, std::string_view text) {
    if (path.has_parent_path()) {
        fs::create_directories(path.parent_path());
    }
    std::ofstream file(path, std::ios::binary);
    if (!file.write(text.data(), static_cast<std::streamsize>(text.size()))) {
        throw std::runtime_error("Cannot write " + path.string());
    }
}

fs::path FindFileAnyCase(const fs::path& directory, const std::string& name) {
    const fs::path exact = directory / name;
    if (fs::exists(exact)) {
        return exact;
    }
    const std::string wanted = ToLower(name);
    std::error_code error;
    for (const fs::directory_entry& entry : fs::directory_iterator(directory, error)) {
        if (ToLower(entry.path().filename().string()) == wanted) {
            return entry.path();
        }
    }
    return exact;
}

#ifdef _WIN32

MappedFile::MappedFile(const fs::path& path) {
    // Others may delete or replace the file while it is mapped, as on POSIX systems: caches evict
    // and rewrite entries that live textures still map, which keep their views until closed.
    file_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr,
                        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_ == INVALID_HANDLE_VALUE) {
        file_ = nullptr;
        throw std::runtime_error("Cannot open " + path.string());
    }
    LARGE_INTEGER size{};
    GetFileSizeEx(file_, &size);
    size_ = static_cast<std::size_t>(size.QuadPart);
    if (size_ == 0) {
        return;
    }
    mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping_ != nullptr) {
        data_ = MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
    }
    if (data_ == nullptr) {
        if (mapping_ != nullptr) {
            CloseHandle(mapping_);
        }
        CloseHandle(file_);
        throw std::runtime_error("Cannot map " + path.string());
    }
}

MappedFile::~MappedFile() {
    if (data_ != nullptr) {
        UnmapViewOfFile(data_);
    }
    if (mapping_ != nullptr) {
        CloseHandle(mapping_);
    }
    if (file_ != nullptr) {
        CloseHandle(file_);
    }
}

#else

MappedFile::MappedFile(const fs::path& path) {
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::runtime_error("Cannot open " + path.string());
    }
    struct stat info{};
    if (fstat(fd, &info) != 0) {
        close(fd);
        throw std::runtime_error("Cannot read " + path.string());
    }
    size_ = static_cast<std::size_t>(info.st_size);
    if (size_ > 0) {
        void* data = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Cannot map " + path.string());
        }
        data_ = data;
    }
    // The mapping keeps the file; the descriptor is not needed.
    close(fd);
}

MappedFile::~MappedFile() {
    if (data_ != nullptr) {
        munmap(data_, size_);
    }
}

#endif

}  // namespace ernie
