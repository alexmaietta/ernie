// SPDX-License-Identifier: BSD-3-Clause
/// @file mapped_file.cpp
/// @brief Read-only file mapping on POSIX and Windows.
#include "ernie/util/mapped_file.h"

#include <stdexcept>
#include <string>

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

#ifdef _WIN32

MappedFile::MappedFile(const std::filesystem::path& path) {
    file_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                        FILE_ATTRIBUTE_NORMAL, nullptr);
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

MappedFile::MappedFile(const std::filesystem::path& path) {
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
