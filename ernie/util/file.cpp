// SPDX-License-Identifier: BSD-3-Clause
/// @file file.cpp
/// @brief Whole-file reads and file lookups.
#include "ernie/util/file.h"

#include "ernie/util/text.h"

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <system_error>

namespace ernie {

namespace fs = std::filesystem;

std::string ReadTextFile(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot read " + path.string());
    }
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
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

}  // namespace ernie
