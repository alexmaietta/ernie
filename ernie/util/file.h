// SPDX-License-Identifier: BSD-3-Clause
/// @file file.h
/// @brief Whole-file reads and file lookups.
#pragma once

#include <filesystem>
#include <string>

namespace ernie {

/// @brief Reads a whole file as text.
/// @param path The file.
/// @return Its bytes.
/// @throws std::runtime_error If it cannot be read.
std::string ReadTextFile(const std::filesystem::path& path);

/// @brief Finds a file by name in any case, as archives named in capitals often unpack in small
/// letters.
/// @param directory The directory to look in.
/// @param name      The file's name.
/// @return The file with that name, exactly if there is one, else in any case; `directory / name`
///         if there is neither.
std::filesystem::path FindFileAnyCase(const std::filesystem::path& directory,
                                      const std::string& name);

}  // namespace ernie
