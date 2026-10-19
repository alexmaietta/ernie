// SPDX-License-Identifier: BSD-3-Clause
/// @file process.h
/// @brief What the operating system says about this process.
#pragma once

#include "ernie/export.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ernie {

/// @return The bytes of host memory the process has resident; none where the system does not say.
ERNIE_API std::optional<std::uint64_t> ReadResidentBytes();

/// @return A program's executable on the PATH, as `name` or, on Windows, `name.exe`; empty when
/// there is none.
ERNIE_API std::filesystem::path FindProgram(std::string_view name);

/// @brief Runs a program without a shell and waits for it, its output discarded.
/// @param command The program, then its arguments.
/// @return Its exit status, or -1 when it could not start.
ERNIE_API int RunProcess(const std::vector<std::string>& command);

}  // namespace ernie
