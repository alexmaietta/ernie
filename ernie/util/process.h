// SPDX-License-Identifier: BSD-3-Clause
/// @file process.h
/// @brief What the operating system says about this process.
#pragma once

#include "ernie/export.h"

#include <cstdint>
#include <optional>

namespace ernie {

/// @return The bytes of host memory the process has resident; none where the system does not say.
ERNIE_API std::optional<std::uint64_t> ReadResidentBytes();

}  // namespace ernie
