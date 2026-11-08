// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/util/error.h"

#include <fmt/format.h>

namespace ernie {
namespace {

std::string ApiMessage(std::string_view call, std::string_view errorName, long code,
                       std::string_view description) {
    if (description.empty()) {
        return fmt::format("{} failed: {} ({})", call, errorName, code);
    }
    return fmt::format("{} failed: {} ({}): {}", call, errorName, code, description);
}

}

Error::Error(const std::string& message) : std::runtime_error(message) {}

Error::~Error() = default;

UnsupportedError::~UnsupportedError() = default;

DeviceLostError::~DeviceLostError() = default;

ApiError::ApiError(std::string call, std::string errorName, long code, std::string_view description)
    : Error(ApiMessage(call, errorName, code, description)), m_call(std::move(call)),
      m_errorName(std::move(errorName)), m_code(code) {}

ApiError::~ApiError() = default;

}
