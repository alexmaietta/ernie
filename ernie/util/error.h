// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#if __has_include(<version>)
#include <version>
#endif
#ifdef __cpp_lib_format
#include <format>
#endif

#include "ernie/export.h"

namespace ernie {

class ERNIE_API Error : public std::runtime_error {
  public:
    explicit Error(const std::string& message);

#ifdef __cpp_lib_format
    template <class... Args>
        requires(sizeof...(Args) > 0)
    explicit Error(std::format_string<Args...> format, Args&&... args)
        : Error(std::format(format, std::forward<Args>(args)...)) {}
#endif

    ~Error() override;
};

class ERNIE_API UnsupportedError : public Error {
  public:
    using Error::Error;
    ~UnsupportedError() override;
};

class ERNIE_API DeviceLostError : public Error {
  public:
    using Error::Error;
    ~DeviceLostError() override;
};

class ERNIE_API ApiError : public Error {
  public:
    ApiError(std::string call, std::string errorName, long code, std::string_view description = {});
    ~ApiError() override;

    const std::string& Call() const {
        return m_call;
    }

    const std::string& ErrorName() const {
        return m_errorName;
    }

    long Code() const {
        return m_code;
    }

  private:
    std::string m_call;
    std::string m_errorName;
    long m_code = 0;
};

}
