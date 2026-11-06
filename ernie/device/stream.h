// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>

namespace ernie {

class Stream {
  public:
    static constexpr Stream Compute(void* handle) {
        return Stream(Kind::Compute, handle);
    }

    static constexpr Stream Default() {
        return Stream(Kind::Compute, nullptr);
    }

    static constexpr Stream Host() {
        return Stream(Kind::Host, nullptr);
    }

    static constexpr Stream None() {
        return Stream(Kind::None, nullptr);
    }

    constexpr bool IsHost() const {
        return m_kind == Kind::Host;
    }

    constexpr bool IsNone() const {
        return m_kind == Kind::None;
    }

    constexpr bool IsCompute() const {
        return m_kind == Kind::Compute;
    }

    constexpr bool IsDefault() const {
        return m_kind == Kind::Compute && m_handle == nullptr;
    }

    constexpr void* Handle() const {
        return m_handle;
    }

  private:
    enum class Kind : uint8_t { None, Host, Compute };

    constexpr Stream(Kind kind, void* handle) : m_kind(kind), m_handle(handle) {}

    Kind m_kind;
    void* m_handle;
};

}
