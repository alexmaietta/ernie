// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <unordered_map>

namespace ernie {

struct Token {
    uint32_t id = 0;

    bool operator==(const Token& other) const {
        return id == other.id;
    }

    bool operator!=(const Token& other) const {
        return !(*this == other);
    }
};

class Tokens {
  public:
    Tokens();
    Tokens(const Tokens&) = delete;
    Tokens& operator=(const Tokens&) = delete;
    Tokens(Tokens&&) = default;
    Tokens& operator=(Tokens&&) = default;

    Token Intern(std::string_view text);
    Token Find(std::string_view text) const;
    std::string_view Text(Token token) const;

    uint32_t Size() const {
        return uint32_t(m_texts.size());
    }

  private:
    std::deque<std::string> m_texts;
    std::unordered_map<std::string_view, uint32_t> m_ids;
};

}
