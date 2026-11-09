// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/stage/structure/tokens.h"

#include "ernie/util/error.h"

namespace ernie {

Tokens::Tokens() {
    Intern("");
}

Token Tokens::Intern(std::string_view text) {
    if (const auto found = m_ids.find(text); found != m_ids.end()) {
        return {found->second};
    }
    const uint32_t id = uint32_t(m_texts.size());
    const std::string& stored = m_texts.emplace_back(text);
    m_ids.emplace(stored, id);
    return {id};
}

Token Tokens::Find(std::string_view text) const {
    const auto found = m_ids.find(text);
    return found == m_ids.end() ? Token{} : Token{found->second};
}

std::string_view Tokens::Text(Token token) const {
    if (token.id >= m_texts.size()) {
        throw Error("No token {}", token.id);
    }
    return m_texts[token.id];
}

}
