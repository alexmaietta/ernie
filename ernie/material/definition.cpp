// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/material/definition.h"

#include <algorithm>
#include <string_view>

namespace ernie {

namespace {

uint64_t Mix(uint64_t hash, std::string_view bytes) {
    for (const char byte : bytes) {
        hash = (hash ^ uint8_t(byte)) * 0x100000001b3ull;
    }
    return hash;
}

}

DefinitionSet::DefinitionSet(std::vector<DefinitionModule> definitions,
                             std::map<std::string, std::string, std::less<>> files)
    : m_definitions(std::move(definitions)), m_files(std::move(files)),
      m_hash(0xcbf29ce484222325ull) {
    for (const auto& [path, source] : m_files) {
        m_hash = Mix(m_hash, path);
        m_hash = Mix(m_hash, source);
    }
    std::ranges::sort(m_definitions, {}, &DefinitionModule::id);
    for (const DefinitionModule& definition : m_definitions) {
        m_hash = Mix(m_hash, std::to_string(definition.id));
        m_hash = Mix(m_hash, definition.type);
        m_hash = Mix(m_hash, definition.source);
        for (const auto& [path, source] : definition.files) {
            m_hash = Mix(m_hash, path);
            m_hash = Mix(m_hash, source);
        }
    }
}

bool DefinitionSet::Contains(uint32_t id) const {
    return id < kFirstGeneratedDefinition ||
           std::ranges::binary_search(m_definitions, id, {}, &DefinitionModule::id);
}

}
