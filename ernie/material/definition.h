// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ernie {

inline constexpr uint32_t kFirstGeneratedDefinition = 4;

struct DefinitionModule {
    uint32_t id = 0;
    std::string type;
    std::string module;
    std::string source;
    std::vector<std::pair<std::string, std::string>> files;
    std::vector<std::pair<std::string, std::string>> origins;
};

class DefinitionSet {
  public:
    explicit DefinitionSet(std::vector<DefinitionModule> definitions = {});

    std::span<const DefinitionModule> Definitions() const {
        return m_definitions;
    }

    uint64_t Hash() const {
        return m_hash;
    }

    bool Contains(uint32_t id) const;

  private:
    std::vector<DefinitionModule> m_definitions;
    uint64_t m_hash = 0;
};

using DefinitionSetRef = std::shared_ptr<const DefinitionSet>;

}
