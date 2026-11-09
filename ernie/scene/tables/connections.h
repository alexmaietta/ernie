// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <initializer_list>
#include <span>
#include <string_view>
#include <vector>

#include "ernie/stage/schema/ids.h"
#include "ernie/stage/schema/schema.h"

namespace ernie {

class Structure;
struct ChangeSet;

class ConnectionCache {
  public:
    struct Connection {
        std::string_view input;
        uint32_t source = kNoPrim;
        std::string_view output;
    };

    ConnectionCache(Structure& structure, std::initializer_list<SchemaId> schemas)
        : m_structure(structure), m_schemas(schemas) {}

    void Update(const ChangeSet& changes);

    std::span<const Connection> Get(uint32_t prim) const {
        return prim < m_connections.size() ? std::span<const Connection>(m_connections[prim])
                                           : std::span<const Connection>();
    }

  private:
    void Read(std::span<const uint32_t> owners);

    Structure& m_structure;
    std::vector<SchemaId> m_schemas;
    std::vector<std::vector<Connection>> m_connections;
    std::vector<uint32_t> m_rowPrims;
};

}
