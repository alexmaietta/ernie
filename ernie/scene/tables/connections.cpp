// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/scene/tables/connections.h"

#include <algorithm>
#include <cstddef>

#include "ernie/scene/schema/columns.h"
#include "ernie/stage/publish/change_set.h"
#include "ernie/stage/structure/structure.h"
#include "ernie/util/profiler.h"
#include "ernie/util/span.h"

namespace ernie {

void ConnectionCache::Update(const ChangeSet& changes) {
    ERNIE_ZONE("Connections/Update");
    const Prims& prims = m_structure.GetPrims();
    const auto accepted = [&](uint32_t prim) {
        return std::ranges::any_of(
            m_schemas, [&](SchemaId schema) { return IsA(prims.Record(prim).schema, schema); });
    };
    if (!changes.resync && changes.added.empty() && changes.removed.empty() &&
        changes.apis.empty() && m_connections.size() == prims.Size()) {
        std::vector<uint32_t> owners;
        bool known = true;
        for (const ColumnRef column :
             {kConnectableConnectionsSources, kConnectableConnectionsInputs,
              kConnectableConnectionsOutputs}) {
            for (const uint32_t row : changes.ArrayRows(column)) {
                const uint32_t prim = row < m_rowPrims.size() ? m_rowPrims[row] : kNoPrim;
                known = known && prim != kNoPrim;
                if (prim != kNoPrim && accepted(prim) &&
                    std::ranges::find(owners, prim) == owners.end()) {
                    owners.push_back(prim);
                }
            }
        }
        if (known) {
            Read(owners);
            return;
        }
    }
    std::vector<uint32_t> owners;
    m_rowPrims.clear();
    for (uint32_t prim = 0; prim < prims.Size(); ++prim) {
        if (!prims.Alive(prim) || !IsA(prims.Record(prim).schema, SchemaId::Connectable)) {
            continue;
        }
        const uint32_t row = prims.Row(prim, SchemaId::Connectable);
        if (row >= m_rowPrims.size()) {
            m_rowPrims.resize(row + 1, kNoPrim);
        }
        m_rowPrims[row] = prim;
        if (accepted(prim)) {
            owners.push_back(prim);
        }
    }
    m_connections.assign(prims.Size(), {});
    Read(owners);
}

void ConnectionCache::Read(std::span<const uint32_t> owners) {
    const Prims& prims = m_structure.GetPrims();
    ColumnStore& store = m_structure.GetColumnStore();
    std::vector<ArrayRef> sources;
    std::vector<ArrayRef> tokens;
    for (const uint32_t prim : owners) {
        const uint32_t row = prims.Row(prim, SchemaId::Connectable);
        sources.push_back(store.Array(kConnectableConnectionsSources, row));
        tokens.push_back(store.Array(kConnectableConnectionsInputs, row));
    }
    for (const uint32_t prim : owners) {
        tokens.push_back(
            store.Array(kConnectableConnectionsOutputs, prims.Row(prim, SchemaId::Connectable)));
    }
    const std::vector<std::vector<std::byte>> sourceBytes =
        store.Arrays(ColumnType::PrimArray).Read(sources);
    const std::vector<std::vector<std::byte>> tokenBytes =
        store.Arrays(ColumnType::TokenArray).Read(tokens);
    const Tokens& text = m_structure.GetTokens();
    for (size_t k = 0; k < owners.size(); ++k) {
        const std::vector<uint32_t> from = Items<uint32_t>(sourceBytes[k]);
        const std::vector<uint32_t> to = Items<uint32_t>(tokenBytes[k]);
        const std::vector<uint32_t> out = Items<uint32_t>(tokenBytes[owners.size() + k]);
        std::vector<Connection>& connections = m_connections[owners[k]];
        connections.clear();
        const size_t count = std::min({from.size(), to.size(), out.size()});
        for (size_t i = 0; i < count; ++i) {
            const uint32_t source = from[i] != kNoPrim && prims.Alive(from[i]) ? from[i] : kNoPrim;
            connections.push_back({text.Text(Token{to[i]}), source, text.Text(Token{out[i]})});
        }
    }
}

}
