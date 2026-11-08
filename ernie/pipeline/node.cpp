// SPDX-License-Identifier: BSD-3-Clause
#include "ernie/pipeline/node.h"

#include "ernie/util/error.h"

namespace ernie::pipeline {

uint32_t NodeRegistry::Add(NodeDef def) {
    const auto found = m_ids.find(def.id);
    if (found != m_ids.end()) {
        m_defs[found->second] = std::move(def);
        return found->second;
    }
    if (def.kind == NodeKind::Source && !def.inputs.empty()) {
        throw Error("Pipeline node {} is a source, so it takes no signal inputs", def.id);
    }
    const auto index = uint32_t(m_defs.size());
    m_ids.emplace(def.id, index);
    m_defs.push_back(std::move(def));
    return index;
}

uint32_t NodeRegistry::Find(std::string_view id) const {
    const auto found = m_ids.find(std::string(id));
    return found == m_ids.end() ? kNoDef : found->second;
}

}
