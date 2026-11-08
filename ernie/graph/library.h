// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ernie/graph/shape.h"

namespace ernie::graph {

template <class Definition> class DefinitionSet {
  public:
    struct Entry {
        uint32_t id = 0;
        std::shared_ptr<const Definition> definition;
    };

    DefinitionSet(std::vector<Entry> entries, uint64_t version)
        : m_entries(std::move(entries)), m_version(version) {}

    std::span<const Entry> Entries() const {
        return m_entries;
    }

    const Definition* Find(uint32_t id) const {
        const auto found = std::ranges::lower_bound(m_entries, id, {}, &Entry::id);
        return found != m_entries.end() && found->id == id ? found->definition.get() : nullptr;
    }

    uint64_t Version() const {
        return m_version;
    }

  private:
    std::vector<Entry> m_entries;
    uint64_t m_version = 0;
};

template <class Definition> class DefinitionLibrary {
  public:
    using Set = DefinitionSet<Definition>;

    explicit DefinitionLibrary(uint32_t firstId = 0) : m_next(firstId) {}

    template <class Make> uint32_t Intern(const ShapeKey& key, Make&& make) {
        const std::lock_guard lock(m_mutex);
        const auto found = m_ids.find(key);
        if (found != m_ids.end()) {
            return found->second;
        }
        const uint32_t id = m_next++;
        m_entries.push_back({id, std::make_shared<const Definition>(make(id))});
        m_ids.emplace(key, id);
        m_snapshot.reset();
        return id;
    }

    std::shared_ptr<const Set> Snapshot() const {
        const std::lock_guard lock(m_mutex);
        if (!m_snapshot) {
            m_snapshot = std::make_shared<const Set>(m_entries, ++m_version);
        }
        return m_snapshot;
    }

  private:
    mutable std::mutex m_mutex;
    std::unordered_map<ShapeKey, uint32_t> m_ids;
    std::vector<typename Set::Entry> m_entries;
    mutable std::shared_ptr<const Set> m_snapshot;
    mutable uint64_t m_version = 0;
    uint32_t m_next = 0;
};

}
