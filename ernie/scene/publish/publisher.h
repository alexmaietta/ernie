// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "ernie/device/operation.h"
#include "ernie/device/stream.h"
#include "ernie/scene/builders/builder.h"
#include "ernie/scene/kernels/schema/schema.h"
#include "ernie/scene/publish/change_set.h"
#include "ernie/scene/structure/structure.h"
#include "ernie/scene/tables/render_tables.h"
#include "ernie/scene/types.h"

namespace ernie {

struct PinnedTables {
    Ref<device::Buffer> header;
    RenderTables tables{};
    std::vector<Ref<const RefCounted>> held;

    static const PinnedTables& From(const Snapshot& snapshot) {
        return *static_cast<const PinnedTables*>(snapshot.pinned.get());
    }
};

class Publisher {
  public:
    explicit Publisher(Structure& structure) : m_structure(structure) {}

    Ordinal Published() const {
        return m_sealed.empty() ? 0 : m_sealed.back().ordinal;
    }

    bool Pending() const;
    void Finish();

    uint32_t Outstanding() const {
        return m_outstanding;
    }

    void Add(TableBuilder& system);
    device::Operation Publish(Ordinal ordinal, Stream producer = Stream::None());
    ChangeSet Changes(Ordinal since, std::span<const ColumnRef> columns = {});

    Snapshot Acquire(Stream consumer = Stream::None());
    void Release(const Snapshot& snapshot, Stream consumer = Stream::None());

  private:
    struct Sealed {
        Ordinal ordinal = 0;
        Ordinal stamp = 0;
    };

    struct Scheduled {
        TableBuilder* system = nullptr;
        Ordinal stamp = 0;
        uint64_t writes = 0;
        uint64_t epoch = 0;
        bool ran = false;
        uint64_t logged = 0;
        std::vector<Ref<device::Buffer>> changed;
    };

    Ordinal StampAt(Ordinal ordinal) const;
    Ordinal OrdinalAt(Ordinal stamp, Ordinal sealing) const;
    ChangeSet Build(Ordinal since, Ordinal until, Ordinal sealing,
                    std::span<const ColumnRef> columns,
                    std::vector<Ref<device::Buffer>>* reused = nullptr);
    void RunBuilders(Ordinal stamp, Ordinal ordinal);
    void Seal();

    Structure& m_structure;
    std::vector<Sealed> m_sealed;
    std::vector<Scheduled> m_systems;
    uint64_t m_writes = 0;
    uint64_t m_epoch = 0;
    uint32_t m_outstanding = 0;
    RenderTables m_tables{};
    TableWrites m_tableWrites;
    std::shared_ptr<const PinnedTables> m_pinned;
};

}
