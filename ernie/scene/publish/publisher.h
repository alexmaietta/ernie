// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "ernie/device/operation.h"
#include "ernie/device/stream.h"
#include "ernie/scene/publish/change_set.h"
#include "ernie/scene/publish/header.h"
#include "ernie/scene/publish/table.h"
#include "ernie/scene/schema/schema.h"
#include "ernie/scene/structure/structure.h"
#include "ernie/scene/types.h"
#include "ernie/util/error.h"

namespace ernie {

struct PinnedTables {
    Ref<device::Buffer> buffer;
    std::vector<std::byte> header;
    std::vector<Ref<const RefCounted>> held;
    uint64_t content = 0;
    std::vector<std::pair<std::string_view, uint64_t>> versions;

    static const PinnedTables& From(const Snapshot& snapshot) {
        return *static_cast<const PinnedTables*>(snapshot.pinned.get());
    }

    template <class T> const T& Header() const {
        if (header.size() != sizeof(T)) {
            throw Error("A snapshot header holds {} bytes, not {}", header.size(), sizeof(T));
        }
        return *reinterpret_cast<const T*>(header.data());
    }
};

class Publisher {
  public:
    explicit Publisher(Structure& structure, SnapshotHeader* header = nullptr)
        : m_structure(structure), m_header(header) {}

    Ordinal Published() const {
        return m_sealed.empty() ? 0 : m_sealed.back().ordinal;
    }

    bool Pending() const;
    void Finish();

    uint32_t Outstanding() const {
        return m_outstanding;
    }

    void Add(Table& table);
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
        Table* table = nullptr;
        Ordinal stamp = 0;
        uint64_t writes = 0;
        uint64_t epoch = 0;
        bool ran = false;
        uint64_t logged = 0;
        std::vector<Ref<device::Buffer>> changed;
    };

    Ordinal SealedStamp(Ordinal ordinal) const;
    Ordinal SealedOrdinal(Ordinal stamp, Ordinal sealing) const;
    ChangeSet Build(Ordinal since, Ordinal until, Ordinal sealing,
                    std::span<const ColumnRef> columns, std::span<const ColumnRef> rows,
                    std::vector<Ref<device::Buffer>>* reused = nullptr);
    void RunTables(Ordinal stamp, Ordinal ordinal);
    void Seal();

    Structure& m_structure;
    SnapshotHeader* m_header = nullptr;
    std::vector<Sealed> m_sealed;
    std::vector<Scheduled> m_scheduled;
    uint64_t m_writes = 0;
    uint64_t m_epoch = 0;
    uint32_t m_outstanding = 0;
    TableVersions m_versions;
    uint64_t m_tableRuns = 0;
    std::shared_ptr<const PinnedTables> m_pinned;
};

}
