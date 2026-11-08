// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <unordered_map>
#include <vector>

#include "ernie/scene/columns/column_store.h"
#include "ernie/scene/kernels/ids.h"
#include "ernie/scene/kernels/schema/schema.h"

namespace ernie {

struct PrimRecord {
    SchemaId schema = SchemaId::None;
    uint32_t parent = kNoPrim;
    uint32_t world = kGlobalWorld;
    uint32_t flags = 0;
    uint32_t slots = kNoRow;
    uint32_t generation = 0;
    uint64_t apis = 0;
};

struct ApiInstance {
    uint32_t instance = 0;
    uint32_t row = kNoRow;
};

struct BlockShape {
    std::vector<SchemaId> schemas;
    std::vector<uint32_t> parents;
    std::vector<uint64_t> apis;
};

struct BlockTable {
    SchemaId table = SchemaId::None;
    uint32_t first = 0;
    uint32_t perCopy = 0;
    uint32_t count = kNoRow;
};

struct PrimBlock {
    uint32_t firstPrim = kNoPrim;
    uint32_t count = 0;
    uint32_t copies = 0;
    uint32_t firstWorld = kGlobalWorld;
    uint32_t worldsPerCopy = 0;
    uint32_t firstSlot = kNoRow;
    uint32_t slotsPerCopy = 0;
    std::vector<BlockTable> tables;

    uint32_t Prim(uint32_t copy, uint32_t index) const {
        return firstPrim + copy * count + index;
    }

    uint32_t PrimCount() const {
        return count * copies;
    }
};

class Prims {
  public:
    explicit Prims(Store& store) : m_store(store) {}

    uint32_t Size() const {
        return uint32_t(m_records.size());
    }

    uint32_t LiveCount() const {
        return m_live;
    }

    uint32_t WorldCount() const {
        return m_worldCount;
    }

    bool Alive(uint32_t prim) const;
    uint32_t Generation(uint32_t prim) const;
    const PrimRecord& At(uint32_t prim) const;
    uint32_t Row(uint32_t prim, SchemaId table) const;
    uint32_t FindRow(uint32_t prim, SchemaId table) const;
    bool HasApi(uint32_t prim, SchemaId api) const;
    std::span<const ApiInstance> Instances(uint32_t prim, SchemaId api) const;
    uint32_t InstanceRow(uint32_t prim, SchemaId api, uint32_t instance) const;

    PrimBlock Add(const BlockShape& shape, std::span<const uint32_t> parents,
                  uint32_t reuseWorld = kGlobalWorld, bool pageAligned = false);
    void CheckRemovable(uint32_t prim) const;
    void Remove(uint32_t prim);
    BlockTable Apply(std::span<const uint32_t> prims, SchemaId api);
    void Unapply(std::span<const uint32_t> prims, SchemaId api);
    BlockTable ApplyInstance(std::span<const uint32_t> prims, SchemaId api, uint32_t instance);
    void UnapplyInstance(std::span<const uint32_t> prims, SchemaId api, uint32_t instance);
    void StageLinks(std::span<const uint32_t> prims, SchemaId api);
    void Reparent(uint32_t prim, uint32_t parent);
    void Renew(uint32_t prim);
    void MarkSession(uint32_t prim);
    bool Session(uint32_t prim) const;

  private:
    void Check(const BlockShape& shape, std::span<const uint32_t> parents) const;
    PrimBlock AllocateOne(SchemaId schema, uint64_t apis);
    PrimBlock AllocateBlock(const BlockShape& shape, uint32_t copies, bool pageAligned);
    void Write(uint32_t prim, const PrimRecord& record, std::span<const uint32_t> rows);
    void StageRecord(uint32_t prim);
    uint32_t AcquireSlots(uint32_t count);
    void Regroup(uint32_t prim, uint64_t apis, SchemaId api, uint32_t row);
    void ReleaseRow(SchemaId table, uint32_t row);
    void SetHead(uint32_t prim, SchemaId api, uint32_t row);

    Store& m_store;
    std::vector<PrimRecord> m_records;
    std::vector<uint32_t> m_slotRows;
    std::vector<uint32_t> m_freePrims;
    std::map<uint32_t, std::vector<uint32_t>> m_freeSlots;
    std::unordered_map<uint64_t, std::vector<ApiInstance>> m_instances;
    uint32_t m_live = 0;
    uint32_t m_worldCount = 0;
};

}
