// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "ernie/algorithms/radix_sort.h"
#include "ernie/bvh/builder.h"
#include "ernie/device/buffer.h"
#include "ernie/device/program.h"
#include "ernie/device/ref.h"
#include "ernie/scene/records/points.h"
#include "ernie/scene/structure/structure.h"
#include "ernie/scene/tables/storage.h"
#include "ernie/scene/tables/table.h"

namespace ernie {

struct PointsStats {
    uint32_t clouds = 0;
    uint32_t built = 0;
    uint32_t refit = 0;
    uint32_t sorted = 0;
    uint32_t points = 0;
    uint32_t capacity = 0;
};

class PointsTable : public RenderTable {
  public:
    PointsTable(Structure& structure, bvh::Builder& builder, uint32_t maxRefits = 16,
                uint32_t pointsPerBox = 1)
        : m_structure(structure), m_builder(builder), m_maxRefits(maxRefits),
          m_pointsPerBox(std::max(pointsPerBox, 1u)) {}

    std::string_view Name() const override {
        return "points";
    }

    std::span<const ColumnRef> Inputs() const override;
    std::span<const TableId> TableOutputs() const override;
    void Describe(RenderTables& tables) const override;
    void Prepare() override;
    void Run(const ChangeSet& changes, TableVersions& versions) override;
    void Hold(std::vector<Ref<const RefCounted>>& held) const override;

    const PointsStats& Stats() const {
        return m_stats;
    }

    const PointsRecord& Record(uint32_t pointsRow) const {
        return m_records.Row(pointsRow);
    }

  private:
    struct Cloud {
        uint32_t prim = kNoPrim;
        uint32_t generation = 0;
        uint32_t pointBasedRow = kNoRow;
        uint32_t row = kNoRow;
        uint32_t count = 0;
        uint32_t capacity = 0;
        uint32_t refits = 0;
        Ref<bvh::Blas> blas;
        Ref<device::Buffer> boxes;
        Ref<device::Buffer> order;
        Ref<device::Buffer> packed;
        ArrayRef bounds{};
        bool update = false;
        bool fresh = false;
    };

    void Place(uint32_t prim);
    void Forget(uint32_t prim);
    void Link(Cloud& cloud);
    void Build(std::vector<uint32_t>& updated);
    device::Program& Kernels();

    Structure& m_structure;
    bvh::Builder& m_builder;
    uint32_t m_maxRefits;
    uint32_t m_pointsPerBox;
    std::unique_ptr<algorithms::RadixSort> m_sort;
    uint32_t m_sortCapacity = 0;
    Ref<device::Buffer> m_keys;
    Ref<device::Buffer> m_values;
    Ref<device::Program> m_program;
    std::unordered_map<uint32_t, Cloud> m_clouds;
    std::unordered_map<uint32_t, uint32_t> m_byPointBasedRow;
    std::unordered_map<uint32_t, uint32_t> m_byRow;
    RecordTable<PointsRecord> m_records{"points records"};
    std::vector<uint32_t> m_relinked;
    PointsStats m_stats;
};

}
